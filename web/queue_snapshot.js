import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FORMAT = "comfyui-ex-rvc-queue-v1";
const MAX_JOBS = 10000;
const MAX_BYTES = 100 * 1024 * 1024;
const clone = value => JSON.parse(JSON.stringify(value));
const object = value => value && typeof value === "object" && !Array.isArray(value);

export function validateSnapshot(snapshot) {
    if (!object(snapshot) || snapshot.format !== FORMAT || typeof snapshot.snapshot_id !== "string" ||
        !snapshot.snapshot_id || snapshot.snapshot_id.length > 128 || !Array.isArray(snapshot.jobs) || snapshot.jobs.length > MAX_JOBS) {
        throw new Error("Keine gültige RVC-Queue-Sicherung.");
    }
    const seen = new Set();
    for (const job of snapshot.jobs) {
        if (!object(job) || typeof job.source_prompt_id !== "string" || !job.source_prompt_id || seen.has(job.source_prompt_id) ||
            typeof job.was_running !== "boolean" || !object(job.prompt) || !Object.keys(job.prompt).length || !object(job.extra_data) ||
            !Array.isArray(job.outputs_to_execute) || job.outputs_to_execute.some(id => typeof id !== "string")) {
            throw new Error("Ungültiger oder doppelter Job in der Queue-Sicherung.");
        }
        for (const node of Object.values(job.prompt)) {
            if (!object(node) || typeof node.class_type !== "string" || !object(node.inputs)) throw new Error("Ungültiger API-Workflow.");
        }
        seen.add(job.source_prompt_id);
    }
    return snapshot;
}

export function makeSnapshot(queue, snapshotId = crypto.randomUUID(), createdAt = new Date().toISOString()) {
    if (!object(queue) || !Array.isArray(queue.queue_running) || !Array.isArray(queue.queue_pending)) throw new Error("Ungültige Queue-Antwort.");
    const jobs = [];
    for (const [rows, running] of [[queue.queue_running, true], [queue.queue_pending, false]]) {
        // ComfyUI returns the pending heap, NOT a sorted execution list.
        for (const row of [...rows].sort((a, b) => a[0] - b[0] || String(a[1]).localeCompare(String(b[1])))) {
            if (!Array.isArray(row) || row.length < 5 || !Number.isFinite(row[0])) throw new Error("Ungültiger Queue-Eintrag.");
            const extra = clone(row[3] || {});
            delete extra.auth_token_comfy_org;
            delete extra.api_key_comfy_org;
            delete extra.client_id;
            jobs.push({ source_prompt_id: row[1], was_running: running, prompt: clone(row[2]),
                extra_data: extra, outputs_to_execute: clone(row[4]) });
            // Deliberately never copy the 6th queue tuple member (credentials).
        }
    }
    return validateSnapshot({ format: FORMAT, snapshot_id: snapshotId, created_at: createdAt, jobs });
}

export async function requestJson(route, options = {}) {
    const response = await api.fetchApi(route, { cache: "no-store", ...options });
    let body;
    try { body = await response.json(); } catch { throw new Error(`HTTP ${response.status}: ungültige Serverantwort`); }
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${body.error?.message || JSON.stringify(body.error || body)}`);
    return body;
}

export async function restoreSnapshot(snapshot, transport = requestJson, storage = localStorage, clientId = api.clientId, onProgress = () => {}) {
    validateSnapshot(snapshot);
    const key = `ex-rvc-queue-restore:${snapshot.snapshot_id}`;
    let journal;
    try { journal = JSON.parse(storage.getItem(key) || '{}'); } catch { throw new Error("Wiederherstellungsprotokoll beschädigt; nicht erneut einreihen."); }
    if (!object(journal)) throw new Error("Ungültiges Wiederherstellungsprotokoll.");
    const persist = () => storage.setItem(key, JSON.stringify(journal));
    // A persistent generated UUID is reserved BEFORE submission. Lost HTTP responses
    // can therefore be reconciled against the queue/history without making duplicates.
    let restored = 0, skipped = 0;
    for (const job of snapshot.jobs) {
        let saved = journal[job.source_prompt_id];
        // Never trust accepted alone: a server restart may have lost this queued job.
        // Reconcile its reserved ID with the current queue/history every time.
        const live = await transport('/queue');
        const rows = [...(live.queue_running || []), ...(live.queue_pending || [])];
        const existing = rows.some(row => row[1] === job.source_prompt_id || row[1] === saved?.prompt_id ||
            (row[3]?.ex_rvc_queue_backup?.snapshot_id === snapshot.snapshot_id &&
             row[3]?.ex_rvc_queue_backup?.source_prompt_id === job.source_prompt_id));
        const originalHistory = await transport(`/history/${encodeURIComponent(job.source_prompt_id)}`);
        const completed = originalHistory[job.source_prompt_id]?.status?.completed === true &&
            originalHistory[job.source_prompt_id]?.status?.status_str === 'success';
        if (existing || completed) { skipped++; onProgress({ restored, skipped }); continue; }
        if (saved?.prompt_id) {
            const recovered = await transport(`/history/${encodeURIComponent(saved.prompt_id)}`);
            if (recovered[saved.prompt_id]) {
                saved.accepted = true; persist(); skipped++; onProgress({ restored, skipped }); continue;
            }
        } else {
            saved = { prompt_id: crypto.randomUUID(), accepted: false };
            Object.defineProperty(journal, job.source_prompt_id, { value: saved, enumerable: true, writable: true, configurable: true });
            persist(); // Refuse to POST if browser persistence is unavailable/full.
        }
        const extra = clone(job.extra_data);
        delete extra.auth_token_comfy_org;
        delete extra.api_key_comfy_org;
        delete extra.client_id;
        extra.ex_rvc_queue_backup = { snapshot_id: snapshot.snapshot_id, source_prompt_id: job.source_prompt_id, was_running: job.was_running };
        const payload = { prompt: clone(job.prompt), extra_data: extra, partial_execution_targets: clone(job.outputs_to_execute),
            prompt_id: saved.prompt_id, front: false };
        if (clientId) payload.client_id = clientId;
        const response = await transport('/prompt', { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        if (response.prompt_id !== saved.prompt_id) throw new Error("Server hat die Wiederherstellungs-ID nicht bestätigt. Abgebrochen.");
        saved.accepted = true; persist(); restored++;
        onProgress({ restored, skipped });
    }
    return { restored, skipped };
}

export function installQueueButtons() {
    window.__exRvcQueueCleanup?.();
    const save = document.createElement('button');
    const load = document.createElement('button');
    const input = document.createElement('input');
    const group = document.createElement('span');
    const status = document.createElement('span');
    group.id = 'ex-rvc-queue-backup';
    group.style.cssText = 'display:inline-flex;gap:4px;align-items:center;margin-left:4px';
    save.id = 'ex-rvc-save-queue'; save.textContent = 'Queue speichern';
    load.id = 'ex-rvc-load-queue'; load.textContent = 'Queue laden';
    for (const button of [save, load]) {
        button.type = 'button';
        button.style.cssText = 'height:32px;border:1px solid var(--border-color,#666);border-radius:6px;background:var(--comfy-menu-bg,#222);color:var(--input-text,#eee);padding:4px 8px;cursor:pointer;white-space:nowrap';
    }
    save.title = 'Wartende und laufende Jobs als JSON sichern. Die Queue läuft unverändert weiter.';
    load.title = 'Gesicherte Jobs anhängen. Der zuvor laufende Job startet von vorn.';
    input.type = 'file'; input.accept = '.json,application/json'; input.hidden = true;
    status.setAttribute('role', 'status'); status.style.fontSize = '11px';
    group.append(save, load, status, input);
    let busy = false, stopped = false;
    const setBusy = value => { busy = value; save.disabled = load.disabled = value; };
    function mount() {
        const run = document.querySelector('[data-testid="queue-button"]') || document.querySelector('#queue-button') || document.querySelector('#comfy-queue-button');
        const pause = document.querySelector('#ex-rvc-sampling-pause');
        const anchor = pause?.parentNode === run?.parentNode ? pause : run;
        if (anchor?.parentNode && anchor.nextElementSibling !== group) anchor.insertAdjacentElement('afterend', group);
    }
    save.addEventListener('click', async () => {
        if (busy) return;
        setBusy(true); status.textContent = 'Sichern …';
        try {
            const snapshot = makeSnapshot(await requestJson('/queue'));
            const json = JSON.stringify(snapshot, null, 2);
            if (new Blob([json]).size > MAX_BYTES) throw new Error('Queue-Sicherung überschreitet 100 MB.');
            const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
            const link = document.createElement('a'); link.href = url;
            link.download = `comfyui-queue-${snapshot.created_at.replace(/[:.]/g, '-')}.json`;
            document.body.append(link); link.click(); link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 60000);
            status.textContent = `${snapshot.jobs.length} Jobs gesichert`;
        } catch (error) { status.textContent = 'Fehler'; window.alert(`Queue nicht gesichert: ${error.message}`); }
        finally { if (!stopped) setBusy(false); }
    });
    load.addEventListener('click', () => { if (!busy) { input.value = ''; input.click(); } });
    input.addEventListener('change', async () => {
        const file = input.files?.[0]; if (!file || busy) return;
        setBusy(true); status.textContent = 'Prüfen …';
        try {
            if (file.size > MAX_BYTES) throw new Error('Sicherung ist größer als 100 MB.');
            const snapshot = validateSnapshot(JSON.parse(await file.text()));
            const running = snapshot.jobs.filter(job => job.was_running).length;
            if (!window.confirm(`${snapshot.jobs.length} gesicherte Jobs wieder einreihen?\n${running} zuvor laufende Jobs starten von vorn.\nVorhandene Queue bleibt erhalten; bekannte Jobs werden übersprungen.\nJobs können sofort anlaufen. Nur vertrauenswürdige Sicherungen laden.`)) return;
            const progress = counts => { status.textContent = `${counts.restored} geladen · ${counts.skipped} übersprungen`; };
            // Serialize restores across tabs of this browser when Web Locks exists.
            const operation = () => restoreSnapshot(snapshot, requestJson, localStorage, api.clientId, progress);
            const result = navigator.locks ? await navigator.locks.request('ex-rvc-queue-restore', operation) : await operation();
            progress(result);
        } catch (error) {
            window.alert(`Wiederherstellung gestoppt: ${error.message}\nBereits bestätigte Jobs bleiben erhalten. Dieselbe Datei kann erneut geladen werden.`);
        } finally { if (!stopped) setBusy(false); }
    });
    const observer = new MutationObserver(mount);
    observer.observe(document.body, { childList: true, subtree: true });
    function cleanup() { stopped = true; observer.disconnect(); group.remove(); window.removeEventListener('pagehide', cleanup); }
    window.__exRvcQueueCleanup = cleanup;
    window.addEventListener('pagehide', cleanup);
    mount();
}

app.registerExtension({ name: 'Amin.QueueSnapshot', setup: installQueueButtons });
