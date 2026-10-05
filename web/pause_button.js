import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export function buttonView(state) {
    const active = state.sampling_active && state.supported;
    const resume = state.paused || state.pause_requested;
    return {
        disabled: !active,
        icon: state.paused ? "▶" : state.pause_requested ? "■ …" : "■",
        tooltip: !state.sampling_active ? "No active sampling operation" :
            !state.supported ? "This sampler has no verified safe pause boundary" :
            state.paused ? "Resume sampling" :
            state.pause_requested ? "Pause requested — finishing current step (click to cancel pause)" :
            "Pause after current sampling step",
        route: resume ? "/sampling_resume" : "/sampling_pause",
    };
}

export function installPauseButton() {
    const id = "ex-rvc-sampling-pause";
    // Avoid duplicate observers/pollers when extensions are hot-loaded.
    if (window.__exRvcPauseCleanup) window.__exRvcPauseCleanup();
    document.getElementById(id)?.remove();
    const button = document.createElement("button");
    button.id = id;
    button.type = "button";
    button.dataset.testid = "sampling-pause-button";
    button.style.cssText = "min-width:36px;height:32px;border:1px solid var(--border-color,#666);border-radius:6px;background:var(--comfy-menu-bg,#222);color:var(--input-text,#eee);margin-left:4px;padding:4px 8px;cursor:pointer;font-size:16px;flex-shrink:0";
    let state = {}, busy = false, stopped = false, timer;
    function render() {
        const view = buttonView(state);
        button.textContent = view.icon;
        button.title = view.tooltip;
        button.setAttribute("aria-label", view.tooltip);
        button.setAttribute("aria-pressed", String(!!state.paused));
        button.disabled = view.disabled || busy;
        button.style.opacity = button.disabled ? "0.45" : "1";
        button.dataset.state = state.paused ? "paused" : state.pause_requested ? "pending" : state.sampling_active ? "running" : "idle";
    }
    function mount() {
        // Current Vue toolbar plus pre-Vue/legacy menu. Never replace Run/Interrupt.
        const run = document.querySelector('[data-testid="queue-button"]') ||
            document.querySelector('#queue-button') || document.querySelector('#comfy-queue-button');
        if (run?.parentNode && run.nextElementSibling !== button) {
            run.insertAdjacentElement("afterend", button);
        }
    }
    async function refresh() {
        try {
            const response = await api.fetchApi("/sampling_pause/status", { cache: "no-store" });
            if (!response.ok) throw new Error(`Pause status HTTP ${response.status}`);
            state = await response.json();
        } catch (error) {
            state = {}; // Network loss/old backend must not leave a stale Resume control.
        }
        if (!stopped) { render(); mount(); }
    }
    button.addEventListener("click", async () => {
        if (busy || button.disabled) return;
        busy = true; render();
        try {
            const response = await api.fetchApi(buttonView(state).route, { method: "POST" });
            if (!response.ok) throw new Error(`Pause command HTTP ${response.status}`);
            state = await response.json();
        } catch (error) {
            console.warn("[SamplerPause]", error);
        } finally {
            busy = false;
            if (!stopped) { render(); await refresh(); }
        }
    });
    const observer = new MutationObserver(mount);
    observer.observe(document.body, { childList: true, subtree: true });
    async function poll() {
        await refresh();
        if (!stopped) timer = setTimeout(poll, 750);
    }
    const onStatus = () => { if (!stopped) refresh(); };
    api.addEventListener("executing", onStatus);
    api.addEventListener("execution_error", onStatus);
    api.addEventListener("execution_interrupted", onStatus);
    function cleanup() {
        stopped = true; clearTimeout(timer); observer.disconnect(); button.remove();
        for (const event of ["executing", "execution_error", "execution_interrupted"]) api.removeEventListener(event, onStatus);
        window.removeEventListener("pagehide", cleanup);
    }
    window.__exRvcPauseCleanup = cleanup;
    window.addEventListener("pagehide", cleanup);
    render(); mount(); poll();
    return cleanup;
}

app.registerExtension({ name: "Amin.SamplerPause", setup: installPauseButton });
