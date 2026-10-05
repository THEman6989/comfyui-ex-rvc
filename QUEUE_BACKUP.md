# Queue speichern und nach Neustart wieder einreihen

## Bedienung

1. Browser neu laden (gegebenenfalls Strg+F5). **Kein ComfyUI-Neustart nötig:** die neue JavaScript-Erweiterung wird vom laufenden Server bereits ausgeliefert.
2. **Queue speichern** neben Run lädt eine JSON-Datei herunter. Sie enthält alle laufenden und wartenden Jobs zum Zeitpunkt der Server-Abfrage.
3. Nach einem Server-Neustart **Queue laden** wählen und die JSON-Datei öffnen. Die Bestätigung zeigt Anzahl und Neustart-Verhalten.
4. Wiederherstellung hängt Jobs an die vorhandene Queue an. Zuerst die zuvor laufenden Jobs, dann die wartenden Jobs in ihrer ursprünglichen Prioritätsreihenfolge. Jobs können sofort anlaufen.

**Ein zuvor laufender Job startet von vorn**, mit dem gesicherten API-Workflow und dessen Einstellungen. Das ist kein dauerhafter Checkpoint eines Samplers: Tensoren, Modellzustand und bisherige Sampling-Schritte werden nicht gespeichert. Die separate Sampling-Pause hält nur innerhalb desselben Serverprozesses an.

## Enthalten

- API-Prompt mit den bereits eingereihten Inputs (inklusive festen Seedwerten)
- Workflow-Metadaten (`extra_pnginfo`), falls im Queue-Eintrag vorhanden
- gewünschte Output-Nodes / Teil-Ausführungsziele
- Kennzeichnung der laufenden Jobs, Original-Prompt-ID und eindeutige Sicherungs-ID

Nicht enthalten: Modelle, Input-Dateien, bereits erzeugte Bilder, RAM/VRAM oder Browser-Verbindungen. Referenzierte Modelle und Dateien müssen nach dem Neustart weiterhin vorhanden sein. Nodes, die bei jeder Ausführung neue Zufallswerte ermitteln, können trotz unverändertem API-Prompt neue Ergebnisse liefern.

ComfyUI-Authentifizierung (`auth_token_comfy_org`, `api_key_comfy_org`) und sensitive sechste Queue-Tupel-Felder werden nicht exportiert. Provider-Anmeldungen können nach Neustart erneut nötig sein. **Andere Geheimnisse direkt in Workflow-Inputs werden nicht automatisch entfernt. Sicherungen nicht unkontrolliert teilen.** Nur vertrauenswürdige Dateien importieren: Workflows können installierte Nodes ausführen.

## Wiederholter Import und Fehler

- Bereits laufende oder wartende Originaljobs werden übersprungen.
- Seit der Sicherung erfolgreich abgeschlossene Originaljobs werden übersprungen, wenn ihre History noch verfügbar ist.
- Jede Übermittlung bekommt vorab eine persistierte Prompt-ID im Browser. Noch vorhandene wiederhergestellte Jobs werden über Queue und History erkannt.
- Selbst wenn ein vorheriger Import vom Browser bestätigt wurde, wird der Job nach einem Server-Neustart erneut eingereiht, falls er weder Queue noch History enthält.
- Bei fehlenden Nodes, HTTP-Fehlern oder ungültigen Workflows stoppt der Import. Bereits eingereihte Jobs bleiben bestehen. Dieselbe Datei erneut laden, um fortzufahren.
- Im selben Browser verhindert ein Web Lock parallele Imports in mehreren Tabs (wenn verfügbar).
- Die Duplikaterkennung ist kein serverweiter transaktionaler Import: andere Browser, gelöschter Browserspeicher, gelöschte History und gleichzeitige externe Queue-Manipulation können ihre Grenzen überschreiten.
- Maximal 100 MB und 10.000 Jobs pro Datei. Bei nicht verfügbarem Browser-Persistenzspeicher wird vor der Übermittlung abgebrochen.

Die Live-Queue wird weder geleert noch pausiert, und keine vorhandenen Jobs werden gelöscht. Dies ist eine manuelle Momentaufnahme, kein automatisches Backup zukünftiger Queue-Änderungen.

## Sicherung ohne Browser

Im Custom-Node-Verzeichnis:

```sh
python3 queue_backup.py --url http://127.0.0.1:8188 --output-dir ../../user/queue_backups
```

Der Helper nutzt nur GET `/queue`, schreibt eine neue Datei mit Modus 0600, überschreibt nichts und liest die Datei zur Verifikation zurück. Das Format ist mit **Queue laden** kompatibel.

## Verifikation

```sh
python3 tests/test_queue_backup.py
uv run --no-project --with playwright --python ../../venv/bin/python tests/queue_snapshot_browser.py
uv run --no-project --with playwright --python ../../venv/bin/python tests/queue_restore_boot_smoke.py
```

Der Browser-Test prüft Reihenfolge, laufende Jobs, feste Seeds, Metadaten, Credential-Ausschluss, Duplikate, partiellen Fehler/retry, verlorene Jobs nach Neustart, aktive Originaljobs, Dateivalidierung und Toolbar.

Der echte Boot-Smoke startet ausschließlich separate CPU-ComfyUI-Prozesse auf einem freien Port, mit eigenen Input-/Output-/User-/Temp-Verzeichnissen und ohne Zugriff auf die Produktionsqueue. Er erzeugt 8×8-Testbilder, bestätigt die History, stoppt den isolierten Server, startet ihn erneut und prüft den erneuten Import. Zusätzlich werden Toolbar-Download und Datei-Upload geprüft. Testartefakte bleiben im Hermes-Scratch-Verzeichnis erhalten.
