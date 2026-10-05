"""Real Chromium DOM smoke with isolated HTTP API; never targets the live queue."""
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
state = dict(sampling_active=False, supported=False, paused=False, pause_requested=False)
posts = []
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == '/':
            content = b'<html><body><div id="toolbar"><button data-testid="queue-button">Run</button><button id="interrupt">Interrupt</button></div><script type="module" src="/extensions/ex-rvc/pause_button.js"></script></body></html>'
            mime = 'text/html'
        elif self.path.endswith('app.js'):
            content = b'export const app = {registerExtension(ext) { window.testExtension = ext; ext.setup(); }};'
            mime = 'text/javascript'
        elif self.path.endswith('api.js'):
            content = b'export const api = Object.assign(new EventTarget(), {fetchApi(path, options) { return fetch(path, options); }});'
            mime = 'text/javascript'
        elif self.path.endswith('pause_button.js'):
            content = (ROOT / 'web/pause_button.js').read_bytes()
            mime = 'text/javascript'
        elif self.path.startswith('/sampling_pause/status'):
            content = json.dumps(state).encode()
            mime = 'application/json'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type', mime); self.end_headers(); self.wfile.write(content)
    def do_POST(self):
        posts.append(self.path)
        if self.path == '/sampling_pause': state['pause_requested'] = True
        elif self.path == '/sampling_resume': state.update(paused=False, pause_requested=False)
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps(state).encode())

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox','--disable-gpu'])
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}')
        button = page.locator('#ex-rvc-sampling-pause')
        button.wait_for()
        assert button.is_disabled()
        assert page.evaluate('document.querySelector("[data-testid=queue-button]").nextElementSibling.id') == 'ex-rvc-sampling-pause'
        state.update(sampling_active=True, supported=True)
        page.wait_for_function('!document.querySelector("#ex-rvc-sampling-pause").disabled')
        button.click()
        page.wait_for_function('document.querySelector("#ex-rvc-sampling-pause").dataset.state === "pending"')
        assert posts == ['/sampling_pause']
        state.update(paused=True, pause_requested=False)
        page.wait_for_function('document.querySelector("#ex-rvc-sampling-pause").textContent === "▶"')
        button.click()
        page.wait_for_function('document.querySelector("#ex-rvc-sampling-pause").dataset.state === "running"')
        assert posts[-1] == '/sampling_resume'
        page.evaluate('document.querySelector("#toolbar").innerHTML = `<button data-testid="queue-button">Run</button><button id="interrupt">Interrupt</button>`')
        page.wait_for_function('document.querySelector("[data-testid=queue-button]").nextElementSibling.id === "ex-rvc-sampling-pause"')
        page.evaluate('() => { window.testExtension.setup(); }')
        page.wait_for_function('document.querySelectorAll("#ex-rvc-sampling-pause").length === 1')
        assert button.count() == 1, page.locator('body').inner_html()
        assert page.locator('#interrupt').count() == 1
        assert page.locator('[data-testid=queue-button]').count() == 1
        assert not errors, errors
        browser.close()
        print('PASS Chromium: adjacent placement, idle disable, pending, resume POST, rerender, no duplicates, Run/Interrupt preserved')
finally:
    server.shutdown()
