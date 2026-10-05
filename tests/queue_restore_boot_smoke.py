"""Real CPU ComfyUI restore/restart test with an isolated browser proxy.

Only 8x8 EmptyImage/SaveImage workflows are submitted; production 8188 is never used.
"""
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT=pathlib.Path(__file__).resolve().parents[3]
PACKAGE=pathlib.Path(__file__).resolve().parents[1]
SCRATCH=pathlib.Path(tempfile.mkdtemp(prefix='queue-restore-smoke-',dir=os.environ['TMPDIR']))
for name in ('input','output','temp','user'): (SCRATCH/name).mkdir()
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
base=f'http://127.0.0.1:{port}'
command=[str(ROOT/'venv/bin/python'),str(ROOT/'main.py'),'--cpu','--listen','127.0.0.1','--port',str(port),
         '--disable-auto-launch','--disable-partner-nodes','--disable-all-custom-nodes',
         '--whitelist-custom-nodes','comfyui-ex-rvc','--database-url','sqlite:///:memory:']
for name in ('input','output','temp','user'): command.extend(['--'+name+'-directory',str(SCRATCH/name)])
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self): self.proxy()
    def do_GET(self):
        if self.path=='/': data=b'<html><body><div><button data-testid="queue-button">Run</button><button id="interrupt">Interrupt</button></div></body></html>'; mime='text/html'
        elif self.path.endswith('app.js'): data=b'export const app={registerExtension(){}}'; mime='text/javascript'
        elif self.path.endswith('api.js'): data=b'export const api={fetchApi:(...args)=>fetch(...args),clientId:"isolated-test"}'; mime='text/javascript'
        elif self.path.endswith('queue_snapshot.js'): data=(PACKAGE/'web/queue_snapshot.js').read_bytes(); mime='text/javascript'
        else: self.proxy(); return
        self.send_response(200); self.send_header('Content-Type',mime); self.end_headers(); self.wfile.write(data)
    def proxy(self):
        if not (self.path=='/queue' or self.path=='/prompt' or self.path.startswith('/history/')): self.send_error(404); return
        data=self.rfile.read(int(self.headers.get('Content-Length',0))) if self.command=='POST' else None
        request=urllib.request.Request(base+self.path,data=data,headers={'Content-Type':'application/json'},method=self.command)
        try:
            response=urllib.request.urlopen(request,timeout=30)
        except urllib.error.HTTPError as error: response=error
        with response:
            self.send_response(response.status); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(response.read())
proxy=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=proxy.serve_forever,daemon=True).start()
process=None
log=None
def stop():
    global process, log
    if process:
        process.terminate()
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        process=None
    if log: log.close(); log=None

def start(phase):
    global process,log
    log=(SCRATCH/f'server-{phase}.log').open('w')
    process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        if process.poll() is not None: raise RuntimeError(f'Server exited; logs {SCRATCH}')
        try:
            with urllib.request.urlopen(base+'/extensions',timeout=1) as response: extensions=json.load(response)
            assert any(p.endswith('/queue_snapshot.js') for p in extensions)
            return
        except (OSError,AssertionError): time.sleep(.25)
    raise RuntimeError(f'Server boot timed out; logs {SCRATCH}')

try:
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(accept_downloads=True)
        page.goto(f'http://127.0.0.1:{proxy.server_port}')
        page.evaluate('''async()=>{
            const m=await import('/extensions/ex-rvc/queue_snapshot.js'); window.m=m;
            const graph=n=>({'1':{class_type:'EmptyImage',inputs:{width:8,height:8,batch_size:1,color:n}},'2':{class_type:'SaveImage',inputs:{images:['1',0],filename_prefix:'queue_restore_smoke'}}});
            window.snapshot=m.makeSnapshot({queue_running:[[0,'source-running',graph(1),{extra_pnginfo:{workflow:{nodes:[]}}},['2']]],queue_pending:[[1,'source-pending',graph(2),{},['2']]]},'restart-smoke');
            m.installQueueButtons();
        }''')
        for phase in (1,2):
            start(phase)
            result=page.evaluate('async()=>await window.m.restoreSnapshot(window.snapshot)')
            assert result=={'restored':2,'skipped':0},result
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                done=page.evaluate('''async()=>{const journal=JSON.parse(localStorage.getItem('ex-rvc-queue-restore:restart-smoke')); const results=await Promise.all(Object.values(journal).map(async j=>{const h=await window.m.requestJson('/history/'+j.prompt_id); return h[j.prompt_id]?.status?.status_str==='success'}));return results.every(Boolean)}''')
                if done: break
                time.sleep(.1)
            else: raise AssertionError(f'Image jobs did not complete; logs {SCRATCH}')
            again=page.evaluate('async()=>await window.m.restoreSnapshot(window.snapshot)')
            assert again=={'restored':0,'skipped':2},again
            if phase==1: stop()
        # Actual toolbar download, then real input-file restore against the actual server.
        # Mock only GET /queue during SAVE to export the known source snapshot.
        page.route('**/queue',lambda route:route.fulfill(json={'queue_running':[[0,'download-running',{'1':{'class_type':'EmptyImage','inputs':{'width':8,'height':8,'batch_size':1,'color':3}},'2':{'class_type':'SaveImage','inputs':{'images':['1',0],'filename_prefix':'queue_restore_download'}}},{},['2']]],'queue_pending':[]}))
        with page.expect_download() as info: page.click('#ex-rvc-save-queue')
        path=SCRATCH/'toolbar-backup.json'; info.value.save_as(path)
        page.unroute('**/queue')
        saved=json.loads(path.read_text()); assert len(saved['jobs'])==1 and saved['jobs'][0]['was_running']
        page.on('dialog',lambda dialog:dialog.accept())
        page.set_input_files('#ex-rvc-queue-backup input',str(path))
        page.wait_for_function("document.querySelector('[role=status]').textContent.includes('1 geladen')",timeout=30000)
        assert page.locator('#interrupt').count()==1
        # Validate the actual production snapshot, without restoring it anywhere.
        backups=sorted((ROOT/'user/queue_backups').glob('comfyui-queue-*.json'))
        if backups:
            production=json.loads(backups[-1].read_text())
            count=page.evaluate('(s)=>window.m.validateSnapshot(s).jobs.length',production)
            print('Production backup schema validated (read-only):',count,'jobs')
        browser.close()
        print('PASS real CPU ComfyUI: two image jobs restored, completed, deduplicated, then restored after actual restart; toolbar download/import works')
        print('Smoke artifacts:',SCRATCH)
finally:
    stop(); proxy.shutdown()
