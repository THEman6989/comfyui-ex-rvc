"""Queue backup behavior tests in Chromium; never writes to production ComfyUI."""
import pathlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parents[1]
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        if self.path == '/': data=b'<html><body></body></html>'; mime='text/html'
        elif self.path.endswith('app.js'): data=b'export const app={registerExtension(){}}'; mime='text/javascript'
        elif self.path.endswith('api.js'): data=b'export const api={fetchApi:fetch,clientId:"new-client"}'; mime='text/javascript'
        elif self.path.endswith('queue_snapshot.js') and (ROOT/'web/queue_snapshot.js').exists(): data=(ROOT/'web/queue_snapshot.js').read_bytes(); mime='text/javascript'
        else: self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type',mime); self.end_headers(); self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
    assert (ROOT/'web/queue_snapshot.js').exists(), 'Queue snapshot extension missing'
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(); page.goto(f'http://127.0.0.1:{server.server_port}')
        result=page.evaluate('''async () => {
          const m=await import('/extensions/ex-rvc/queue_snapshot.js');
          const assert=(v,msg)=>{if(!v)throw Error(msg)};
          const row=(n,id)=>[n,id,{"1":{class_type:"SaveImage",inputs:{seed:42}}},{client_id:"old",extra_pnginfo:{workflow:{nodes:[]}},auth_token_comfy_org:"SECRET"},["1"],{api_key_comfy_org:"SECRET"}];
          const q={queue_running:[row(0,'running')],queue_pending:[row(3,'third'),row(1,'first'),row(2,'second')]};
          const snap=m.makeSnapshot(q,'snapshot-1','2026-10-05T10:00:00Z');
          assert(snap.jobs.map(j=>j.source_prompt_id).join(',')==='running,first,second,third','heap must be sorted');
          assert(snap.jobs[0].was_running,'running marker');
          assert(!JSON.stringify(snap).includes('SECRET'),'sensitive fields exported');
          assert(snap.jobs[0].extra_data.extra_pnginfo.workflow,'workflow metadata lost');
          const posts=[]; const live={queue_running:[],queue_pending:[]}; const history={};
          const transport=async (route,options={})=>{
            if(route==='/queue')return live;
            if(route.startsWith('/history/'))return {};
            if(route==='/prompt') {const body=JSON.parse(options.body); posts.push(body); live.queue_pending.push(row(10+posts.length,body.prompt_id)); return {prompt_id:body.prompt_id};}
            throw Error(route);
          };
          const journal=new Map(); const store={getItem:k=>journal.get(k)||null,setItem:(k,v)=>journal.set(k,v)};
          let r=await m.restoreSnapshot(snap,transport,store,'new-client');
          assert(r.restored===4,'not all jobs restored');
          assert(posts[0].client_id==='new-client','old WS client');
          assert(posts[0].partial_execution_targets[0]==='1','output targets lost');
          assert(posts[0].prompt['1'].inputs.seed===42,'seed changed');
          assert(posts[0].extra_data.ex_rvc_queue_backup.source_prompt_id==='running','restore lineage');
          r=await m.restoreSnapshot(snap,transport,store,'new-client'); assert(r.restored===0&&r.skipped===4,'duplicate restore');
          const oldIds=posts.slice(0,4).map(p=>p.prompt_id);
          live.queue_pending=[]; // A restart loses queued jobs despite accepted journal entries.
          r=await m.restoreSnapshot(snap,transport,store,'new-client');
          assert(r.restored===4,'restart-lost accepted jobs not restored');
          assert(posts.slice(-4).every((p,i)=>p.prompt_id===oldIds[i]),'retry must reuse reserved IDs');
          const partial=m.makeSnapshot(q,'snapshot-partial'); let count=0;
          const flaky=async(route,opts)=>{if(route==='/prompt' && ++count===2)throw Error('validation failed'); return transport(route,opts)};
          try{await m.restoreSnapshot(partial,flaky,store,'new-client'); throw Error('expected stop')}catch(e){assert(e.message.includes('validation failed'),'wrong failure')}
          assert(count===2,'must stop on failure');
          const before=posts.length; r=await m.restoreSnapshot(partial,transport,store,'new-client'); assert(r.restored===3,'partial retry duplicated first job');
          const activeTransport=async(route,opts)=>route==='/queue'?{queue_running:[row(0,'running')],queue_pending:[]}:transport(route,opts);
          const activeSnap=m.makeSnapshot(q,'snapshot-active'); r=await m.restoreSnapshot(activeSnap,activeTransport,store,'new-client'); assert(r.skipped===1,'original live job duplicated');
          try {m.validateSnapshot({format:'garbage',jobs:[]});throw Error('expected reject')}catch(e){assert(e.message!=='expected reject','bad file accepted')}
          m.installQueueButtons(); document.body.insertAdjacentHTML('beforeend','<div><button data-testid="queue-button">Run</button><button id="interrupt">Interrupt</button></div>');
          await new Promise(r=>setTimeout(r,50));
          assert(document.querySelectorAll('#ex-rvc-save-queue').length===1,'Save mount');
          assert(document.querySelectorAll('#ex-rvc-load-queue').length===1,'Restore mount');
          m.installQueueButtons(); assert(document.querySelectorAll('#ex-rvc-save-queue').length===1,'duplicate mount');
          assert(document.querySelector('#interrupt'),'Interrupt removed');
          return 'PASS snapshot order, running job, metadata, seed, secrets, append, dedup, partial retry, active skip, invalid file, toolbar';
        }''')
        print(result); browser.close()
finally: server.shutdown()
