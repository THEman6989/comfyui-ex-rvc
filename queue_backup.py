"""Save a running ComfyUI queue without restarting it (stdlib, no CUDA).

Restoration uses the browser's Queue laden button and normal /prompt validation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import uuid

def make_snapshot(queue):
    jobs=[]
    for key,running in (('queue_running',True),('queue_pending',False)):
        for row in sorted(queue[key], key=lambda item:(item[0], item[1])):
            extra=copy.deepcopy(row[3])
            for key in ('client_id','auth_token_comfy_org','api_key_comfy_org'):
                extra.pop(key,None)
            jobs.append(dict(source_prompt_id=row[1],was_running=running,prompt=copy.deepcopy(row[2]),
                             extra_data=extra,outputs_to_execute=copy.deepcopy(row[4])))
    return dict(format='comfyui-ex-rvc-queue-v1',snapshot_id=str(uuid.uuid4()),
                created_at=datetime.now(timezone.utc).isoformat(),jobs=jobs)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8188')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    with urllib.request.urlopen(args.url.rstrip('/')+'/queue',timeout=30) as response:
        snapshot=make_snapshot(json.load(response))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir / ('comfyui-queue-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+snapshot['snapshot_id'][:8]+'.json')
    # New unique private file; never overwrite another backup or save credentials.
    data=json.dumps(snapshot,ensure_ascii=False,indent=2).encode()
    if len(data)>100*1024*1024: raise ValueError('Snapshot exceeds browser import limit of 100MB')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    with path.open() as handle: verified=json.load(handle)
    assert verified==snapshot
    print(json.dumps(dict(path=str(path),jobs=len(snapshot['jobs']),running=sum(j['was_running'] for j in snapshot['jobs']),bytes=len(data))))

if __name__=='__main__': main()
