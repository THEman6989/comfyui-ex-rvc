"""Start a separate real ComfyUI CPU server; keep the production queue untouched."""
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix='sampler-pause-boot-', dir=os.environ['TMPDIR']))
for name in ('input', 'output', 'temp', 'user'): (SCRATCH / name).mkdir()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
command = [str(ROOT / 'venv/bin/python'), str(ROOT / 'main.py'), '--cpu', '--listen', '127.0.0.1', '--port', str(port),
           '--disable-auto-launch', '--disable-partner-nodes', '--disable-all-custom-nodes',
           '--whitelist-custom-nodes', 'comfyui-ex-rvc', '--database-url', 'sqlite:///:memory:']
for name in ('input', 'output', 'temp', 'user'): command.extend(['--' + name + '-directory', str(SCRATCH / name)])
with (SCRATCH / 'server.log').open('w') as log:
    process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError(f'ComfyUI exited: {process.returncode}; log {SCRATCH}/server.log')
            try:
                with urllib.request.urlopen(base + '/sampling_pause/status', timeout=1) as response:
                    state = json.load(response)
                break
            except Exception: time.sleep(.25)
        else: raise RuntimeError(f'Boot timeout; log {SCRATCH}/server.log')
        assert state['sampling_active'] is False
        assert state['paused'] is False
        with urllib.request.urlopen(base + '/extensions', timeout=3) as response:
            extensions = json.load(response)
        script = next(p for p in extensions if p.endswith('/pause_button.js'))
        with urllib.request.urlopen(base + script, timeout=3) as response:
            assert b'Amin.SamplerPause' in response.read()
        with urllib.request.urlopen(base + '/object_info', timeout=10) as response:
            info = json.load(response)
        assert 'RVC_Terminal_Node' in info and 'TrueRandomSeed' in info
        print('PASS real ComfyUI boot: pause routes, frontend delivery, existing RVC nodes registered; CPU-only; no workflows submitted')
        print('Smoke artifacts:', SCRATCH)
    finally:
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        print('Isolated server stopped:', process.poll() is not None)
