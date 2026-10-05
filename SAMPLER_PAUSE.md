# Quick Sampling Pause / Resume

Implemented inside `comfyui-ex-rvc`; requires a ComfyUI restart to load Python hooks.
Do not restart while a production job or pending queue must be preserved. Browser
reload alone cannot activate the backend in an already-running process.

## Controls

A square button is inserted directly after Run. Click it to request a pause.
The current solver iteration finishes, including predictor/corrector forwards,
noise addition and multistep-history updates. Pending shows `■ …`; paused shows
`▶`. Click again to resume the same live Python invocation. Clicking pending
cancels the request. Run and Interrupt are not replaced.

No latent serialization, sampling reconstruction, model unloading, tensor
migration, process suspension or CPU busy-loop is used. CUDA synchronization
runs only when a pause is requested and only for existing primary contexts.
Other processes / parallel sampling invocations are not paused: globally idle
GPU utilization is only expected when this is the sole GPU workload.

## Supported path and limits

Hooks wrap `comfy.samplers.KSAMPLER.sample`, shared by standard KSampler,
SamplerCustom and SamplerCustomAdvanced. Existing preview callbacks and their
arguments are passed through untouched. **Callbacks are not the pause point**:
for example, Heun's callback precedes its second model forward. The pause gate
is in a thread-local wrapper of the standard k-diffusion `trange` iterator,
after the preceding loop body returns and before the next iteration starts.
The final iteration finishes normally without waiting for a nonexistent next step.

An explicit allowlist covers the standard step-loop solvers, including Euler,
Heun, DPM++ 2M and `res_multistep` (the H3 workflow's selected solver). The
implementation was inspected against the installed ComfyUI 0.38.0 sources.
Model code is not changed: native Wan/H3 workflows using this stack use the
same hook. Actual Wan/H3 GPU workflow equivalence remains unverified.

Not supported in this version: `dpm_fast`, `dpm_adaptive`, UniPC, AR-video
block loops, and third-party samplers that bypass/replace this standard stack.
Unknown KSAMPLER functions report `supported: false` and reject pause requests.
Samplers bypassing KSAMPLER do not expose an active controllable invocation.
Do not claim universal support for arbitrary custom-node sampling loops.

## Safety

Normal interrupt handling sets its existing interrupt flag before waking the
paused worker, avoiding a resume-before-interrupt race. A 200ms Event wait
also checks normal interruption, covering callers holding an older interrupt
function reference. The pause state is cleared in the invocation's `finally`,
on queue clear and on aiohttp shutdown / process exit. An interrupt prevents
new pause requests for that invocation. A parallel invocation cannot reset or
block the owner of an existing pause.

## API

- `POST /sampling_pause`: request pause; HTTP 409 if idle/unsupported/interrupted.
- `POST /sampling_resume`: resume or cancel a pending pause.
- `GET /sampling_pause/status`: `paused`, `pause_requested`, `sampling_active`,
  `supported`, `sampler`, `completed_steps`, `total_steps`.

The frontend polls every 750ms; HTTP and WebSocket processing remain unblocked.
No Deep Pause/offload mode is implemented.

## Verification

Run from this custom-node directory, with `COMFY` set to the owning ComfyUI root:

```sh
COMFY=/home/amin/experi/sabilitymatrix/Data/Packages/ComfyUI_nacked
SAMPLER_PAUSE_HOLD=60 CUDA_VISIBLE_DEVICES='' "$COMFY/venv/bin/python" tests/test_sampler_pause.py
"$COMFY/venv/bin/python" tests/test_sampling_pause_api.py
uv run --no-project --with playwright --python "$COMFY/venv/bin/python" tests/pause_button_browser.py
python3 tests/sampler_pause_boot_smoke.py
```

Verified:

- 8 controller / real sampling-code tests and 1 HTTP API test pass.
- Actual KSAMPLER execution with small CPU tensors and a deterministic toy
  denoiser: Euler, Heun, DPM++ 2M and res_multistep outputs are bit-identical
  after multiple pauses and on the next invocation.
- res_multistep held for 60 seconds without advancing or busy-spinning.
- Interrupt while paused exits before another step; preview error/final-step
  pause requests do not leak; unsupported/idle requests are rejected.
- Real Chromium: placement beside Run, disabled idle state, pending/play,
  resume requests, toolbar re-render and duplicate prevention; Interrupt remains.
- A separate real CPU ComfyUI boot loads the extension, serves API and JS, and
  preserves registration of existing RVC nodes. No workflows were submitted;
  the isolated server was stopped after testing.
- Full Python suite: 176 passed, 1 skipped, 1 failed. The failure is the existing
  BeatThis test requiring missing `beat_this` in the active runtime; not a pause test.
- Optional Node state-test assertions succeed but installed Node processes
  abort during shutdown in this environment (even a trivial `console.log`).
  Real Chromium is the verified frontend test path instead.

Still required before calling this fully live-validated: activate after a safe
production restart, test real normal/custom/Wan/H3 model workflows, measure GPU
idle and allocation retention, and compare GPU outputs with uninterrupted
same-seed runs. No live queue was cleared, interrupted, or restarted for this work.
