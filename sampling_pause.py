"""Bootstrap owned by comfyui-ex-rvc; no core files or model code edited."""
import atexit
import logging
from .pause_controller import PauseController
from .sampling_hook import SamplingHooks
from .sampling_pause_api import register_routes

_hooks = None
_controller = None

def install(prompt_server=None):
    global _hooks, _controller
    if _hooks is not None:
        return _hooks
    import torch
    import comfy.samplers as samplers
    import comfy.k_diffusion.sampling as sampling
    import comfy.model_management as management
    if prompt_server is None:
        from server import PromptServer
        prompt_server = PromptServer.instance
    if prompt_server is None:
        raise RuntimeError('SamplerPause requires initialized PromptServer')

    def synchronize():
        # Multi-device operations may have outstanding work on either GPU.
        # Drain each device context in this process only when actually pausing.
        # No allocations, model unloads, or tensor/device moves are requested.
        if torch.cuda.is_initialized():
            has_context = getattr(torch._C, '_cuda_hasPrimaryContext', None)
            if has_context is None:
                raise RuntimeError('Cannot safely identify initialized CUDA contexts on this Torch version')
            for device in range(torch.cuda.device_count()):
                if has_context(device):
                    torch.cuda.synchronize(device)

    controller = PauseController(management.throw_exception_if_processing_interrupted, synchronize)
    hooks = SamplingHooks(controller)
    register_routes(prompt_server, controller)
    hooks.install(samplers, sampling, management)
    atexit.register(controller.release, 'shutdown')
    _controller, _hooks = controller, hooks
    logging.info('[SamplerPause] Ready: step-boundary pause/resume (VRAM retained)')
    return hooks
