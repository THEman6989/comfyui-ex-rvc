"""Reversible in-memory hooks, restricted to audited standard step iterators.

Comfy preview callbacks often run BEFORE solver corrections; never wait there.
The trange iterator resumes only when the entire prior loop body has finished.
"""
import functools
import threading

class SamplingHooks:
    def __init__(self, controller):
        self.controller = controller
        self.local = threading.local()
        self._installed = False

    def run_sampler(self, sampler_function, operation):
        if getattr(self.local, 'active', False):
            return operation()
        supported = sampler_function in self._supported
        if not self.controller.begin(getattr(sampler_function, '__name__', 'custom'), supported):
            return operation()  # Parallel foreign invocation must not reset/pause the owner.
        self.local.active = supported
        try:
            return operation()
        finally:
            self.local.active = False
            self.controller.finish()

    def install(self, samplers, sampling, management):
        if self._installed:
            return
        if getattr(samplers.KSAMPLER.sample, '_sampler_pause', False):
            raise RuntimeError('SamplerPause already installed')
        self.samplers, self.sampling, self.management = samplers, sampling, management
        # Only standard functions whose outer trange loop is one full solver step.
        # DPM adaptive/fast use different loops. AR video iterates blocks, not steps.
        audited_names = {
            'euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp',
            'heun', 'heunpp2', 'exp_heun_2_x0', 'exp_heun_2_x0_sde',
            'dpm_2', 'dpm_2_ancestral', 'lms', 'dpmpp_2s_ancestral',
            'dpmpp_2s_ancestral_cfg_pp', 'dpmpp_sde', 'dpmpp_sde_gpu',
            'dpmpp_2m', 'dpmpp_2m_cfg_pp', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
            'dpmpp_2m_sde_heun', 'dpmpp_2m_sde_heun_gpu', 'dpmpp_3m_sde',
            'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ipndm', 'ipndm_v', 'deis',
            'cfgpp_ud10_ab', 'res_multistep', 'res_multistep_cfg_pp',
            'res_multistep_ancestral', 'res_multistep_ancestral_cfg_pp',
            'gradient_estimation', 'gradient_estimation_cfg_pp', 'er_sde',
            'seeds_2', 'seeds_3', 'sa_solver', 'sa_solver_pece',
        }
        self._supported = {getattr(sampling, 'sample_' + name) for name in audited_names
                           if name in samplers.KSAMPLER_NAMES
                           and hasattr(sampling, 'sample_' + name)}
        self._original_trange = sampling.trange
        self._original_sample = samplers.KSAMPLER.sample
        self._original_interrupt = management.interrupt_current_processing
        hooks = self

        def step_range(*args, **kwargs):
            progress = hooks._original_trange(*args, **kwargs)
            if not getattr(hooks.local, 'active', False):
                return progress
            def iterate():
                total = len(progress)
                try:
                    for index, value in enumerate(progress):
                        if index > 0:
                            hooks.controller.boundary(index, total)
                        yield value
                finally:
                    progress.close()
            return iterate()

        @functools.wraps(self._original_sample)
        def sample(instance, *args, **kwargs):
            # Preserve the existing callback and argument ordering unchanged.
            return hooks.run_sampler(instance.sampler_function,
                                     lambda: hooks._original_sample(instance, *args, **kwargs))
        sample._sampler_pause = True

        @functools.wraps(self._original_interrupt)
        def interrupt(value=True):
            # Set normal interrupt flag BEFORE waking so no next forward can race.
            # The original takes its own interrupt lock; never hold ours across it.
            result = hooks._original_interrupt(value)
            if value:
                hooks.controller.release('interrupt')
            return result

        self._step_range, self._sample, self._interrupt = step_range, sample, interrupt
        sampling.trange = step_range
        samplers.KSAMPLER.sample = sample
        management.interrupt_current_processing = interrupt
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        self.controller.release('shutdown')
        for owner, name, installed, original in (
            (self.sampling, 'trange', self._step_range, self._original_trange),
            (self.samplers.KSAMPLER, 'sample', self._sample, self._original_sample),
            (self.management, 'interrupt_current_processing', self._interrupt, self._original_interrupt)):
            if getattr(owner, name) is installed:
                setattr(owner, name, original)
        self._installed = False
