"""Threaded tests without importing the node package's heavyweight __init__."""
import importlib.util
import pathlib
import sys
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class PauseTests(unittest.TestCase):
    def test_request_wait_resume_same_invocation(self):
        path = ROOT / 'pause_controller.py'
        self.assertTrue(path.exists(), 'Pause controller is not implemented')
        c = load('pause_controller').PauseController()
        c.begin('euler', True)
        self.assertTrue(c.request_pause())
        self.assertTrue(c.status()['pause_requested'])
        done = threading.Event()
        thread = threading.Thread(target=lambda: (c.boundary(2, 5), done.set()))
        thread.start()
        try:
            for _ in range(100):
                if c.status()['paused']: break
                time.sleep(.01)
            self.assertTrue(c.status()['paused'])
            self.assertFalse(done.wait(.05))
        finally:
            c.release('resume')
            thread.join(2)
        self.assertTrue(done.is_set())
        c.finish()
        self.assertFalse(c.status()['sampling_active'])

class HookTests(unittest.TestCase):
    def test_heun_completes_both_forwards_before_pause(self):
        self.assertTrue((ROOT / 'sampling_hook.py').exists(), 'Sampling hook is not implemented')
        sys.path.insert(0, str(ROOT.parents[1]))
        sys.argv = [sys.argv[0], '--cpu']
        import comfy.options
        comfy.options.enable_args_parsing()
        import torch
        import comfy.k_diffusion.sampling as kd
        import comfy.samplers as samplers
        import comfy.model_management as mm
        controller = load('pause_controller').PauseController()
        hooks = load('sampling_hook').SamplingHooks(controller)
        hooks.install(samplers, kd, mm)
        calls = []
        previews = []
        result = []
        def model(x, sigma, **kwargs):
            calls.append(float(sigma[0]))
            return x * .25
        def callback(info):
            previews.append(info['i'])
            if info['i'] == 0: controller.request_pause()
        x = torch.ones(1, 1, 2, 2)
        sigmas = torch.tensor([1., .7, .4, 0.])
        baseline = kd.sample_heun(model, x.clone(), sigmas, disable=True)
        calls.clear()
        thread = threading.Thread(target=lambda: result.append(hooks.run_sampler(kd.sample_heun, lambda: kd.sample_heun(model, x.clone(), sigmas, callback=callback, disable=True))))
        thread.start()
        try:
            for _ in range(200):
                if controller.status()['paused']: break
                time.sleep(.01)
            self.assertTrue(controller.status()['paused'])
            self.assertEqual(len(calls), 2, 'Heun second forward must finish before blocking')
            self.assertEqual(previews, [0])
        finally:
            controller.release()
            thread.join(3)
            hooks.uninstall()
        self.assertFalse(thread.is_alive())
        self.assertTrue(torch.equal(result[0], baseline))
        self.assertEqual(previews, [0, 1, 2])

class SafetyTests(unittest.TestCase):
    def setUp(self):
        import types
        sys.path.insert(0, str(ROOT.parents[1]))
        sys.argv = [sys.argv[0], '--cpu']
        import comfy.options
        comfy.options.enable_args_parsing()
        import torch
        import comfy.samplers as samplers
        import comfy.k_diffusion.sampling as kd
        import comfy.model_management as mm
        self.torch, self.samplers, self.kd, self.mm = torch, samplers, kd, mm
        self.c = load('pause_controller').PauseController(mm.throw_exception_if_processing_interrupted)
        self.hooks = load('sampling_hook').SamplingHooks(self.c)
        self.hooks.install(samplers, kd, mm)
        sampling = types.SimpleNamespace(sigma_max=1., noise_scale=1.,
            noise_scaling=lambda sigma, noise, latent, maximum: noise.clone(),
            inverse_noise_scaling=lambda sigma, x: x)
        patcher = types.SimpleNamespace(model=types.SimpleNamespace(), get_model_object=lambda key: sampling)
        class Model:
            inner_model = types.SimpleNamespace(model_sampling=sampling)
            model_patcher = patcher
            cfg = 1.
            def __call__(self, x, sigma, **kwargs): return x * .25 + .05
        self.model = Model()
        self.noise = torch.ones(1, 1, 2, 2)
        self.sigmas = torch.tensor([1., .8, .6, .4, .2, 0.])
        self.mm.interrupt_current_processing(False)

    def tearDown(self):
        self.c.release()
        self.hooks.uninstall()
        self.mm.interrupt_current_processing(False)

    def wait_paused(self):
        for _ in range(300):
            if self.c.status()['paused']: return
            time.sleep(.01)
        self.fail('Worker did not reach pause boundary')

    def sample(self, name, callback=None):
        return self.samplers.ksampler(name).sample(self.model, self.sigmas, {'seed':42}, callback, self.noise, disable_pbar=True)

    def test_multistep_two_pause_cycles_and_next_invocation_equal(self):
        import os
        for name in ('euler', 'heun', 'dpmpp_2m', 'res_multistep'):
            with self.subTest(sampler=name):
                baseline = self.sample(name)
                result, errors = [], []
                def preview(step, *args):
                    if step in (1, 3): self.c.request_pause()
                def run():
                    try: result.append(self.sample(name, preview))
                    except BaseException as e: errors.append(e)
                t = threading.Thread(target=run)
                t.start()
                try:
                    for cycle in range(2):
                        self.wait_paused()
                        before = self.c.status()['completed_steps']
                        hold = float(os.environ.get('SAMPLER_PAUSE_HOLD', '.05')) if name == 'res_multistep' and cycle == 0 else .05
                        cpu = time.process_time()
                        time.sleep(hold)
                        self.assertEqual(self.c.status()['completed_steps'], before)
                        if hold >= 60: self.assertLess(time.process_time() - cpu, .5, 'Paused worker should not busy-spin')
                        self.c.release()
                        # Avoid accidentally observing the old paused state.
                        deadline = time.monotonic() + 3
                        while self.c.status()['completed_steps'] == before and t.is_alive() and time.monotonic() < deadline:
                            time.sleep(.005)
                    t.join(3)
                finally:
                    self.c.release(); t.join(3)
                self.assertFalse(t.is_alive())
                self.assertFalse(errors, errors)
                self.assertTrue(self.torch.equal(result[0], baseline))
                self.assertTrue(self.torch.equal(self.sample(name), baseline))

    def test_interrupt_while_paused_exits_without_next_step(self):
        errors, previews = [], []
        def preview(step, *args):
            previews.append(step)
            if step == 0: self.c.request_pause()
        def run():
            try: self.sample('heun', preview)
            except BaseException as e: errors.append(e)
        t = threading.Thread(target=run); t.start()
        try:
            self.wait_paused()
            self.mm.interrupt_current_processing(True)
            t.join(2)
        finally:
            self.c.release(); t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(previews, [0])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], self.mm.InterruptProcessingException)
        self.assertFalse(self.c.status()['sampling_active'])
        self.assertFalse(self.c.request_pause())

    def test_exception_and_final_step_request_do_not_leak(self):
        def fail(*args): raise ValueError('preview failed')
        with self.assertRaises(ValueError): self.sample('euler', fail)
        self.assertFalse(self.c.status()['sampling_active'])
        self.sample('euler', lambda step, *args: self.c.request_pause() if step == 4 else None)
        self.assertFalse(self.c.status()['pause_requested'])
        self.assertFalse(self.c.status()['sampling_active'])

    def test_other_invocation_is_not_broken_by_active_pause(self):
        result, errors = [], []
        def preview(step, *args):
            if step == 0: self.c.request_pause()
        def run():
            try: result.append(self.sample('euler', preview))
            except BaseException as e: errors.append(e)
        t = threading.Thread(target=run); t.start()
        try:
            self.wait_paused()
            # This thread has no ownership of the paused invocation.
            other = self.sample('euler')
            self.assertEqual(tuple(other.shape), tuple(self.noise.shape))
            self.assertTrue(self.c.status()['paused'])
        finally:
            self.c.release(); t.join(2)
        self.assertFalse(errors, errors)

    def test_unsupported_and_idle_requests_are_rejected(self):
        self.assertFalse(self.c.request_pause())
        def operation():
            self.assertFalse(self.c.status()['supported'])
            self.assertFalse(self.c.request_pause())
        self.hooks.run_sampler(self.kd.sample_dpm_adaptive, operation)
        self.hooks.run_sampler(lambda: None, operation)

class StartupTests(unittest.TestCase):
    def test_install_is_idempotent(self):
        self.assertTrue((ROOT / 'sampling_pause.py').exists(), 'Startup integration is missing')
        import types
        import importlib
        from aiohttp import web
        package = types.ModuleType('sampler_pause_test_package')
        package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + '.sampling_pause')
        server = types.SimpleNamespace(routes=web.RouteTableDef(), app=web.Application())
        hooks = module.install(server)
        try:
            self.assertIs(module.install(server), hooks)
            self.assertEqual(len(server.routes), 3)
        finally:
            hooks.uninstall()

if __name__ == '__main__': unittest.main(argv=[sys.argv[0]])
