"""Invocation-scoped, non-busy sampler pause state. No tensor storage/migration."""
import logging
import threading

log = logging.getLogger('SamplerPause')

class PauseController:
    def __init__(self, check_interrupt=lambda: None, synchronize=lambda: None):
        self._lock = threading.RLock()
        self._event = threading.Event()
        self._event.set()
        self._check_interrupt = check_interrupt
        self._synchronize = synchronize
        self._active = self._supported = self._requested = self._paused = False
        self._cancelled = False
        self._sampler = None
        self._step = self._total = 0

    def status(self):
        with self._lock:
            return dict(paused=self._paused, sampling_active=self._active,
                        pause_requested=self._requested and not self._paused,
                        supported=self._supported, sampler=self._sampler,
                        completed_steps=self._step, total_steps=self._total)

    def begin(self, sampler, supported):
        with self._lock:
            if self._active:
                return False  # Only the owner can control/reset this invocation.
            self._active = True
            self._supported = supported
            self._sampler = sampler
            self._requested = self._paused = self._cancelled = False
            self._step = self._total = 0
            self._event.set()
            return True

    def request_pause(self):
        with self._lock:
            if not self._active or not self._supported or self._cancelled:
                return False
            self._requested = True
            self._event.clear()
            log.info('[SamplerPause] Pause requested')
            return True

    def release(self, reason='resume'):
        with self._lock:
            had_pause = self._requested or self._paused
            self._requested = self._paused = False
            if reason in ('interrupt', 'shutdown'):
                self._cancelled = True
            self._event.set()
            if had_pause:
                if reason == 'resume':
                    log.info('[SamplerPause] Resuming at step %s/%s', self._step + 1, self._total)
                else:
                    log.info('[SamplerPause] Pause released because %s', reason)

    def finish(self):
        with self._lock:
            self.release('sampling finished')
            self._active = self._supported = False
            self._sampler = None
            log.info('[SamplerPause] Sampling finished')

    def boundary(self, completed, total):
        self._check_interrupt()
        with self._lock:
            self._step, self._total = completed, total
            requested = self._requested and self._active and not self._cancelled
        if not requested:
            return
        # Drain already submitted CUDA work only on a pause request. Never migrate.
        self._synchronize()
        with self._lock:
            if not self._requested or self._cancelled:
                return
            self._paused = True
            log.info('[SamplerPause] Paused after step %s/%s', completed, total)
        try:
            while not self._event.wait(.2):
                self._check_interrupt()
            self._check_interrupt()
        finally:
            with self._lock:
                self._paused = False
