"""Regression tests for ComfyUI's queue-scoped suspend-only inhibitor.

Run directly so pytest does not import the heavyweight ComfyUI package first:
``python tests/test_suspend_only_inhibit.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


MODULE_FILE = Path(__file__).parents[1] / "suspend_only_inhibit.py"
HELPER_FILE = Path(__file__).parents[1] / "plasma_suspend_helper.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("suspend_only_inhibit_test", MODULE_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("plasma_suspend_helper_test", HELPER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SuspendOnlyInhibitTests(unittest.TestCase):
    def test_session_bus_environment_is_recovered_for_headless_pm2_process(self):
        module = _load_module()

        with (
            mock.patch.object(module.os, "getuid", return_value=1000),
            mock.patch.object(module.Path, "is_dir", return_value=True),
            mock.patch.object(module.Path, "is_socket", return_value=True),
        ):
            env = module._session_bus_environment({"HOME": "/home/amin"})

        self.assertEqual(env["XDG_RUNTIME_DIR"], "/run/user/1000")
        self.assertEqual(
            env["DBUS_SESSION_BUS_ADDRESS"],
            "unix:path=/run/user/1000/bus",
        )
        self.assertEqual(env["HOME"], "/home/amin")

    def test_preexisting_screen_inhibition_does_not_reject_suspend_only_lock(self):
        helper = _load_helper_module()

        self.assertTrue(helper._screen_policy_unchanged(True, True))
        self.assertTrue(helper._screen_policy_unchanged(False, False))
        self.assertFalse(helper._screen_policy_unchanged(False, True))

    def test_plasma_helper_blocks_session_interrupt_but_not_screen_settings(self):
        helper = HELPER_FILE.read_text(encoding="utf-8")

        self.assertIn(
            'POLICY_INTERFACE = "org.kde.Solid.PowerManagement.PolicyAgent"',
            helper,
        )
        self.assertIn("INTERRUPT_SESSION = 1", helper)
        self.assertIn("CHANGE_SCREEN_SETTINGS = 4", helper)
        self.assertIn("HasInhibition(INTERRUPT_SESSION)", helper)
        self.assertIn("HasInhibition(CHANGE_SCREEN_SETTINGS)", helper)
        self.assertNotIn("org.freedesktop.PowerManagement.Inhibit", helper)

    def test_plasma_backend_keeps_verified_helper_alive(self):
        module = _load_module()
        calls = []

        class FakePipe:
            def __init__(self, line=""):
                self.line = line
                self.writes = []

            def readline(self):
                return self.line

            def write(self, value):
                self.writes.append(value)

            def flush(self):
                pass

        class FakeProcess:
            def __init__(self):
                self.stdin = FakePipe()
                self.stdout = FakePipe("47\n")
                self.stderr = FakePipe()
                self.returncode = None
                self.waited = False

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waited = True
                self.returncode = 0
                return 0

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        process = FakeProcess()

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        with (
            mock.patch.object(module, "_find_dbus_python", return_value="/usr/bin/python3"),
            mock.patch.object(module, "_readline_with_timeout", return_value="47\n"),
            mock.patch.object(module.subprocess, "Popen", side_effect=fake_popen),
        ):
            backend, holder = module.start_suspend_only_inhibit(
                "ComfyUI", "queue active"
            )

        self.assertEqual(backend, "kde-powerdevil-suspend-only")
        self.assertIs(holder, process)
        self.assertEqual(
            calls[0][0],
            [
                "/usr/bin/python3",
                str(module.HELPER_FILE),
                "ComfyUI",
                "queue active",
            ],
        )

        with mock.patch.object(module, "reset_plasma_idle_clock", return_value=True) as reset:
            module.stop_suspend_only_inhibit(holder)
        self.assertEqual(process.stdin.writes, ["\n"])
        self.assertTrue(process.waited)
        reset.assert_called_once_with()

    def test_idle_reset_uses_plasma_simulate_user_activity(self):
        module = _load_module()
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return mock.Mock(returncode=0)

        with (
            mock.patch.object(module.shutil, "which", return_value="/usr/bin/qdbus6"),
            mock.patch.object(
                module,
                "_session_bus_environment",
                return_value={"DBUS_SESSION_BUS_ADDRESS": "test-bus"},
            ),
            mock.patch.object(module.subprocess, "run", side_effect=fake_run),
        ):
            self.assertTrue(module.reset_plasma_idle_clock())

        self.assertEqual(
            calls,
            [
                (
                    [
                        "/usr/bin/qdbus6",
                        "org.freedesktop.ScreenSaver",
                        "/ScreenSaver",
                        "org.freedesktop.ScreenSaver.SimulateUserActivity",
                    ],
                    {
                        "stdout": module.subprocess.DEVNULL,
                        "stderr": module.subprocess.DEVNULL,
                        "timeout": 3,
                        "check": False,
                        "env": {"DBUS_SESSION_BUS_ADDRESS": "test-bus"},
                    },
                )
            ],
        )

    def test_systemd_fallback_uses_sleep_only_never_idle(self):
        module = _load_module()
        calls = []

        class FakeProcess:
            stderr = None

            def poll(self):
                return None

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return FakeProcess()

        with (
            mock.patch.object(module, "_start_plasma_inhibit", return_value=None),
            mock.patch.object(module.shutil, "which", return_value="/usr/bin/systemd-inhibit"),
            mock.patch.object(module.time, "sleep"),
            mock.patch.object(module.subprocess, "Popen", side_effect=fake_popen),
        ):
            backend, holder = module.start_suspend_only_inhibit(
                "ComfyUI", "queue active"
            )

        self.assertEqual(backend, "systemd-suspend-only")
        self.assertIsNotNone(holder)
        command = calls[0][0]
        self.assertIn("--what=sleep", command)
        self.assertNotIn("--what=idle", command)
        self.assertNotIn("--what=idle:sleep", command)

    def test_source_has_no_screensaver_heartbeat(self):
        source = (MODULE_FILE.parent / "__init__.py").read_text(encoding="utf-8")

        self.assertNotIn("_heartbeat_keep_awake", source)
        self.assertNotIn('xdg-screensaver", "reset"', source)
        self.assertNotIn('xset", "s", "reset"', source)

    def test_guard_is_queue_scoped_not_comfyui_process_scoped(self):
        source = (MODULE_FILE.parent / "__init__.py").read_text(encoding="utf-8")

        self.assertIn("COMFYUI QUEUE-SCOPED SUSPEND-ONLY GUARD", source)
        self.assertIn("def _monitor_comfy_queue", source)
        self.assertIn("prompt_server.prompt_queue.get_current_queue()", source)
        self.assertIn("if queue_active:", source)
        self.assertIn("_start_auto_guard()", source)
        self.assertIn("else:\n                _stop_auto_guard()", source)
        self.assertIn(
            "threading.Thread(target=_monitor_comfy_queue, daemon=True).start()",
            source,
        )
        self.assertNotIn("ComfyUI PM2 process active", source)
        self.assertNotIn("atexit.register(_stop_auto_guard)\n_start_auto_guard()", source)

if __name__ == "__main__":
    unittest.main(verbosity=2)
