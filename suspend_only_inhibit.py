"""Suspend-only process holders for the ComfyUI queue guard.

The preferred Plasma backend requests only ``InterruptSession``. It explicitly
verifies that ``ChangeScreenSettings`` is not inhibited, so display blanking and
screen locking remain available while renders run.
"""

from __future__ import annotations

import functools
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path


HELPER_FILE = Path(__file__).resolve().with_name("plasma_suspend_helper.py")


def _session_bus_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Recover the graphical user bus for PM2 processes with a minimal env."""
    env = dict(os.environ if environ is None else environ)
    runtime_dir = Path(f"/run/user/{os.getuid()}")
    session_bus = runtime_dir / "bus"

    if "XDG_RUNTIME_DIR" not in env and runtime_dir.is_dir():
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    if "DBUS_SESSION_BUS_ADDRESS" not in env and session_bus.is_socket():
        env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={session_bus}"

    return env


@functools.lru_cache(maxsize=1)
def _find_dbus_python() -> str | None:
    candidates = ["/usr/bin/python3", "/usr/bin/python", shutil.which("python3")]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-c", "import dbus"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                return candidate
        except Exception:
            continue
    return None


def _readline_with_timeout(stream, timeout: float = 8.0) -> str:
    result: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(stream.readline())
        except Exception:
            result.put("")

    threading.Thread(target=read, daemon=True).start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        return ""


def _terminate(process: subprocess.Popen[str]) -> None:
    try:
        process.terminate()
        process.wait(timeout=1)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _start_plasma_inhibit(who: str, why: str) -> subprocess.Popen[str] | None:
    python = _find_dbus_python()
    if not python or not HELPER_FILE.is_file():
        return None

    try:
        process = subprocess.Popen(
            [python, str(HELPER_FILE), who, why],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_session_bus_environment(),
        )
    except Exception:
        return None

    cookie = _readline_with_timeout(process.stdout).strip() if process.stdout else ""
    if process.poll() is not None or not cookie:
        _terminate(process)
        return None
    return process


def _start_systemd_sleep_inhibit(
    who: str, why: str
) -> subprocess.Popen[str] | None:
    """Portable fallback; never use logind's broader ``idle`` inhibitor."""
    if not shutil.which("systemd-inhibit"):
        return None
    try:
        process = subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=sleep",
                "--mode=block",
                f"--who={who}",
                f"--why={why}",
                "sleep",
                "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.35)
        if process.poll() is not None:
            return None
        return process
    except Exception:
        return None


def start_suspend_only_inhibit(
    who: str, why: str
) -> tuple[str | None, subprocess.Popen[str] | None]:
    process = _start_plasma_inhibit(who, why)
    if process is not None:
        return "kde-powerdevil-suspend-only", process

    process = _start_systemd_sleep_inhibit(who, why)
    if process is not None:
        return "systemd-suspend-only", process

    # No idle inhibitor and no heartbeat fallback: both can keep the display on.
    return None, None


def reset_plasma_idle_clock() -> bool:
    """Restart Plasma's configured idle countdown from zero."""
    qdbus = shutil.which("qdbus6")
    if not qdbus:
        return False
    try:
        result = subprocess.run(
            [
                qdbus,
                "org.freedesktop.ScreenSaver",
                "/ScreenSaver",
                "org.freedesktop.ScreenSaver.SimulateUserActivity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
            env=_session_bus_environment(),
        )
    except Exception:
        return False
    return result.returncode == 0


def stop_suspend_only_inhibit(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    try:
        if process.stdin:
            process.stdin.write("\n")
            process.stdin.flush()
        process.wait(timeout=2)
    except Exception:
        _terminate(process)
    finally:
        # PowerDevil keeps the old idle age while inhibited. Reset it only when
        # real queue work ends, so the user's normal 15-minute timer starts over
        # instead of suspending immediately after a long render.
        reset_plasma_idle_clock()
