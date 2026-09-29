#!/usr/bin/env python3
"""Hold only KDE's suspend policy while this helper's stdin remains open."""

from __future__ import annotations

import sys
import time


SERVICE = "org.kde.Solid.PowerManagement"
PATH = "/org/kde/Solid/PowerManagement/PolicyAgent"
POLICY_INTERFACE = "org.kde.Solid.PowerManagement.PolicyAgent"
INTERRUPT_SESSION = 1
CHANGE_SCREEN_SETTINGS = 4
POWERDEVIL_ENFORCEMENT_DELAY_SECONDS = 5.5


def _screen_policy_unchanged(
    screen_blocked_before: bool, screen_blocked_after: bool
) -> bool:
    """Accept an existing screen inhibitor, but never introduce a new one."""
    return screen_blocked_before or not screen_blocked_after


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: plasma_suspend_helper.py APPLICATION REASON", file=sys.stderr)
        return 2

    import dbus

    application, reason = sys.argv[1:]
    bus = dbus.SessionBus()
    proxy = bus.get_object(SERVICE, PATH)
    interface = dbus.Interface(proxy, dbus_interface=POLICY_INTERFACE)
    screen_blocked_before = bool(interface.HasInhibition(CHANGE_SCREEN_SETTINGS))
    cookie = interface.AddInhibition(
        dbus.UInt32(INTERRUPT_SESSION), application, reason
    )

    try:
        # Plasma 6 deliberately delays application inhibitors for five seconds.
        time.sleep(POWERDEVIL_ENFORCEMENT_DELAY_SECONDS)
        suspend_blocked = bool(interface.HasInhibition(INTERRUPT_SESSION))
        screen_blocked_after = bool(interface.HasInhibition(CHANGE_SCREEN_SETTINGS))
        screen_policy_unchanged = _screen_policy_unchanged(
            screen_blocked_before, screen_blocked_after
        )
        if not suspend_blocked or not screen_policy_unchanged:
            print(
                "PowerDevil did not establish a suspend-only inhibition "
                f"(suspend={suspend_blocked}, "
                f"screen_before={screen_blocked_before}, "
                f"screen_after={screen_blocked_after})",
                file=sys.stderr,
            )
            return 1

        print(int(cookie), flush=True)
        # Newline means normal release; EOF means the ComfyUI parent vanished.
        sys.stdin.readline()
        return 0
    finally:
        try:
            interface.ReleaseInhibition(cookie)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
