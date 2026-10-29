"""What happens when the policy fires. Each action reacts to alert/clear events; the
shield additionally owns a window and needs pumping from the main (GUI) thread."""

from __future__ import annotations

import logging
import os
import platform
import shlex
import shutil
import subprocess

from ..policy import Event

log = logging.getLogger("peepguard")


class Action:
    name = "action"

    def on_event(self, ev: Event) -> None: ...

    def pump(self, key: int = -1) -> tuple[str, float] | None:
        """Called every loop iteration from the main thread with the last highgui key
        (-1 if none). Returns (reason, snooze_seconds) when the user chose to reveal the
        screen: ("dismissed", short) or ("allowed", the minutes they picked)."""
        return None

    def close(self) -> None: ...


class LogAction(Action):
    name = "log"

    def on_event(self, ev: Event) -> None:
        if ev.kind == "alert":
            log.warning("ALERT  %s (%d onlooker%s)", ev.reason, ev.onlookers, "" if ev.onlookers == 1 else "s")
        else:
            log.info("CLEAR  %s", ev.reason)


class NotifyAction(Action):
    """A native notification. Deliberately content-free: the notification itself is
    visible to the onlooker, so it says nothing beyond 'heads up'."""

    name = "notify"

    def __init__(self):
        self._cmd = self._resolve()

    @staticmethod
    def _resolve():
        sysname = platform.system()
        if sysname == "Darwin" and shutil.which("osascript"):
            return lambda msg: ["osascript", "-e", f'display notification "{msg}" with title "PeepGuard"']
        if sysname == "Linux" and shutil.which("notify-send"):
            return lambda msg: ["notify-send", "-u", "normal", "-a", "PeepGuard", "PeepGuard", msg]
        return None

    def on_event(self, ev: Event) -> None:
        if ev.kind != "alert" or self._cmd is None:
            return
        msg = "Screen paused for privacy" if ev.reason == "onlooker" else "Screen paused while you were away"
        subprocess.Popen(self._cmd(msg), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class HookAction(Action):
    """Run a user command on every transition, e.g. toggle Do Not Disturb, lock the
    session, or pause a screen share. Gets PEEPGUARD_EVENT / _REASON / _ONLOOKERS."""

    name = "hook"

    def __init__(self, cmd: str):
        self._argv = shlex.split(cmd)

    def on_event(self, ev: Event) -> None:
        env = dict(os.environ, PEEPGUARD_EVENT=ev.kind, PEEPGUARD_REASON=ev.reason,
                   PEEPGUARD_ONLOOKERS=str(ev.onlookers))
        try:
            subprocess.Popen(self._argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            log.error("hook failed: %s", e)


def build_actions(names, hook: str = "", allow_minutes: float = 15.0) -> list[Action]:
    from .shield import ShieldAction

    out: list[Action] = []
    for n in names:
        n = n.strip()
        if n == "log":
            out.append(LogAction())
        elif n == "notify":
            out.append(NotifyAction())
        elif n == "shield":
            out.append(ShieldAction(allow_minutes))
        elif n == "hook":
            if not hook:
                raise ValueError("action 'hook' needs --hook 'command'")
        elif n:
            raise ValueError(f"unknown action: {n}")
    if hook:
        out.append(HookAction(hook))
    return out
