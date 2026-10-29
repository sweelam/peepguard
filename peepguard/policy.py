"""Temporal policy: a single noisy frame should never blur your screen.

    CLEAR --onlooker seen--> SUSPECT --persists >= dwell_s--> ALERT --clear >= release_s--> CLEAR
                                |                                  (or user dismisses, if sticky)
                                +--gone > gap_tolerance_s--> CLEAR

Frames flagged DEGRADED (e.g. too dark) carry no evidence either way: they neither
advance an alert nor count as "clear".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import Config
from .scene import Observation


class State(str, Enum):
    CLEAR = "clear"
    SUSPECT = "suspect"
    ALERT = "alert"


@dataclass(frozen=True)
class Event:
    kind: str        # "alert" | "clear"
    reason: str      # "onlooker" | "stranger" | "away" | "dismissed" | "allowed" | "resolved"
    t: float
    onlookers: int = 0


class Policy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State.CLEAR
        self.reason: str | None = None
        self.degraded: str | None = None
        self._since = 0.0
        self._last_threat = -1e9
        self._last_primary: float | None = None
        self._snoozed_until = -1e9

    def step(self, obs: Observation) -> Event | None:
        t = obs.t
        self.degraded = obs.degraded
        if obs.degraded:
            return None

        if obs.primary is not None or self._last_primary is None:
            self._last_primary = t
        n = len(obs.onlookers)
        away = self.cfg.away_lock_s > 0 and t - self._last_primary >= self.cfg.away_lock_s
        threat = (n > 0 or away) and t >= self._snoozed_until
        if threat:
            self._last_threat = t

        if self.state is State.CLEAR:
            if away and threat:  # walking away is unambiguous: no dwell needed
                return self._enter_alert(t, "away", n)
            if threat:
                self.state, self._since = State.SUSPECT, t
            return None

        if self.state is State.SUSPECT:
            if threat and t - self._since >= self.cfg.dwell_s:
                who = "stranger" if obs.owner_enrolled and obs.primary is None else "onlooker"
                return self._enter_alert(t, "away" if away and n == 0 else who, n)
            if t - self._last_threat > self.cfg.gap_tolerance_s:
                self.state = State.CLEAR
            return None

        # ALERT
        if not self.cfg.sticky and t - self._last_threat >= self.cfg.release_s:
            return self._clear(t, "resolved")
        return None

    def dismiss(self, t: float, snooze_s: float = 10.0, reason: str = "dismissed") -> Event | None:
        """User chose to reveal the screen. Snooze so we don't re-blur instantly while the
        same person is still there: briefly ("dismissed"), or for as long as they picked
        ("allowed", e.g. 15 min while showing a teammate something)."""
        if self.state is not State.ALERT:
            return None
        self._snoozed_until = t + snooze_s
        return self._clear(t, reason)

    def allow(self, t: float, snooze_s: float) -> Event | None:
        """Let people look for a while. Clears the shield if it's up; works when it isn't."""
        if self.state is State.ALERT:
            return self.dismiss(t, snooze_s, "allowed")
        self._snoozed_until = t + snooze_s
        self.state = State.CLEAR
        return None

    def resume(self) -> None:
        self._snoozed_until = -1e9

    def snoozed_for(self, t: float) -> float:
        return max(0.0, self._snoozed_until - t)

    def _enter_alert(self, t: float, reason: str, n: int) -> Event:
        self.state, self.reason, self._since = State.ALERT, reason, t
        return Event("alert", reason, t, n)

    def _clear(self, t: float, reason: str) -> Event:
        self.state, self.reason = State.CLEAR, None
        self._last_threat = -1e9
        return Event("clear", reason, t)
