"""Main loop: capture -> analyse -> policy -> actions, at a few frames per second."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field, replace

import cv2

from .actions import Action, build_actions
from .camera import FrameSource
from .config import Config
from .detector import make_detector
from .hud import WINDOW as HUD_WINDOW, annotate
from .owner import load_matcher
from .policy import Event, Policy, State
from .scene import SceneAnalyzer

log = logging.getLogger("peepguard")
_AUTO = object()  # Guard(owner=...): load the enrolled owner unless told otherwise


@dataclass
class RunStats:
    frames: int = 0
    events: list[Event] = field(default_factory=list)
    detector: str = ""
    analysis_ms: float = 0.0  # mean per-frame analysis time


class Guard:
    def __init__(self, cfg: Config, actions: list[Action] | None = None, detector=None, owner=_AUTO):
        self.cfg = cfg
        cv2.setNumThreads(cfg.threads)
        self.detector = detector or make_detector(cfg.detector, cfg.score_threshold)
        if owner is _AUTO:
            owner = load_matcher(cfg) if self.detector.name == "yunet" else None
        self.scene = SceneAnalyzer(cfg, self.detector, owner=owner)
        self.policy = Policy(cfg)
        self.actions = build_actions(cfg.actions, cfg.hook, cfg.allow_minutes) if actions is None else actions
        self._quit = False

    def process(self, t: float, frame) -> tuple[Event | None, object, object]:
        small = self.scene.prepare(frame)
        obs = self.scene.analyze(small, now=t)
        ev = self.policy.step(obs)
        if ev:
            self._dispatch(ev)
        return ev, obs, small

    def _dispatch(self, ev: Event) -> None:
        for a in self.actions:
            try:
                a.on_event(ev)
            except Exception:  # an action must never take the guard down
                log.exception("action %s failed", a.name)

    # ------------------------------------------------- controls (used by the UI panel)
    def allow(self, t: float, minutes: float | None = None) -> None:
        """Stop pausing for a while (showing a teammate something), shield up or not."""
        snooze_s = 60.0 * (self.cfg.allow_minutes if minutes is None else minutes)
        dev = self.policy.allow(t, snooze_s)
        until = time.strftime("%H:%M", time.localtime(time.time() + snooze_s))
        log.info("sharing allowed for %.0f min (until %s)", snooze_s / 60, until)
        if dev:
            self._stats.events.append(dev)
            self._dispatch(dev)

    def resume(self) -> None:
        self.policy.resume()
        log.info("protection resumed")

    def reconfigure(self, **changes) -> None:
        self.cfg = replace(self.cfg, **changes)
        self.scene.cfg = self.policy.cfg = self.cfg

    def reload_owner(self) -> None:
        """After enrolling or forgetting: pick up the new profile, re-judge every face."""
        self.scene.owner = load_matcher(self.cfg) if self.detector.name == "yunet" else None
        for t in self.scene.tracker.tracks:
            t.owner, t.owner_sim, t.owner_checked, t.owner_strikes = None, None, -1e9, 0

    PREVIEW_FPS = 30.0  # the panel's camera view; analysis still runs at cfg.fps

    def run(self, max_frames: int | None = None, ui=None) -> RunStats:
        # With the panel open, frames arrive at preview rate so the picture is smooth, and
        # only every 1/fps-th second one is analysed. Without it, frames arrive at cfg.fps.
        rate = max(self.PREVIEW_FPS, self.cfg.fps) if ui is not None else self.cfg.fps
        src = FrameSource(self.cfg.source, rate, self.cfg.frame_width,
                          self.cfg.flip_horizontal, self.cfg.rotate)
        stats = self._stats = RunStats(detector=self.detector.name)
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_quit", True))
        log.info("watching %s at %.1f fps with %s detector", self.cfg.source, self.cfg.fps, self.detector.name)
        if self.detector.name == "haar":
            log.warning("Haar mode: no head pose, any live frontal face counts as looking")
        fps_ema, last, busy = self.cfg.fps, time.monotonic(), 0.0
        obs, next_analysis = None, -1e9
        try:
            while not self._quit:
                tick = time.monotonic()
                got = src.read()
                if got is None:
                    break
                t, frame = got
                if obs is None or t >= next_analysis or src.is_file:
                    next_analysis = max(next_analysis + 1.0 / self.cfg.fps, t)
                    a0 = time.perf_counter()
                    ev, obs, small = self.process(t, frame)
                    busy += time.perf_counter() - a0
                    stats.frames += 1
                    if ev:
                        stats.events.append(ev)
                    now = time.monotonic()
                    fps_ema = 0.8 * fps_ema + 0.2 / max(now - last, 1e-3)
                    last = now
                else:  # preview-only frame: fresh picture, last analysis' annotations
                    small = self.scene.prepare(frame)
                del frame  # nothing outlives the iteration (preview aside)

                key = -1
                if ui is not None:  # the panel pumps highgui for every window, shield included
                    key = ui.update(self, small, obs, fps_ema)
                else:
                    if self.cfg.hud:
                        cv2.imshow(HUD_WINDOW, annotate(small, obs, self.policy, fps_ema))
                    if self.cfg.hud or any(getattr(a, "active", False) for a in self.actions):
                        k = cv2.waitKey(1)
                        key = -1 if k == -1 else k & 0xFF
                    if self.cfg.hud and key == ord("q") and self.policy.state is not State.ALERT:
                        break
                choice = next((c for a in self.actions if (c := a.pump(key))), None)
                if choice:
                    reason, snooze_s = choice
                    if reason == "allowed":
                        self.allow(t, snooze_s / 60)
                    elif dev := self.policy.dismiss(t, snooze_s, reason):
                        stats.events.append(dev)
                        self._dispatch(dev)

                if ui is not None and ui.wants_enroll:  # the camera is ours; lend it to setup
                    src.close()
                    from .enroll import enroll
                    enroll(self.cfg)
                    ui.wants_enroll = False
                    self.reload_owner()
                    src = FrameSource(self.cfg.source, rate, self.cfg.frame_width,
                                      self.cfg.flip_horizontal, self.cfg.rotate)
                    continue

                if ui is not None and ui.wants_camera_restart:  # rotation changed
                    src.close()
                    ui.wants_camera_restart = False
                    src = FrameSource(self.cfg.source, rate, self.cfg.frame_width,
                                      self.cfg.flip_horizontal, self.cfg.rotate)
                    continue

                if max_frames and stats.frames >= max_frames:
                    break
                if not src.is_file:  # pace live capture; replays run as fast as possible
                    time.sleep(max(0.0, src.period - (time.monotonic() - tick)))
        finally:
            src.close()
            for a in self.actions:
                a.close()
            if self.cfg.hud or ui is not None:
                cv2.destroyAllWindows()
        stats.analysis_ms = 1000 * busy / max(stats.frames, 1)
        return stats
