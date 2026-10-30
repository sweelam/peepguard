"""Turns raw detections into a judgement for one frame: who is the user, who else is
there, and which of those others are alive and pointed at the screen."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import Config
from .detector import Face
from .pose import Camera, estimate_pose, range_from_box
from .tracker import SceneMotion, Track, Tracker, _iou


class Role(str, Enum):
    PRIMARY = "primary"
    ONLOOKER = "onlooker"   # alive, not the user, facing the screen
    BYSTANDER = "bystander"  # alive, not the user, looking elsewhere
    PICTURE = "picture"     # never moved: poster / photo / paused TV
    PENDING = "pending"     # not enough evidence yet


@dataclass
class TrackView:
    track: Track
    role: Role
    facing: bool | None
    screen_angle: float | None
    distance_mm: float | None
    behind_user: bool


@dataclass
class Observation:
    t: float
    views: list[TrackView] = field(default_factory=list)
    primary: TrackView | None = None
    degraded: str | None = None  # reason we can't trust this frame, if any
    owner_enrolled: bool = False  # recognition is on, so "no primary" means "not you"

    @property
    def onlookers(self) -> list[TrackView]:
        return [v for v in self.views if v.role is Role.ONLOOKER]


class SceneAnalyzer:
    MAX_STATIC_ZONES = 16

    def __init__(self, cfg: Config, detector, clock=time.monotonic, owner=None):
        self.cfg = cfg
        self.detector = detector
        self.clock = clock
        self.owner = owner  # OwnerMatcher, or None: largest-face heuristic only
        self.tracker = Tracker(max_gap_s=max(1.5, cfg.gap_tolerance_s * 2))
        self.motion = SceneMotion()
        self.cam: Camera | None = None
        self._primary_id: int | None = None
        self._static_zones: list[np.ndarray] = []

    # ------------------------------------------------------------------ public
    def analyze(self, bgr: np.ndarray, now: float | None = None) -> Observation:
        now = self.clock() if now is None else now
        bgr = self.prepare(bgr)
        h, w = bgr.shape[:2]
        if self.cam is None or (self.cam.width, self.cam.height) != (w, h):
            self.cam = Camera.approximate(w, h, self.cfg.hfov_deg)

        obs = Observation(t=now, owner_enrolled=self.owner is not None)
        luma = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
        if luma < self.cfg.min_lux:
            obs.degraded = f"too dark (luma {luma:.0f} < {self.cfg.min_lux:.0f})"

        faces = [f for f in self.detector.detect(bgr) if f.box[2] >= self.cfg.min_face_px]
        shift = self.motion.update(bgr, [f.box for f in faces])
        tracks = self.tracker.update(faces, now, shift)
        for t in tracks:
            self._update_pose(t, now)
            if self.owner is not None and not obs.degraded:
                self.owner.check(t, bgr, now)

        primary = self._pick_primary(tracks)
        for t in tracks:
            view = self._classify(t, primary, now)
            obs.views.append(view)
            if view.role is Role.PRIMARY:
                obs.primary = view
        return obs

    # ----------------------------------------------------------------- helpers
    def prepare(self, bgr: np.ndarray) -> np.ndarray:
        """Downscale to the analysis width (idempotent)."""
        h, w = bgr.shape[:2]
        if w == self.cfg.frame_width:
            return bgr
        s = self.cfg.frame_width / w
        return cv2.resize(bgr, (self.cfg.frame_width, round(h * s)), interpolation=cv2.INTER_AREA)

    def _update_pose(self, t: Track, now: float) -> None:
        f = t.face
        t.pose = None
        if f.landmarks is not None:
            p = estimate_pose(f.landmarks, self.cam, self.cfg.screen_offset_mm)
            # Landmarks that don't fit a face model (occlusion, extreme profile) are noise.
            if p is not None and p.reproj_px <= 0.12 * f.box[2]:
                t.pose = p
        if t.pose is not None:
            a = t.pose.screen_angle
            if t.screen_angle_ema is None:
                t.screen_angle_ema = a
            else:  # time-based, so smoothing doesn't shrink as fps goes up
                k = 1.0 - np.exp(-(now - t.screen_angle_t) / self.cfg.pose_smoothing_s)
                t.screen_angle_ema += k * (a - t.screen_angle_ema)
            t.screen_angle_t = now
            t.distance_mm = t.pose.distance_mm
        else:
            t.distance_mm = range_from_box(f.box[2], self.cam)

    def _pick_primary(self, tracks: list[Track]) -> Track | None:
        """A recognised owner always wins. Otherwise the largest, most central face that isn't
        a known stranger wins; the incumbent is sticky so a passer-by who leans in for a
        moment doesn't steal the 'user' role.

        Special case: if there's exactly ONE face detected and recognition temporarily fails,
        treat them as primary anyway - a lone person shouldn't trigger their own blur shield."""
        owners = [t for t in tracks if t.owner]
        if owners:
            best = max(owners, key=lambda t: t.owner_sim or 0.0)
            self._primary_id = best.id
            return best
        non_strangers = [t for t in tracks if t.owner is not False]
        # If only one face is detected but recognition failed, still treat them as primary.
        # This prevents the blur from triggering when the enrolled user is alone but
        # temporarily misrecognized (bad angle, lighting, etc.)
        if not non_strangers:
            if len(tracks) == 1:
                self._primary_id = tracks[0].id
                return tracks[0]
            self._primary_id = None
            return None
        tracks = non_strangers
        cx, cy = self.cam.width / 2, self.cam.height / 2
        diag = np.hypot(cx, cy)

        def score(t: Track) -> float:
            off = np.hypot(*(t.face.center - (cx, cy))) / diag
            return t.face.area * (1.0 - 0.6 * off)

        best = max(tracks, key=score)
        current = next((t for t in tracks if t.id == self._primary_id), None)
        if current is not None and score(current) >= 0.6 * score(best):
            return current
        self._primary_id = best.id
        return best

    def _classify(self, t: Track, primary: Track | None, now: float) -> TrackView:
        cfg = self.cfg
        angle = t.screen_angle_ema if t.pose is not None else None
        if angle is not None:
            facing = angle <= cfg.attention_cone_deg
        elif t.face.landmarks is None:
            facing = True  # Haar frontal cascade: a hit means roughly frontal
        else:
            facing = None  # YuNet saw a face but landmarks were unusable

        behind = bool(
            primary is not None and t is not primary and primary.distance_mm
            and t.distance_mm and t.distance_mm > 1.15 * primary.distance_mm
        )
        view = TrackView(t, Role.PENDING, facing, angle, t.distance_mm, behind)

        if primary is not None and t.id == primary.id:
            view.role = Role.PRIMARY
            return view

        in_static_zone = any(_iou(t.face.box, z) > 0.5 for z in self._static_zones)
        needed = cfg.static_motion_frac * (3.0 if in_static_zone else 1.0)
        alive = t.update_liveness(now, cfg.static_window_s, needed)

        if not alive:
            if t.age >= cfg.static_window_s:
                view.role = Role.PICTURE
                self._remember_static(t.face.box)
            return view

        view.role = Role.ONLOOKER if facing else Role.BYSTANDER
        return view

    def _remember_static(self, box: np.ndarray) -> None:
        if any(_iou(box, z) > 0.7 for z in self._static_zones):
            return
        self._static_zones.append(box.copy())
        del self._static_zones[: -self.MAX_STATIC_ZONES]
