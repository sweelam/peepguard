"""Short-term face tracking plus a liveness signal that rejects posters and photos.

Identity here is purely geometric (box overlap between consecutive frames). No face
embeddings, nothing that could recognise a person across sessions.

Liveness idea: a real head is never perfectly still. Over a few seconds it sways,
nods, shifts weight. A poster, a framed photo, or a face on a TV paused behind you
does not. We measure each track's motion *relative to the scene* (the whole frame's
shift is removed with phase correlation, so nudging the laptop doesn't make the
poster "come alive") and require some within a sliding window.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Face
from .pose import Pose


@dataclass
class Track:
    id: int
    face: Face
    born: float
    last_seen: float
    hits: int = 1
    pose: Pose | None = None
    screen_angle_ema: float | None = None
    screen_angle_t: float | None = None  # when screen_angle_ema was last updated
    distance_mm: float | None = None
    alive_since: float | None = None
    # owner recognition: True = you, False = someone else, None = not sure (yet)
    owner: bool | None = None
    owner_sim: float | None = None
    owner_checked: float = -1e9
    owner_strikes: int = 0
    # cumulative scene-relative position of the face centre, normalised by face width
    _base_w: float = 1.0
    _residual: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _trail: deque = field(default_factory=deque)  # (t, residual_x, residual_y, log_scale)

    @property
    def age(self) -> float:
        return self.last_seen - self.born

    def motion(self, now: float, window: float) -> float:
        """Peak-to-peak scene-relative movement inside the window, as a fraction of face
        width (translation) or relative size change (approach/retreat), whichever is larger."""
        pts = [p for p in self._trail if now - p[0] <= window]
        if len(pts) < 2:
            return 0.0
        a = np.array(pts)[:, 1:]
        if len(a) >= 3:  # rolling median of 3 kills single-frame detector glitches
            a = np.median(np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)
        span = np.ptp(a, axis=0)
        # Size estimates are noisier than position, so approach/retreat is down-weighted.
        return float(max(span[0], span[1], 0.5 * span[2]))

    def update_liveness(self, now: float, window: float, motion_frac: float) -> bool:
        """Latching: once a track has shown life it stays alive for as long as it is tracked,
        so a real onlooker who freezes mid-read does not silently drop out."""
        if self.alive_since is None and self.motion(now, window) >= motion_frac:
            self.alive_since = now
        return self.alive_since is not None


def _anchor(f: Face) -> np.ndarray:
    """Landmark centroid is far steadier frame-to-frame than the box centre."""
    return f.landmarks.mean(axis=0) if f.landmarks is not None else f.center


def _size(f: Face) -> float:
    if f.landmarks is not None:  # eye-to-mouth span: stable, rotation-tolerant
        eyes = f.landmarks[:2].mean(axis=0)
        mouth = f.landmarks[3:].mean(axis=0)
        return max(float(np.linalg.norm(eyes - mouth)) * 2.6, 1.0)
    return max(float(f.box[2]), 1.0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def _affinity(a: Face, b: Face) -> float:
    """IoU, or for a fast mover (someone walking in at 4 fps jumps most of a face width
    per frame) a centre-distance score, as long as the size is consistent."""
    iou = _iou(a.box, b.box)
    wa, wb = float(a.box[2]), float(b.box[2])
    ratio = min(wa, wb) / max(wa, wb, 1e-9)
    if ratio < 0.7:
        return iou
    dist = float(np.linalg.norm(a.center - b.center)) / max(wa, wb)
    return max(iou, 0.5 * (1.0 - dist / 1.5)) if dist < 1.5 else iou


class SceneMotion:
    """Global frame-to-frame translation (camera shake, laptop nudged).

    Sparse LK optical flow on background corners, median displacement. People are
    masked out first: the user is often the largest moving thing in a webcam frame,
    and a whole-frame method (phase correlation) happily locks onto their swaying,
    which would make a static poster look alive."""

    def __init__(self, width: int = 320, max_corners: int = 120):
        self._w = width
        self._max = max_corners
        self._prev: np.ndarray | None = None
        self._prev_mask_boxes: list[np.ndarray] = []

    def update(self, bgr: np.ndarray, people: list[np.ndarray] = ()) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self._w / w
        g = cv2.cvtColor(cv2.resize(bgr, (self._w, round(h * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        boxes = [b * s for b in people]
        shift = np.zeros(2)
        if self._prev is not None and self._prev.shape == g.shape:
            mask = _background_mask(g.shape, boxes + self._prev_mask_boxes)
            pts = cv2.goodFeaturesToTrack(self._prev, self._max, 0.01, 6, mask=mask)
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, g, pts, None, winSize=(15, 15), maxLevel=2)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    d = (nxt[ok] - pts[ok]).reshape(-1, 2)
                    med = np.median(d, axis=0)
                    if np.hypot(*med) > 0.15:  # sub-pixel noise floor at this width
                        shift = med / s
        self._prev, self._prev_mask_boxes = g, boxes
        return shift


def _background_mask(shape, boxes) -> np.ndarray:
    """Everything except faces and the body hanging below each of them."""
    m = np.full(shape, 255, np.uint8)
    H, W = shape
    for x, y, w, h in boxes:
        x0, x1 = int(max(0, x - 0.6 * w)), int(min(W, x + 1.6 * w))
        y0 = int(max(0, y - 0.4 * h))
        m[y0:H, x0:x1] = 0
    return m


class Tracker:
    def __init__(self, max_gap_s: float = 1.0, iou_min: float = 0.2, trail_s: float = 12.0):
        self._tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)
        self._max_gap = max_gap_s
        self._iou_min = iou_min
        self._trail_s = trail_s

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, faces: list[Face], now: float, scene_shift: np.ndarray) -> list[Track]:
        live = list(self._tracks.values())
        pairs = sorted(
            ((_affinity(t.face, f), ti, fi) for ti, t in enumerate(live) for fi, f in enumerate(faces)),
            reverse=True,
        )
        used_t, used_f = set(), set()
        for aff, ti, fi in pairs:
            if aff < self._iou_min or ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            self._advance(live[ti], faces[fi], now, scene_shift)

        for fi, f in enumerate(faces):
            if fi not in used_f:
                tid = next(self._ids)
                t = Track(id=tid, face=f, born=now, last_seen=now, _base_w=_size(f))
                t._trail.append((now, 0.0, 0.0, 0.0))
                self._tracks[tid] = t

        for tid in [tid for tid, t in self._tracks.items() if now - t.last_seen > self._max_gap]:
            del self._tracks[tid]
        return [t for t in self._tracks.values() if t.last_seen == now]

    def _advance(self, t: Track, f: Face, now: float, shift: np.ndarray) -> None:
        w = max(float(f.box[2]), 1.0)
        delta = (_anchor(f) - _anchor(t.face)) - shift
        t._residual = t._residual + delta / w
        log_scale = float(np.log(_size(f) / t._base_w))
        t._trail.append((now, t._residual[0], t._residual[1], log_scale))
        while t._trail and now - t._trail[0][0] > self._trail_s:
            t._trail.popleft()
        t.face = f
        t.last_seen = now
        t.hits += 1
