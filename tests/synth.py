"""Synthetic webcam scenes built from one real face photo, so end-to-end tests exercise
the real detector, pose, tracker and policy without needing a camera or people.

The sample photo's head is turned ~25-30 deg off the camera axis. Blending it with its
own mirror image (about the eye/mouth midline) gives a near-frontal face (~10-15 deg).
So we get two "people" from one photo: FRONTAL reads the screen, TURNED looks past it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

FACE = Path(__file__).parent / "data" / "face.jpg"
W, H = 640, 480


MIDLINE_X = 295.6  # eye/mouth midline of the sample photo (from YuNet landmarks)


def _face_patch(frontal: bool) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(FACE))
    if frontal:
        M = np.float32([[-1, 0, 2 * MIDLINE_X], [0, 1, 0]])
        img = cv2.addWeighted(img, 0.5, cv2.warpAffine(img, M, img.shape[1::-1], borderMode=cv2.BORDER_REFLECT), 0.5, 0)
    patch = img[150:430, 170:420]  # head and a bit of hair
    ph, pw = patch.shape[:2]
    mask = np.zeros((ph, pw), np.float32)
    cv2.ellipse(mask, (pw // 2, ph // 2), (pw // 2 - 4, ph // 2 - 4), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    return patch, mask


PATCHES = {True: _face_patch(True), False: _face_patch(False)}


def background(seed: int = 7) -> np.ndarray:
    """An 'office wall': low-frequency texture plus a few hard-edged objects, so phase
    correlation has something to lock onto."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(60, 170, (12, 16, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    for _ in range(9):
        x, y = int(rng.integers(0, W - 80)), int(rng.integers(0, H - 80))
        cv2.rectangle(base, (x, y), (x + int(rng.integers(30, 120)), y + int(rng.integers(20, 90))),
                      tuple(int(c) for c in rng.integers(20, 230, 3)), -1)
    return base


@dataclass
class Actor:
    """place(t) -> (cx, cy, face_width_px) or None when out of frame."""
    place: Callable[[float], tuple[float, float, float] | None]
    frontal: bool = True


@dataclass
class Scene:
    actors: list[Actor] = field(default_factory=list)
    camera_shift: Callable[[float], tuple[float, float]] = lambda t: (0.0, 0.0)
    brightness: Callable[[float], float] = lambda t: 1.0
    noise: float = 4.0
    seed: int = 1

    def __post_init__(self):
        self._bg = background()
        self._rng = np.random.default_rng(self.seed)

    def frame(self, t: float) -> np.ndarray:
        # world = background + actors; then the "camera" moves over the world
        pad = 40
        world = cv2.copyMakeBorder(self._bg, pad, pad, pad, pad, cv2.BORDER_REFLECT).astype(np.float32)
        for a in self.actors:
            p = a.place(t)
            if p is not None:
                _paste(world, p[0] + pad, p[1] + pad, p[2], a.frontal)
        dx, dy = self.camera_shift(t)
        M = np.float32([[1, 0, -pad - dx], [0, 1, -pad - dy]])
        img = cv2.warpAffine(world, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = img * self.brightness(t) + self._rng.normal(0, self.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


# The detector's face box is ~0.58 of the patch width; size actors by face width.
FACE_FRAC = 0.58


def _paste(world: np.ndarray, cx: float, cy: float, face_w: float, frontal: bool) -> None:
    PATCH, MASK = PATCHES[frontal]
    pw = max(8, int(round(face_w / FACE_FRAC)))
    ph = int(round(pw * PATCH.shape[0] / PATCH.shape[1]))
    patch = cv2.resize(PATCH, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = cv2.resize(MASK, (pw, ph))[..., None]
    x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    x1, y1 = x0 + pw, y0 + ph
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x1, world.shape[1]), min(y1, world.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = world[Y0:Y1, X0:X1]
    p = patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    m = mask[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    world[Y0:Y1, X0:X1] = sub * (1 - m) + p * m


# ---- canned actors -------------------------------------------------------------

def user(t: float):
    """The person at the laptop: big, central, low in frame, breathing/swaying."""
    return (330 + 6 * math.sin(t * 1.3), 300 + 4 * math.sin(t * 0.9), 170 + 3 * math.sin(t * 0.7))


def poster(x: float = 520, y: float = 95, w: float = 50):
    return lambda t: (x, y, w)


def walker(enter: float, leave: float, x: float, y: float = 130, w: float = 54, from_x: float = -60):
    """Walks in from the side over 1 s, stands (swaying), walks out."""
    def place(t):
        if t < enter or t > leave + 1.0:
            return None
        if t < enter + 1.0:
            k = t - enter
            return (from_x + (x - from_x) * k, y, w)
        if t > leave:
            k = t - leave
            return (x + (from_x - x) * k, y, w)
        return (x + 4 * math.sin(t * 2.1), y + 3 * math.sin(t * 1.7), w + 2 * math.sin(t))
    return place
