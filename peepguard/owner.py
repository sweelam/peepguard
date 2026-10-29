"""Owner recognition (opt-in). Without it, "the user" is simply the largest, most central
face, which is wrong whenever someone leans in closer than you or sits down at your
laptop while you're away. With it, the user is *you*.

What is kept: a handful of 128-number SFace signatures in one file only you can read.
No photos. `peepguard forget` deletes it. Recognition never runs unless you enrolled.

Matching is deliberately three-way. A face is the OWNER only on a confident match, a
STRANGER only on a confident mismatch, and anything in between stays unknown and is
treated the old way. A bad angle or dim light must never turn you into an onlooker of
your own screen.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

import cv2
import numpy as np

from .detector import Face, _fetch

log = logging.getLogger("peepguard")

SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"


def data_dir() -> Path:
    """Per-user, private: ~/Library/Application Support/PeepGuard or $XDG_DATA_HOME/peepguard."""
    if platform.system() == "Darwin":
        d = Path.home() / "Library" / "Application Support" / "PeepGuard"
    else:
        d = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "peepguard"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def owner_file() -> Path:
    return data_dir() / "owner.npz"


def skip_file() -> Path:
    """Marker: the user declined onboarding, so `run` stops offering it."""
    return data_dir() / "onboarding-declined"


def fetch_sface() -> Path:
    # Lives in the user's data dir, not the package: frozen builds are read-only.
    return _fetch(SFACE_URL, data_dir() / "face_recognition_sface_2021dec.onnx", SFACE_SHA256)


class Recognizer:
    def __init__(self, model: Path | None = None):
        self._rec = cv2.FaceRecognizerSF.create(str(model or fetch_sface()), "")

    def embed(self, bgr: np.ndarray, face: Face) -> np.ndarray | None:
        """Unit-length signature of one YuNet face, or None without landmarks."""
        if face.landmarks is None:
            return None
        row = np.concatenate([face.box, face.landmarks.ravel(), [face.score]]).astype(np.float32)[None]
        f = self._rec.feature(self._rec.alignCrop(bgr, row)).ravel().astype(np.float32)
        n = float(np.linalg.norm(f))
        return f / n if n > 0 else None


class OwnerProfile:
    def __init__(self, templates: np.ndarray):
        self.templates = np.asarray(templates, np.float32).reshape(-1, 128)

    def similarity(self, feat: np.ndarray) -> float:
        """Mean of the best few cosine matches: one lucky template can't carry a stranger."""
        sims = np.sort(self.templates @ feat)[::-1]
        return float(sims[: min(3, len(sims))].mean())

    def consistency(self) -> float:
        """How alike the enrolled samples are; low means a bad enrollment (two people, blur)."""
        s = self.templates @ self.templates.T
        n = len(self.templates)
        return float((s.sum() - n) / max(n * (n - 1), 1))

    def save(self, path: Path | None = None) -> Path:
        path = path or owner_file()
        tmp = path.with_name(path.name + ".part")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            np.savez(f, templates=self.templates, version=np.array(1))
        tmp.replace(path)
        return path

    @staticmethod
    def load(path: Path | None = None) -> "OwnerProfile | None":
        path = path or owner_file()
        if not path.is_file():
            return None
        with np.load(path) as z:
            return OwnerProfile(z["templates"])


def forget(path: Path | None = None) -> bool:
    path = path or owner_file()
    existed = path.is_file()
    path.unlink(missing_ok=True)
    return existed


class OwnerMatcher:
    """Labels tracks owner / stranger / unknown (track.owner = True / False / None)."""

    UNKNOWN_RECHECK_S = 0.25
    MIN_FACE_PX = 48       # smaller than this, a signature is mostly noise
    MAX_YAW_DEG = 40       # strongly turned heads don't match reliably; don't judge them

    def __init__(self, profile: OwnerProfile, recognizer: Recognizer, match: float, reject: float,
                 recheck_s: float):
        self.profile, self.recognizer = profile, recognizer
        self.match, self.reject, self.recheck_s = match, reject, recheck_s

    def check(self, t, bgr: np.ndarray, now: float) -> None:
        wait = self.recheck_s if t.owner is not None else self.UNKNOWN_RECHECK_S
        if now - t.owner_checked < wait or t.face.box[2] < self.MIN_FACE_PX:
            return
        if t.pose is not None and abs(t.pose.yaw) > self.MAX_YAW_DEG:
            return
        feat = self.recognizer.embed(bgr, t.face)
        if feat is None:
            return
        t.owner_checked = now
        sim = self.profile.similarity(feat)
        t.owner_sim = sim if t.owner_sim is None else 0.5 * t.owner_sim + 0.5 * sim
        if t.owner_sim >= self.match:
            t.owner, t.owner_strikes = True, 0
        elif t.owner_sim <= self.reject:
            # A recognised owner keeps the role through one bad look; it takes two.
            t.owner_strikes += 1
            if t.owner is not True or t.owner_strikes >= 2:
                t.owner = False


def load_matcher(cfg) -> OwnerMatcher | None:
    if cfg.owner == "off":
        return None
    profile = OwnerProfile.load()
    if profile is None:
        return None
    try:
        rec = Recognizer()
    except Exception as e:
        log.warning("owner recognition unavailable (%s); using largest-face heuristic", e)
        return None
    log.info("owner recognition on (%d enrolled samples)", len(profile.templates))
    return OwnerMatcher(profile, rec, cfg.owner_match, cfg.owner_reject, cfg.owner_check_s)
