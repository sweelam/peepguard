"""Onboarding: teach PeepGuard the owner's face. Offered once on the first `run`, or any
time with `peepguard enroll`.

The welcome card comes first and the camera opens only after you agree. Then four short
looks (straight, left, right, chin up) collect a few signatures each, so the profile
covers the angles you actually sit at. Preview frames are shown, never kept; the only
thing written is the signature file (see owner.py).
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .actions.shield import SOFT, _rounded_rect, _shield_icon, _text, frost
from .camera import FrameSource
from .config import Config
from .detector import YuNetDetector
from .owner import OwnerProfile, Recognizer, fetch_sface, owner_file, skip_file

log = logging.getLogger("peepguard")

WINDOW = "PeepGuard setup"
W, H = 960, 720
_ROTATED_90 = False  # Set by _capture based on config
STEPS = [
    "Look at the middle of your screen",
    "Now turn your head a little to the left",
    "And a little to the right",
    "Last one: tilt your chin up slightly",
]
PER_STEP = 4
MIN_FACE_PX = 90        # at the 640 px analysis width: close enough for a clean signature
MIN_GAP_S = 0.3         # samples spaced out in time...
MAX_SAME = 0.97         # ...and actually different from the previous one
MIN_CONSISTENCY = 0.45  # enrolled samples must agree with each other (one person, in focus)
ESC, ENTER = 27, (10, 13)


def _setup_window_size(cfg: Config) -> None:
    """Set window dimensions based on camera rotation."""
    global W, H, _ROTATED_90
    _ROTATED_90 = cfg.rotate in (90, 270)
    if _ROTATED_90:
        W, H = 720, 960  # Portrait mode
    else:
        W, H = 960, 720  # Landscape mode


def onboard(cfg: Config) -> bool:
    """First-run offer. Declining is remembered, so `run` asks only once."""
    _setup_window_size(cfg)
    choice = _card("Let PeepGuard recognise you",
                   ["So it can tell you apart from the people around you,",
                    "PeepGuard can learn your face. It keeps only a short",
                    "numeric signature on this computer, never photos."],
                   "Enter: set up (about 15 seconds)     Esc: not now")
    if choice != "ok":
        skip_file().touch()
        _close()
        log.info("onboarding skipped; run `peepguard enroll` whenever you like")
        return False
    return enroll(cfg, intro=False)


def enroll(cfg: Config, intro: bool = True) -> bool:
    _setup_window_size(cfg)
    try:
        if intro and _card("Let PeepGuard recognise you",
                           ["Sit as you normally do and follow the prompts.",
                            "Only you in view, please. It takes about 15 seconds."],
                           "Enter: start     Esc: cancel") != "ok":
            return _close(False)
        _card("Getting ready...", ["Fetching the face recognition model (37 MB, first time only)."],
              "", wait=False)
        cv2.setNumThreads(cfg.threads)
        rec = Recognizer(fetch_sface())
        det = YuNetDetector(cfg.score_threshold)
        while True:
            samples = _capture(cfg, det, rec)
            if samples is None:
                return _close(False)
            profile = OwnerProfile(np.stack(samples))
            if profile.consistency() >= MIN_CONSISTENCY:
                break
            log.info("enrollment samples disagree (%.2f); asking to retry", profile.consistency())
            if _card("That didn't quite work",
                     ["The samples didn't look like one clear face.",
                      "Try again with a bit more light and only you in view."],
                     "Enter: try again     Esc: cancel") != "ok":
                return _close(False)
        path = profile.save()
        skip_file().unlink(missing_ok=True)
        log.info("owner enrolled: %d samples, consistency %.2f -> %s",
                 len(profile.templates), profile.consistency(), path)
        _card("All set", ["PeepGuard will recognise you from now on.",
                          "Run `peepguard forget` to remove your face signature."], "", wait=False)
        _pause(1.8)
        return _close(True)
    except cv2.error as e:
        log.error("setup needs a display (OpenCV highgui): %s", e)
        return False


def is_enrolled() -> bool:
    return owner_file().is_file()


# ------------------------------------------------------------------- capture

def _capture(cfg: Config, det: YuNetDetector, rec: Recognizer) -> list[np.ndarray] | None:
    src = FrameSource(cfg.source, 30, cfg.frame_width, cfg.flip_horizontal, cfg.rotate)
    samples: list[np.ndarray] = []
    last_t, last_f = -1e9, None
    try:
        while len(samples) < PER_STEP * len(STEPS):
            got = src.read()
            if got is None:
                return None
            now, frame = got
            small = _resize(frame, cfg.frame_width)
            faces = [f for f in det.detect(small) if f.box[2] >= 28]
            luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

            hint, ok_face = None, None
            if luma < cfg.min_lux:
                hint = "A little more light would help"
            elif not faces:
                hint = "Centre your face in the oval"
            elif len(faces) > 1:
                hint = "Just you in view, please"
            elif faces[0].box[2] < MIN_FACE_PX:
                hint = "Come a little closer"
            else:
                ok_face = faces[0]

            if ok_face is not None and now - last_t >= MIN_GAP_S:
                feat = rec.embed(small, ok_face)
                if feat is not None and (last_f is None or float(feat @ last_f) < MAX_SAME):
                    samples.append(feat)
                    last_t, last_f = now, feat

            step = min(len(samples) // PER_STEP, len(STEPS) - 1)
            cv2.imshow(WINDOW, _capture_view(small, step, len(samples), hint, ok_face is not None))
            if cv2.waitKey(1) & 0xFF == ESC:
                return None
            time.sleep(max(0.0, src.period - (time.monotonic() - now)))
        return samples
    finally:
        src.close()


_OVAL: np.ndarray | None = None
_OVAL_SIZE: tuple[int, int] | None = None


def _oval_params() -> tuple[tuple[int, int], tuple[int, int]]:
    """Get oval center and axes based on current window size."""
    return (W // 2, H // 2 - 40), (min(175, W // 5), min(230, H // 4))


def _oval_mask() -> np.ndarray:
    """Soft-edged oval weight map, rebuilt if size changes."""
    global _OVAL, _OVAL_SIZE
    if _OVAL is None or _OVAL_SIZE != (W, H):
        oval_c, oval_axes = _oval_params()
        mask = np.zeros((H, W), np.uint8)
        cv2.ellipse(mask, oval_c, oval_axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
        _OVAL = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255
        _OVAL_SIZE = (W, H)
    return _OVAL


def _capture_view(small: np.ndarray, step: int, n: int, hint: str | None, good: bool) -> np.ndarray:
    view = cv2.resize(cv2.flip(small, 1), (W, H), interpolation=cv2.INTER_LINEAR)  # mirror, like a selfie
    c, axes = _oval_params()
    # Blur at quarter size and scale back up: same look, a fraction of the cost per frame.
    soft = cv2.resize(cv2.GaussianBlur(cv2.resize(view, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3.5),
                      (W, H), interpolation=cv2.INTER_LINEAR)
    dim = cv2.addWeighted(soft, 0.35, np.full_like(view, (40, 32, 28)), 0.65, 0)
    m = _oval_mask()
    out = cv2.blendLinear(view, dim, m, 1.0 - m)

    total = PER_STEP * len(STEPS)
    cv2.ellipse(out, c, (axes[0] + 14, axes[1] + 14), 0, 0, 360, (90, 84, 80), 3, cv2.LINE_AA)
    cv2.ellipse(out, c, (axes[0] + 14, axes[1] + 14), -90, 0, 360 * n / total,
                SOFT if good else (150, 170, 130), 5, cv2.LINE_AA)

    _text(out, f"Step {step + 1} of {len(STEPS)}", (40, 52), 0.55, (180, 180, 188), 1)
    # Responsive card layout
    margin = min(140, W // 7)
    x0, y0, cw, ch = margin, H - 150, W - margin * 2, 110
    card = out.copy()
    _rounded_rect(card, (x0, y0), (x0 + cw, y0 + ch), 22, (44, 38, 36))
    out = cv2.addWeighted(card, 0.75, out, 0.25, 0)
    _text(out, STEPS[step], (x0 + 32, y0 + 46), 0.65, (240, 240, 244), 1)
    _text(out, hint or "Hold still for a moment...", (x0 + 32, y0 + 82), 0.48,
          (150, 150, 160) if hint is None else (150, 196, 230), 1)
    return out


# --------------------------------------------------------------------- cards

def _card(title: str, lines: list[str], hint: str, wait: bool = True) -> str | None:
    """A calm full-window message, same look as the shield. Returns "ok" / "cancel"."""
    # Use current W, H values (may be portrait or landscape)
    img = frost(None, (W, H))
    margin = min(110, W // 8)
    x0, cw = margin, W - margin * 2
    ch = 150 + 34 * len(lines)
    y0 = (H - ch) // 2
    card = img.copy()
    _rounded_rect(card, (x0, y0), (x0 + cw, y0 + ch), 26, (44, 38, 36))
    img = cv2.addWeighted(card, 0.8, img, 0.2, 0)
    _shield_icon(img, (x0 + 52, y0 + 64), 20, SOFT)
    _text(img, title, (x0 + 95, y0 + 74), 0.75, (240, 240, 244), 1)
    for i, line in enumerate(lines):
        _text(img, line, (x0 + 95, y0 + 120 + 34 * i), 0.5, (200, 200, 208), 1)
    if hint:
        _text(img, hint, (x0 + 95, y0 + ch - 30), 0.42, (150, 150, 160), 1)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)  # no Qt toolbar
    cv2.imshow(WINDOW, img)
    cv2.waitKey(1)
    if not wait:
        return None
    while True:
        k = cv2.waitKey(50)
        if k != -1 and (k & 0xFF) in ENTER:
            return "ok"
        if k != -1 and (k & 0xFF) == ESC:
            return "cancel"
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # closed with the X
            return "cancel"


def _pause(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        cv2.waitKey(30)


def _close(result: bool = False) -> bool:
    try:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
    except cv2.error:
        pass
    return result


def _resize(bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w == width:
        return bgr
    return cv2.resize(bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
