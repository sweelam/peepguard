"""The privacy shield: a full-screen frosted copy of your own screen with a small
warning card. The frosted look matters. A black box screams "something secret here";
a blur reads as "screen asleep" to the person behind you, while you still recognise
your layout and can reveal it with one key or click."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from . import Action
from ..policy import Event

log = logging.getLogger("peepguard")
WINDOW = "PeepGuard shield"


def _wayland_grab() -> np.ndarray | None:
    """Whole desktop through the compositor's own tool. X11-style grabs come back black on
    Wayland. grim streams to stdout; spectacle and gnome-screenshot need a file, so it is a
    private (0600) file in the RAM-backed runtime dir, read and unlinked immediately."""
    if shutil.which("grim"):  # wlroots: Sway, Hyprland, ...
        try:
            png = subprocess.run(["grim", "-t", "png", "-"], capture_output=True, timeout=3).stdout
            img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) if png else None
            if img is not None:
                return img
        except Exception as e:
            log.debug("grim failed: %s", e)
    tools = [("spectacle", lambda f: ["spectacle", "-b", "-n", "-f", "-o", f]),   # KDE
             ("gnome-screenshot", lambda f: ["gnome-screenshot", "-f", f])]       # GNOME
    for exe, cmd in tools:
        if not shutil.which(exe):
            continue
        fd, path = tempfile.mkstemp(suffix=".png", prefix="peepguard-", dir=os.environ.get("XDG_RUNTIME_DIR"))
        os.close(fd)
        try:
            subprocess.run(cmd(path), capture_output=True, timeout=5)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception as e:
            log.debug("%s failed: %s", exe, e)
        finally:
            os.unlink(path)
    return None


def _crop(desktop: np.ndarray, virtual: dict, mon: dict) -> np.ndarray | None:
    """Cut one monitor out of a whole-desktop screenshot (which may be HiDPI-scaled)."""
    sx, sy = desktop.shape[1] / virtual["width"], desktop.shape[0] / virtual["height"]
    x0, y0 = round((mon["left"] - virtual["left"]) * sx), round((mon["top"] - virtual["top"]) * sy)
    part = desktop[y0:y0 + round(mon["height"] * sy), x0:x0 + round(mon["width"] * sx)]
    return part if part.size else None


def grab_screens() -> list[tuple[dict | None, np.ndarray | None]]:
    """One (monitor geometry, screenshot) per physical display. Geometry is None when mss
    is missing; a screenshot is None when nothing could capture it, so the shield falls
    back to a plain card."""
    try:
        import mss

        with mss.MSS() if hasattr(mss, "MSS") else mss.mss() as sct:
            virtual, mons = sct.monitors[0], sct.monitors[1:] or sct.monitors[:1]
            out = []
            for mon in mons:
                try:
                    shot = cv2.cvtColor(np.asarray(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
                    out.append((mon, shot if shot.max() > 8 else None))  # black = Wayland
                except Exception as e:
                    log.debug("screen grab of %s failed: %s", mon, e)
                    out.append((mon, None))
    except Exception as e:  # no mss, no permission (macOS Screen Recording), headless
        log.debug("monitor enumeration unavailable: %s", e)
        virtual, out = None, [(None, None)]

    if any(shot is None for _, shot in out) and platform.system() == "Linux":
        desktop = _wayland_grab()
        if desktop is not None:
            if virtual is None:
                return [(None, desktop)]
            out = [(mon, shot if shot is not None else _crop(desktop, virtual, mon)) for mon, shot in out]
    return out


def grab_screen() -> np.ndarray | None:
    """The primary display only (used by preview-shield)."""
    screens = grab_screens()
    return next((shot for mon, shot in screens if mon and mon.get("is_primary")), screens[0][1])


COPY = {
    "onlooker": ("Paused for your privacy",
                 "Someone nearby might be able to see your screen."),
    "stranger": ("Paused for your privacy",
                 "This screen is resting until its owner is back."),
    "away": ("Paused while you were away",
             "Your screen is resting until you're back."),
}


def frost(screen: np.ndarray | None, size: tuple[int, int] = (1440, 900)) -> np.ndarray:
    """Your own screen, blurred past legibility: layout and colours survive, text doesn't."""
    w, h = size if screen is None else (screen.shape[1], screen.shape[0])
    if screen is None:
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        base = (np.array([62, 52, 46], np.float32) * (1 - y) + np.array([38, 32, 28], np.float32) * y)
        return np.broadcast_to(base, (h, w, 3)).astype(np.uint8).copy()
    small = cv2.resize(screen, (max(w // 32, 8), max(h // 32, 8)), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 12)
    return cv2.addWeighted(out, 0.8, np.full_like(out, (48, 40, 36)), 0.2, 0)


def card_layout(size: tuple[int, int], reason: str = "onlooker", allow_label: str = "") -> dict:
    """Where everything on the card goes, so drawing and click hit-testing agree."""
    w, h = size
    sc = h / 900  # scale UI with the display
    title, body = COPY.get(reason, COPY["onlooker"])
    font = cv2.FONT_HERSHEY_DUPLEX
    tw = max(cv2.getTextSize(title, font, 0.7 * sc, 1)[0][0], cv2.getTextSize(body, font, 0.5 * sc, 1)[0][0])
    pad, bh = int(36 * sc), int(42 * sc)
    cont_w = cv2.getTextSize("Continue", font, 0.52 * sc, 1)[0][0] + int(44 * sc)
    allow_w = cv2.getTextSize(allow_label or "Allow for 000 min", font, 0.52 * sc, 1)[0][0] + int(44 * sc)
    tx_off = pad + int(60 * sc)
    cw = max(tw, cont_w + allow_w + int(14 * sc)) + tx_off + pad
    ch = int(236 * sc)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    tx, by = x0 + tx_off, y0 + int(126 * sc)
    return {
        "sc": sc, "title": title, "body": body, "card": (x0, y0, cw, ch), "tx": tx, "pad": pad,
        "continue": (tx, by, cont_w, bh),
        "allow": (tx + cont_w + int(14 * sc), by, allow_w, bh),
    }


def render_shield(screen: np.ndarray | None, size: tuple[int, int] = (1440, 900),
                  reason: str = "onlooker", base: np.ndarray | None = None,
                  allow_minutes: str = "15", typing: bool = False) -> np.ndarray:
    frosted = frost(screen, size) if base is None else base
    h, w = frosted.shape[:2]
    allow_label = f"Allow for {allow_minutes}{'_' if typing else ''} min"
    L = card_layout((w, h), reason, allow_label)
    sc, (x0, y0, cw, ch), tx = L["sc"], L["card"], L["tx"]
    r = int(26 * sc)

    # Frosted-glass card: a translucent tint with a hairline edge, not an opaque slab.
    card = frosted.copy()
    _rounded_rect(card, (x0, y0), (x0 + cw, y0 + ch), r, (44, 38, 36))
    out = cv2.addWeighted(card, 0.62, frosted, 0.38, 0)
    edge = out.copy()
    _rounded_rect(edge, (x0, y0), (x0 + cw, y0 + ch), r, (120, 112, 108), outline=max(1, int(sc)))
    out = cv2.addWeighted(edge, 0.35, out, 0.65, 0)

    _shield_icon(out, (x0 + L["pad"] + int(22 * sc), y0 + int(70 * sc)), int(20 * sc), SOFT)
    _text(out, L["title"], (tx, y0 + int(62 * sc)), 0.7 * sc, (240, 240, 244), 1)
    _text(out, L["body"], (tx, y0 + int(94 * sc)), 0.5 * sc, (196, 196, 204), 1)

    # Two choices: reveal now, or let people look for a while (showing a teammate).
    bx, by, bw, bh = L["continue"]
    _rounded_rect(out, (bx, by), (bx + bw, by + bh), bh // 2, (74, 66, 62))
    _text(out, "Continue", (bx + int(22 * sc), by + int(27 * sc)), 0.52 * sc, (236, 236, 240), 1)
    ax, ay, aw, ah = L["allow"]
    if typing:
        _rounded_rect(out, (ax, ay), (ax + aw, ay + ah), ah // 2, (64, 74, 58))
    _rounded_rect(out, (ax, ay), (ax + aw, ay + ah), ah // 2, SOFT, outline=max(1, int(1.5 * sc)))
    _text(out, allow_label, (ax + int(22 * sc), ay + int(27 * sc)), 0.52 * sc, SOFT, 1)

    hint = "Any key continues.  Type minutes, then A or Enter, to allow."
    _text(out, hint, (tx, y0 + ch - int(30 * sc)), 0.42 * sc, (150, 150, 160), 1)
    return out


SOFT = (196, 214, 168)  # calm green, shared with onboarding


def _text(img, s, org, scale, color, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, max(1, thick), cv2.LINE_AA)


def _rounded_rect(img, p1, p2, r, color, outline: int = 0):
    (x1, y1), (x2, y2) = p1, p2
    if outline:
        t = outline
        for a, b in (((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                     ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))):
            cv2.line(img, a, b, color, t, cv2.LINE_AA)
        for (cx, cy), ang in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                              ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, t, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _shield_icon(img, c, r, color):
    """PeepGuard logo: a shield with an eye inside, representing privacy protection."""
    cx, cy = c
    t = max(2, r // 6)

    # Shield outline
    shield_pts = np.array([
        (cx, cy - r),
        (cx + int(r * 0.9), cy - int(r * 0.55)),
        (cx + int(r * 0.8), cy + int(r * 0.35)),
        (cx, cy + r),
        (cx - int(r * 0.8), cy + int(r * 0.35)),
        (cx - int(r * 0.9), cy - int(r * 0.55)),
    ], np.int32)
    cv2.polylines(img, [shield_pts], True, color, t, cv2.LINE_AA)

    # Eye inside shield
    eye_cy = cy - int(r * 0.1)
    eye_rx, eye_ry = int(r * 0.4), int(r * 0.22)
    cv2.ellipse(img, (cx, eye_cy), (eye_rx, eye_ry), 0, 0, 360, color, t, cv2.LINE_AA)

    # Pupil
    pupil_r = int(r * 0.15)
    cv2.circle(img, (cx, eye_cy), pupil_r, color, -1, cv2.LINE_AA)


class ShieldAction(Action):
    """Full-screen pause on every monitor. pump() reports the user's choice:
    ("dismissed", seconds) to reveal now, or ("allowed", seconds) to stop pausing for a while."""

    name = "shield"
    FADE_MS, FADE_STEPS = 320, 10
    REVEAL_SNOOZE_S = 10.0
    MAX_MINUTES = 480
    BACKSPACE, ENTER, ALLOW_KEYS = (8, 127), (10, 13), (ord("a"), ord("A"))

    def __init__(self, allow_minutes: float = 15.0):
        self.allow_minutes = allow_minutes
        self._windows: list[tuple[str, np.ndarray, tuple[int, int]]] = []  # name, frosted base, size
        self._reason = "onlooker"
        self._typed = ""
        self._click: str | None = None
        self._broken = False

    def on_event(self, ev: Event) -> None:
        if ev.kind == "alert":
            self._show(ev.reason)
        else:
            self._hide()

    def _default_minutes(self) -> str:
        return f"{self.allow_minutes:g}"

    def _render(self, base, size) -> np.ndarray:
        return render_shield(None, size, self._reason, base=base,
                             allow_minutes=self._typed or self._default_minutes(), typing=bool(self._typed))

    def _show(self, reason: str) -> None:
        if self._broken or self._windows:
            return
        self._reason, self._typed, self._click = reason, "", None
        screens = grab_screens()  # grab BEFORE our windows cover them
        try:
            frames = []
            for i, (mon, shot) in enumerate(screens):
                size = (mon["width"], mon["height"]) if mon else (1440, 900)
                base = frost(shot, size)
                if base.shape[1::-1] != size:  # HiDPI screenshot: match the window
                    base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
                name = WINDOW if i == 0 else f"{WINDOW} {i + 1}"
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                if mon:  # fullscreen lands on whichever display the window sits on
                    cv2.moveWindow(name, mon["left"], mon["top"])
                    cv2.resizeWindow(name, *size)
                cv2.imshow(name, base)
                cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                if hasattr(cv2, "WND_PROP_TOPMOST"):
                    try:
                        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                    except cv2.error:
                        pass
                cv2.setMouseCallback(name, self._on_mouse, size)
                self._windows.append((name, base, size))
                frames.append((name, base, self._render(base, size)))
            for k in range(1, self.FADE_STEPS + 1):  # blur is instant; only the card eases in
                a = (k / self.FADE_STEPS) ** 2 * (3 - 2 * k / self.FADE_STEPS)  # smoothstep
                for name, base, final in frames:
                    cv2.imshow(name, cv2.addWeighted(final, a, base, 1 - a, 0))
                cv2.waitKey(self.FADE_MS // self.FADE_STEPS)
        except cv2.error as e:
            self._hide()
            self._broken = True
            log.error("shield needs a display (OpenCV highgui): %s", e)

    def _on_mouse(self, event, x, y, _flags, size):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        label = f"Allow for {self._typed or self._default_minutes()}{'_' if self._typed else ''} min"
        L = card_layout(size, self._reason, label)
        ax, ay, aw, ah = L["allow"]
        self._click = "allow" if ax <= x <= ax + aw and ay <= y <= ay + ah else "continue"

    def _redraw(self) -> None:
        for name, base, size in self._windows:
            cv2.imshow(name, self._render(base, size))

    def _hide(self) -> None:
        if self._windows:
            for name, *_ in self._windows:
                try:
                    cv2.destroyWindow(name)
                except cv2.error:
                    pass
            cv2.waitKey(1)
            self._windows = []

    @property
    def active(self) -> bool:
        return bool(self._windows)

    def pump(self, key: int = -1) -> tuple[str, float] | None:
        if not self._windows:
            return None
        if self._click == "continue":
            return "dismissed", self.REVEAL_SNOOZE_S
        if self._click == "allow":
            return self._allow()
        if key == -1:
            return None
        if ord("0") <= key <= ord("9"):  # typing the minutes for "Allow"
            if len(self._typed) < 3:
                self._typed = (self._typed + chr(key)).lstrip("0")
                self._redraw()
            return None
        if key in self.BACKSPACE:
            self._typed = self._typed[:-1]
            self._redraw()
            return None
        if key in self.ALLOW_KEYS or (key in self.ENTER and self._typed):
            return self._allow()
        return "dismissed", self.REVEAL_SNOOZE_S

    def _allow(self) -> tuple[str, float]:
        minutes = float(self._typed) if self._typed else self.allow_minutes
        return "allowed", 60.0 * min(max(minutes, 1.0), self.MAX_MINUTES)

    def close(self) -> None:
        self._hide()
