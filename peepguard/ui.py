"""`peepguard ui`: one small window with the camera on the right and a control panel on
the left. It shows what PeepGuard thinks right now (status, who's in view, whether it
knows you) and puts the everyday controls one click away: allow for a while, resume,
sensitivity, face setup, hide the preview. Drawn with OpenCV like the shield, so it
needs nothing beyond what `run` already uses. Preview frames are shown, never kept.
"""

from __future__ import annotations

import cv2
import numpy as np

from .actions.shield import _rounded_rect, _text
from .hud import COLORS, annotate
from .owner import forget, owner_file
from .policy import State
from .scene import Observation, Role

WINDOW = "PeepGuard"
PANEL_W, CAM_W, CAM_H = 380, 800, 600

# Modern dark theme colors
BG = (18, 18, 20)
PANEL_BG = (24, 24, 28)
CARD = (36, 36, 42)
CARD_HOVER = (44, 44, 52)
DIVIDER = (48, 48, 56)

# Text colors
TEXT_PRIMARY = (245, 245, 250)
TEXT_SECONDARY = (180, 180, 190)
TEXT_MUTED = (110, 110, 120)

# Accent colors
ACCENT_GREEN = (160, 220, 140)       # Primary green
ACCENT_GREEN_DIM = (100, 160, 90)    # Darker green
ACCENT_AMBER = (80, 180, 240)        # Warning amber (BGR)
ACCENT_RED = (100, 100, 230)         # Alert red (BGR)
ACCENT_BLUE = (220, 160, 100)        # Info blue (BGR)


def _draw_logo(img: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
    """Draw the PeepGuard logo: a shield with an eye symbol inside."""
    cx, cy = center
    r = size

    # Draw shield outline
    shield_pts = np.array([
        (cx, cy - r),                           # top
        (cx + int(r * 0.9), cy - int(r * 0.55)),  # top right
        (cx + int(r * 0.8), cy + int(r * 0.35)),  # bottom right
        (cx, cy + r),                           # bottom
        (cx - int(r * 0.8), cy + int(r * 0.35)),  # bottom left
        (cx - int(r * 0.9), cy - int(r * 0.55)),  # top left
    ], np.int32)

    t = max(2, r // 6)
    cv2.polylines(img, [shield_pts], True, color, t, cv2.LINE_AA)

    # Draw eye inside shield
    eye_cy = cy - int(r * 0.1)
    eye_rx, eye_ry = int(r * 0.4), int(r * 0.22)

    # Eye outline (almond shape)
    cv2.ellipse(img, (cx, eye_cy), (eye_rx, eye_ry), 0, 0, 360, color, t, cv2.LINE_AA)

    # Pupil
    pupil_r = int(r * 0.15)
    cv2.circle(img, (cx, eye_cy), pupil_r, color, -1, cv2.LINE_AA)

    # Highlight dot in pupil
    highlight_r = max(1, pupil_r // 3)
    cv2.circle(img, (cx - pupil_r // 3, eye_cy - pupil_r // 3), highlight_r, PANEL_BG, -1, cv2.LINE_AA)


def _draw_gradient_rect(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int],
                        color1: tuple[int, int, int], color2: tuple[int, int, int]) -> None:
    """Draw a vertical gradient rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for y in range(y1, y2):
        alpha = (y - y1) / max(y2 - y1, 1)
        color = tuple(int(c1 * (1 - alpha) + c2 * alpha) for c1, c2 in zip(color1, color2))
        cv2.line(img, (x1, y), (x2, y), color, 1)


def _draw_section_header(img: np.ndarray, x: int, y: int, w: int, text: str) -> None:
    """Draw a section header with subtle styling."""
    _text(img, text, (x, y), 0.38, TEXT_MUTED, 1)
    # Subtle line after text
    text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)[0][0]
    cv2.line(img, (x + text_w + 12, y - 4), (x + w, y - 4), DIVIDER, 1, cv2.LINE_AA)


def _draw_status_badge(img: np.ndarray, x: int, y: int, w: int, h: int,
                       label: str, color: tuple[int, int, int], icon_color: tuple[int, int, int]) -> None:
    """Draw a status badge with icon and label."""
    _rounded_rect(img, (x, y), (x + w, y + h), h // 2, CARD)

    # Status dot with glow effect
    dot_x, dot_y = x + 24, y + h // 2
    # Glow
    for r, a in [(10, 0.1), (7, 0.15), (5, 0.2)]:
        overlay = img.copy()
        cv2.circle(overlay, (dot_x, dot_y), r, icon_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, a, img, 1 - a, 0, img)
    # Core dot
    cv2.circle(img, (dot_x, dot_y), 5, icon_color, -1, cv2.LINE_AA)

    _text(img, label, (x + 44, y + h // 2 + 5), 0.48, TEXT_PRIMARY, 1)


def _draw_info_card(img: np.ndarray, x: int, y: int, w: int,
                    title: str, subtitle: str, icon: str | None = None) -> int:
    """Draw an info card and return height used."""
    h = 58
    _rounded_rect(img, (x, y), (x + w, y + h), 12, CARD)
    _text(img, title, (x + 16, y + 24), 0.5, TEXT_PRIMARY, 1)
    _text(img, subtitle, (x + 16, y + 44), 0.38, TEXT_SECONDARY, 1)
    return h


class ControlPanel:
    def __init__(self):
        self._buttons: list[tuple[tuple[int, int, int, int], str]] = []
        self._click: tuple[int, int] | None = None
        self._opened = False
        self.show_camera = True
        self.wants_enroll = False
        self.wants_camera_restart = False

    # ----------------------------------------------------------------- frame
    def update(self, guard, small: np.ndarray, obs: Observation, fps: float) -> int:
        if not self._opened:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            cv2.setMouseCallback(WINDOW, self._on_mouse)
            self._opened = True

        img = np.full((CAM_H, PANEL_W + CAM_W, 3), BG, np.uint8)
        self._buttons = []
        self._panel(img, guard, obs)
        self._camera(img, guard, small, obs, fps)
        cv2.imshow(WINDOW, img)

        k = cv2.waitKey(1)
        key = -1 if k == -1 else k & 0xFF
        if self._click is not None:
            self._handle(guard, obs, *self._click)
            self._click = None
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            guard._quit = True
        return key if any(getattr(a, "active", False) for a in guard.actions) else -1

    # ----------------------------------------------------------------- panel
    def _panel(self, img: np.ndarray, guard, obs: Observation) -> None:
        # Panel background with subtle gradient
        panel = img[:, :PANEL_W]
        _draw_gradient_rect(panel, (0, 0), (PANEL_W, CAM_H), (28, 28, 32), (20, 20, 24))

        x, w = 24, PANEL_W - 48

        # Logo and title
        _draw_logo(img, (x + 22, 44), 18, ACCENT_GREEN)
        _text(img, "PeepGuard", (x + 52, 52), 0.75, TEXT_PRIMARY, 1)
        _text(img, "Privacy Guardian", (x + 52, 72), 0.35, TEXT_MUTED, 1)

        # Divider
        cv2.line(img, (x, 95), (x + w, 95), DIVIDER, 1, cv2.LINE_AA)

        # Status badge
        label, icon_color = self._status(guard, obs)
        _draw_status_badge(img, x, 110, w, 44, label, CARD, icon_color)

        # Owner section
        y = 175
        _draw_section_header(img, x, y, w, "OWNER")
        enrolled = guard.scene.owner is not None
        if not enrolled:
            owner_line, sub = "Not enrolled", "Tap Face Setup to begin"
        elif obs.primary is not None and obs.primary.track.owner:
            owner_line, sub = "Verified", f"Confidence: {obs.primary.track.owner_sim:.0%}"
        elif obs.primary is not None:
            owner_line, sub = "Verifying...", "Please wait"
        else:
            owner_line, sub = "Not in view", "Screen locks when strangers appear"
        _draw_info_card(img, x, y + 14, w, owner_line, sub)

        # In view section
        y = 260
        _draw_section_header(img, x, y, w, "DETECTION")
        others = [v for v in obs.views if v is not obs.primary]
        looking = sum(v.role is Role.ONLOOKER for v in others)
        if not others:
            view_line, view_sub = "Clear", "No one else detected"
        else:
            people = f"{len(others)} {'person' if len(others) == 1 else 'people'}"
            view_sub = f"{looking} looking at screen" if looking else "Not looking at screen"
            view_line = people
        _draw_info_card(img, x, y + 14, w, view_line, view_sub)

        # Role indicators
        if others:
            cx = x + 16
            for v in others[:8]:
                cv2.circle(img, (cx, y + 82), 4, COLORS[v.role], -1, cv2.LINE_AA)
                cx += 14

        # Sharing section
        y = 350
        _draw_section_header(img, x, y, w, "SHARING")
        left = guard.policy.snoozed_for(obs.t)
        if left > 15:
            self._button(img, (x, y + 14, w, 42), f"Resume  ({left / 60:.0f} min left)", "resume", style="outline")
        else:
            self._button(img, (x, y + 14, w, 42), f"Allow for {guard.cfg.allow_minutes:g} min", "allow", style="accent")

        # Settings section
        y = 420
        _draw_section_header(img, x, y, w, "SETTINGS")
        self._stepper(img, (x, y + 14, w), "Detection cone", f"{guard.cfg.attention_cone_deg:.0f}°", "cone")
        self._stepper(img, (x, y + 48, w), "Alert delay", f"{guard.cfg.dwell_s:.1f}s", "dwell")
        self._stepper(img, (x, y + 82, w), "Rotation", f"{guard.cfg.rotate}°", "rotate")

        # Bottom buttons
        y = CAM_H - 58
        bw = (w - 16) // 3
        self._button(img, (x, y, bw, 40), "Hide" if self.show_camera else "Show", "camera", style="subtle")
        self._button(img, (x + bw + 8, y, bw, 40), "Forget" if enrolled else "Setup",
                     "forget" if enrolled else "enroll", style="subtle")
        self._button(img, (x + 2 * (bw + 8), y, bw, 40), "Quit", "quit", style="subtle")

    def _status(self, guard, obs: Observation) -> tuple[str, tuple[int, int, int]]:
        p = guard.policy
        if p.degraded:
            return "Low light", TEXT_MUTED
        if p.state is State.ALERT:
            return "Screen protected", ACCENT_RED
        if p.snoozed_for(obs.t) > 15:
            return "Sharing mode", ACCENT_AMBER
        if p.state is State.SUSPECT:
            return "Checking...", ACCENT_AMBER
        return "Watching", ACCENT_GREEN

    def _stepper(self, img, rect, label, value, key) -> None:
        x, y, w = rect
        h = 32

        # Label and value
        _text(img, label, (x, y + 20), 0.42, TEXT_SECONDARY, 1)
        _text(img, value, (x + w - 130, y + 20), 0.42, TEXT_PRIMARY, 1)

        # Stepper buttons
        btn_w, btn_h = 36, 28
        self._button(img, (x + w - 78, y + 2, btn_w, btn_h), "-", key + "-", style="stepper")
        self._button(img, (x + w - 38, y + 2, btn_w, btn_h), "+", key + "+", style="stepper")

    def _button(self, img, rect, label, action, style: str = "default") -> None:
        x, y, w, h = rect
        r = min(h // 2, 14)

        if style == "accent":
            # Green accent button
            _rounded_rect(img, (x, y), (x + w, y + h), r, ACCENT_GREEN_DIM)
            _rounded_rect(img, (x, y), (x + w, y + h), r, ACCENT_GREEN, outline=2)
            text_color = ACCENT_GREEN
        elif style == "outline":
            _rounded_rect(img, (x, y), (x + w, y + h), r, CARD)
            _rounded_rect(img, (x, y), (x + w, y + h), r, TEXT_MUTED, outline=1)
            text_color = TEXT_PRIMARY
        elif style == "stepper":
            _rounded_rect(img, (x, y), (x + w, y + h), r, CARD)
            text_color = TEXT_SECONDARY
        elif style == "subtle":
            _rounded_rect(img, (x, y), (x + w, y + h), r, CARD)
            text_color = TEXT_SECONDARY
        else:
            _rounded_rect(img, (x, y), (x + w, y + h), r, CARD)
            text_color = TEXT_PRIMARY

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
        _text(img, label, (x + (w - tw) // 2, y + (h + th) // 2), 0.44, text_color, 1)
        self._buttons.append((rect, action))

    # ---------------------------------------------------------------- camera
    def _camera(self, img, guard, small, obs, fps) -> None:
        if self.show_camera:
            view = annotate(small, obs, guard.policy, fps, bar=False)
            view = cv2.resize(view, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

            # FPS overlay
            fps_text = f"{fps:.1f} fps"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
            # Background pill
            px, py = CAM_W - tw - 24, CAM_H - 24
            _rounded_rect(view, (px - 8, py - th - 4), (px + tw + 8, py + 6), 10, (0, 0, 0))
            _text(view, fps_text, (px, py), 0.4, TEXT_SECONDARY, 1)
        else:
            view = np.full((CAM_H, CAM_W, 3), BG, np.uint8)
            _draw_logo(view, (CAM_W // 2, CAM_H // 2 - 30), 40, TEXT_MUTED)
            msg = "Camera hidden"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            _text(view, msg, ((CAM_W - tw) // 2, CAM_H // 2 + 50), 0.55, TEXT_MUTED, 1)
            sub = "PeepGuard is still watching"
            (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
            _text(view, sub, ((CAM_W - tw) // 2, CAM_H // 2 + 80), 0.4, TEXT_MUTED, 1)

        img[:, PANEL_W:] = view

    # ---------------------------------------------------------------- input
    def _on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def _handle(self, guard, obs: Observation, x: int, y: int) -> None:
        action = next((a for (bx, by, bw, bh), a in self._buttons
                       if bx <= x <= bx + bw and by <= y <= by + bh), None)
        cfg = guard.cfg
        if action == "allow":
            guard.allow(obs.t)
        elif action == "resume":
            guard.resume()
        elif action in ("cone-", "cone+"):
            step = 2.0 if action.endswith("+") else -2.0
            guard.reconfigure(attention_cone_deg=float(np.clip(cfg.attention_cone_deg + step, 10, 60)))
        elif action in ("dwell-", "dwell+"):
            step = 0.2 if action.endswith("+") else -0.2
            guard.reconfigure(dwell_s=round(float(np.clip(cfg.dwell_s + step, 0.4, 5.0)), 1))
        elif action in ("rotate-", "rotate+"):
            rotations = [0, 90, 180, 270]
            idx = rotations.index(cfg.rotate) if cfg.rotate in rotations else 0
            idx = (idx + (1 if action.endswith("+") else -1)) % 4
            guard.reconfigure(rotate=rotations[idx])
            self.wants_camera_restart = True
        elif action == "camera":
            self.show_camera = not self.show_camera
        elif action == "enroll":
            self.wants_enroll = True
        elif action == "forget":
            forget(owner_file())
            guard.reload_owner()
        elif action == "quit":
            guard._quit = True
