"""Debug overlay for tuning. Shown only with --hud; frames are still never written."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .policy import Policy, State
from .scene import Observation, Role

WINDOW = "PeepGuard HUD"
COLORS = {
    Role.PRIMARY: (110, 200, 90),
    Role.ONLOOKER: (60, 60, 235),
    Role.BYSTANDER: (40, 180, 240),
    Role.PICTURE: (150, 150, 150),
    Role.PENDING: (220, 160, 80),
}
STATE_COLORS = {State.CLEAR: (110, 200, 90), State.SUSPECT: (40, 180, 240), State.ALERT: (60, 60, 235)}


def annotate(frame: np.ndarray, obs: Observation, policy: Policy, fps: float, bar: bool = True) -> np.ndarray:
    img = frame.copy()
    for v in obs.views:
        x, y, w, h = (int(a) for a in v.track.face.box)
        col = COLORS[v.role]
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)
        bits = [v.role.value]
        if v.screen_angle is not None:
            bits.append(f"{v.screen_angle:.0f}deg")
        if v.distance_mm:
            bits.append(f"{v.distance_mm / 1000:.1f}m")
        if v.behind_user:
            bits.append("behind")
        if v.track.owner_sim is not None:
            who = {True: "you", False: "not you", None: "?"}[v.track.owner]
            bits.append(f"{who} {v.track.owner_sim:.2f}")
        _label(img, " ".join(bits), (x, y - 6), col)
        lm, pose = v.track.face.landmarks, v.track.pose
        if lm is not None:
            for p in lm:
                cv2.circle(img, tuple(int(c) for c in p), 2, col, -1, cv2.LINE_AA)
            if pose is not None:  # where the head points
                nose = lm[2]
                L = w * 0.9
                tip = (int(nose[0] + L * math.sin(math.radians(pose.yaw))),
                       int(nose[1] + L * math.sin(math.radians(pose.pitch))))
                cv2.arrowedLine(img, tuple(int(c) for c in nose), tip, col, 2, cv2.LINE_AA, tipLength=0.25)

    if not bar:
        return img
    bar = f"{policy.state.value.upper()}"
    if policy.reason:
        bar += f" ({policy.reason})"
    left = policy.snoozed_for(obs.t)
    if left > 60:
        bar += f"  |  allowed, {left / 60:.0f} min left"
    if policy.degraded:
        bar += f"  |  DEGRADED: {policy.degraded}"
    bar += f"  |  {fps:.1f} fps  |  q quits"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (24, 22, 20), -1)
    cv2.putText(img, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COLORS[policy.state], 1, cv2.LINE_AA)
    return img


def _label(img, text, org, col):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = org
    y = max(y, th + 30)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 3), (24, 22, 20), -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
