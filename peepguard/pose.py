"""Head pose from YuNet's five landmarks via solvePnP, and the question that actually
matters: is this face pointed at the *screen*?

Coordinate frame (OpenCV camera convention): x -> image right, y -> image down,
z -> away from the lens. The model below is a generic adult face, nose tip at the
origin, written so that a face looking straight into the lens has R ~= identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Generic face model (mm). Order matches YuNet: subject's right eye (image-left),
# left eye, nose tip, right mouth corner, left mouth corner.
FACE_MODEL_MM = np.array(
    [
        [-32.0, -35.0, 32.0],
        [32.0, -35.0, 32.0],
        [0.0, 0.0, 0.0],
        [-25.0, 33.0, 22.0],
        [25.0, 33.0, 22.0],
    ],
    dtype=np.float64,
)
AVG_FACE_WIDTH_MM = 150.0  # bounding-box width proxy for range when there are no landmarks
FACE_OUT = np.array([0.0, 0.0, -1.0])  # direction the face "looks", in model frame


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    width: int
    height: int

    @staticmethod
    def approximate(width: int, height: int, hfov_deg: float) -> "Camera":
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
        return Camera(K=K, width=width, height=height)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])


@dataclass(frozen=True)
class Pose:
    yaw: float              # +: head points toward image right
    pitch: float            # +: head points down (toward the screen, below the lens)
    roll: float
    position_mm: np.ndarray  # nose tip in camera frame
    screen_angle: float     # degrees between face direction and the screen centre
    reproj_px: float        # mean reprojection error; large => landmarks don't fit a face

    @property
    def distance_mm(self) -> float:
        return float(np.linalg.norm(self.position_mm))


def estimate_pose(landmarks: np.ndarray, cam: Camera, screen_offset_mm: float) -> Pose | None:
    img = landmarks.astype(np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(FACE_MODEL_MM, img, cam.K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(FACE_MODEL_MM, img, cam.K, None, rvec, tvec)
    t = tvec.ravel()
    if t[2] <= 0:  # behind the camera: degenerate solution
        return None

    proj, _ = cv2.projectPoints(FACE_MODEL_MM, rvec, tvec, cam.K, None)
    reproj = float(np.linalg.norm(proj.reshape(-1, 2) - landmarks, axis=1).mean())

    R, _ = cv2.Rodrigues(rvec)
    # Angles straight from the facing vector: no Euler-order ambiguity.
    fx, fy, fz = R @ FACE_OUT
    yaw = math.degrees(math.atan2(fx, -fz))
    pitch = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
    eyes = landmarks[1] - landmarks[0]
    roll = math.degrees(math.atan2(float(eyes[1]), float(eyes[0])))

    return Pose(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        position_mm=t,
        screen_angle=screen_gaze_angle(R, t, screen_offset_mm),
        reproj_px=reproj,
    )


def screen_gaze_angle(R: np.ndarray, t: np.ndarray, screen_offset_mm: float) -> float:
    """Angle between where the head points and the ray from the head to the screen centre.

    Using the ray *to the screen* (not to the lens) matters at the edges of the frame:
    someone standing far to the side must turn their head toward the laptop to read it,
    and that turn is exactly what this measures."""
    facing = R @ FACE_OUT
    screen = np.array([0.0, screen_offset_mm, 0.0])
    to_screen = screen - t
    cos = float(np.dot(facing, to_screen) / (np.linalg.norm(to_screen) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def range_from_box(box_w_px: float, cam: Camera) -> float:
    return cam.fx * AVG_FACE_WIDTH_MM / max(box_w_px, 1.0)
