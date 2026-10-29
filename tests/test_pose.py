import math

import cv2
import numpy as np
import pytest

from peepguard.pose import FACE_MODEL_MM, Camera, estimate_pose, screen_gaze_angle

CAM = Camera.approximate(640, 480, 70)


def look_at(face_pos, target):
    """Rotation whose face-out direction (-z in model frame) points from face_pos at target."""
    fwd = np.asarray(target, float) - np.asarray(face_pos, float)
    fwd /= np.linalg.norm(fwd)
    z = -fwd                      # model +z points into the head
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def project(R, t, noise=0.0, seed=0):
    p, _ = cv2.projectPoints(FACE_MODEL_MM, cv2.Rodrigues(R)[0], np.asarray(t, float), CAM.K, None)
    p = p.reshape(-1, 2)
    if noise:
        p = p + np.random.default_rng(seed).normal(0, noise, p.shape)
    return p.astype(np.float32)


SCREEN = (0.0, 110.0, 0.0)


@pytest.mark.parametrize("pos", [(0, 0, 700), (350, -100, 1400), (-500, -200, 1800), (200, 150, 900)])
def test_face_reading_the_screen_is_inside_the_cone(pos):
    R = look_at(pos, SCREEN)
    pose = estimate_pose(project(R, pos, noise=0.8), CAM, SCREEN[1])
    assert pose is not None
    assert pose.screen_angle < 12, pose
    assert pose.distance_mm == pytest.approx(np.linalg.norm(pos), rel=0.15)


@pytest.mark.parametrize("yaw", [45, -45, 70])
def test_face_looking_away_is_outside_the_cone(yaw):
    pos = (100, -80, 1200)
    target = (pos[0] + 1000 * math.sin(math.radians(yaw)), pos[1], pos[2] - 1000 * math.cos(math.radians(yaw)))
    pose = estimate_pose(project(look_at(pos, target), pos, noise=0.8), CAM, SCREEN[1])
    assert pose.screen_angle > 30, pose
    assert np.sign(pose.yaw) == np.sign(yaw)


def test_screen_ray_not_camera_ray():
    """Someone far to the side has to turn their head to read the laptop. A head turned
    by the same amount in the middle of the frame is looking *past* it."""
    side = np.array([700.0, -50.0, 1300.0])
    R_turned = look_at(side, SCREEN)
    assert screen_gaze_angle(R_turned, side, SCREEN[1]) < 2
    centre = np.array([0.0, -50.0, 1300.0])
    assert screen_gaze_angle(R_turned, centre, SCREEN[1]) > 25


def test_looking_at_lens_vs_screen_offset():
    """Screen sits below the lens: a head aimed exactly at the lens from close up is a
    few degrees above the screen centre, which the cone comfortably absorbs."""
    pos = np.array([0.0, -30.0, 600.0])
    a = screen_gaze_angle(look_at(pos, (0, 0, 0)), pos, SCREEN[1])
    assert 5 < a < 15
