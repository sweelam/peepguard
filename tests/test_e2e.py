"""End-to-end: real YuNet / Haar, real pose, tracker and policy, on synthetic scenes."""

import cv2
import numpy as np
import pytest

from peepguard.app import Guard
from peepguard.config import Config
from peepguard.scene import Role
from synth import Actor, Scene, poster, user, walker

FPS = 8.0


def run_scene(scene, seconds, owner=None, **cfg):
    g = Guard(Config(actions=(), **cfg), actions=[], owner=owner)
    evs, last = [], None
    for i in range(int(seconds * FPS)):
        t = i / FPS
        ev, last, _ = g.process(t, scene.frame(t))
        if ev:
            evs.append(ev)
    return evs, g, last


def kinds(evs):
    return [(e.kind, e.reason) for e in evs]


@pytest.mark.parametrize("detector", ["yunet", "haar"])
def test_onlooker_walks_in_reads_and_leaves(detector):
    scene = Scene(actors=[Actor(user, frontal=False), Actor(walker(4, 12, 140))])
    evs, *_ = run_scene(scene, 16, detector=detector)
    assert kinds(evs) == [("alert", "onlooker"), ("clear", "resolved")]
    assert 5.0 <= evs[0].t <= 6.5          # ~arrival + 1.2 s dwell
    assert 12.5 <= evs[1].t <= 15.0        # ~departure + 1.5 s release


def test_posters_never_alert():
    scene = Scene(actors=[Actor(user, frontal=False), Actor(poster()), Actor(poster(90, 110, 46))])
    evs, g, last = run_scene(scene, 20)
    assert evs == []
    roles = sorted(v.role for v in last.views if v.role is not Role.PRIMARY)
    assert roles == [Role.PICTURE, Role.PICTURE]


def test_nudging_the_laptop_does_not_wake_the_posters():
    def shove(t):
        return (0, 0) if t < 7 else (14, -5) if t < 7.5 else (22, -9)
    scene = Scene(actors=[Actor(user, frontal=False), Actor(poster()), Actor(poster(90, 110, 46))],
                  camera_shift=shove)
    evs, *_ = run_scene(scene, 16)
    assert evs == []


def test_onlooker_still_detected_next_to_a_poster():
    scene = Scene(actors=[Actor(user, frontal=False), Actor(poster()), Actor(walker(6, 14, 150))])
    evs, *_ = run_scene(scene, 18)
    assert kinds(evs) == [("alert", "onlooker"), ("clear", "resolved")]


def test_person_looking_past_the_screen_is_a_bystander():
    scene = Scene(actors=[Actor(user, frontal=False), Actor(walker(2, 14, 140), frontal=False)])
    evs, g, last = run_scene(scene, 10, attention_cone_deg=20)
    assert evs == []
    assert [v.role for v in last.views if v.role is not Role.PRIMARY] == [Role.BYSTANDER]


def test_dark_room_is_degraded_not_alarming():
    scene = Scene(actors=[Actor(user, frontal=False), Actor(walker(1, 9, 140))], brightness=lambda t: 0.18)
    evs, g, _ = run_scene(scene, 10)
    assert evs == []
    assert g.policy.degraded and "dark" in g.policy.degraded


def test_walk_away_lock():
    def leaves(t):
        return user(t) if t < 4 else None
    scene = Scene(actors=[Actor(leaves, frontal=False)])
    evs, *_ = run_scene(scene, 10, away_lock_s=3.0)
    assert kinds(evs) == [("alert", "away")]


def test_replay_from_video_file(tmp_path):
    """Same pipeline through FrameSource: file timeline, frame skipping to the analysis rate."""
    path = str(tmp_path / "scene.avi")
    vfps = 12
    scene = Scene(actors=[Actor(user, frontal=False), Actor(walker(3, 9, 140))])
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), vfps, (640, 480))
    for i in range(12 * vfps):
        vw.write(scene.frame(i / vfps))
    vw.release()

    stats = Guard(Config(source=path, actions=()), actions=[], owner=None).run()
    assert [(e.kind, e.reason) for e in stats.events] == [("alert", "onlooker"), ("clear", "resolved")]
    assert 65 <= stats.frames <= 80      # 12 s at 12 fps, every 2nd frame for 8 fps
