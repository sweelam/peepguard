from types import SimpleNamespace

from peepguard.config import Config
from peepguard.policy import Policy, State
from peepguard.scene import Observation, Role


def obs(t, onlookers=0, primary=True, degraded=None):
    o = Observation(t=t, degraded=degraded)
    o.views = [SimpleNamespace(role=Role.ONLOOKER) for _ in range(onlookers)]
    o.primary = SimpleNamespace(role=Role.PRIMARY) if primary else None
    return o


def run(policy, frames):
    return [e for f in frames if (e := policy.step(f))]


CFG = Config(dwell_s=2.0, gap_tolerance_s=0.75, release_s=1.5)


def test_single_frame_blip_never_alerts():
    p = Policy(CFG)
    assert run(p, [obs(0), obs(0.25, 1), obs(0.5), obs(1.5), obs(3.0)]) == []
    assert p.state is State.CLEAR


def test_dwell_then_alert_then_release():
    p = Policy(CFG)
    frames = [obs(i / 4, 1) for i in range(12)] + [obs(3 + i / 4) for i in range(10)]
    evs = run(p, frames)
    assert [(e.kind, e.reason) for e in evs] == [("alert", "onlooker"), ("clear", "resolved")]
    assert evs[0].t == 2.0
    assert evs[1].t >= evs[0].t + 1.5


def test_missed_detections_inside_dwell_are_tolerated():
    p = Policy(CFG)
    ts = [0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0]   # gaps of 0.5 s (< tolerance)
    frames = [obs(t, 1 if t in ts else 0) for t in [i / 4 for i in range(9)]]
    assert [e.kind for e in run(p, frames)] == ["alert"]


def test_gap_longer_than_tolerance_resets():
    p = Policy(CFG)
    frames = [obs(0, 1), obs(1.0, 1), obs(1.25), obs(2.0), obs(2.25, 1), obs(3.0, 1)]
    assert run(p, frames) == []


def test_sticky_requires_dismiss_then_snoozes():
    p = Policy(Config(dwell_s=1.0, sticky=True))
    run(p, [obs(i / 4, 1) for i in range(5)])
    assert p.state is State.ALERT
    run(p, [obs(5 + i, 0) for i in range(5)])   # onlooker long gone, still hidden
    assert p.state is State.ALERT
    assert p.dismiss(10).reason == "dismissed"
    run(p, [obs(10 + i / 4, 1) for i in range(12)])  # same person still there: snoozed
    assert p.state is State.CLEAR


def test_degraded_frames_carry_no_evidence():
    p = Policy(CFG)
    frames = [obs(i / 4, 1, degraded="too dark") for i in range(20)]
    assert run(p, frames) == []
    assert p.degraded == "too dark"


def test_away_lock():
    p = Policy(Config(away_lock_s=3.0))
    evs = run(p, [obs(0)] + [obs(1 + i / 2, primary=False) for i in range(8)] + [obs(6 + i / 2) for i in range(6)])
    assert [(e.kind, e.reason) for e in evs] == [("alert", "away"), ("clear", "resolved")]
    assert evs[0].t == 3.0
