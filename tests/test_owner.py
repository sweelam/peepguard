"""Owner recognition: profile storage, three-way matching, and how identity changes who
counts as the user. Scenes use a stand-in matcher (every synthetic actor is the same
photo, so real SFace would call them all 'you')."""

import stat
from types import SimpleNamespace

import numpy as np

from peepguard.owner import OwnerMatcher, OwnerProfile, forget
from peepguard.scene import Role
from synth import Actor, Scene, user, walker
from test_e2e import kinds, run_scene


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def basis(i, n=128):
    v = np.zeros(n, np.float32)
    v[i] = 1
    return v


def test_profile_roundtrip_is_private(tmp_path):
    path = tmp_path / "owner.npz"
    OwnerProfile(np.stack([basis(0), basis(1)])).save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert OwnerProfile.load(path).templates.shape == (2, 128)
    assert forget(path) and not path.exists() and not forget(path)


def test_similarity_needs_more_than_one_lucky_template():
    prof = OwnerProfile(np.stack([basis(0), basis(1), basis(2), basis(3)]))
    assert prof.similarity(basis(0)) < 0.4          # one exact hit, three misses
    prof = OwnerProfile(np.stack([basis(0)] * 4))
    assert prof.similarity(basis(0)) > 0.99


class FakeRecognizer:
    def __init__(self):
        self.next = None

    def embed(self, bgr, face):
        return self.next


def track(yaw=0.0, w=120.0):
    return SimpleNamespace(owner=None, owner_sim=None, owner_checked=-1e9, owner_strikes=0,
                           face=SimpleNamespace(box=np.array([0, 0, w, w])), pose=SimpleNamespace(yaw=yaw))


def test_three_way_matching_and_owner_latch():
    rec = FakeRecognizer()
    m = OwnerMatcher(OwnerProfile(np.stack([basis(0)] * 3)), rec, match=0.363, reject=0.22, recheck_s=1.0)
    t = track()

    rec.next = unit(np.r_[0.3, 0.95, np.zeros(126)])  # a so-so look: neither yes nor no
    m.check(t, None, 0.0)
    assert t.owner is None

    rec.next = basis(0)
    m.check(t, None, 0.5)
    assert t.owner is True

    rec.next = basis(5)                     # bad looks: smoothed, then two strikes needed
    m.check(t, None, 2.0)
    m.check(t, None, 2.2)                   # (too soon: not re-checked)
    m.check(t, None, 3.5)
    assert t.owner is True
    m.check(t, None, 5.0)
    assert t.owner is False


def test_turned_or_tiny_faces_are_not_judged():
    rec = FakeRecognizer()
    rec.next = basis(9)
    m = OwnerMatcher(OwnerProfile(np.stack([basis(0)] * 3)), rec, 0.363, 0.22, 1.0)
    turned, tiny = track(yaw=60), track(w=30)
    m.check(turned, None, 0.0)
    m.check(tiny, None, 0.0)
    assert turned.owner is None and tiny.owner is None


class PositionMatcher:
    """Stand-in identity: whoever sits near x=330 (the user's seat) is the owner."""

    def __init__(self, owner_x=330.0):
        self.owner_x = owner_x

    def check(self, t, bgr, now):
        is_owner = abs(float(t.face.center[0]) - self.owner_x) < 60
        t.owner, t.owner_sim = is_owner, 0.8 if is_owner else 0.05


def test_stranger_at_your_laptop_is_caught_only_with_recognition():
    seat = Scene(actors=[Actor(user, frontal=True)])
    # Without recognition, whoever sits down is "the user".
    evs, *_ = run_scene(seat, 6)
    assert evs == []
    # With it, someone who isn't you, alive and facing the screen, pauses it.
    evs, g, last = run_scene(seat, 6, owner=PositionMatcher(owner_x=-999))
    assert kinds(evs) == [("alert", "stranger")]
    assert last.primary is None


def test_owner_stays_the_user_when_someone_leans_in_closer():
    # A bigger face arrives beside you; recognition keeps you as the user.
    scene = Scene(actors=[Actor(user, frontal=False), Actor(walker(2, 12, 120, y=200, w=200))])
    evs, g, last = run_scene(scene, 8, owner=PositionMatcher())
    assert last.primary is not None and abs(float(last.primary.track.face.center[0]) - 330) < 60
    others = [v.role for v in last.views if v is not last.primary]
    assert others and Role.PRIMARY not in others
