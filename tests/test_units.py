import numpy as np

from peepguard.actions.shield import render_shield
from peepguard.config import Config
from peepguard.detector import Face
from peepguard.tracker import Tracker


def face(x, y, w=60):
    return Face(box=np.array([x, y, w, w], np.float32), score=0.9, landmarks=None)


def test_fast_mover_keeps_its_identity():
    tr = Tracker()
    ids = set()
    for i, x in enumerate(range(0, 300, 45)):  # 45 px/frame on a 60 px face: IoU ~ 0.14
        (t,) = tr.update([face(x, 100)], i * 0.25, np.zeros(2))
        ids.add(t.id)
    assert len(ids) == 1


def test_single_frame_glitch_is_not_life():
    tr = Tracker()
    for i in range(8):
        (t,) = tr.update([face(100 + (9 if i == 3 else 0), 100)], i * 0.25, np.zeros(2))
    assert not t.update_liveness(2.0, 5.0, 0.06)


def test_liveness_latches():
    tr = Tracker()
    for i in range(8):
        (t,) = tr.update([face(100 + (6 if 3 <= i <= 5 else 0), 100)], i * 0.25, np.zeros(2))
    assert t.update_liveness(2.0, 5.0, 0.06)
    for i in range(8, 40):  # perfectly still afterwards
        (t,) = tr.update([face(100, 100)], i * 0.25, np.zeros(2))
    assert t.update_liveness(10.0, 5.0, 0.06)


def test_static_face_with_camera_shift_stays_dead():
    tr = Tracker()
    for i in range(20):
        dx = 15.0 if i == 10 else 0.0  # whole scene jumped; face moved with it
        (t,) = tr.update([face(100 + (15 if i >= 10 else 0), 100)], i * 0.25, np.array([dx, 0.0]))
    assert not t.update_liveness(5.0, 5.0, 0.06)


def test_shield_renders_with_and_without_screenshot():
    blank = render_shield(None, (1280, 800))
    assert blank.shape == (800, 1280, 3)
    shot = np.random.default_rng(0).integers(0, 255, (900, 1440, 3), dtype=np.uint8)
    out = render_shield(shot)
    assert out.shape == shot.shape
    # Frosted means no fine detail survives: high-frequency energy collapses.
    hf = lambda im: np.abs(np.diff(im.astype(np.int16), axis=1)).mean()
    assert hf(out[:100]) < 0.05 * hf(shot[:100])


def test_config_overrides_and_unknown_keys(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('fps = 3\nattention-cone-deg = 22\nactions = ["log"]\n')
    cfg = Config.load(f, dwell_s=1.0, hud=None)
    assert (cfg.fps, cfg.attention_cone_deg, cfg.actions, cfg.dwell_s) == (3, 22, ("log",), 1.0)
    f.write_text("nope = 1\n")
    try:
        Config.load(f)
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError


def test_shield_choices_continue_or_allow_for_typed_minutes():
    from peepguard.actions.shield import ShieldAction

    sh = ShieldAction(allow_minutes=15)
    sh._windows = [("w", None, (1440, 900))]
    sh._redraw = lambda: None
    assert sh.pump(-1) is None
    assert sh.pump(ord("x")) == ("dismissed", 10.0)         # any key reveals
    assert sh.pump(ord("a")) == ("allowed", 15 * 60.0)      # A: the default allowance
    for k in "45":
        assert sh.pump(ord(k)) is None                      # typing minutes doesn't reveal
    assert sh.pump(13) == ("allowed", 45 * 60.0)            # Enter confirms what was typed
    sh._typed = ""
    sh._click = "allow"
    assert sh.pump(-1) == ("allowed", 15 * 60.0)


def test_allow_keeps_the_screen_unpaused_for_that_long():
    from peepguard.policy import Policy, State
    from test_policy import obs

    p = Policy(Config(dwell_s=1.0, release_s=1.5))
    t = 0.0
    while p.state is not State.ALERT:
        p.step(obs(t, onlookers=1))
        t += 0.25
    assert p.dismiss(t, 15 * 60, "allowed").reason == "allowed"
    for _ in range(int(14 * 60 / 0.25)):            # teammate looks on for 14 minutes
        t += 0.25
        assert p.step(obs(t, onlookers=1)) is None
    assert p.snoozed_for(t) < 70
    evs = [p.step(obs(t + 1 + k * 0.25, onlookers=1)) for k in range(int(70 / 0.25))]
    assert [e.reason for e in evs if e] == ["onlooker"]   # pauses again once it runs out


def test_control_panel_buttons_drive_the_guard(monkeypatch):
    import peepguard.ui as U
    from peepguard.app import Guard, RunStats
    from synth import Actor, Scene, user

    for name in ("imshow", "namedWindow", "setMouseCallback"):
        monkeypatch.setattr(U.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(U.cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(U.cv2, "getWindowProperty", lambda *a: 1)

    g = Guard(Config(actions=()), actions=[], owner=None)
    g._stats = RunStats()
    _, obs, small = g.process(0.0, Scene(actors=[Actor(user, frontal=False)]).frame(0.0))
    panel = U.ControlPanel()

    def click(action):
        panel.update(g, small, obs, 8.0)
        (x, y, w, h), _ = next(b for b in panel._buttons if b[1] == action)
        panel._click = (x + w // 2, y + h // 2)
        panel.update(g, small, obs, 8.0)

    click("allow")
    assert g.policy.snoozed_for(obs.t) == 15 * 60
    click("resume")
    assert g.policy.snoozed_for(obs.t) == 0
    click("cone+")
    assert g.cfg.attention_cone_deg == 30 and g.policy.cfg is g.cfg and g.scene.cfg is g.cfg
    click("camera")
    assert not panel.show_camera
    click("quit")
    assert g._quit
