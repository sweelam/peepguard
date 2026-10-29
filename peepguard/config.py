"""Runtime configuration. Every threshold that matters lives here so it can be tuned
from the CLI or a TOML file without touching detection code."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


def _default_config_paths() -> list[Path]:
    """Return standard config file locations to check."""
    paths = []
    home = Path.home()
    if sys.platform == "darwin":
        paths.append(home / "Library/Application Support/peepguard/config.toml")
    paths.append(home / ".config/peepguard/config.toml")
    paths.append(home / ".config/peepguard.toml")
    return paths


@dataclass(frozen=True)
class Config:
    # --- capture -----------------------------------------------------------
    source: str = "0"                 # camera index or path to a video file (replay)
    fps: float = 8.0                  # analysis rate; ~20 ms/frame, so 8 fps is ~15% of a core
    frame_width: int = 640            # analysis resolution (frames are downscaled to this)
    threads: int = 1                  # OpenCV worker threads. Idle workers spin, so the default
                                      # 8 cost ~85% of a core for ~5 ms/frame saved; 1 is ~24%
    flip_horizontal: bool = False     # mirror the camera horizontally (selfie view)
    rotate: int = 0                   # rotation in degrees: 0, 90, 180, or 270

    # --- detection ---------------------------------------------------------
    detector: str = "auto"            # auto | yunet | haar
    min_face_px: int = 28             # ignore faces smaller than this (far background)
    score_threshold: float = 0.75

    # --- geometry: is a face looking at *the screen*? ----------------------
    # The screen centre sits below the webcam. We aim the "attention cone" at that
    # point instead of at the lens, which is what separates "looking at your screen"
    # from "looking at your camera".
    screen_offset_mm: float = 110.0   # webcam -> screen centre, downward (13-15" laptop ~ 100-130)
    attention_cone_deg: float = 28.0  # half-angle of the cone that counts as "facing the screen"
    hfov_deg: float = 70.0            # webcam horizontal field of view (laptops: 65-78)
    pose_smoothing_s: float = 0.36    # time constant of the head-angle average (0.5/frame at 4 fps)

    # --- liveness: posters, photos, the framed print behind you -----------
    static_window_s: float = 5.0      # how long a face must be motionless to be called a picture
    static_motion_frac: float = 0.06  # peak-to-peak scene-relative motion (x face width) that counts as alive

    # --- owner recognition (only after `peepguard enroll`) --------------------
    owner: str = "auto"               # auto: recognise the enrolled owner if there is one | off
    owner_match: float = 0.363        # SFace cosine at/above which a face is you (OpenCV's threshold)
    owner_reject: float = 0.22        # at/below which it is someone else; in between stays unknown
    owner_check_s: float = 1.0        # re-verify a recognised face this often

    # --- policy ------------------------------------------------------------
    dwell_s: float = 1.2              # onlooker must persist this long before we alert
    gap_tolerance_s: float = 0.75     # tolerate missed detections inside the dwell window
    release_s: float = 1.5            # clear for this long before the shield drops
    sticky: bool = False              # True: shield stays until the user dismisses it
    allow_minutes: float = 15.0       # "Allow" on the shield: stop pausing this long (type to change)
    away_lock_s: float = 0.0          # >0: also shield when the primary user leaves for N s
    min_lux: float = 35.0             # mean luma below this => DEGRADED (don't trust, don't alarm)

    # --- actions -----------------------------------------------------------
    actions: tuple[str, ...] = ("log", "shield", "notify")
    hook: str = ""                    # shell command run on transitions; PEEPGUARD_EVENT is set
    hud: bool = False                 # debug window with annotations (frames still never stored)

    @staticmethod
    def load(path: str | Path | None, **overrides) -> "Config":
        cfg = Config()
        # Auto-load from standard locations if no explicit path given
        if path:
            config_file = Path(path)
        else:
            config_file = next((p for p in _default_config_paths() if p.is_file()), None)
        if config_file and config_file.is_file():
            data = tomllib.loads(config_file.read_text())
            cfg = _apply(cfg, data)
        return _apply(cfg, {k: v for k, v in overrides.items() if v is not None})


def _apply(cfg: Config, data: dict) -> Config:
    known = {f.name: f for f in fields(Config)}
    clean = {}
    for k, v in data.items():
        k = k.replace("-", "_")
        if k not in known:
            raise ValueError(f"unknown config key: {k}")
        if k == "actions" and isinstance(v, (list, str)):
            v = tuple(v.split(",")) if isinstance(v, str) else tuple(v)
        clean[k] = v
    return replace(cfg, **clean)
