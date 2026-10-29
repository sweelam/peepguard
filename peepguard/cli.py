from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="TOML file with any Config fields")
    p.add_argument("--source", help="camera index (0) or a video file to replay")
    p.add_argument("--fps", type=float)
    p.add_argument("--flip", dest="flip_horizontal", action="store_true", default=None,
                   help="mirror the camera horizontally")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270],
                   help="rotate camera image (degrees)")
    p.add_argument("--detector", choices=["auto", "yunet", "haar"])
    p.add_argument("--cone", dest="attention_cone_deg", type=float, help="attention cone half-angle, degrees")
    p.add_argument("--screen-offset", dest="screen_offset_mm", type=float, help="webcam->screen centre, mm")
    p.add_argument("--dwell", dest="dwell_s", type=float)
    p.add_argument("--sticky", action="store_true", default=None, help="shield stays until you dismiss it")
    p.add_argument("--allow-minutes", dest="allow_minutes", type=float,
                   help="default for the shield's 'Allow for N min' button (15)")
    p.add_argument("--away-lock", dest="away_lock_s", type=float, help="also shield after N s without you")
    p.add_argument("--actions", help="comma list: log,shield,notify")
    p.add_argument("--hook", help="command to run on alert/clear (env: PEEPGUARD_EVENT, _REASON)")
    p.add_argument("--hud", action="store_true", default=None, help="debug window with annotations")
    p.add_argument("--owner", choices=["auto", "off"], help="recognise the enrolled owner (auto) or not")
    p.add_argument("--report", action="store_true", help="print a JSON event report at the end")
    p.add_argument("-v", "--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="peepguard", description="On-device shoulder-surfing detector.")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="watch the webcam (default)")
    _run_args(run)
    ui = sub.add_parser("ui", help="window with the camera and a control panel")
    _run_args(ui)

    sub.add_parser("fetch-model", help="download and verify the YuNet face model")
    enr = sub.add_parser("enroll", help="teach PeepGuard your face, so it knows you're the owner")
    enr.add_argument("--source", help="camera index (0)")
    sub.add_parser("forget", help="delete your enrolled face signature")
    prev = sub.add_parser("preview-shield", help="render the shield to an image file to see it")
    prev.add_argument("out")
    prev.add_argument("--blank", action="store_true", help="don't capture your screen")

    args = p.parse_args(argv)
    cmd = args.cmd
    if cmd is None:  # double-clicked app / bare binary: the window; plain `peepguard` in a shell: run
        cmd = "ui" if getattr(sys, "frozen", False) and sys.stdin is not None and not sys.stdin.isatty() else "run"
        args = (ui if cmd == "ui" else run).parse_args([])

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    if cmd == "fetch-model":
        from .detector import fetch_model
        print(fetch_model())
        return 0

    if cmd == "enroll":
        from .enroll import enroll
        return 0 if enroll(Config.load(None, source=args.source)) else 1

    if cmd == "forget":
        from .owner import forget, owner_file
        print(f"removed {owner_file()}" if forget() else "nothing enrolled")
        return 0

    if cmd == "preview-shield":
        import cv2
        from .actions.shield import grab_screen, render_shield
        cv2.imwrite(args.out, render_shield(None if args.blank else grab_screen()))
        print(args.out)
        return 0

    overrides = {k: v for k, v in vars(args).items()
                 if k not in {"config", "report", "verbose", "cmd"}}
    cfg = Config.load(args.config, **overrides)

    _maybe_onboard(cfg, interactive=cmd == "ui")
    from .app import Guard
    panel = None
    if cmd == "ui":
        from .ui import ControlPanel
        panel = ControlPanel()
    stats = Guard(cfg).run(ui=panel)
    if args.report:
        json.dump({
            "detector": stats.detector, "frames": stats.frames,
            "analysis_ms_per_frame": round(stats.analysis_ms, 1),
            "events": [{"t": round(e.t, 2), "kind": e.kind, "reason": e.reason, "onlookers": e.onlookers}
                       for e in stats.events],
        }, sys.stdout, indent=2)
        print()
    return 0


def _maybe_onboard(cfg: Config, interactive: bool = False) -> None:
    """First interactive run: offer once to learn the owner's face."""
    from .owner import owner_file, skip_file
    if (cfg.owner == "off" or not cfg.source.isdigit() or cfg.detector == "haar"
            or not (interactive or "shield" in cfg.actions or cfg.hud)
            or owner_file().is_file() or skip_file().exists()):
        return
    from .enroll import onboard
    onboard(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
