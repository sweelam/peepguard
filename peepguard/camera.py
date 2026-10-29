"""Frame source. A webcam keeps delivering ~30 fps whether we want it or not, and most
backends buffer, so naively sleeping then reading returns a stale frame from seconds
ago. A reader thread drains the camera continuously and keeps only the newest decoded
frame; read() hands that over without waiting on the camera. (Grabbing in the thread and
retrieving in read() under one lock made every read wait out the next frame interval,
which capped a 30 fps camera at ~18 fps and added a frame of lag.)

Frames live only in memory and are overwritten on the next read. Nothing touches disk.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


class FrameSource:
    def __init__(self, source: str, fps: float, width: int = 640,
                 flip_horizontal: bool = True, rotate: int = 0):
        self.is_file = not source.isdigit()
        if self.is_file:
            self._cap = cv2.VideoCapture(source)
        else:
            idx = int(source)
            # Use AVFoundation on macOS for smoother capture
            if sys.platform == "darwin":
                self._cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r} (camera permission?)")
        if not self.is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.period = 1.0 / fps
        self._flip_h = flip_horizontal and not self.is_file
        self._rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
        self._lock = threading.Lock()
        self._fresh = threading.Condition(self._lock)
        self._latest: tuple[float, np.ndarray] | None = None
        self._seq = self._taken = 0
        self._stop = threading.Event()
        self._grabber: threading.Thread | None = None
        self._file_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._file_t = 0.0
        if not self.is_file:
            self._grabber = threading.Thread(target=self._drain, daemon=True)
            self._grabber.start()

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        if self._rotate is not None:
            frame = cv2.rotate(frame, self._rotate)
        if self._flip_h:
            frame = cv2.flip(frame, 1)
        return frame

    def _drain(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()  # blocks on the camera, outside the lock
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame = self._transform(frame)
            with self._fresh:
                self._latest, self._seq = (time.monotonic(), frame), self._seq + 1
                self._fresh.notify_all()

    def read(self) -> tuple[float, np.ndarray] | None:
        """Returns (timestamp_s, frame). For files, timestamps follow the video timeline
        and frames are sampled at the analysis rate, so replays behave like live runs."""
        if self.is_file:
            skip = max(1, round(self._file_fps * self.period))
            for _ in range(skip - 1):
                self._cap.grab()
            ok, frame = self._cap.read()
            if not ok:
                return None
            self._file_t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            return self._file_t, frame
        with self._fresh:  # newest frame we haven't handed out yet; wait only if none
            if not self._fresh.wait_for(lambda: self._seq > self._taken, timeout=2.0):
                return None  # camera stopped delivering
            self._taken = self._seq
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._grabber:
            self._grabber.join(timeout=1)
        self._cap.release()
