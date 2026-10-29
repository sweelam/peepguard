"""Face detection. YuNet (OpenCV DNN, ~230 KB ONNX) is preferred because it returns five
landmarks per face, which is exactly what head-pose estimation needs. Haar is the
zero-download fallback: no landmarks, but the frontal cascade only fires on roughly
frontal faces, so a hit is itself weak evidence of "facing the camera"."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_DIR = Path(__file__).parent / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
# OpenCV 5 wheels no longer ship the cascade XMLs, so the fallback fetches its own too.
HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
HAAR_SHA256 = "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0"
HAAR_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


@dataclass
class Face:
    box: np.ndarray                 # x, y, w, h (float32)
    score: float
    landmarks: np.ndarray | None    # (5, 2): r-eye, l-eye, nose, r-mouth, l-mouth (subject's L/R)

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2], dtype=np.float32)

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


def fetch_model(force: bool = False) -> Path:
    """Download YuNet once, verify its hash, cache it next to the package."""
    return _fetch(YUNET_URL, YUNET_PATH, YUNET_SHA256, force)


def _fetch(url: str, path: Path, sha: str, force: bool = False) -> Path:
    if path.exists() and not force and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.info("downloading %s", url)
    urllib.request.urlretrieve(url, tmp)
    digest = _sha256(tmp)
    if digest != sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {path.name}: {digest}")
    tmp.replace(path)
    return path


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class YuNetDetector:
    name = "yunet"

    def __init__(self, score_threshold: float = 0.75, model: Path | None = None):
        path = model or fetch_model()
        self._net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, 0.3, 50)
        self._size: tuple[int, int] | None = None

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, raw = self._net.detect(bgr)
        if raw is None:
            return []
        return [
            Face(box=r[0:4].astype(np.float32), score=float(r[14]),
                 landmarks=r[4:14].reshape(5, 2).astype(np.float32))
            for r in raw
        ]


class HaarDetector:
    name = "haar"

    def __init__(self, **_):
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        bundled = Path(data_dir) / HAAR_PATH.name if data_dir else HAAR_PATH
        path = bundled if bundled.is_file() else _fetch(HAAR_URL, HAAR_PATH, HAAR_SHA256)
        self._cc = cv2.CascadeClassifier(str(path))
        if self._cc.empty():
            raise RuntimeError(f"could not load Haar cascade from {path}")

    def detect(self, bgr: np.ndarray) -> list[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        boxes = self._cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
        return [Face(box=np.array(b, dtype=np.float32), score=1.0, landmarks=None) for b in boxes]


def make_detector(kind: str = "auto", score_threshold: float = 0.75):
    if kind in ("auto", "yunet"):
        try:
            return YuNetDetector(score_threshold)
        except Exception as e:  # offline, blocked, old OpenCV...
            if kind == "yunet":
                raise
            log.warning("YuNet unavailable (%s); falling back to Haar (no head pose)", e)
    return HaarDetector()
