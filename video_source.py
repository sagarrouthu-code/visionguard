"""Video input helpers: VideoSource (webcam or .mp4) and FPSCounter."""

import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent


class VideoSource:
    """Wraps cv2.VideoCapture for either a webcam index or a video file."""

    def __init__(self, source):
        self.source = source
        self.is_webcam = isinstance(source, int)
        self.cap = None
        self.native_fps = 0.0

    def open(self):
        """Open the source. Returns (ok, error_message)."""
        if self.is_webcam:
            target = self.source
        else:
            path = Path(self.source)
            if not path.is_absolute():
                path = BASE_DIR / path
            if not path.exists():
                return False, f"Video file not found: {path}"
            target = str(path)

        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            kind = "webcam" if self.is_webcam else "video file"
            return False, f"Could not open {kind}: {self.source}"

        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        return True, ""

    def read(self):
        """Return (ok, frame). ok is False at end of video or on read failure."""
        return self.cap.read()

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap is not None:
            self.cap.release()


class FPSCounter:
    """Smoothed FPS (exponential moving average)."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = instant if self.fps == 0 else (
                self.smoothing * self.fps + (1 - self.smoothing) * instant
            )
        return self.fps