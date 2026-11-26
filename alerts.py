"""Alert handling: on-screen banner, console message, sound, and a saved snapshot."""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    import winsound   # Windows only
except ImportError:
    winsound = None

BASE_DIR = Path(__file__).resolve().parent

RED = (0, 0, 200)
ORANGE = (0, 110, 255)


class AlertManager:
    def __init__(self, snapshot_dir="alerts", banner_seconds=2.5, flag_seconds=2.5):
        self.snapshot_dir = BASE_DIR / snapshot_dir
        self.banner_seconds = banner_seconds
        self.flag_seconds = flag_seconds
        self.line_count = 0
        self.bag_count = 0
        self._banner_until = 0.0
        self._banner_text = ""
        self._banner_color = RED
        self._flagged = {}            # track_id -> expiry time
        self._pending_snapshot = None

    def trigger(self, track_ids, kind="line"):
        """kind = 'line' (person crossed the line) or 'bag' (unattended bag)."""
        now = time.time()
        ids_text = " ".join(f"#{t}" for t in track_ids)
        for tid in track_ids:
            self._flagged[tid] = now + self.flag_seconds
            if kind == "line":
                self.line_count += 1
                print(f"[ALERT] {datetime.now():%H:%M:%S} - person #{tid} crossed the line "
                      f"(line alerts: {self.line_count})")
            else:
                self.bag_count += 1
                print(f"[ALERT] {datetime.now():%H:%M:%S} - UNATTENDED BAG #{tid} "
                      f"(bag alerts: {self.bag_count})")

        if kind == "line":
            self._banner_text, self._banner_color = f"ALERT: PERSON CROSSED THE LINE  {ids_text}", RED
        else:
            self._banner_text, self._banner_color = f"ALERT: UNATTENDED BAG  {ids_text}", ORANGE
        self._banner_until = now + self.banner_seconds
        self._pending_snapshot = kind
        self._beep()

    def is_flagged(self, track_id):
        return self._flagged.get(track_id, 0) > time.time()

    def draw(self, frame):
        """Draw banner + counters; save a snapshot if an alert just fired."""
        h, w = frame.shape[:2]
        if time.time() < self._banner_until:
            cv2.rectangle(frame, (0, h - 50), (w, h), self._banner_color, -1)
            cv2.putText(frame, self._banner_text, (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        for i, text in enumerate((f"Line alerts: {self.line_count}",
                                  f"Bag alerts: {self.bag_count}")):
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            pos = (w - size[0] - 10, 28 + 28 * i)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)

        if self._pending_snapshot is not None:
            self._save_snapshot(frame, self._pending_snapshot)
            self._pending_snapshot = None

    def _save_snapshot(self, frame, kind):
        try:
            self.snapshot_dir.mkdir(exist_ok=True)
            name = f"{kind}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(self.snapshot_dir / name), frame)
        except Exception as e:      # never let a disk problem crash the loop
            print(f"[WARN] Could not save snapshot: {e}")

    @staticmethod
    def _beep():
        if winsound is not None:
            winsound.MessageBeep(winsound.MB_ICONHAND)   # non-blocking
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()