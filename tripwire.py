"""Virtual tripwire: a line on screen; reports when a tracked point crosses it."""

import time


class Tripwire:
    def __init__(self, start_rel, end_rel, deadband_px=4, cooldown_s=2.0):
        # Endpoints stored as fractions of frame size, so they work at any resolution
        self.start_rel = start_rel
        self.end_rel = end_rel
        self.deadband_px = deadband_px   # ignore jitter right on the line
        self.cooldown_s = cooldown_s     # min seconds between alerts for one ID
        self._last_side = {}             # track_id -> +1 / -1
        self._last_alert = {}            # track_id -> timestamp
        self.pending_click = None        # first click while the user redraws the line

    def endpoints(self, size):
        w, h = size
        a = (int(self.start_rel[0] * w), int(self.start_rel[1] * h))
        b = (int(self.end_rel[0] * w), int(self.end_rel[1] * h))
        return a, b

    def click(self, x, y, size):
        """Mouse clicks redraw the line: first click = start, second = end."""
        w, h = size
        if self.pending_click is None:
            self.pending_click = (x, y)
        else:
            sx, sy = self.pending_click
            self.start_rel = (sx / w, sy / h)
            self.end_rel = (x / w, y / h)
            self.pending_click = None
            self._last_side.clear()      # old side info is meaningless for a new line

    def update(self, track_id, point, size):
        """Feed one tracked point. Returns True if it just crossed the line."""
        a, b = self.endpoints(size)
        abx, aby = b[0] - a[0], b[1] - a[1]
        length_sq = abx * abx + aby * aby
        if length_sq == 0:
            return False
        length = length_sq ** 0.5

        px, py = point
        cross = abx * (py - a[1]) - aby * (px - a[0])
        distance = cross / length            # signed distance from the line
        if abs(distance) < self.deadband_px:
            return False                     # too close to call; keep old side

        side = 1 if distance > 0 else -1
        previous = self._last_side.get(track_id)
        self._last_side[track_id] = side
        if previous is None or previous == side:
            return False

        # Side flipped. Only count it if it happened along the segment (not
        # around its ends).
        t = ((px - a[0]) * abx + (py - a[1]) * aby) / length_sq
        if not 0.0 <= t <= 1.0:
            return False

        now = time.time()
        if now - self._last_alert.get(track_id, 0) < self.cooldown_s:
            return False
        self._last_alert[track_id] = now
        return True