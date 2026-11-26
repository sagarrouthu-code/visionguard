"""Unattended-baggage logic.

A bag is UNATTENDED when it has stayed (roughly) still and no person has been
near it for `unattended_seconds`. Pure logic: no OpenCV, easy to test.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BagState:
    anchor: tuple                       # where the bag was when it last "settled"
    last_seen: float
    alone_since: Optional[float] = None
    alone_for: float = 0.0
    status: str = "ok"                  # ok | alone | unattended
    alerted: bool = False


def _dist_point_to_box(point, box):
    """Distance from a point to a rectangle (0 if the point is inside it)."""
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


class BaggageMonitor:
    def __init__(self, unattended_seconds=5.0, near_fraction=0.12,
                 still_fraction=0.03, lost_grace_seconds=10.0):
        self.unattended_seconds = unattended_seconds
        self.near_fraction = near_fraction      # "near a person", as a share of frame width
        self.still_fraction = still_fraction    # movement below this = "not moving"
        self.lost_grace_seconds = lost_grace_seconds
        self.states = {}                        # track_id -> BagState

    def reset(self):
        self.states.clear()

    def info(self, track_id):
        return self.states.get(track_id)

    def _adopt_lost_state(self, center, now, still_px, current_ids):
        """If a bag's track ID changed (detector flicker), reuse the old state."""
        for old_id, old in list(self.states.items()):
            if old_id in current_ids:
                continue
            if now - old.last_seen > self.lost_grace_seconds:
                continue
            if math.dist(center, old.anchor) <= still_px:
                return self.states.pop(old_id)
        return None

    def update(self, bags, persons, now, frame_width):
        """bags/persons: lists of Detection. Returns IDs that just became unattended."""
        near_px = self.near_fraction * frame_width
        still_px = self.still_fraction * frame_width
        current_ids = {b.track_id for b in bags}
        newly_unattended = []

        for bag in bags:
            center = bag.center
            state = self.states.get(bag.track_id)
            if state is None:
                state = (self._adopt_lost_state(center, now, still_px, current_ids)
                         or BagState(anchor=center, last_seen=now))
                self.states[bag.track_id] = state
            state.last_seen = now

            # Bag moved -> someone is carrying it; restart the clock
            if math.dist(center, state.anchor) > still_px:
                state.anchor = center
                state.alone_since = None
                state.alerted = False

            person_near = any(_dist_point_to_box(center, p.box) < near_px
                              for p in persons)
            if person_near:
                state.alone_since = None
                state.alerted = False
            elif state.alone_since is None:
                state.alone_since = now

            state.alone_for = 0.0 if state.alone_since is None else now - state.alone_since
            if state.alone_for >= self.unattended_seconds:
                state.status = "unattended"
                if not state.alerted:
                    state.alerted = True
                    newly_unattended.append(bag.track_id)
            elif state.alone_since is not None:
                state.status = "alone"
            else:
                state.status = "ok"

        # Forget bags that have been gone for a long time
        for tid in list(self.states):
            if tid not in current_ids and now - self.states[tid].last_seen > self.lost_grace_seconds:
                del self.states[tid]

        return newly_unattended