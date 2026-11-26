"""Owner-aware unattended-bag logic. Pure Python: no OpenCV / YOLO needed.

Idea
----
The person who is with a bag when it is put down becomes its OWNER. The bag is
"attended" only while that owner is still around it. Strangers walking past do
not count, so a bag whose owner has walked away still raises the alarm even in
a crowd. (The older logic only asked "is *anyone* near the bag?".)

Tracker IDs can change when a person is hidden for a moment, so two safety
nets stop that from causing a false alarm:
  * hand-off: a brand-new track that appears right where the owner was last
    seen inherits ownership;
  * custody: if the owner's track is gone and someone else stays right beside
    the bag for `custody_s` seconds, that person is treated as its keeper.

The bag entry `e` is a dict owned by server.py. This module reads and writes:
    alone    timestamp when the bag became unattended, or None
    alerted  reset to False whenever the bag is attended again
    owner, owner_checked, owner_seen, owner_pos, close   (owner bookkeeping)
"""

import math


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def dist_to_box(pt, b):
    """Distance from a point to a rectangle (0 when the point is inside it)."""
    dx = max(b[0] - pt[0], 0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0, pt[1] - b[3])
    return math.hypot(dx, dy)


def near_dwell(e, near_now, now, dwell_s):
    """True once a person has stayed near the bag for `dwell_s` seconds.
    Someone just walking past (or across the camera's view of the bag) does not count.
    dwell_s = 0 gives the old behaviour: anyone near counts at once."""
    if not near_now:
        e["near_since"] = None
        return False
    if e.get("near_since") is None:
        e["near_since"] = now
    return now - e["near_since"] >= dwell_s


def bag_moved(e, c, still_px, now, move_s):
    """Has the bag really moved? A shift must last `move_s` seconds, or be large,
    so a box that jumps because someone walks in front of the bag is ignored.
    move_s = 0 gives the old behaviour: any shift counts at once."""
    shift = math.dist(c, e["anchor"])
    if shift <= still_px:
        e["move_since"] = None
        return False
    if e.get("move_since") is None:
        e["move_since"] = now
    return shift > 3 * still_px or now - e["move_since"] >= move_s


def _attend(e, attended, now):
    if attended:
        e["alone"] = None
        e["alerted"] = False
    elif e.get("alone") is None:
        e["alone"] = now


def _adopt(e, tid, box, now):
    e["owner"] = tid
    e["owner_seen"] = now
    e["owner_pos"] = _center(box)
    e["close"] = {}


def update_owner(e, c, moved, now, people, fallback_boxes,
                 near_px, away_px, handoff_px, close_px, first_seen,
                 owner_aware=True, handoff_s=3.0, custody_s=4.0, dwell_s=0.0):
    """Update one bag's attended / unattended state for the current frame.

    c               bag centre (x, y)
    moved           True if the bag moved since it last settled (being carried)
    people          [(track_id, box)] tracked people in this frame
    fallback_boxes  boxes of ALL people, tracked or not (used when no owner is known)
    near_px         "with the bag" distance, used to pick the owner
    away_px         owner counts as still around the bag within this distance
    handoff_px      how close to the owner's last position a new track must appear
    close_px        "right beside the bag" distance (custody rule)
    first_seen      {track_id: time first seen}, maintained by the caller
    """
    by_id = dict(people)
    dist = {tid: dist_to_box(c, box) for tid, box in people}
    nearest = min((t for t in dist if dist[t] < near_px),
                  key=lambda t: dist[t], default=None)

    # Switch off -> the old rule: a person staying near the bag keeps it attended
    if not owner_aware:
        near_now = any(dist_to_box(c, b) < near_px for b in fallback_boxes)
        _attend(e, near_dwell(e, near_now, now, dwell_s), now)
        return

    # Pick the owner: the person with the bag while it is carried or first seen
    first = not e.get("owner_checked")
    owner_with_bag = e.get("owner") in by_id and dist[e["owner"]] < near_px
    if first or (moved and not owner_with_bag):
        # (if the current owner is still beside the bag, a "move" is just
        #  detection jitter or the owner carrying it: keep the owner)
        e["owner"] = nearest
        e["owner_checked"] = True
        e["close"] = {}
        if nearest is not None:
            e["owner_seen"] = now
            e["owner_pos"] = _center(by_id[nearest])
    owner = e.get("owner")

    # Nobody was with the bag when it appeared: fall back to the old rule
    if owner is None:
        near_now = any(dist_to_box(c, b) < near_px for b in fallback_boxes)
        _attend(e, near_dwell(e, near_now, now, dwell_s), now)
        return

    # Owner is still around the bag
    if owner in by_id and dist[owner] < away_px:
        e["owner_seen"] = now
        e["owner_pos"] = _center(by_id[owner])
        e["close"] = {}
        _attend(e, True, now)
        return

    # Owner's track vanished: maybe the tracker just gave them a new ID
    if owner not in by_id:
        pos, seen = e.get("owner_pos"), e.get("owner_seen")
        if pos is not None and seen is not None and now - seen <= handoff_s:
            for tid, box in people:
                is_new = now - first_seen.get(tid, now) <= handoff_s
                if is_new and math.dist(_center(box), pos) < handoff_px:
                    _adopt(e, tid, box, now)
                    _attend(e, True, now)
                    return

        # Or someone else has stayed right beside the bag: they are looking after it
        close = e.setdefault("close", {})
        for tid in list(close):
            if tid not in dist or dist[tid] >= close_px:
                del close[tid]
        for tid, d in dist.items():
            if d < close_px:
                close.setdefault(tid, now)
        for tid, since in close.items():
            if now - since >= custody_s:
                _adopt(e, tid, by_id[tid], now)
                _attend(e, True, now)
                return

    # Owner is away from the bag
    _attend(e, False, now)