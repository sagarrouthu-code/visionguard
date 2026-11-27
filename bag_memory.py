"""Remember bags the detector has lost, and still raise the alarm for them.

Problem: YOLO often loses a bag the moment it is put down flat (small, low
contrast, partly hidden by legs). The old logic only timed bags that were
detected in the CURRENT frame, so a bag that vanished after being set down
never reached 15 seconds.

Idea (a bag that was seen reliably becomes a "ghost" when detection is lost):
    1. remember its last position                       (bags dict keeps it)
    2. while nobody is near that spot, the timer runs   (same rule as live bags)
    3. a person standing near the spot resets the timer (they may be picking it up)
    4. when the timer passes the limit, VERIFY: zoom into that spot and look
       again with a lower threshold. Alert only if a bag is really there.
       (A bag carried away by its owner leaves nothing to find -> no false alarm.)

Pure Python, no YOLO needed: `verify` is passed in as a function.
"""

import math

from bag_owner import dist_to_box, near_dwell


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def update_ghosts(bags, used, now, owner_boxes, live_centers, near_px,
                  refound_px, limit_s, verify, min_hits=3, verify_every=2.0,
                  max_tries=6, dwell_s=0.0):
    """Advance the timers of remembered (undetected) bags.

    bags          dict id -> entry (owned by server.py)
    used          ids that were detected in this frame (those are not ghosts)
    owner_boxes   boxes of every person in this frame
    live_centers  centres of the bags detected in this frame
    verify        function(entry) -> box (x1,y1,x2,y2) if a bag is found at the spot, else None
    Returns the list of entries that should raise an alert now.
    """
    to_alert = []
    for e in list(bags.values()):
        if e["id"] in used or e.get("hits", 0) < min_hits or "c" not in e:
            continue
        c = e["c"]

        # A live detection appeared close by: the bag was re-found, drop the ghost
        if any(math.dist(c, lc) < refound_px for lc in live_centers):
            del bags[e["id"]]
            continue

        # Someone has stayed at the spot (maybe picking it up): attended.
        # A person only walking past does not reset the timer.
        near_now = any(dist_to_box(c, ob) < near_px for ob in owner_boxes)
        if near_dwell(e, near_now, now, dwell_s):
            e["alone"] = None
            e["alerted"] = False
            e["for"] = 0.0
            continue

        if e.get("alone") is None:
            e["alone"] = now
        e["for"] = now - e["alone"]

        if e["for"] >= limit_s and not e["alerted"]:
            if e.get("tries", 0) >= max_tries:
                continue                      # give up: bounds the extra CPU work
            if now - e.get("verify_at", -1e9) < verify_every:
                continue
            e["verify_at"] = now
            e["tries"] = e.get("tries", 0) + 1
            found = verify(e)
            if found is not None:
                e["alerted"] = True
                e["box"] = list(found)
                e["c"] = _center(found)
                to_alert.append(e)
    return to_alert


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def suppress_overlapping(dets, iou_thr=0.5, contain_thr=0.7):
    """One physical bag often comes back twice in the same frame, under two
    labels (handbag + backpack). Keep the most confident box and drop the rest.
    Also drops a small box mostly inside a larger one."""
    kept = []
    for d in sorted(dets, key=lambda d: d.get("conf", 0.0), reverse=True):
        a = d["box"]
        dup = False
        for k in kept:
            b = k["box"]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if _iou(a, b) >= iou_thr or (smaller > 0 and inter / smaller >= contain_thr):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept