"""Remember bags the detector has lost and handle unattended alerts.

Features:
1. Remember a bag when YOLO temporarily loses it.
2. Preserve its owner information.
3. Continue the unattended timer while the bag is missing.
4. If someone returns near the bag, reset the timer.
5. After the unattended limit, try verification.
6. Do not lose the alert just because verification misses the bag.
"""

import math

from bag_owner import dist_to_box, near_dwell


def _center(box):
    return (
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2,
    )


def update_ghosts(
    bags,
    used,
    now,
    owner_boxes,
    live_centers,
    near_px,
    refound_px,
    limit_s,
    verify,
    min_hits=3,
    verify_every=2.0,
    max_tries=6,
    dwell_s=0.0,
):
    """Advance timers for remembered bags that YOLO temporarily lost."""

    to_alert = []

    for e in list(bags.values()):

        # Currently visible bags are handled by server.py.
        if (
            e["id"] in used
            or e.get("hits", 0) < min_hits
            or "c" not in e
        ):
            continue

        c = e["c"]

        # -------------------------------------------------
        # BAG FOUND AGAIN
        # Preserve owner + timer information.
        # -------------------------------------------------

        refound = False

        for live in bags.values():

            if (
                live["id"] in used
                and math.dist(c, live["c"]) < refound_px
            ):
                # Preserve owner
                if e.get("owner") is not None:
                    live["owner"] = e["owner"]

                # Preserve unattended timer
                if (
                    e.get("alone") is not None
                    and live.get("alone") is None
                ):
                    live["alone"] = e["alone"]

                # Preserve owner memory
                if "owner_seen" in e:
                    live["owner_seen"] = e["owner_seen"]

                refound = True
                break

        if refound:
            continue

        # -------------------------------------------------
        # CHECK WHETHER SOMEONE IS BACK NEAR THE BAG
        # -------------------------------------------------

        near_now = any(
            dist_to_box(c, ob) < near_px
            for ob in owner_boxes
        )

        # Someone staying near the bag means it is attended.
        if near_dwell(
            e,
            near_now,
            now,
            dwell_s,
        ):
            e["alone"] = None
            e["alerted"] = False
            e["for"] = 0.0
            continue

        # -------------------------------------------------
        # START / CONTINUE UNATTENDED TIMER
        # -------------------------------------------------

        if e.get("alone") is None:
            e["alone"] = now

        e["for"] = now - e["alone"]

        # -------------------------------------------------
        # 15-SECOND ALERT
        # -------------------------------------------------

        if (
            e["for"] >= limit_s
            and not e["alerted"]
        ):

            # ---------------------------------------------
            # FINAL SAFETY CHECK
            # Someone may have returned right at the
            # alert threshold.
            # ---------------------------------------------

            near_now = any(
                dist_to_box(c, ob) < near_px
                for ob in owner_boxes
            )

            if near_now:
                e["alone"] = None
                e["for"] = 0.0
                e["alerted"] = False
                continue

            # ---------------------------------------------
            # TRY TO VERIFY THE BAG
            # Verification is optional.
            # ---------------------------------------------

            found = None

            if (
                now - e.get("verify_at", -1e9)
                >= verify_every
                and e.get("tries", 0) < max_tries
            ):
                e["verify_at"] = now
                e["tries"] = e.get("tries", 0) + 1

                try:
                    found = verify(e)
                except Exception:
                    found = None

            # If verification finds the bag,
            # update its latest position.
            if found is not None:
                e["box"] = list(found)
                e["c"] = _center(found)

            # ---------------------------------------------
            # FINAL SAFETY CHECK AGAIN
            #
            # This catches someone returning while the
            # verification step was running.
            # ---------------------------------------------

            near_now = any(
                dist_to_box(e["c"], ob) < near_px
                for ob in owner_boxes
            )

            if near_now:
                e["alone"] = None
                e["for"] = 0.0
                e["alerted"] = False
                continue

            # ---------------------------------------------
            # ALERT
            # ---------------------------------------------

            e["alerted"] = True
            to_alert.append(e)

    return to_alert