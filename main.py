"""
VISIONGUARD - AI CCTV security monitoring.

Features:
    - Person detection + tracking
    - Dynamic virtual tripwire
    - Intrusion alerts
    - Backpack / handbag / suitcase detection
    - Unattended baggage detection

Controls:
    Left-click twice = redraw tripwire
    Q = quit
"""

import sys
import time

import cv2
import numpy as np

from alerts import AlertManager
from detector import PersonTracker
from tripwire import Tripwire
from video_source import FPSCounter, VideoSource
from backpack import BaggageMonitor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SOURCE = "videos/testvideo.mp4"

LOOP_VIDEO = False

WINDOW_NAME = "VISIONGUARD (Q = quit, click twice = redraw line)"

MAX_DISPLAY_WIDTH = 960

# Tripwire
LINE_START = (0.15, 0.65)
LINE_END = (0.85, 0.65)

# YOLO
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.4        # people shown + used for the tripwire
BAG_CONF = 0.25         # bags are harder to detect, so a lower bar
OWNER_CONF = 0.25       # weaker person detections still count as "owner nearby"
INFERENCE_SIZE = 640

# Unattended baggage
# 5 seconds for TESTING.
# Change to 15.0 for final version.
UNATTENDED_SECONDS = 5.0


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------

def fit_width(frame, max_width):
    h, w = frame.shape[:2]

    if w <= max_width:
        return frame

    scale = max_width / w

    return cv2.resize(
        frame,
        (max_width, int(h * scale))
    )


def put_text(
    frame,
    text,
    pos,
    color=(0, 255, 0),
    scale=0.7
):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


def draw_info(frame, fps, source_size, persons, bags):

    h, w = frame.shape[:2]

    put_text(
        frame,
        f"FPS: {fps:.1f}",
        (10, 28)
    )

    res = f"Resolution: {source_size[0]}x{source_size[1]}"

    if (w, h) != tuple(source_size):
        res += f" (shown {w}x{h})"

    put_text(
        frame,
        res,
        (10, 56)
    )

    put_text(
        frame,
        f"PEOPLE: {len(persons)}",
        (10, 86),
        (0, 255, 0)
    )

    put_text(
        frame,
        f"BAGS: {len(bags)}",
        (10, 114),
        (0, 255, 255)
    )


def draw_tripwire(frame, tripwire):

    h, w = frame.shape[:2]

    a, b = tripwire.endpoints((w, h))

    cv2.line(
        frame,
        a,
        b,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        a,
        6,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        frame,
        b,
        6,
        (0, 255, 255),
        -1
    )

    put_text(
        frame,
        "TRIPWIRE",
        (a[0], max(20, a[1] - 10)),
        (0, 255, 255),
        0.6
    )

    if tripwire.pending_click is not None:

        cv2.circle(
            frame,
            tripwire.pending_click,
            7,
            (255, 0, 255),
            -1
        )

        put_text(
            frame,
            "Click the line's end point",
            (10, 145),
            (255, 0, 255),
            0.6
        )


def draw_persons(frame, persons, alerts):

    for person in persons:

        x1, y1, x2, y2 = person.box

        color = (
            (0, 0, 255)
            if alerts.is_flagged(person.track_id)
            else (0, 255, 0)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        put_text(
            frame,
            f"person #{person.track_id} {person.conf:.2f}",
            (x1, max(18, y1 - 8)),
            color,
            0.5
        )

        cv2.circle(
            frame,
            person.foot,
            4,
            color,
            -1
        )


def draw_bags(frame, bags, baggage_monitor):

    for bag in bags:

        x1, y1, x2, y2 = bag.box

        state = baggage_monitor.info(bag.track_id)

        if state is None:
            color = (0, 255, 255)
            status = "DETECTED"
            timer = ""

        else:

            if state.status == "unattended":

                color = (0, 0, 255)
                status = "UNATTENDED"

            elif state.status == "alone":

                color = (0, 165, 255)
                status = "ALONE"

            else:

                color = (0, 255, 255)
                status = "WITH PERSON"

            timer = f" {state.alone_for:.1f}s"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = (
            f"{bag.label} #{bag.track_id} "
            f"{status}{timer}"
        )

        put_text(
            frame,
            label,
            (x1, max(18, y1 - 8)),
            color,
            0.55
        )


def show_message_window(message):

    img = np.zeros(
        (300, 900, 3),
        dtype=np.uint8
    )

    cv2.putText(
        img,
        "VISIONGUARD",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    line, y = "", 130

    for word in message.split():

        trial = f"{line} {word}".strip()

        if len(trial) > 60:

            cv2.putText(
                img,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            line = word
            y += 35

        else:
            line = trial

    cv2.putText(
        img,
        line,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "Press any key to close.",
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    try:
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(0)
    except cv2.error:
        pass


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run(source):

    video = VideoSource(source)

    ok, error = video.open()

    if not ok:

        print(f"[ERROR] {error}")

        show_message_window(error)

        cv2.destroyAllWindows()

        return 1

    try:

        tracker = PersonTracker(
            MODEL_PATH,
            CONFIDENCE,
            INFERENCE_SIZE,
            bag_conf=BAG_CONF,
            owner_conf=OWNER_CONF,
        )

    except RuntimeError as e:

        print(f"[ERROR] {e}")

        show_message_window(str(e))

        video.release()

        cv2.destroyAllWindows()

        return 1

    tripwire = Tripwire(
        LINE_START,
        LINE_END
    )

    alerts = AlertManager()

    baggage_monitor = BaggageMonitor(
        unattended_seconds=UNATTENDED_SECONDS
    )

    fps_counter = FPSCounter()

    state = {
        "size": (1, 1)
    }

    def on_mouse(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            tripwire.click(
                x,
                y,
                state["size"]
            )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        on_mouse
    )

    print(f"[INFO] Source opened: {source}")

    print(
        "[INFO] Press Q to quit. "
        "Click twice to redraw the line."
    )

    print(
        f"[INFO] Unattended bag threshold: "
        f"{UNATTENDED_SECONDS} seconds"
    )

    frame_delay_ms = 1

    if not video.is_webcam and video.native_fps > 0:

        frame_delay_ms = max(
            1,
            int(1000 / video.native_fps)
        )

    # Timers use VIDEO time (frame number / fps), not wall-clock time, so
    # "5 seconds alone" means 5 seconds of footage even on a slow CPU.
    video_fps = video.native_fps if video.native_fps > 0 else 30.0
    frame_index = 0

    failed_reads = 0

    try:

        while True:

            loop_start = time.perf_counter()

            ok, frame = video.read()

            if not ok or frame is None:

                if not video.is_webcam:

                    if LOOP_VIDEO:

                        video.restart()

                        tracker.reset()

                        baggage_monitor.reset()

                        frame_index = 0

                        continue

                    print(
                        "[INFO] End of video reached."
                    )

                    break

                failed_reads += 1

                if failed_reads > 30:

                    print(
                        "[ERROR] Lost connection to webcam."
                    )

                    show_message_window(
                        "Lost connection to the webcam."
                    )

                    return 1

                continue

            failed_reads = 0

            source_size = (
                frame.shape[1],
                frame.shape[0]
            )

            frame = fit_width(
                frame,
                MAX_DISPLAY_WIDTH
            )

            h, w = frame.shape[:2]

            state["size"] = (w, h)

            video_now = (
                time.time() if video.is_webcam
                else frame_index / video_fps
            )
            frame_index += 1

            # -------------------------------------------------------------
            # 1. YOLO + ByteTrack
            # -------------------------------------------------------------

            detections = tracker.track(frame)

            all_people, bags = tracker.split_detections(detections)
            # all_people includes weak detections (used to decide if a bag has an owner)
            persons = [p for p in all_people if p.conf >= CONFIDENCE]

            # -------------------------------------------------------------
            # 2. Tripwire
            # -------------------------------------------------------------

            crossed = [
                d.track_id
                for d in persons
                if tripwire.update(
                    d.track_id,
                    d.foot,
                    (w, h)
                )
            ]

            if crossed:
                alerts.trigger(crossed)

            # -------------------------------------------------------------
            # 3. Unattended baggage
            # -------------------------------------------------------------

            newly_unattended = baggage_monitor.update(
                bags=bags,
                persons=all_people,
                now=video_now,
                frame_width=w
            )

            if newly_unattended:
                alerts.trigger(newly_unattended, kind="bag")

            # -------------------------------------------------------------
            # 4. Draw UI
            # -------------------------------------------------------------

            draw_tripwire(
                frame,
                tripwire
            )

            draw_persons(
                frame,
                persons,
                alerts
            )

            draw_bags(
                frame,
                bags,
                baggage_monitor
            )

            alerts.draw(frame)

            draw_info(
                frame,
                fps_counter.tick(),
                source_size,
                persons,
                bags
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # -------------------------------------------------------------
            # 5. Keyboard
            # -------------------------------------------------------------

            elapsed_ms = int(
                (time.perf_counter() - loop_start)
                * 1000
            )

            wait = max(
                1,
                frame_delay_ms - elapsed_ms
            )

            if cv2.waitKey(wait) & 0xFF in (
                ord("q"),
                ord("Q")
            ):

                print(
                    "[INFO] Quit requested."
                )

                break

    finally:

        video.release()

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":

    sys.exit(
        run(SOURCE)
    )