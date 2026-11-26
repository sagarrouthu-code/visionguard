"""VisionGuard backend: detection thread + FastAPI/WebSocket.
Run:  python server.py   then open http://localhost:8000
Env:  VG_SOURCE (file path or webcam index), VG_MODEL, VG_IMGSZ
"""

import asyncio
import base64
import importlib.util
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from bag_owner import update_owner, bag_moved   # owner-aware unattended-bag logic
from bag_memory import update_ghosts   # remember bags the detector loses
from ultralytics import YOLO


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BASE = Path(__file__).parent
SNAP = BASE / "snapshots"
SNAP.mkdir(exist_ok=True)

# Uploaded videos live in videos/uploads. Only files inside videos/ can be played
# from the dashboard (a webcam number is also allowed).
ROOT = BASE.resolve()
VIDEO_DIR = ROOT / "videos"
UPLOAD_DIR = VIDEO_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_MB = int(os.getenv("VG_MAX_UPLOAD_MB", "500"))
KEEP_UPLOADS = 30     # older uploads are deleted automatically

# Uploading needs the python-multipart package. Without it the rest still works.
HAVE_UPLOAD = bool(
    importlib.util.find_spec("python_multipart")
    or importlib.util.find_spec("multipart")
)
if not HAVE_UPLOAD:
    print("[WARN] Video upload is off. Run: pip install python-multipart")

MODEL = os.getenv("VG_MODEL", "yolov8n.pt")
IMGSZ = int(os.getenv("VG_IMGSZ", "640"))
EVERY = int(os.getenv("VG_EVERY", "2"))
# confidence used when zooming into the spot where a remembered bag should be
VERIFY_CONF = float(os.getenv("VG_VERIFY_CONF", "0.10"))
# a person must stay near a bag this long (seconds) to count as looking after it;
# someone walking past does not reset the timer. 0 = old behaviour.
ATTEND_S = float(os.getenv("VG_ATTEND_S", "2.0"))
# a bag must keep shifting this long (seconds) to count as moved, so a box that
# jumps when someone walks in front of it is ignored. 0 = old behaviour.
MOVE_S = float(os.getenv("VG_MOVE_S", "1.0"))

OUT_W = 960

BAGS = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

cfg = {
    "unattended_seconds": 15.0,
    "person_conf": 0.4,
    "bag_conf": 0.2,
    "owner_conf": 0.25,   # lower bar for "a person is standing near the bag"
    "remember_lost": 1.0,   # 1 = keep timing bags the detector loses, re-check the spot before alerting
    "owner_aware": 0.0,   # 0 = old tested rule (anyone nearby counts), 1 = alert when the bag's OWNER leaves (experimental)
}

zones = {
    "line": None,
    "polygon": None,
}

state = {
    "seq": 0,
    "jpg": None,
    "persons": [],
    "bags": [],
    "fps": 0.0,
    "events": [],

    "source": os.getenv("VG_SOURCE", "videos/testvideo.mp4"),

    "restart": True,
    # Start idle: nothing plays until you press Play in the dashboard.
    # Set VG_AUTOPLAY=1 to start playing as soon as the server starts.
    "playing": os.getenv("VG_AUTOPLAY", "0") == "1",
    "ended": False,
}

lock = threading.Lock()


# ---------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------

def seg_cross(p1, p2, p3, p4):
    def c(a, b, p):
        return (
            (p[1] - a[1]) * (b[0] - a[0])
            - (b[1] - a[1]) * (p[0] - a[0])
        )

    return (
        c(p3, p4, p1) * c(p3, p4, p2) < 0
        and c(p1, p2, p3) * c(p1, p2, p4) < 0
    )


def dist_to_box(pt, b):
    return math.hypot(
        max(b[0] - pt[0], 0, pt[0] - b[2]),
        max(b[1] - pt[1], 0, pt[1] - b[3]),
    )


# ---------------------------------------------------------
# EVENTS
# ---------------------------------------------------------

def add_event(kind, text, frame, box=None):
    img = frame.copy()

    if box:
        cv2.rectangle(
            img,
            tuple(map(int, box[:2])),
            tuple(map(int, box[2:])),
            (0, 0, 255),
            3,
        )

    eid = int(time.time() * 1000)
    name = f"{eid}_{kind}.jpg"

    cv2.imwrite(str(SNAP / name), img)

    ev = {
        "id": eid,
        "kind": kind,
        "text": text,
        "t": time.strftime("%H:%M:%S"),
        "img": name,
    }

    with lock:
        state["events"] = (state["events"] + [ev])[-30:]


# ---------------------------------------------------------
# AI WORKER
# ---------------------------------------------------------

def verify_bag_at(model, frame, e, conf):
    """Zoom into the area around a remembered bag and look for it again.
    Returns the bag box in frame coordinates, or None if nothing is there."""

    h, w = frame.shape[:2]
    bx = e["box"]
    cx, cy = e["c"]

    r = max(0.12 * w, 1.8 * max(bx[2] - bx[0], bx[3] - bx[1]))
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(w, cx + r)), int(min(h, cy + r))

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    try:
        res = model.predict(
            crop,
            imgsz=640,
            conf=conf,
            classes=list(BAGS),
            verbose=False,
        )[0]
    except Exception as ex:
        print(f"[WARN] bag re-check failed: {ex}")
        return None

    if res.boxes is None or not len(res.boxes):
        return None

    i = int(res.boxes.conf.argmax())
    b = res.boxes.xyxy[i].cpu().numpy()

    return [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]


def worker():

    model = YOLO(MODEL)

    cap = None
    idx = 0
    t0 = 0.0
    vfps = 25.0
    is_file = True

    bags = {}
    nid = 1

    prev_foot = {}
    inside = {}
    flagged = {}
    first_seen = {}      # person track id -> time first seen
    vm = {"model": None, "failed": False}   # 2nd YOLO, only used to re-check remembered bags

    ema = 0.0
    last = time.time()
    paused_at = None

    while True:

        # -------------------------------------------------
        # PAUSE
        # -------------------------------------------------

        if not state["playing"]:
            if paused_at is None:
                paused_at = time.time()
            time.sleep(0.1)
            continue

        # Resuming: shift the playback clock by the time spent paused, otherwise
        # the real-time catch-up below fast-forwards the video (and the bag timers).
        if paused_at is not None:
            t0 += time.time() - paused_at
            paused_at = None

        # -------------------------------------------------
        # OPEN / RESTART VIDEO
        # -------------------------------------------------

        if state["restart"] or cap is None:

            state["restart"] = False

            if cap:
                cap.release()

            src = state["source"]

            is_file = not src.isdigit()

            cap = cv2.VideoCapture(
                str(BASE / src) if is_file else int(src)
            )

            if not cap.isOpened():
                print(f"[ERROR] Could not open source: {src}")
                state["playing"] = False
                time.sleep(1)
                continue

            vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            idx = 0
            t0 = time.time()

            bags.clear()
            prev_foot.clear()
            inside.clear()
            flagged.clear()
            first_seen.clear()

            state["ended"] = False

            try:
                model.predictor.trackers[0].reset()
            except Exception:
                pass

        # -------------------------------------------------
        # KEEP FILE VIDEO REAL-TIME
        # -------------------------------------------------

        if is_file:

            target = (time.time() - t0) * vfps

            while idx < target - 1 and cap.grab():
                idx += 1

        # -------------------------------------------------
        # READ FRAME
        # -------------------------------------------------

        ok, frame = cap.read()
        idx += 1

        # -------------------------------------------------
        # VIDEO ENDED
        # -------------------------------------------------

        if not ok:

            if is_file:

                cap.release()
                cap = None

                state["playing"] = False
                state["ended"] = True

                print("[INFO] Video ended.")

            else:
                time.sleep(0.2)

            continue

        # -------------------------------------------------
        # VIDEO TIME
        # -------------------------------------------------

        now = (
            idx / vfps
            if is_file
            else time.time()
        )

        # -------------------------------------------------
        # RESIZE
        # -------------------------------------------------

        h0, w0 = frame.shape[:2]

        if w0 > OUT_W:

            frame = cv2.resize(
                frame,
                (
                    OUT_W,
                    int(h0 * OUT_W / w0),
                ),
            )

        h, w = frame.shape[:2]

        # -------------------------------------------------
        # DISPLAY-ONLY FRAME
        # -------------------------------------------------

        if idx % EVERY:

            ok2, j = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )

            if ok2:

                with lock:
                    state.update(
                        jpg=base64.b64encode(j).decode(),
                        seq=state["seq"] + 1,
                    )

            continue

        # -------------------------------------------------
        # YOLO
        # -------------------------------------------------

        r = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMGSZ,
            verbose=False,
            conf=min(
                cfg["person_conf"],
                cfg["bag_conf"],
            ),
            classes=[0, 24, 26, 28],
        )[0]

        persons = []
        bag_dets = []
        owner_boxes = []
        owner_people = []     # (track id, box) of tracked people, owner-level confidence

        if r.boxes is not None and len(r.boxes):

            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)

            ids = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else [None] * len(cl)
            )

            for b, c, k, i in zip(
                xyxy,
                cf,
                cl,
                ids,
            ):

                # Any reasonably confident person (tracked or not) can be the
                # owner. Missing an owner causes a false unattended-bag alert.
                if k == 0 and c >= cfg["owner_conf"]:
                    owner_boxes.append(b.tolist())
                    if i is not None:
                        owner_people.append((int(i), b.tolist()))
                        first_seen.setdefault(int(i), now)

                # PERSON
                if (
                    k == 0
                    and i is not None
                    and c >= cfg["person_conf"]
                ):

                    persons.append(
                        {
                            "id": int(i),
                            "box": b.tolist(),
                        }
                    )

                # BAG
                elif (
                    k in BAGS
                    and c >= cfg["bag_conf"]
                ):

                    bag_dets.append(
                        {
                            "label": BAGS[k],
                            "box": b.tolist(),
                        }
                    )

        # -------------------------------------------------
        # ZONES
        # -------------------------------------------------

        line = (
            [(x * w, y * h) for x, y in zones["line"]]
            if zones["line"]
            else None
        )

        poly = (
            np.array(
                [
                    (x * w, y * h)
                    for x, y in zones["polygon"]
                ],
                np.float32,
            )
            if zones["polygon"]
            else None
        )

        # -------------------------------------------------
        # PERSON / TRIPWIRE
        # -------------------------------------------------

        for p in persons:

            b = p["box"]

            foot = (
                (b[0] + b[2]) / 2,
                b[3],
            )

            tid = p["id"]
            hit = None

            # Tripwire
            if (
                line
                and tid in prev_foot
                and seg_cross(
                    prev_foot[tid],
                    foot,
                    line[0],
                    line[1],
                )
            ):

                hit = "crossed the tripwire"

            # Restricted polygon
            if poly is not None:

                now_in = (
                    cv2.pointPolygonTest(
                        poly,
                        foot,
                        False,
                    )
                    >= 0
                )

                if now_in and not inside.get(
                    tid,
                    False,
                ):
                    hit = "entered restricted zone"

                inside[tid] = now_in

            prev_foot[tid] = foot

            # Alert
            if (
                hit
                and now - flagged.get(tid, -99) > 5
            ):

                flagged[tid] = now

                add_event(
                    "intrusion",
                    f"Person #{tid} {hit}",
                    frame,
                    b,
                )

            p["flag"] = (
                now - flagged.get(tid, -99) < 3
            )

        # -------------------------------------------------
        # UNATTENDED BAGGAGE
        # -------------------------------------------------

        match_px = 0.07 * w
        still_px = 0.02 * w
        near_px = 0.10 * w

        used = set()

        for d in bag_dets:

            b = d["box"]

            c = (
                (b[0] + b[2]) / 2,
                (b[1] + b[3]) / 2,
            )

            best = min(
                (
                    e
                    for e in bags.values()
                    if e["id"] not in used
                ),
                key=lambda e: math.dist(
                    c,
                    e["c"],
                ),
                default=None,
            )

            if (
                best is None
                or math.dist(
                    c,
                    best["c"],
                ) > match_px
            ):

                best = {
                    "id": nid,
                    "anchor": c,
                    "alone": None,
                    "alerted": False,
                    "hits": 0,
                }

                bags[nid] = best
                nid += 1

            used.add(best["id"])

            # Bag moved (picked up / being carried)
            moved = bag_moved(best, c, still_px, now, MOVE_S)

            if moved:
                best.update(
                    anchor=c,
                    alone=None,
                    alerted=False,
                )

            best.update(
                c=c,
                box=b,
                label=d["label"],
                seen=now,
            )

            best["hits"] = best.get("hits", 0) + 1

            # Who is looking after this bag? (see bag_owner.py)
            update_owner(
                best,
                c,
                moved,
                now,
                people=owner_people,
                fallback_boxes=owner_boxes,
                near_px=near_px,
                away_px=0.22 * w,
                handoff_px=0.07 * w,
                close_px=0.05 * w,
                first_seen=first_seen,
                owner_aware=cfg["owner_aware"] >= 0.5,
                dwell_s=ATTEND_S,
            )

            best["for"] = (
                0.0
                if best["alone"] is None
                else now - best["alone"]
            )

            # Unattended alert
            if (
                best["for"]
                >= cfg["unattended_seconds"]
                and not best["alerted"]
            ):

                best["alerted"] = True

                if best.get("owner") is not None and cfg["owner_aware"] >= 0.5:
                    text = (
                        f"Unattended {best['label']} #{best['id']}. "
                        f"Owner #{best['owner']} left "
                        f"{int(best['for'])}s ago"
                    )
                else:
                    text = (
                        f"Unattended {best['label']} "
                        f"#{best['id']} "
                        f"for {int(best['for'])}s"
                    )

                add_event(
                    "baggage",
                    text,
                    frame,
                    b,
                )

        # Bags the detector lost: keep timing them, re-check the spot before alerting
        if cfg["remember_lost"] >= 0.5:

            def verify(e):
                if vm["model"] is None and not vm["failed"]:
                    try:
                        vm["model"] = YOLO(MODEL)
                    except Exception as ex:
                        vm["failed"] = True
                        print(f"[WARN] re-check model unavailable: {ex}")
                if vm["model"] is None:
                    return None
                return verify_bag_at(vm["model"], frame, e, VERIFY_CONF)

            live_centers = [
                (
                    (d["box"][0] + d["box"][2]) / 2,
                    (d["box"][1] + d["box"][3]) / 2,
                )
                for d in bag_dets
            ]

            for e in update_ghosts(
                bags,
                used,
                now,
                owner_boxes,
                live_centers,
                near_px,
                0.15 * w,
                cfg["unattended_seconds"],
                verify,
                dwell_s=ATTEND_S,
            ):
                add_event(
                    "baggage",
                    f"Unattended {e['label']} "
                    f"#{e['id']} "
                    f"for {int(e['for'])}s "
                    f"(remembered)",
                    frame,
                    e["box"],
                )

        # Remove lost bags
        for k in [
            k
            for k, e in bags.items()
            if now - e["seen"] > (30 if cfg["remember_lost"] >= 0.5 else 8)
        ]:
            del bags[k]

        live = [
            e
            for e in bags.values()
            if e["id"] in used
        ]

        # remembered bags that are currently being timed (not visible to the detector)
        ghosts = (
            [
                e
                for e in bags.values()
                if e["id"] not in used
                and e.get("hits", 0) >= 3
                and e.get("alone") is not None
            ]
            if cfg["remember_lost"] >= 0.5
            else []
        )

        # -------------------------------------------------
        # FPS
        # -------------------------------------------------

        t = time.time()

        ema = (
            0.9 * ema
            + 0.1 / max(t - last, 1e-3)
        )

        last = t

        # -------------------------------------------------
        # NORMALIZED BOXES
        # -------------------------------------------------

        def normalize_box(b):

            return [
                b[0] / w,
                b[1] / h,
                b[2] / w,
                b[3] / h,
            ]

        # -------------------------------------------------
        # JPEG
        # -------------------------------------------------

        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 72],
        )

        if not ok:
            continue

        # -------------------------------------------------
        # UPDATE STATE
        # -------------------------------------------------

        with lock:

            state.update(
                jpg=base64.b64encode(
                    jpg
                ).decode(),

                fps=ema,

                seq=state["seq"] + 1,

                persons=[
                    {
                        "id": p["id"],
                        "box": normalize_box(
                            p["box"]
                        ),
                        "flag": p["flag"],
                    }
                    for p in persons
                ],

                bags=[
                    {
                        "id": e["id"],
                        "label": e["label"],
                        "box": normalize_box(
                            e["box"]
                        ),
                        "alone": round(
                            e["for"],
                            1,
                        ),
                        "owner": (
                            e.get("owner")
                            if cfg["owner_aware"] >= 0.5
                            else None
                        ),
                        "ghost": e["id"] not in used,
                        "status":
                            "unattended"
                            if e["alerted"]
                            else
                            "alone"
                            if e["alone"] is not None
                            else "ok",
                    }
                    for e in live + ghosts
                ],
            )


# ---------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------

app = FastAPI()

app.mount(
    "/snapshots",
    StaticFiles(directory=SNAP),
    name="snaps",
)


# ---------------------------------------------------------
# START AI THREAD
# ---------------------------------------------------------

@app.on_event("startup")
def _start():

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


# ---------------------------------------------------------
# FRONTEND
# ---------------------------------------------------------

@app.get("/")
def index():

    return FileResponse(
        BASE / "index.html"
    )


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------

@app.get("/api/state")
def get_state():

    return {
        "cfg": cfg,
        "zones": zones,
        "source": state["source"],
        "playing": state["playing"],
        "ended": state["ended"],
    }


# ---------------------------------------------------------
# ZONES
# ---------------------------------------------------------

@app.post("/api/zones")
def set_zones(
    body: dict = Body(...)
):

    zones["line"] = body.get("line")
    zones["polygon"] = body.get("polygon")

    return zones


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

@app.post("/api/config")
def set_cfg(
    body: dict = Body(...)
):

    for k in cfg:

        if k in body:
            cfg[k] = float(body[k])

    return cfg


# ---------------------------------------------------------
# VIDEO SOURCE
# ---------------------------------------------------------

def resolve_video(src):
    """Return a safe source string (webcam number or a video file inside videos/),
    or None if the request is not allowed."""

    src = str(src or "").strip()

    if src.isdigit():
        return src

    p = (ROOT / src).resolve()

    if (
        VIDEO_DIR not in p.parents
        or p.suffix.lower() not in VIDEO_EXT
        or not p.is_file()
    ):
        return None

    return p.relative_to(ROOT).as_posix()


def first_frame(path):
    cap = cv2.VideoCapture(str(path))

    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
    finally:
        cap.release()

    return frame if ok else None


def prune_uploads():
    """Keep only the newest KEEP_UPLOADS uploads (never the one being played)."""

    files = sorted(
        (p for p in UPLOAD_DIR.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    current = (
        None
        if state["source"].isdigit()
        else (ROOT / state["source"]).resolve()
    )

    for p in files[KEEP_UPLOADS:]:
        if p.resolve() != current:
            try:
                p.unlink()
            except OSError:
                pass


@app.post("/api/source")
def set_source(
    body: dict = Body(...)
):

    src = resolve_video(body.get("source"))

    if src is None:
        raise HTTPException(
            400,
            "Choose an uploaded video, a file in the videos folder, "
            "or a webcam number.",
        )

    state["source"] = src

    state["restart"] = True
    # play=false loads the video and waits, so you can draw zones first
    state["playing"] = bool(body.get("play", True))
    state["ended"] = False

    return {
        "ok": True,
        "source": state["source"],
    }


@app.get("/api/videos")
def list_videos():

    items = []

    for p in sorted(VIDEO_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            items.append(
                {
                    "source": p.relative_to(ROOT).as_posix(),
                    "name": p.name,
                    "uploaded": UPLOAD_DIR in p.parents,
                    "mb": round(p.stat().st_size / 1e6, 1),
                }
            )

    return {
        "videos": items,
        "current": state["source"],
        "upload": HAVE_UPLOAD,
        "max_mb": MAX_UPLOAD_MB,
    }


@app.get("/api/preview")
def preview(source: str):
    """First frame of a video, so zones can be drawn before pressing Play."""

    src = resolve_video(source)

    if src is None or src.isdigit():
        raise HTTPException(404, "No preview available.")

    frame = first_frame(ROOT / src)

    if frame is None:
        raise HTTPException(404, "No preview available.")

    h, w = frame.shape[:2]

    if w > OUT_W:
        frame = cv2.resize(frame, (OUT_W, int(h * OUT_W / w)))

    ok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    return Response(buf.tobytes(), media_type="image/jpeg")


if HAVE_UPLOAD:

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):

        name = Path(file.filename or "video").name
        ext = Path(name).suffix.lower()

        if ext not in VIDEO_EXT:
            raise HTTPException(
                400,
                "Unsupported file type. Use "
                + ", ".join(sorted(VIDEO_EXT))
                + ".",
            )

        stem = (
            re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem)[:40].strip("._")
            or "video"
        )

        dest = UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"

        limit = MAX_UPLOAD_MB * 1024 * 1024
        size = 0

        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    size += len(chunk)

                    if size > limit:
                        raise HTTPException(
                            413,
                            f"File is larger than {MAX_UPLOAD_MB} MB.",
                        )

                    out.write(chunk)

        except HTTPException:
            dest.unlink(missing_ok=True)
            raise

        except Exception:
            dest.unlink(missing_ok=True)
            raise HTTPException(500, "Upload failed. Please try again.")

        # make sure it really is a readable video
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, first_frame, dest)

        if frame is None:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                400, "This file could not be read as a video."
            )

        prune_uploads()

        return {
            "ok": True,
            "source": dest.relative_to(ROOT).as_posix(),
            "name": dest.name,
        }


# ---------------------------------------------------------
# PLAY
# ---------------------------------------------------------

@app.post("/api/video/play")
def play_video():

    state["playing"] = True

    return {
        "ok": True,
        "status": "playing",
    }


# ---------------------------------------------------------
# PAUSE
# ---------------------------------------------------------

@app.post("/api/video/pause")
def pause_video():

    state["playing"] = False

    return {
        "ok": True,
        "status": "paused",
    }


# ---------------------------------------------------------
# STOP
# ---------------------------------------------------------

@app.post("/api/video/stop")
def stop_video():

    state["playing"] = False
    state["restart"] = True
    state["ended"] = False

    return {
        "ok": True,
        "status": "stopped",
    }


# ---------------------------------------------------------
# RESTART
# ---------------------------------------------------------

@app.post("/api/video/restart")
def restart_video():

    state["playing"] = True
    state["restart"] = True
    state["ended"] = False

    return {
        "ok": True,
        "status": "restarted",
    }


# ---------------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------------

@app.websocket("/ws")
async def ws(
    sock: WebSocket
):

    await sock.accept()

    last = -1

    try:

        while True:

            if (
                state["jpg"]
                and state["seq"] != last
            ):

                with lock:
                    s = dict(state)

                last = s["seq"]

                await sock.send_text(
                    json.dumps(
                        {
                            "img": s["jpg"],
                            "persons": s["persons"],
                            "bags": s["bags"],
                            "fps": round(
                                s["fps"],
                                1,
                            ),
                            "events": s["events"][-10:],
                            "playing": s["playing"],
                            "ended": s["ended"],
                        }
                    )
                )

            await asyncio.sleep(0.01)

    except Exception:
        pass


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )