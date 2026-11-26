"""Object detection + tracking using YOLOv8n + ByteTrack."""

from dataclasses import dataclass


@dataclass
class Detection:
    track_id: int
    box: tuple
    conf: float
    class_id: int
    label: str

    @property
    def foot(self):
        """Bottom-centre of the detection box."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, y2)

    @property
    def center(self):
        """Centre of the detection box."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class PersonTracker:
    # COCO class IDs
    PERSON_CLASS_ID = 0
    BACKPACK_CLASS_ID = 24
    HANDBAG_CLASS_ID = 26
    SUITCASE_CLASS_ID = 28

    DETECTED_CLASSES = [
        PERSON_CLASS_ID,
        BACKPACK_CLASS_ID,
        HANDBAG_CLASS_ID,
        SUITCASE_CLASS_ID,
    ]

    CLASS_NAMES = {
        0: "person",
        24: "backpack",
        26: "handbag",
        28: "suitcase",
    }

    def __init__(self, model_path="yolov8n.pt", conf=0.4, imgsz=640,
                 bag_conf=0.25, owner_conf=0.25):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from e

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load model '{model_path}'. "
                f"On first run an internet connection is needed "
                f"to download it. ({e})"
            ) from e

        self.conf = conf                # people shown / used for the tripwire
        self.bag_conf = bag_conf        # bags are harder to detect: lower bar
        self.owner_conf = owner_conf    # weaker person detections still count as "owner nearby"
        self.imgsz = imgsz
        self._fresh = True

    def reset(self):
        """Forget all track IDs when video restarts."""
        self._fresh = True

    def track(self, frame):
        """Detect and track people + baggage."""

        results = self.model.track(
            frame,
            persist=not self._fresh,
            tracker="bytetrack.yaml",
            classes=self.DETECTED_CLASSES,
            conf=min(self.conf, self.bag_conf, self.owner_conf),
            imgsz=self.imgsz,
            verbose=False,
        )

        self._fresh = False

        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()

        detections = []

        for track_id, box, conf, class_id in zip(
            ids, xyxy.tolist(), confs, classes
        ):
            is_person = class_id == self.PERSON_CLASS_ID
            min_conf = min(self.conf, self.owner_conf) if is_person else self.bag_conf
            if conf < min_conf:
                continue

            detections.append(
                Detection(
                    track_id=track_id,
                    box=tuple(box),
                    conf=conf,
                    class_id=class_id,
                    label=self.CLASS_NAMES.get(
                        class_id,
                        f"class_{class_id}"
                    ),
                )
            )

        return detections

    def split_detections(self, detections):
        """Separate people and bags."""

        persons = [
            d for d in detections
            if d.class_id == self.PERSON_CLASS_ID
        ]

        bags = [
            d for d in detections
            if d.class_id in (
                self.BACKPACK_CLASS_ID,
                self.HANDBAG_CLASS_ID,
                self.SUITCASE_CLASS_ID,
            )
        ]

        return persons, bags