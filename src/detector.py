"""YOLO26 detection + multi-object tracking wrapper (Ultralytics).

The rest of the pipeline only ever sees a clean list of `Track` objects, so the
detector is swappable: to move off Ultralytics (e.g. to RF-DETR under a permissive
license) you reimplement this one class and nothing downstream changes.

YOLO26 is the current Ultralytics generation (NMS-free, edge-optimized). Any
Ultralytics detection checkpoint works here — pass model_path="yolo11n.pt" to use
YOLO11 instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# COCO class ids relevant to traffic scenes (the classes YOLO26 ships pretrained).
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    conf: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def is_vehicle(self) -> bool:
        return self.class_name in VEHICLE_NAMES


class Detector:
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        # Imported lazily so this module (and the geometry/rule unit tests) can be
        # imported without ultralytics installed.
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.device = device
        # Inference resolution. Cost scales ~quadratically, so on CPU this is the
        # single biggest speed knob: 416 is roughly 2x faster than 640. Small or
        # distant objects are the first thing you lose — check recall before
        # keeping a lower value.
        self.imgsz = imgsz
        self.classes = list(classes) if classes is not None else list(COCO_CLASSES)

    def track(self, frame) -> List[Track]:
        """Run detection + tracking on one BGR frame, returning tracked objects."""
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        tracks: List[Track] = []
        if not results:
            return tracks
        r = results[0]
        # No detections, or the tracker hasn't assigned IDs yet this frame.
        if r.boxes is None or r.boxes.id is None:
            return tracks
        boxes = r.boxes
        ids = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        for tid, cid, cf, box in zip(ids, clss, confs, xyxy):
            tracks.append(Track(
                track_id=int(tid),
                class_id=int(cid),
                class_name=COCO_CLASSES.get(int(cid), str(cid)),
                conf=float(cf),
                bbox=(box[0], box[1], box[2], box[3]),
            ))
        return tracks
