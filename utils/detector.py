"""
detector.py
-----------
Car detection using YOLOv8 (ultralytics).
COCO class IDs used: 2=car, 5=bus, 7=truck
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

# Lazy import so the module loads quickly even without ultralytics installed yet
_model = None
_model_name_cached: str | None = None

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def _get_model(model_name: str = "yolov8n.pt"):
    """Load and cache the YOLOv8 model."""
    global _model, _model_name_cached
    if _model is not None:
        if _model_name_cached != model_name:
            print(
                f"[detector] WARNING: model '{model_name}' requested but "
                f"'{_model_name_cached}' is already cached. Using cached model."
            )
        return _model
    from ultralytics import YOLO  # type: ignore
    print(f"[detector] Loading YOLOv8 model '{model_name}' …")
    _model = YOLO(model_name)
    _model_name_cached = model_name
    print("[detector] Model ready.")
    return _model


def detect_vehicles(
    image: np.ndarray,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    classes: set[int] | None = None,
) -> list[Detection]:
    """Run YOLOv8 on *image* and return vehicle detections.

    Parameters
    ----------
    image:          BGR numpy array (as returned by cv2.imread).
    model_name:     YOLOv8 weights file (auto-downloaded on first run).
    conf_threshold: Minimum confidence to keep a detection.
    classes:        COCO class IDs to keep; defaults to VEHICLE_CLASSES.
    """
    if classes is None:
        classes = set(VEHICLE_CLASSES.keys())

    model = _get_model(model_name)
    # Use smaller imgsz to dramatically speed up YOLO on CPU (default is 640)
    results = model(image, imgsz=320, verbose=False)[0]

    detections: list[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in classes:
            continue
        conf = float(box.conf[0].item())
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_name=VEHICLE_CLASSES.get(cls_id, str(cls_id)),
            )
        )

    # Sort largest first so the most prominent vehicle is processed first
    detections.sort(key=lambda d: d.area, reverse=True)
    return detections
