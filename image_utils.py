"""
image_utils.py
--------------
Shared image helpers used across the pipeline.
"""

import cv2
import numpy as np


def safe_crop(image: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """Crop *image* to *bbox* = (x1, y1, x2, y2), clamping to image bounds.
    Returns None if the cropped region is empty."""
    h, w = image.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def resize_keep_aspect(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Resize *image* so its longest side is at most *max_side*, preserving AR."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_boxes(
    image: np.ndarray,
    car_bbox: tuple,
    plate_bbox_in_car: tuple | None,
    plate_text: str,
    car_color: tuple = (0, 200, 0),
    plate_color: tuple = (200, 100, 0),
) -> np.ndarray:
    """Return an annotated copy of *image* with bounding boxes and label."""
    out = image.copy()
    cx1, cy1, cx2, cy2 = (int(v) for v in car_bbox[:4])
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), car_color, 2)
    cv2.putText(out, "car", (cx1, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, car_color, 2)

    if plate_bbox_in_car is not None:
        px1, py1, px2, py2 = (int(v) for v in plate_bbox_in_car[:4])
        # Convert plate coords from car-crop space to full-image space
        px1 += cx1; py1 += cy1; px2 += cx1; py2 += cy1
        cv2.rectangle(out, (px1, py1), (px2, py2), plate_color, 2)
        label = plate_text if plate_text else "plate"
        cv2.putText(out, label, (px1, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, plate_color, 2)

    return out
