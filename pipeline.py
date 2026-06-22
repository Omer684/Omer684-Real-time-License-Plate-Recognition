"""
pipeline.py
-----------
Core orchestration: car detection → plate localization → OCR.

The pipeline has two modes:
  1. Normal: YOLO detects a vehicle → crop vehicle → find plate inside → OCR.
  2. Fallback: No vehicle detected → search the full frame for a plate → OCR.
     This handles close-ups, phone screens, and cropped plate images.
"""

from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from utils.detector import detect_vehicles, Detection
from utils.plate_locator import locate_plate
from utils.ocr import read_plate
from utils.image_utils import safe_crop, draw_boxes, resize_keep_aspect


@dataclass
class PlateResult:
    """Holds all outputs for one detected vehicle."""
    vehicle: Detection
    plate_bbox: tuple[int, int, int, int] | None   # in car-crop coordinates
    plate_image: np.ndarray | None
    plate_text: str


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_pipeline(
    image: np.ndarray,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    ocr_languages: list[str] | None = None,
    max_vehicles: int = 3,
) -> list[PlateResult]:
    """End-to-end pipeline on a single BGR image.

    Parameters
    ----------
    image:          BGR numpy array.
    model_name:     YOLOv8 weights to use.
    conf_threshold: Minimum detection confidence.
    ocr_languages:  EasyOCR language list (default ['en']).
    max_vehicles:   Maximum number of vehicles to process.

    Returns
    -------
    List of PlateResult objects (one per detected vehicle).
    """
    ocr_languages = ocr_languages or ["en"]

    # Resize image to a max of 640px to prevent huge images from tanking CPU
    image = resize_keep_aspect(image, max_side=640)

    # 1. Detect vehicles
    vehicles = detect_vehicles(image, model_name=model_name, conf_threshold=conf_threshold)
    vehicles = vehicles[:max_vehicles]

    results: list[PlateResult] = []

    # ------------------------------------------------------------------
    # Fallback: no vehicles found → search the full frame for a plate.
    # Handles: phone/screen showing a plate, close-up shots, cropped images.
    # ------------------------------------------------------------------
    if not vehicles:
        direct = _run_direct_plate(image, ocr_languages)
        if direct is not None:
            results.append(direct)
        return results

    for vehicle in vehicles:
        # 2. Crop the vehicle region
        car_crop = safe_crop(image, vehicle.bbox)
        if car_crop is None:
            results.append(PlateResult(vehicle=vehicle, plate_bbox=None, plate_image=None, plate_text=""))
            continue

        # 3. Locate the licence plate inside the car crop
        plate_bbox = locate_plate(car_crop)

        if plate_bbox is None:
            # Sub-fallback: try the full vehicle crop with direct OCR (skip plate localisation)
            plate_text = read_plate(car_crop, languages=ocr_languages)
            results.append(PlateResult(vehicle=vehicle, plate_bbox=None, plate_image=car_crop, plate_text=plate_text))
            continue

        # 4. Crop the plate
        plate_crop = safe_crop(car_crop, plate_bbox)
        if plate_crop is None:
            results.append(PlateResult(vehicle=vehicle, plate_bbox=plate_bbox, plate_image=None, plate_text=""))
            continue

        # 5. Run OCR
        plate_text = read_plate(plate_crop, languages=ocr_languages)

        results.append(
            PlateResult(
                vehicle=vehicle,
                plate_bbox=plate_bbox,
                plate_image=plate_crop,
                plate_text=plate_text,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_direct_plate(
    image: np.ndarray,
    ocr_languages: list[str],
) -> PlateResult | None:
    """Find and read a plate directly from the full *image* (no vehicle step)."""
    plate_bbox = locate_plate(image)

    if plate_bbox is None:
        # DO NOT run OCR on the full frame directly as it brings CPU to a halt.
        return None
    else:
        plate_crop = safe_crop(image, plate_bbox)
        if plate_crop is None:
            return None
        plate_text = read_plate(plate_crop, languages=ocr_languages)

    h, w = image.shape[:2]
    synthetic_vehicle = Detection(
        x1=0, y1=0, x2=w, y2=h,
        confidence=1.0,
        class_name="direct",
    )

    return PlateResult(
        vehicle=synthetic_vehicle,
        plate_bbox=plate_bbox,
        plate_image=plate_crop,
        plate_text=plate_text,
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_image(image: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    """Draw bounding boxes and plate text on *image*. Returns annotated copy."""
    out = image.copy()
    for r in results:
        out = draw_boxes(
            out,
            car_bbox=r.vehicle.bbox,
            plate_bbox_in_car=r.plate_bbox,
            plate_text=r.plate_text,
        )
    return out
