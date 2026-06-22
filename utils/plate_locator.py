"""
plate_locator.py
----------------
Locates licence plates inside a (pre-cropped) car image.

Strategy:
1. Try OpenCV's Haar cascade for Russian/European style plates.
2. Fallback: contour-based detection looking for rectangles whose
   aspect ratio is roughly 2:1 – 6:1 (covers most plate formats).
"""

from __future__ import annotations
import os
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Haar cascade — we ship a small wrapper that auto-downloads the cascade XML
# from the OpenCV GitHub repository if it is not bundled with the local install.
# ---------------------------------------------------------------------------
_CASCADE_NAME = "haarcascade_russian_plate_number.xml"
_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    + _CASCADE_NAME
)
_CASCADE_LOCAL = os.path.join(os.path.dirname(__file__), _CASCADE_NAME)

_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier | None:
    global _cascade
    if _cascade is not None:
        return _cascade

    # 1. Try loading from OpenCV's own data directory
    built_in = os.path.join(cv2.data.haarcascades, _CASCADE_NAME)  # type: ignore[attr-defined]
    for path in (built_in, _CASCADE_LOCAL):
        if os.path.exists(path):
            cc = cv2.CascadeClassifier(path)
            if not cc.empty():
                _cascade = cc
                return _cascade

    # 2. Download from GitHub
    try:
        print(f"[plate_locator] Downloading {_CASCADE_NAME} …")
        urllib.request.urlretrieve(_CASCADE_URL, _CASCADE_LOCAL)
        cc = cv2.CascadeClassifier(_CASCADE_LOCAL)
        if not cc.empty():
            _cascade = cc
            return _cascade
    except Exception as exc:
        print(f"[plate_locator] Could not download cascade: {exc}")

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def locate_plate(car_image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the best licence-plate bounding box in *car_image*.

    Returns (x1, y1, x2, y2) in *car_image* coordinates, or None.
    """
    gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

    # --- Method 1: Haar cascade ------------------------------------------
    result = _try_haar(gray)
    if result is not None:
        return result

    # --- Method 2: Contour heuristics ------------------------------------
    return _try_contours(gray, car_image)


def _try_haar(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = _get_cascade()
    if cascade is None:
        return None

    plates = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 15),
    )
    if len(plates) == 0:
        return None

    # Pick the largest detection
    best = max(plates, key=lambda r: r[2] * r[3])
    x, y, w, h = best
    return x, y, x + w, y + h


def _try_contours(gray: np.ndarray, color: np.ndarray) -> tuple[int, int, int, int] | None:
    """Contour-based plate detection as a fallback."""
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Edge map
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = gray.shape

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        area_ratio = (w * h) / (img_w * img_h)
        # Typical plate is narrow & wide; ignore tiny or huge regions
        if 1.0 <= aspect <= 8.0 and 0.005 <= area_ratio <= 0.50:
            # Prefer candidates in the lower half of the car image
            score = aspect * area_ratio * (1 + int(y > img_h * 0.4))
            candidates.append((score, x, y, x + w, y + h))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, x1, y1, x2, y2 = candidates[0]
    return x1, y1, x2, y2
