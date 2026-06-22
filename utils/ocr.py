"""
ocr.py
------
EasyOCR wrapper with pre-processing for licence plates.
"""

from __future__ import annotations

import cv2
import numpy as np

_reader = None
_reader_langs_cached: list[str] | None = None


def _get_reader(languages: list[str] = None):
    global _reader, _reader_langs_cached
    if _reader is not None:
        if languages and languages != _reader_langs_cached:
            print(
                f"[ocr] WARNING: languages {languages} requested but "
                f"{_reader_langs_cached} are already cached. Using cached reader."
            )
        return _reader
    import easyocr  # type: ignore
    langs = languages or ["en"]
    print(f"[ocr] Initialising EasyOCR for languages: {langs} …")
    _reader = easyocr.Reader(langs, gpu=False)
    _reader_langs_cached = langs
    print("[ocr] EasyOCR ready.")
    return _reader


def _preprocess(plate_image: np.ndarray) -> np.ndarray:
    """Enhance contrast and reduce noise for better OCR accuracy."""
    # Upscale small plates
    h, w = plate_image.shape[:2]
    if w < 200:
        scale = 200 / w
        plate_image = cv2.resize(
            plate_image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Slight Gaussian blur to reduce sensor noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Back to BGR so EasyOCR can handle it uniformly
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def read_plate(
    plate_image: np.ndarray,
    languages: list[str] | None = None,
    detail: bool = False,
) -> str:
    """Run OCR on a cropped licence-plate image.

    Parameters
    ----------
    plate_image: BGR numpy array of the plate region.
    languages:   EasyOCR language codes; defaults to ['en'].
    detail:      If True, returns list of (bbox, text, conf) tuples instead.

    Returns
    -------
    Cleaned plate text string (uppercase, spaces removed between segments).
    """
    reader = _get_reader(languages)
    processed = _preprocess(plate_image)
    
    # Restrict characters to alphanumeric and dash to speed up transformer decoding
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    results = reader.readtext(processed, allowlist=allowlist)

    if detail:
        return results  # type: ignore[return-value]

    # Concatenate all detected text segments, keeping a space between them
    parts = [text.strip() for (_, text, conf) in results if conf > 0.35]
    return " ".join(parts).upper()
