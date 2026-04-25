"""
Scale bar auto-detection for microscopy images.

Finds the horizontal scale bar in the bottom portion of an image, reads its
physical length label via OCR (pytesseract), and computes µm-per-pixel
calibration.

Optional dependency: pytesseract + Tesseract binary.
Install with:
    pip install pytesseract
    # Then install the Tesseract binary for your OS:
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
    # Linux:   sudo apt install tesseract-ocr
    # macOS:   brew install tesseract
"""

import re
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_scale_label(text: str) -> Optional[float]:
    """
    Parse a scale-bar label string and return its value in micrometres (µm).

    Handles common formats:
        "500 µm", "500 um", "500um", "1 mm", "0.5mm", "200 nm", "50μm"

    Returns float in µm, or None if the text cannot be parsed.
    """
    if not text:
        return None

    # Normalise various Unicode micro/mu variants to plain 'u'
    text = (text
            .replace('\u03bc', 'u')   # Greek small letter mu (μ)
            .replace('\u00b5', 'u')   # Micro sign (µ)
            .replace('μ', 'u')
            .replace('µ', 'u'))

    match = re.search(r'(\d+(?:[.,]\d+)?)\s*(nm|mm|cm|um)', text.lower())
    if not match:
        return None

    val_str = match.group(1).replace(',', '.')
    unit = match.group(2)
    try:
        val = float(val_str)
    except ValueError:
        return None

    conversions = {'nm': 1e-3, 'um': 1.0, 'mm': 1e3, 'cm': 1e4}
    return val * conversions[unit]


def _find_scalebar_line(gray_crop: np.ndarray,
                        full_img_width: int) -> Optional[Dict]:
    """
    Locate the dominant horizontal bar (scale bar) inside ``gray_crop``.

    Uses morphological opening with a wide horizontal kernel to isolate
    horizontal lines, then picks the widest qualifying contour.

    Returns a dict {x, y, w, h, length_px} in crop-local coordinates,
    or None if nothing was found.
    """
    min_bar_len = max(20, full_img_width // 12)
    best: Optional[Dict] = None

    for invert in (False, True):
        src = (255 - gray_crop) if invert else gray_crop.copy()

        # Try a few thresholds to be robust across different image contrasts
        for thresh_val in (220, 180, 0):           # 0 → Otsu
            if thresh_val == 0:
                _, bw = cv2.threshold(
                    src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, bw = cv2.threshold(src, thresh_val, 255, cv2.THRESH_BINARY)

            # Wide horizontal kernel keeps only long horizontal structures
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (min_bar_len, 1))
            hlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            # Small dilation to bridge minor gaps in the bar
            hlines = cv2.dilate(hlines, np.ones((2, 5), np.uint8))

            for cnt in cv2.findContours(
                    hlines, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[0]:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < min_bar_len:
                    continue
                aspect = cw / max(ch, 1)
                if aspect < 5:          # must be clearly horizontal
                    continue
                if best is None or cw > best['length_px']:
                    best = {'x': x, 'y': y, 'w': cw, 'h': ch,
                            'length_px': cw}

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scalebar(img: np.ndarray) -> Dict[str, Any]:
    """
    Detect a scale bar in a microscopy image and return calibration data.

    Parameters
    ----------
    img : np.ndarray
        H×W (greyscale) or H×W×C (RGB / RGBA) uint8 image.

    Returns
    -------
    dict with keys:
        success        – bool
        scale_um_per_px – µm per pixel in the *original* image (float|None)
        phys_width_um  – total physical image width in µm (float|None)
        phys_height_um – total physical image height in µm (float|None)
        bar_length_px  – detected bar width in pixels (int|None)
        label_text     – raw OCR string (str|None)
        label_um       – parsed physical length in µm (float|None)
        debug_img      – RGB numpy array with detection overlay
        message        – human-readable result description
    """
    result: Dict[str, Any] = {
        'success': False,
        'scale_um_per_px': None,
        'phys_width_um': None,
        'phys_height_um': None,
        'bar_length_px': None,
        'label_text': None,
        'label_um': None,
        'debug_img': None,
        'message': '',
    }

    if img is None:
        result['message'] = 'No image provided.'
        return result

    # --- Normalise to RGB uint8 ---
    arr = np.asarray(img)
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3].copy()
    else:
        rgb = arr.copy()

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    result['debug_img'] = rgb.copy()

    if not _HAS_TESSERACT:
        result['message'] = (
            'pytesseract is not installed – OCR-based scale-bar reading '
            'is unavailable.\n'
            'Install it with:  pip install pytesseract\n'
            'Then install the Tesseract binary for your OS:\n'
            '  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n'
            '  Linux:   sudo apt install tesseract-ocr\n'
            '  macOS:   brew install tesseract'
        )
        return result

    h_full, w_full = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Search only the bottom 28 % of the image
    crop_top = int(h_full * 0.72)
    gray_crop = gray[crop_top:, :]

    bar = _find_scalebar_line(gray_crop, w_full)
    if bar is None:
        result['message'] = (
            'No scale bar detected in the bottom portion of the image.')
        return result

    # Absolute coordinates in the full image
    ax, ay = bar['x'], crop_top + bar['y']
    aw, ah = bar['w'], bar['h']
    result['bar_length_px'] = aw

    # --- OCR region: above and around the bar ---
    ocr_t = max(0, ay - 70)
    ocr_b = ay + ah + 8
    ocr_l = max(0, ax - 30)
    ocr_r = min(w_full, ax + aw + 30)
    text_roi = gray[ocr_t:ocr_b, ocr_l:ocr_r]

    label_um: Optional[float] = None
    label_text: Optional[str] = None

    if text_roi.size > 0:
        # Upscale 3× for better Tesseract accuracy on small text
        roi_up = cv2.resize(text_roi, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)

        for invert_ocr in (False, True):
            src_ocr = (255 - roi_up) if invert_ocr else roi_up
            _, bw_ocr = cv2.threshold(
                src_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            for psm in (7, 11, 6):
                cfg = (f'--oem 3 --psm {psm} '
                       r'-c tessedit_char_whitelist=0123456789.,µμumMnNkK ')
                try:
                    raw = pytesseract.image_to_string(
                        bw_ocr, config=cfg).strip()
                except Exception:
                    continue

                parsed = _parse_scale_label(raw)
                if parsed is not None:
                    label_um = parsed
                    label_text = raw
                    break

            if label_um is not None:
                break

    result['label_text'] = label_text
    result['label_um'] = label_um

    # --- Build debug overlay ---
    debug = rgb.copy()
    # Green rectangle around detected bar
    cv2.rectangle(debug, (ax, ay - 2), (ax + aw, ay + ah + 2),
                  (0, 220, 0), 2)
    # Orange rectangle around OCR region
    cv2.rectangle(debug, (ocr_l, ocr_t), (ocr_r, ocr_b),
                  (255, 165, 0), 1)

    if label_um is not None:
        scale_um_per_px = label_um / aw
        result.update(
            success=True,
            scale_um_per_px=scale_um_per_px,
            phys_width_um=scale_um_per_px * w_full,
            phys_height_um=scale_um_per_px * h_full,
            message=(
                f'Scale bar: {aw} px = {label_um:.1f} µm  →  '
                f'{scale_um_per_px:.4f} µm/px  |  '
                f'Image: {scale_um_per_px * w_full:.1f} × '
                f'{scale_um_per_px * h_full:.1f} µm'
            ),
        )
        cv2.putText(debug, f'{label_um:.0f} um / {aw} px',
                    (ax, max(12, ay - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2,
                    cv2.LINE_AA)
    else:
        result['message'] = (
            f'Scale bar line detected ({aw} px) but label could not be '
            f'read via OCR.  Raw OCR text: "{label_text or "—"}"'
        )
        cv2.putText(debug, f'bar = {aw} px (label unreadable)',
                    (ax, max(12, ay - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2,
                    cv2.LINE_AA)

    result['debug_img'] = debug
    return result
