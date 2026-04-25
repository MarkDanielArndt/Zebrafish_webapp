"""
Scale bar auto-detection for microscopy images.

Finds the horizontal scale bar in the bottom portion of an image and returns
its length in pixels.  The user supplies the corresponding physical length
(µm), which is then used to compute the µm-per-pixel calibration.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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

def detect_scalebar(img: np.ndarray,
                    label_um: Optional[float] = None) -> Dict[str, Any]:
    """
    Detect the scale bar line in a microscopy image.

    Parameters
    ----------
    img : np.ndarray
        H×W (greyscale) or H×W×C (RGB / RGBA) uint8 image.
    label_um : float, optional
        Physical length of the scale bar in µm as provided by the user.
        When supplied, full calibration (µm/px, image physical size) is
        computed.  When None, only the bar pixel length is returned and the
        caller should ask the user for the physical value.

    Returns
    -------
    dict with keys:
        success         – bool  (True only when bar found AND label_um given)
        bar_found       – bool  (True when the bar line was detected)
        scale_um_per_px – µm per pixel in the original image (float|None)
        phys_width_um   – total physical image width in µm (float|None)
        phys_height_um  – total physical image height in µm (float|None)
        bar_length_px   – detected bar width in pixels (int|None)
        debug_img       – RGB numpy array with detection overlay
        message         – human-readable result description
    """
    result: Dict[str, Any] = {
        'success': False,
        'bar_found': False,
        'scale_um_per_px': None,
        'phys_width_um': None,
        'phys_height_um': None,
        'bar_length_px': None,
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

    h_full, w_full = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Search only the bottom 28 % of the image
    crop_top = int(h_full * 0.72)
    gray_crop = gray[crop_top:, :]

    bar = _find_scalebar_line(gray_crop, w_full)
    if bar is None:
        result['message'] = 'No scale bar detected in the bottom portion of the image.'
        return result

    # Absolute coordinates in the full image
    ax, ay = bar['x'], crop_top + bar['y']
    aw, ah = bar['w'], bar['h']
    result['bar_length_px'] = aw
    result['bar_found'] = True

    # --- Build debug overlay ---
    debug = rgb.copy()
    cv2.rectangle(debug, (ax, ay - 2), (ax + aw, ay + ah + 2),
                  (0, 220, 0), 2)

    if label_um is not None and label_um > 0:
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
            f'Scale bar detected: {aw} px.  '
            f'Enter the physical length above and click "Apply" to calibrate.'
        )
        cv2.putText(debug, f'bar = {aw} px',
                    (ax, max(12, ay - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2,
                    cv2.LINE_AA)

    result['debug_img'] = debug
    return result
