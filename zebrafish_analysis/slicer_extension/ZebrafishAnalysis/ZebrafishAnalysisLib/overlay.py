"""
Render OpenCV overlays onto zebrafish images.

make_overlay(result, thumbnail_size) -> uint8 H×W×3 RGB array
make_full_overlay(result)            -> uint8 H×W×3 BGR array (full resolution)
"""

import numpy as np
import cv2

# BGR colours
_MASK_COLOR   = (0,   200, 255)  # yellow
_EYE_COLOR    = (0,   0,   200)  # red
_PATH_COLOR   = (200, 200, 0  )  # cyan
_STRAIGHT_CLR = (200, 0,   200)  # magenta
_MASK_ALPHA   = 0.35
_EYE_ALPHA    = 0.45


def _blend_mask(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float) -> np.ndarray:
    overlay = base_bgr.copy()
    overlay[mask > 0] = color_bgr
    return cv2.addWeighted(base_bgr, 1 - alpha, overlay, alpha, 0)


def _draw_polyline(img: np.ndarray, points: np.ndarray, color: tuple, thickness: int = 2) -> np.ndarray:
    if points is None or len(points) < 2:
        return img
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
    return img


def _resize_to_fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_full_overlay(result: dict) -> np.ndarray:
    """Return full-resolution BGR overlay array."""
    original = result.get("original")
    if original is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    # original is stored as RGB in result dicts
    base = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    h_base, w_base = base.shape[:2]

    mask = result.get("mask")
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (w_base, h_base), interpolation=cv2.INTER_NEAREST)
        base = _blend_mask(base, m, _MASK_COLOR, _MASK_ALPHA)

    eye_mask = result.get("eye_mask")
    if eye_mask is not None:
        em = cv2.resize(eye_mask.astype(np.uint8), (w_base, h_base), interpolation=cv2.INTER_NEAREST)
        base = _blend_mask(base, em, _EYE_COLOR, _EYE_ALPHA)

    path_pts = result.get("path_points")
    if path_pts is not None and len(path_pts) >= 2:
        mask_h = mask.shape[0] if mask is not None else h_base
        mask_w = mask.shape[1] if mask is not None else w_base
        scale_y = h_base / mask_h
        scale_x = w_base / mask_w
        scaled = path_pts.astype(float).copy()
        scaled[:, 0] = scaled[:, 0] * scale_y  # row  → y-axis
        scaled[:, 1] = scaled[:, 1] * scale_x  # col  → x-axis
        _draw_polyline(base, scaled[:, ::-1], _PATH_COLOR, thickness=2)  # (row,col) → (x,y)

    sl_pts = result.get("straight_line_points")
    if sl_pts is not None:
        p0, p1 = sl_pts
        mask_h = mask.shape[0] if mask is not None else h_base
        mask_w = mask.shape[1] if mask is not None else w_base
        scale_y = h_base / mask_h
        scale_x = w_base / mask_w
        x0, y0 = int(p0[1] * scale_x), int(p0[0] * scale_y)
        x1, y1 = int(p1[1] * scale_x), int(p1[0] * scale_y)
        cv2.line(base, (x0, y0), (x1, y1), _STRAIGHT_CLR, 2)

    return base  # BGR


def make_overlay(result: dict, thumbnail_size: int = 150) -> np.ndarray:
    """Return thumbnail-sized RGB overlay array."""
    bgr = make_full_overlay(result)
    thumb = _resize_to_fit(bgr, thumbnail_size)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
