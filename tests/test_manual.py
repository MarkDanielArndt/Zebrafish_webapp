# tests/test_manual.py
import numpy as np
import pytest
import cv2


def _ellipse_mask(h=256, w=256):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (100, 30), 0, 0, 360, 255, -1)
    return mask


def test_compute_manual_length_basic():
    """Points at opposite ends of ellipse mask return positive length."""
    from zebrafish_analysis.core.manual import compute_manual_length
    mask = _ellipse_mask()
    spacing = (1.0, 1.0)
    p1 = (128, 28)   # left end of ellipse
    p2 = (128, 228)  # right end of ellipse
    length, straight, path, sl_pts = compute_manual_length(mask, p1, p2, spacing)
    assert length > 0
    assert straight > 0
    assert len(path) >= 2
    assert len(sl_pts) == 2


def test_compute_manual_length_greater_than_straight():
    """Curved path through mask is >= straight-line distance."""
    from zebrafish_analysis.core.manual import compute_manual_length
    mask = _ellipse_mask()
    spacing = (1.0, 1.0)
    p1 = (128, 28)
    p2 = (128, 228)
    length, straight, path, _ = compute_manual_length(mask, p1, p2, spacing)
    assert length >= straight - 1e-6  # path >= straight line


def test_compute_manual_length_points_outside_mask():
    """Points outside mask are snapped to nearest mask pixel — no crash."""
    from zebrafish_analysis.core.manual import compute_manual_length
    mask = _ellipse_mask()
    spacing = (1.0, 1.0)
    p1 = (0, 0)    # corner — outside ellipse
    p2 = (255, 255)  # corner — outside ellipse
    length, straight, path, sl_pts = compute_manual_length(mask, p1, p2, spacing)
    assert length > 0
    assert len(path) >= 2


def test_compute_manual_length_empty_mask_fallback():
    """Empty mask triggers straight-line fallback — no crash."""
    from zebrafish_analysis.core.manual import compute_manual_length
    mask = np.zeros((256, 256), dtype=np.uint8)
    spacing = (1.0, 1.0)
    p1 = (50, 50)
    p2 = (200, 200)
    length, straight, path, sl_pts = compute_manual_length(mask, p1, p2, spacing)
    assert length > 0
    assert len(path) >= 2


def test_compute_manual_length_spacing_scales_result():
    """Doubling spacing doubles the returned length."""
    from zebrafish_analysis.core.manual import compute_manual_length
    mask = _ellipse_mask()
    p1 = (128, 28)
    p2 = (128, 228)
    length1, _, _, _ = compute_manual_length(mask, p1, p2, (1.0, 1.0))
    length2, _, _, _ = compute_manual_length(mask, p1, p2, (2.0, 2.0))
    assert abs(length2 - 2 * length1) < 5.0  # within 5 µm tolerance
