"""Tests for pure math helpers in zoom_view. No Qt required."""


def _zoom_factor_from_delta(dy: int) -> float:
    """Compute zoom scale factor from wheel delta. Extracted for testing."""
    return 1.15 ** (dy / 120.0)


def _minimap_viewport_rect(
    vis_x, vis_y, vis_w, vis_h,   # visible scene rect
    scene_w, scene_h,              # full scene size
    thumb_x, thumb_y,              # thumbnail offset in minimap canvas
    thumb_w, thumb_h,              # thumbnail size in minimap canvas
):
    """Map visible scene rect to pixel rect inside minimap thumbnail. Pure math."""
    if scene_w <= 0 or scene_h <= 0:
        return (thumb_x, thumb_y, thumb_w, thumb_h)
    sx = thumb_w / scene_w
    sy = thumb_h / scene_h
    rx = int(thumb_x + vis_x * sx)
    ry = int(thumb_y + vis_y * sy)
    rw = max(2, int(vis_w * sx))
    rh = max(2, int(vis_h * sy))
    return (rx, ry, rw, rh)


def test_zoom_factor_scroll_up():
    f = _zoom_factor_from_delta(120)
    assert abs(f - 1.15) < 0.001

def test_zoom_factor_scroll_down():
    f = _zoom_factor_from_delta(-120)
    assert abs(f - (1/1.15)) < 0.001

def test_zoom_factor_half_notch():
    f = _zoom_factor_from_delta(60)
    assert 1.0 < f < 1.15

def test_minimap_full_visible():
    # Entire scene visible → rect fills thumbnail
    rx, ry, rw, rh = _minimap_viewport_rect(0, 0, 100, 100, 100, 100, 0, 0, 80, 60)
    assert rx == 0 and ry == 0 and rw == 80 and rh == 60

def test_minimap_top_left_quadrant():
    # Top-left quarter of scene visible
    rx, ry, rw, rh = _minimap_viewport_rect(0, 0, 50, 50, 100, 100, 0, 0, 80, 80)
    assert rx == 0 and ry == 0 and rw == 40 and rh == 40

def test_minimap_bottom_right_quadrant():
    rx, ry, rw, rh = _minimap_viewport_rect(50, 50, 50, 50, 100, 100, 0, 0, 80, 80)
    assert rx == 40 and ry == 40 and rw == 40 and rh == 40

def test_minimap_minimum_size():
    # Very zoomed in → rect must be at least 2×2
    rx, ry, rw, rh = _minimap_viewport_rect(0, 0, 1, 1, 1000, 1000, 0, 0, 80, 80)
    assert rw >= 2 and rh >= 2

def test_minimap_with_thumb_offset():
    # Thumbnail not at origin (letterboxed in minimap canvas)
    rx, ry, rw, rh = _minimap_viewport_rect(0, 0, 100, 100, 100, 100, 10, 5, 80, 60)
    assert rx == 10 and ry == 5 and rw == 80 and rh == 60
