import numpy as np


def test_make_overlay_returns_rgb_array(synthetic_fish_image, synthetic_fish_mask):
    from zebrafish_analysis.slicer_extension.ZebrafishAnalysis.ZebrafishAnalysisLib.overlay import make_overlay

    result_dict = {
        "original":  synthetic_fish_image,
        "mask":      synthetic_fish_mask,
        "eye_mask":  None,
        "path_points":          np.array([[64, 128], [128, 128], [192, 128]]),
        "straight_line_points": ((64, 128), (192, 128)),
        "length":    1200.0,
        "curvature": 2,
        "ratio":     1.05,
    }

    overlay = make_overlay(result_dict, thumbnail_size=150)
    assert overlay.ndim == 3
    assert overlay.shape[2] == 3
    assert overlay.dtype == np.uint8
    assert overlay.shape[0] <= 150
    assert overlay.shape[1] <= 150


def test_make_overlay_handles_none_mask(synthetic_fish_image):
    from zebrafish_analysis.slicer_extension.ZebrafishAnalysis.ZebrafishAnalysisLib.overlay import make_overlay

    result_dict = {
        "original":             synthetic_fish_image,
        "mask":                 None,
        "eye_mask":             None,
        "path_points":          None,
        "straight_line_points": None,
        "length":               None,
        "curvature":            None,
        "ratio":                None,
    }
    overlay = make_overlay(result_dict, thumbnail_size=150)
    assert overlay is not None
    assert overlay.dtype == np.uint8
