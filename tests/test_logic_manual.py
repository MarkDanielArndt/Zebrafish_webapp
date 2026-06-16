# tests/test_logic_manual.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sys, os

# Point at the lib dir so logic.py is importable without a full Slicer runtime
_LIB = os.path.join(
    os.path.dirname(__file__), "..",
    "zebrafish_analysis", "slicer_extension",
    "ZebrafishAnalysis", "ZebrafishAnalysisLib",
)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)


def _make_result(with_spacing=True, with_mask=True, orig_shape=(256, 256, 3)):
    """Minimal result dict for testing."""
    import cv2
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (100, 30), 0, 0, 360, 255, -1)
    orig = np.zeros(orig_shape, dtype=np.uint8)
    result = {
        "filename": "test.tif",
        "image_path": "/tmp/test.tif",
        "original": orig,
        "mask": mask if with_mask else None,
        "grown": orig.copy(),
        "eye_mask": None,
        "path_points": None,
        "straight_line_points": None,
        "length": 1000.0,
        "curvature": 2,
        "ratio": 1.1,
        "eye_area": None,
        "eye_diameter": None,
        "spacing": (1.0, 1.0) if with_spacing else None,
        "error": None,
        "manual_corrected": False,
    }
    return result


def test_apply_manual_correction_updates_length():
    """apply_manual_correction changes result['length'] from auto value."""
    import logic

    result = _make_result()

    mock_cls = MagicMock()
    mock_cls.item.return_value = 3

    with patch("logic._MODEL_CACHE", {"curvature": MagicMock()}), \
         patch("zebrafish_analysis.core.manual.compute_manual_length",
               return_value=(500.0, 450.0, np.zeros((10, 2), dtype=int), ((0, 0), (9, 9)))), \
         patch("zebrafish_analysis.core.length.classification_curvature",
               return_value=(None, mock_cls)):
        logic.apply_manual_correction(result, (128, 28), (128, 228))

    assert result["length"] == 500.0
    assert result["manual_corrected"] is True


def test_apply_manual_correction_snapshots_auto_values():
    """First call saves auto values; second call does not overwrite the snapshot."""
    import logic

    result = _make_result()
    result["length"] = 1000.0
    result["curvature"] = 2

    mock_cls = MagicMock()
    mock_cls.item.return_value = 3

    with patch("logic._MODEL_CACHE", {"curvature": MagicMock()}), \
         patch("zebrafish_analysis.core.manual.compute_manual_length",
               return_value=(500.0, 450.0, np.zeros((10, 2), dtype=int), ((0, 0), (9, 9)))), \
         patch("zebrafish_analysis.core.length.classification_curvature",
               return_value=(None, mock_cls)):
        logic.apply_manual_correction(result, (128, 28), (128, 228))
        assert result["_auto_length"] == 1000.0
        assert result["_auto_curvature"] == 2

        # Second call — snapshot must not change
        mock_cls2 = MagicMock()
        mock_cls2.item.return_value = 4
        logic.apply_manual_correction(result, (100, 30), (150, 220))
        assert result["_auto_length"] == 1000.0  # still the original auto value
        assert result["_auto_curvature"] == 2


def test_apply_manual_correction_noop_when_spacing_none():
    """Returns unchanged result when spacing is None."""
    import logic

    result = _make_result(with_spacing=False)
    result["length"] = 999.0

    logic.apply_manual_correction(result, (128, 28), (128, 228))
    assert result["length"] == 999.0
    assert not result.get("manual_corrected")


def test_apply_manual_correction_skips_curvature_when_no_model():
    """Curvature unchanged when model not in cache."""
    import logic

    result = _make_result()
    result["curvature"] = 2

    with patch("logic._MODEL_CACHE", {}), \
         patch("zebrafish_analysis.core.manual.compute_manual_length",
               return_value=(500.0, 450.0, np.zeros((10, 2), dtype=int), ((0, 0), (9, 9)))):
        logic.apply_manual_correction(result, (128, 28), (128, 228))

    assert result["curvature"] == 2  # unchanged


def test_revert_manual_correction_restores_auto():
    """revert_manual_correction restores auto-snapshotted values."""
    import logic

    result = _make_result()
    result["_auto_length"] = 1000.0
    result["_auto_ratio"] = 1.1
    result["_auto_path_points"] = None
    result["_auto_straight_line_points"] = None
    result["_auto_curvature"] = 2
    result["length"] = 500.0
    result["ratio"] = 1.05
    result["curvature"] = 3
    result["manual_corrected"] = True

    logic.revert_manual_correction(result)

    assert result["length"] == 1000.0
    assert result["ratio"] == 1.1
    assert result["curvature"] == 2
    assert not result.get("manual_corrected")
    assert "_auto_length" not in result


def test_revert_manual_correction_noop_when_not_corrected():
    """revert_manual_correction is a no-op when manual_corrected not set."""
    import logic

    result = _make_result()
    result["length"] = 1000.0

    logic.revert_manual_correction(result)
    assert result["length"] == 1000.0
