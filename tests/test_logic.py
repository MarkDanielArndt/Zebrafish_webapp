import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_seg_result(synthetic_fish_image, synthetic_fish_mask):
    bgr = synthetic_fish_image[:, :, ::-1].copy()
    return [{
        "filename":             "fish.png",
        "image_path":           "/tmp/fish.png",
        "original":             synthetic_fish_image,
        "mask":                 synthetic_fish_mask,
        "grown":                synthetic_fish_mask.copy(),
        "eye_mask":             None,
        "path_points":          np.array([[64, 128], [128, 128], [192, 128]]),
        "straight_line_points": ((64, 128), (192, 128)),
        "length":               1200.0,
        "curvature":            2,
        "ratio":                1.05,
        "eye_area":             None,
        "eye_diameter":         None,
    }]


def test_analyse_images_returns_one_result_per_image(tmp_path, synthetic_fish_image):
    import cv2
    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_grown = dummy_mask.copy()

    with patch("zebrafish_analysis.core.seg.segmentation_pipeline") as mock_pipeline, \
         patch("zebrafish_analysis.core.length.load_model") as mock_load, \
         patch("zebrafish_analysis.core.length.tube_length_border2border") as mock_length, \
         patch("zebrafish_analysis.core.length.classification_curvature") as mock_curv:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_grown],
        )
        mock_load.return_value = MagicMock()
        mock_length.return_value = (1200.0, 1100.0,
                                    np.array([[64, 128], [192, 128]]),
                                    ((64, 128), (192, 128)))
        mock_curv.return_value = (MagicMock(), MagicMock(item=lambda: 2))

        from zebrafish_analysis.slicer_extension.ZebrafishAnalysis.ZebrafishAnalysisLib.logic import analyse_images
        results = analyse_images(
            [img_path],
            {"length": True, "curvature": True, "ratio": True,
             "eyes": False, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99},
        )

    assert len(results) == 1
    assert results[0]["filename"] == "fish.png"
    assert results[0]["length"] == pytest.approx(1200.0)


def test_analyse_images_error_per_image_does_not_crash(tmp_path, synthetic_fish_image):
    import cv2
    p1 = str(tmp_path / "good.png")
    cv2.imwrite(p1, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)

    with patch("zebrafish_analysis.core.seg.segmentation_pipeline") as mock_pipeline, \
         patch("zebrafish_analysis.core.length.load_model"), \
         patch("zebrafish_analysis.core.length.tube_length_border2border") as mock_length:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_mask.copy()],
        )
        mock_length.side_effect = RuntimeError("synthetic error")

        from zebrafish_analysis.slicer_extension.ZebrafishAnalysis.ZebrafishAnalysisLib.logic import analyse_images
        results = analyse_images(
            [p1],
            {"length": True, "curvature": False, "ratio": False,
             "eyes": False, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99},
        )

    assert len(results) == 1
    assert results[0]["length"] is None


@pytest.mark.skip(reason="widget.py requires Slicer runtime — verify manually in Slicer")
def test_models_to_download_all_cached():
    pass


@pytest.mark.skip(reason="widget.py requires Slicer runtime — verify manually in Slicer")
def test_models_to_download_missing():
    pass
