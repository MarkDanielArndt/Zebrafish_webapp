import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


def _make_dummy_model():
    model = MagicMock()
    dummy_logits = MagicMock()
    dummy_logits.squeeze.return_value.cpu.return_value.numpy.return_value = np.zeros((256, 256), dtype=np.float32)
    model.return_value = dummy_logits
    return model


def _make_pil_mask():
    return Image.fromarray(np.zeros((256, 256), dtype=np.uint8))


def test_segmentation_pipeline_returns_three_tuple(synthetic_fish_image, tmp_path):
    """segmentation_pipeline returns (originals, masks, growns) for include_eyes=False."""
    import cv2

    img_path = tmp_path / "fish.png"
    cv2.imwrite(str(img_path), synthetic_fish_image)

    dummy_model = _make_dummy_model()

    with patch("zebrafish_analysis.core.seg.hf_hub_download") as mock_dl, \
         patch("zebrafish_analysis.core.seg.torch") as mock_torch, \
         patch("zebrafish_analysis.core.seg.Unet") as mock_unet, \
         patch("zebrafish_analysis.core.seg.segment_fish") as mock_seg:

        mock_dl.return_value = "/fake/path/model.pth"
        mock_torch.load.return_value = {}
        mock_torch.device.return_value = "cpu"
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.tensor.return_value = MagicMock()
        mock_unet.return_value = dummy_model

        pil_mask = _make_pil_mask()
        confidence = np.zeros((256, 256), dtype=np.uint8)
        mock_seg.return_value = (pil_mask, confidence)

        from zebrafish_analysis.core.seg import segmentation_pipeline
        result = segmentation_pipeline(
            folder_path=str(tmp_path),
            include_eyes=False,
            body_force_download=False,
        )

    assert len(result) == 3
    originals, masks, growns = result
    assert len(originals) == 1


def test_segmentation_pipeline_accepts_local_model_paths():
    # Slicer passes pre-downloaded weights as body_model_path/eye_model_path.
    # Regression: body_model_path was missing from the signature.
    import inspect
    from zebrafish_analysis.core.seg import segmentation_pipeline

    params = inspect.signature(segmentation_pipeline).parameters
    assert "body_model_path" in params
    assert "eye_model_path" in params


def test_segmentation_pipeline_sorted_order(tmp_path):
    """Output count matches number of images in the folder — regression for filename scrambling bug."""
    import cv2

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    for name in ["b.png", "a.png", "c.png"]:
        cv2.imwrite(str(tmp_path / name), dummy_img)

    dummy_model = _make_dummy_model()
    pil_mask = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
    confidence = np.zeros((256, 256), dtype=np.uint8)

    with patch("zebrafish_analysis.core.seg.hf_hub_download"), \
         patch("zebrafish_analysis.core.seg.torch") as mock_torch, \
         patch("zebrafish_analysis.core.seg.Unet") as mock_unet, \
         patch("zebrafish_analysis.core.seg.segment_fish") as mock_seg:

        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.tensor.return_value = MagicMock()
        mock_unet.return_value = dummy_model
        mock_seg.return_value = (pil_mask, confidence)

        from zebrafish_analysis.core.seg import segmentation_pipeline
        originals, masks, growns = segmentation_pipeline(
            folder_path=str(tmp_path),
            include_eyes=False,
            body_force_download=False,
        )

    assert len(originals) == 3
    assert len(masks) == 3
    assert len(growns) == 3


def test_segmentation_pipeline_include_eyes_returns_four_tuple(tmp_path):
    """include_eyes=True returns a 4-tuple including eye masks."""
    import cv2

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "fish.png"), dummy_img)

    dummy_model = _make_dummy_model()
    pil_mask = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
    confidence = np.zeros((256, 256), dtype=np.uint8)

    with patch("zebrafish_analysis.core.seg.hf_hub_download"), \
         patch("zebrafish_analysis.core.seg.torch") as mock_torch, \
         patch("zebrafish_analysis.core.seg.Unet") as mock_unet, \
         patch("zebrafish_analysis.core.seg.segment_fish") as mock_seg:

        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.tensor.return_value = MagicMock()
        mock_unet.return_value = dummy_model
        mock_seg.return_value = (pil_mask, confidence)

        from zebrafish_analysis.core.seg import segmentation_pipeline
        result = segmentation_pipeline(
            folder_path=str(tmp_path),
            include_eyes=True,
            body_force_download=False,
        )

    assert len(result) == 4
    originals, masks, growns, eyes = result
    assert len(eyes) == 1


def test_segmentation_pipeline_model_load_failure_raises(tmp_path):
    """RuntimeError is raised when the body model cannot be loaded."""
    import cv2
    import zebrafish_analysis.core.seg as seg_mod

    cv2.imwrite(str(tmp_path / "fish.png"), np.zeros((64, 64, 3), dtype=np.uint8))

    # Patch _load_unet_model directly — logic.py may have monkey-patched it at import
    # time, so patching hf_hub_download would be bypassed by the caching wrapper.
    with patch.object(seg_mod, "_load_unet_model", return_value=None):
        from zebrafish_analysis.core.seg import segmentation_pipeline
        with pytest.raises(RuntimeError):
            segmentation_pipeline(
                folder_path=str(tmp_path),
                include_eyes=False,
                body_force_download=False,
            )
