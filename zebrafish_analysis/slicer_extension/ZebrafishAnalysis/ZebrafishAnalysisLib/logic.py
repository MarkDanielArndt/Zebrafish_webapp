"""
Logic layer for the ZebrafishAnalysis Slicer extension.

Wraps core/ functions and provides a clean public API:
  - analyse_images()   — batch segmentation + measurements
  - detect_scalebar()  — thin wrapper around core scalebar

Path setup (sys.path) is handled by ZebrafishAnalysis.py, not here.
Export functions (export_excel, export_csv) live in export.py.
"""

import os
import tempfile
import cv2
import numpy as np

_MODEL_CACHE: dict = {}
_original_load_unet = None  # set on first use by _install_model_cache()


def _install_model_cache():
    """Lazily import seg and install the caching monkey-patch (first call only)."""
    global _original_load_unet
    if _original_load_unet is not None:
        return
    import numpy as np  # noqa: F401 — must precede torch to enable numpy bridge
    import zebrafish_analysis.core.seg as _seg_module
    # On Slicer module reload, logic.py globals reset but seg._load_unet_model
    # still holds the wrapper from the old instance → would recurse infinitely.
    # Stash the true original on seg so it survives logic.py reloads.
    if hasattr(_seg_module, "_load_unet_model_original"):
        _original_load_unet = _seg_module._load_unet_model_original
    else:
        _original_load_unet = _seg_module._load_unet_model
        _seg_module._load_unet_model_original = _original_load_unet
    _seg_module._load_unet_model = _cached_load_unet


def _cached_load_unet(model_path=None, repo_id=None, filename=None, label="model",
                       revision="main", force_download=False, encoder_name="vgg16"):
    """Caching wrapper: first call loads from disk, subsequent calls return cached model."""
    cache_key = f"_unet_{filename}_{encoder_name}"
    if force_download or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _original_load_unet(
            model_path=model_path, repo_id=repo_id, filename=filename,
            label=label, revision=revision, force_download=force_download,
            encoder_name=encoder_name,
        )
    return _MODEL_CACHE[cache_key]


def preload_models(params: dict) -> None:
    """Load and cache all models needed for the given params.

    Safe to call from a background thread — only does file I/O and weight
    deserialization, no parallel OMP inference.
    """
    _install_model_cache()
    from zebrafish_analysis.core.length import load_model

    if params.get("curvature", True) and "curvature" not in _MODEL_CACHE:
        _MODEL_CACHE["curvature"] = load_model()

    body_filename = params.get("body_model_filename", "best_model_body_3400_vgg19.pth")
    body_encoder  = params.get("body_encoder_name",   "vgg19")
    _cached_load_unet(
        model_path=params.get("body_model_path"),
        repo_id="markdanielarndt/Zebrafish_Segmentation",
        filename=body_filename, label="body model",
        revision="main", force_download=False, encoder_name=body_encoder,
    )

    if params.get("eyes", False):
        eye_filename = params.get("eye_model_filename", "best_model_eye_3400.pth")
        _cached_load_unet(
            model_path=params.get("eye_model_path"),
            repo_id="markdanielarndt/Zebrafish_Segmentation",
            filename=eye_filename, label="eye model",
            revision="main", force_download=False, encoder_name="vgg16",
        )

# ---------------------------------------------------------------------------
# Result dict schema — every key must be present, missing values use None
# ---------------------------------------------------------------------------
_RESULT_KEYS = (
    "filename",
    "image_path",
    "original",
    "mask",
    "grown",
    "eye_mask",
    "path_points",
    "straight_line_points",
    "length",
    "curvature",
    "ratio",
    "eye_area",
    "eye_diameter",
    "spacing",
    "error",
)


def _empty_result(image_path: str) -> dict:
    r = {k: None for k in _RESULT_KEYS}
    r["filename"] = os.path.basename(image_path)
    r["image_path"] = image_path
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scalebar(image_path: str, label_um: float | None = None) -> dict:
    """
    Detect scale bar in an image file.

    Returns the dict produced by core detect_scalebar, or a failure dict
    if the image cannot be read.
    """
    import cv2
    from zebrafish_analysis.core.scalebar import detect_scalebar as _detect_scalebar
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"success": False, "bar_found": False,
                "message": "Could not read image."}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return _detect_scalebar(img_rgb, label_um=label_um)


def analyse_images(image_paths: list, params: dict,
                   progress_callback=None) -> list:
    """
    Run segmentation + measurements on a list of image paths.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to input images.
    params : dict
        Keys:
          length, curvature, ratio, eyes : bool
          hitl                           : bool  — use confidence threshold
          threshold                      : float 0–1
          um_per_px                      : float — physical scale (µm/pixel)
          body_model_filename            : str   — HF filename for body U-Net
          body_encoder_name              : str   — encoder (vgg16 / vgg19)
          eye_model_filename             : str   — HF filename for eye U-Net
          body_force_download            : bool  — force re-download of body model
    progress_callback : callable(current, total) | None

    Returns
    -------
    list[dict]
        One result dict per image. Every dict contains all keys from the
        schema. On per-image errors the numeric fields are None and
        ``error`` holds the exception message.
    """
    _install_model_cache()
    from zebrafish_analysis.core.seg import segmentation_pipeline
    from zebrafish_analysis.core.length import (
        load_model,
        tube_length_border2border,
        classification_curvature,
        compute_eye_metrics,
    )

    um_per_px = float(params.get("um_per_px", 22.99))
    include_eyes = params.get("eyes", False)
    body_filename = params.get("body_model_filename",
                               "best_model_body_3400_vgg19.pth")
    body_encoder = params.get("body_encoder_name", "vgg19")
    eye_filename = params.get("eye_model_filename", "best_model_eye_3400.pth")
    force_download = params.get("body_force_download", False)

    # ---- load curvature model once (cached across calls) ----
    if params.get("curvature", True):
        if "curvature" not in _MODEL_CACHE:
            _MODEL_CACHE["curvature"] = load_model()
        curv_model = _MODEL_CACHE["curvature"]
    else:
        curv_model = None

    # ---- per-image segmentation + measurement ----
    # Call segmentation_pipeline once per image so progress_callback fires
    # after each one, keeping the UI responsive. The cached _load_unet_model
    # means model weights are only read from disk once across all calls.
    _seg_kwargs = dict(
        include_eyes=include_eyes,
        body_model_filename=body_filename,
        body_encoder_name=body_encoder,
        body_force_download=force_download,
    )
    if params.get("body_model_path"):
        _seg_kwargs["body_model_path"] = params["body_model_path"]
    if include_eyes and eye_filename:
        _seg_kwargs["eye_model_filename"] = eye_filename
    if params.get("eye_model_path"):
        _seg_kwargs["eye_model_path"] = params["eye_model_path"]

    n = len(image_paths)
    results = []

    for _loop_i, image_path in enumerate(sorted(image_paths)):
        r = _empty_result(image_path)

        try:
            # Segment this single image — model already cached, no disk reload
            with tempfile.TemporaryDirectory() as _tmp:
                os.symlink(image_path, os.path.join(_tmp, os.path.basename(image_path)))
                seg_result = segmentation_pipeline(_tmp, **_seg_kwargs)

            if include_eyes and len(seg_result) == 4:
                originals_bgr, masks, growns, eyes_list = seg_result
            else:
                originals_bgr, masks, growns = seg_result[:3]
                eyes_list = [None]

            orig_bgr = originals_bgr[0] if originals_bgr else None
            mask    = masks[0]      if masks      else None
            grown   = growns[0]     if growns     else None
            eye     = eyes_list[0]  if eyes_list  else None

            if orig_bgr is None:
                r["error"] = "Could not read image."
                results.append(r)
                if progress_callback:
                    progress_callback(_loop_i + 1, n)
                continue

            r["original"]  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            r["mask"]      = mask
            r["grown"]     = grown
            r["eye_mask"]  = eye

            mask_bin = (mask > 0) if mask is not None else None
            eye_bin  = (eye  > 0) if eye  is not None else None

            h_orig, w_orig = orig_bgr.shape[:2]
            mask_h, mask_w = mask.shape[:2] if mask is not None else (256, 256)
            spacing = (
                um_per_px * h_orig / mask_h,
                um_per_px * w_orig / mask_w,
            )
            r["spacing"] = spacing

            # ---- length + ratio ----
            if params.get("length", True) and mask_bin is not None:
                try:
                    length, straight, path_pts, sl_pts = tube_length_border2border(
                        mask_bin,
                        spacing=spacing,
                        return_path=True,
                        return_straight_line=True,
                        mask_eye=eye_bin,
                        return_eye_info=False,
                    )
                    r["length"] = float(length)
                    r["path_points"] = path_pts
                    r["straight_line_points"] = sl_pts
                    if params.get("ratio", True) and straight and straight > 0:
                        r["ratio"] = float(length) / float(straight)
                except Exception as exc:
                    r["error"] = f"Length error: {exc}"

            # ---- curvature ----
            if params.get("curvature", True) and curv_model is not None:
                try:
                    use_thr = params.get("hitl", False)
                    thr = float(params.get("threshold", 0.85))
                    _, cls = classification_curvature(
                        orig_bgr, r["grown"], curv_model, use_thr, thr
                    )
                    r["curvature"] = int(cls.item())
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Curvature error: {exc}"

            # ---- eye metrics ----
            if params.get("eyes", False) and eye_bin is not None and mask_bin is not None:
                try:
                    info = compute_eye_metrics(
                        eye_bin, mask_fish=mask_bin, spacing=spacing
                    )
                    r["eye_area"]     = float(info.get("eye_area",     0))
                    r["eye_diameter"] = float(info.get("eye_diameter", 0))
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Eye metrics error: {exc}"

        except Exception as exc:
            import traceback
            r["error"] = f"Unhandled error: {exc}\n{traceback.format_exc()}"

        results.append(r)
        if progress_callback:
            progress_callback(_loop_i + 1, n)

    return results


# ---------------------------------------------------------------------------
# Manual correction
# ---------------------------------------------------------------------------

def apply_manual_correction(result, point1_orig, point2_orig, params=None):
    """
    Recompute length, ratio, and curvature from manually placed head/tail points.

    Parameters
    ----------
    result : dict
        Result dict (mutated in-place).  Must contain 'mask', 'original', 'spacing'.
    point1_orig, point2_orig : tuple
        (row, col) in original image coordinate space (as clicked on the display).
    params : dict | None
        Optional keys: 'hitl' (bool), 'threshold' (float).
        Used for curvature re-classification.  Defaults to hitl=False, threshold=0.85.

    Returns
    -------
    result : dict
        The same dict, updated in-place.
    """
    if params is None:
        params = {}

    spacing = result.get("spacing")
    if spacing is None:
        print("apply_manual_correction: spacing is None — skipping (fish had an error?)")
        return result

    mask = result.get("mask")
    original = result.get("original")
    if mask is None or original is None:
        print("apply_manual_correction: mask or original missing — skipping")
        return result

    from zebrafish_analysis.core.manual import compute_manual_length
    from zebrafish_analysis.core.length import classification_curvature

    # Snapshot auto values on first correction only
    if "_auto_length" not in result:
        result["_auto_length"] = result.get("length")
        result["_auto_ratio"] = result.get("ratio")
        result["_auto_path_points"] = result.get("path_points")
        result["_auto_straight_line_points"] = result.get("straight_line_points")
        result["_auto_curvature"] = result.get("curvature")

    # Convert original-image coords → mask coords
    orig_h, orig_w = original.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    scale_y = mask_h / orig_h
    scale_x = mask_w / orig_w

    point1_mask = (
        int(np.clip(point1_orig[0] * scale_y, 0, mask_h - 1)),
        int(np.clip(point1_orig[1] * scale_x, 0, mask_w - 1)),
    )
    point2_mask = (
        int(np.clip(point2_orig[0] * scale_y, 0, mask_h - 1)),
        int(np.clip(point2_orig[1] * scale_x, 0, mask_w - 1)),
    )

    # Recompute length + path
    length, straight_length, path_pts, sl_pts = compute_manual_length(
        mask, point1_mask, point2_mask, spacing
    )
    result["length"] = float(length)
    result["ratio"] = float(length / straight_length) if straight_length > 0 else None
    result["path_points"] = path_pts
    result["straight_line_points"] = sl_pts

    # Recompute curvature if model is loaded
    curv_model = _MODEL_CACHE.get("curvature")
    if curv_model is not None:
        try:
            use_thr = params.get("hitl", False)
            thr = float(params.get("threshold", 0.85))
            _, cls = classification_curvature(
                result["original"], result["grown"], curv_model, use_thr, thr
            )
            result["curvature"] = int(cls.item())
        except Exception as exc:
            print(f"apply_manual_correction: curvature recompute failed ({exc})")
    else:
        print("apply_manual_correction: curvature model not in cache — skipping")

    result["manual_corrected"] = True
    return result


def revert_manual_correction(result):
    """
    Restore auto-computed values saved before the first manual correction.
    No-op if result['manual_corrected'] is not set.

    Returns
    -------
    result : dict
        The same dict, updated in-place.
    """
    if not result.get("manual_corrected"):
        return result

    result["length"] = result.pop("_auto_length", result.get("length"))
    result["ratio"] = result.pop("_auto_ratio", result.get("ratio"))
    result["path_points"] = result.pop("_auto_path_points", result.get("path_points"))
    result["straight_line_points"] = result.pop(
        "_auto_straight_line_points", result.get("straight_line_points")
    )
    result["curvature"] = result.pop("_auto_curvature", result.get("curvature"))
    result.pop("manual_corrected", None)
    return result
