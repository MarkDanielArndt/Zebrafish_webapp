import gradio as gr
import tempfile, os, shutil
from typing import List, Optional, Tuple
from seg import segmentation_pipeline
from length import load_model, get_fish_length_circles_fixed, classification_curvature, tube_length_border2border, compute_eye_metrics
import openpyxl, io
from openpyxl.drawing.image import Image as ExcelImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import cv2

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from scalebar import detect_scalebar as _detect_scalebar
    _HAS_SCALEBAR = True
except Exception:
    _HAS_SCALEBAR = False

MODEL = None  # lazy-loaded

def _ensure_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL

def _to_numpy(img):
    if img is None:
        return None
    if _HAS_TORCH and isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, PILImage.Image):
        img = np.array(img)
    img = np.asarray(img)
    while img.ndim > 2 and img.shape[0] in (1,3) and img.shape[-1] not in (1,3):
        if img.ndim == 3:
            img = np.transpose(img, (1,2,0))
        else:
            break
    if img.dtype != np.uint8:
        img_min = float(img.min()) if img.size else 0.0
        img_max = float(img.max()) if img.size else 1.0
        if img_max <= 1.0 and img_min >= 0.0:
            img = (img * 255.0).clip(0,255).astype(np.uint8)
        else:
            denom = (img_max - img_min) if (img_max - img_min) != 0 else 1.0
            img = ((img - img_min) / denom * 255.0).clip(0,255).astype(np.uint8)
    return img

def _make_boxplots_image(fish_lengths, curvatures, ratios, eye_areas=None, eye_diameters=None):
    def _clean_numeric(vals):
        out = []
        for v in (vals or []):
            if isinstance(v, (int, float)) and np.isfinite(v):
                out.append(float(v))
        return out

    fish_lengths_clean = _clean_numeric(fish_lengths)
    curvatures_clean = _clean_numeric(curvatures)
    ratios_clean = _clean_numeric(ratios)
    eye_areas_clean = _clean_numeric(eye_areas)
    eye_diameters_clean = _clean_numeric(eye_diameters)

    # Count how many plots we need
    num_plots = sum([
        bool(fish_lengths_clean),
        bool(curvatures_clean),
        bool(ratios_clean),
        bool(eye_areas_clean),
        bool(eye_diameters_clean),
    ])
    if num_plots == 0:
        num_plots = 1  # At least one subplot
    
    fig = plt.figure(figsize=(5*num_plots, 5))
    plot_idx = 1
    
    if fish_lengths_clean:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(fish_lengths_clean, vert=True, patch_artist=True)
        plt.title("Fish Lengths"); plt.ylabel("Length (µm)")
        plot_idx += 1
    
    if curvatures_clean:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(curvatures_clean, vert=True, patch_artist=True)
        plt.title("Curvatures"); plt.ylabel("Curvature")
        plot_idx += 1
    
    if ratios_clean:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(ratios_clean, vert=True, patch_artist=True)
        plt.title("Length/Straight Line Ratio"); plt.ylabel("Ratio")
        plot_idx += 1

    if eye_areas_clean:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(eye_areas_clean, vert=True, patch_artist=True)
        plt.title("Eye Areas"); plt.ylabel("Area (µm²)")
        plot_idx += 1

    if eye_diameters_clean:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(eye_diameters_clean, vert=True, patch_artist=True)
        plt.title("Eye Diameters"); plt.ylabel("Diameter (µm)")
    
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close(fig)
    img_bytes.seek(0)
    return img_bytes.getvalue()

def write_lengths_to_excel_bytes(
    filenames,
    fish_lengths,
    curvatures,
    ratios,
    eye_areas,
    eye_diameters,
    threshold_used,
    threshold_value,
    boxplot_png_bytes,
):
    wb = openpyxl.Workbook()
    sh = wb.active
    sh.title = "Fish Data"
    # Build header dynamically based on what data we have
    header = ["Filename"]
    if fish_lengths: header.append("Fish Length (µm)")
    if curvatures: header.append("Curvature")
    if ratios: header.append("Length/Straight Line Ratio")
    if eye_areas: header.append("Eye Area (µm²)")
    if eye_diameters: header.append("Eye Diameter (µm)")
    sh.append(header)
    
    for i, fname in enumerate(filenames):
        row = [fname]
        if fish_lengths:
            L = fish_lengths[i] if i < len(fish_lengths) else "N/A"
            row.append(L)
        if curvatures:
            c = curvatures[i] if i < len(curvatures) else "N/A"
            if c == 5:
                c = "Not Classified"
            row.append(c)
        if ratios:
            r = ratios[i] if i < len(ratios) else "N/A"
            row.append(r)
        if eye_areas:
            ea = eye_areas[i] if i < len(eye_areas) and eye_areas[i] is not None else "N/A"
            row.append(ea)
        if eye_diameters:
            ed = eye_diameters[i] if i < len(eye_diameters) and eye_diameters[i] is not None else "N/A"
            row.append(ed)
        sh.append(row)

    def _stats(vals):
        clean_vals = []
        for v in vals:
            if isinstance(v, (int, float)) and np.isfinite(v):
                clean_vals.append(float(v))
        if not clean_vals: return ("N/A",)*5
        vals_sorted = sorted(clean_vals); n = len(vals_sorted)
        median = vals_sorted[n//2]
        p25 = vals_sorted[int(n*0.25)]
        p75 = vals_sorted[int(n*0.75)]
        mean = sum(vals_sorted)/n
        std = (sum((x-mean)**2 for x in vals_sorted)/n)**0.5
        return median, p25, p75, mean, std

    sh.append([])
    if threshold_used:
        sh.append([f"Threshold used; statistics may be unreliable (threshold: {threshold_value})"])
    sh.append(["Statistics"])
    
    if fish_lengths:
        medL,p25L,p75L,meanL,stdL = _stats(fish_lengths)
        sh.append(["Median Length (µm)", medL]); sh.append(["25th Percentile Length (µm)", p25L])
        sh.append(["75th Percentile Length (µm)", p75L]); sh.append(["Mean Length (µm)", meanL])
        sh.append(["Standard Deviation Length (µm)", stdL])
    
    if curvatures:
        medC,p25C,p75C,meanC,stdC = _stats(curvatures)
        sh.append(["Median Curvature", medC]); sh.append(["25th Percentile Curvature", p25C])
        sh.append(["75th Percentile Curvature", p75C]); sh.append(["Mean Curvature", meanC])
        sh.append(["Standard Deviation Curvature", stdC])
    
    if ratios:
        medR,p25R,p75R,meanR,stdR = _stats(ratios)
        sh.append(["Median Ratio", medR]); sh.append(["25th Percentile Ratio", p25R])
        sh.append(["75th Percentile Ratio", p75R]); sh.append(["Mean Ratio", meanR])
        sh.append(["Standard Deviation Ratio", stdR])

    if eye_areas:
        medEA,p25EA,p75EA,meanEA,stdEA = _stats(eye_areas)
        sh.append(["Median Eye Area (µm²)", medEA]); sh.append(["25th Percentile Eye Area (µm²)", p25EA])
        sh.append(["75th Percentile Eye Area (µm²)", p75EA]); sh.append(["Mean Eye Area (µm²)", meanEA])
        sh.append(["Standard Deviation Eye Area (µm²)", stdEA])

    if eye_diameters:
        medED,p25ED,p75ED,meanED,stdED = _stats(eye_diameters)
        sh.append(["Median Eye Diameter (µm)", medED]); sh.append(["25th Percentile Eye Diameter (µm)", p25ED])
        sh.append(["75th Percentile Eye Diameter (µm)", p75ED]); sh.append(["Mean Eye Diameter (µm)", meanED])
        sh.append(["Standard Deviation Eye Diameter (µm)", stdED])
    sh.append([]); sh.append(["Class Distribution"])
    cls_counts = [0,0,0,0,0]
    for c in curvatures:
        idx = 4 if c == 5 else int(c)-1
        if 0 <= idx < 5:
            cls_counts[idx] += 1
    labels = ["Class 1","Class 2","Class 3","Class 4","Not Classified"]
    for i,lbl in enumerate(labels):
        sh.append([f"{lbl}", cls_counts[i]])

    if boxplot_png_bytes:
        img_stream = io.BytesIO(boxplot_png_bytes)
        img = ExcelImage(img_stream); sh.add_image(img, "E2")

    buf = io.BytesIO(); wb.save(buf); buf.seek(0); return buf

def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    m = _to_numpy(mask).astype(np.float32)
    if m.ndim == 3 and m.shape[-1] == 3: m = m[...,0]
    if m.max() <= 1.0: m = (m > 0.5).astype(np.uint8) * 255
    else: m = (m > 127).astype(np.uint8) * 255
    return m

GALLERY_MASK_ALPHA = 0.45
MANUAL_MASK_ALPHA = 0.15

def _make_seg_overlay(original_img, seg_mask, path_points=None, straight_line_points=None, eye_mask=None, mask_alpha=GALLERY_MASK_ALPHA) -> np.ndarray:
    base = _to_numpy(original_img); mask = _normalize_mask(seg_mask)
    if base.ndim == 2: base = np.stack([base]*3, axis=-1)
    if mask.shape[:2] != base.shape[:2]:
        mask = np.array(PILImage.fromarray(mask).resize((base.shape[1], base.shape[0]), resample=PILImage.NEAREST))
    overlay = base.copy().astype(np.float32)
    # fish mask overlay in yellow
    alpha = float(np.clip(mask_alpha, 0.0, 1.0))
    yellow = np.zeros_like(overlay)
    yellow[..., 0] = 255
    yellow[..., 1] = 255
    m = (mask > 0)[..., None].astype(np.float32)
    overlay = overlay * (1 - alpha * m) + yellow * (alpha * m)
    if eye_mask is not None:
        eye_norm = _normalize_mask(eye_mask)
        if eye_norm.shape[:2] != base.shape[:2]:
            eye_norm = np.array(PILImage.fromarray(eye_norm).resize((base.shape[1], base.shape[0]), resample=PILImage.NEAREST))
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        em = (eye_norm > 0)[..., None].astype(np.float32)
        overlay = overlay * (1 - 0.35 * em) + red * (0.35 * em)

    overlay = overlay.clip(0,255).astype(np.uint8)

    h_mask, w_mask = _normalize_mask(seg_mask).shape[:2]
    h_base, w_base = overlay.shape[:2]
    sy = h_base / float(max(1, h_mask))
    sx = w_base / float(max(1, w_mask))

    if path_points is not None:
        try:
            p = np.asarray(path_points)
            if p.ndim == 2 and p.shape[1] == 2 and len(p) >= 2:
                pts = np.stack([
                    np.clip(np.round(p[:, 1] * sx), 0, w_base - 1),
                    np.clip(np.round(p[:, 0] * sy), 0, h_base - 1),
                ], axis=1).astype(np.int32)
                cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        except Exception:
            pass

    if straight_line_points is not None:
        try:
            (r1, c1), (r2, c2) = straight_line_points
            p1 = (int(np.clip(round(c1 * sx), 0, w_base - 1)), int(np.clip(round(r1 * sy), 0, h_base - 1)))
            p2 = (int(np.clip(round(c2 * sx), 0, w_base - 1)), int(np.clip(round(r2 * sy), 0, h_base - 1)))
            cv2.line(overlay, p1, p2, (255, 0, 255), 2)
        except Exception:
            pass

    return overlay  # Return full resolution image without resizing

def _shorten_name(name: str, max_chars: int = 22) -> str:
    base = os.path.basename(name)
    if len(base) <= max_chars: return base
    root, ext = os.path.splitext(base)
    keep = max_chars - len(ext) - 3
    if keep <= 0: return base[:max(1, max_chars-3)] + '...'
    head = keep // 2; tail = keep - head
    return f"{root[:head]}...{root[-tail:]}{ext}"

def _stage_inputs(files: Optional[List[gr.File]], folder_input) -> Tuple[str, list]:
    """
    Normalize inputs into a working directory with all images inside, and a
    sorted list of filenames (basenames) that match what will be processed.
    - If `folder_input` is a list/tuple of paths (Gradio folder upload), copy ALL
      of them into a temp dir and return that dir + filenames.
    - If `folder_input` is a string path to a directory, enumerate it.
    - Otherwise, fall back to `files` (individual uploads) and copy into a temp dir.
    """
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    # Helper: extract plain file paths from a gradio payload item
    def _get_path(x):
        if isinstance(x, str):
            return x
        # Some gradio versions pass objects with `.name`
        return getattr(x, "name", None)

    # Case 1: Folder upload via list/tuple of paths
    if isinstance(folder_input, (list, tuple)) and len(folder_input) > 0:
        src_paths = []
        for item in folder_input:
            p = _get_path(item)
            if p and os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                src_paths.append(p)
        if src_paths:
            tmpdir = tempfile.mkdtemp()
            basenames = []
            for p in src_paths:
                bn = os.path.basename(p)
                dst = os.path.join(tmpdir, bn)
                # If duplicate basenames (rare but possible), disambiguate
                if os.path.exists(dst):
                    root, ext = os.path.splitext(bn)
                    k = 1
                    while os.path.exists(dst):
                        bn = f"{root}_{k}{ext}"
                        dst = os.path.join(tmpdir, bn)
                        k += 1
                shutil.copy(p, dst)
                basenames.append(bn)
            basenames.sort()
            return tmpdir, basenames

    # Case 2: Folder upload as a single directory path (less common)
    if isinstance(folder_input, str) and os.path.isdir(folder_input):
        names = [n for n in os.listdir(folder_input)
                 if os.path.splitext(n)[1].lower() in exts]
        names.sort()
        return folder_input, names

    # Case 3: Individual files upload (UploadButton)
    tmpdir = tempfile.mkdtemp()
    filenames = []
    if files:
        for f in files:
            p = _get_path(f)
            if p and os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                bn = os.path.basename(p)
                dst = os.path.join(tmpdir, bn)
                if os.path.exists(dst):
                    root, ext = os.path.splitext(bn)
                    k = 1
                    while os.path.exists(dst):
                        bn = f"{root}_{k}{ext}"
                        dst = os.path.join(tmpdir, bn)
                        k += 1
                shutil.copy(p, dst)
                filenames.append(bn)
    filenames.sort()
    return tmpdir, filenames


def _safe_float(s, default=None):
    try:
        if s is None: return default
        if isinstance(s, (int, float)): return float(s)
        s = str(s).strip()
        if not s:
            return default

        # remove common thousands separators/spaces
        s = s.replace("\u00A0", "")  # non-breaking space
        s = s.replace(" ", "")
        s = s.replace("_", "")
        s = s.replace("'", "")

        # Handle locale-specific decimal/thousands separators
        if "," in s and "." in s:
            # Assume the last separator is the decimal separator
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")

        return float(s)
    except Exception:
        return default


def _get_first_image_path(folder_input, files) -> Optional[str]:
    """Return the path to the first image in whichever upload was provided."""
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def _get_path(x):
        if isinstance(x, str):
            return x
        return getattr(x, 'name', None)

    # Folder upload (list of file paths)
    if isinstance(folder_input, (list, tuple)) and len(folder_input) > 0:
        paths = []
        for item in folder_input:
            p = _get_path(item)
            if p and os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                paths.append(p)
        if paths:
            return sorted(paths)[0]

    # Folder upload (single directory path)
    if isinstance(folder_input, str) and os.path.isdir(folder_input):
        names = sorted([n for n in os.listdir(folder_input)
                        if os.path.splitext(n)[1].lower() in exts])
        if names:
            return os.path.join(folder_input, names[0])

    # Individual file upload
    if isinstance(files, (list, tuple)) and len(files) > 0:
        for f in files:
            p = _get_path(f)
            if p and os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                return p

    return None


def _run_scalebar_detection(folder_input, files, bar_label_um_str=""):
    """
    Detect the scale bar line from the first uploaded image and, if the user
    has supplied the physical bar length, compute the full calibration.

    Returns (preview_update, status_md, bar_px_update, phys_w_update, phys_h_update)
    """
    no_img_update = gr.update(visible=False)

    first_path = _get_first_image_path(folder_input, files)
    if first_path is None:
        return (no_img_update,
                "Upload images first, then click **Detect Scale Bar**.",
                gr.update(), gr.update(), gr.update())

    # Load image
    try:
        img_bgr = cv2.imread(first_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil = PILImage.open(first_path).convert('RGB')
            img_rgb = np.array(pil)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return (no_img_update, f"⚠ Could not load image: {e}",
                gr.update(), gr.update(), gr.update())

    if not _HAS_SCALEBAR:
        return (gr.update(value=img_rgb, visible=True),
                "⚠ `scalebar` module could not be imported.",
                gr.update(), gr.update(), gr.update())

    label_um = _safe_float(bar_label_um_str, default=None)
    result = _detect_scalebar(img_rgb, label_um=label_um)
    debug_img = result.get('debug_img') if result.get('debug_img') is not None else img_rgb

    bar_px = result.get('bar_length_px')
    bar_px_str = str(bar_px) if bar_px is not None else ""

    if result['success']:
        phys_w = f"{result['phys_width_um']:.1f}"
        phys_h = f"{result['phys_height_um']:.1f}"
        status = f"✅ {result['message']}"
        return (gr.update(value=debug_img, visible=True),
                status,
                gr.update(value=bar_px_str),
                gr.update(value=phys_w),
                gr.update(value=phys_h))
    elif result['bar_found']:
        status = (
            f"📏 Scale bar line detected: **{bar_px} px**.  "
            f"Enter its physical length in the field below, then click **Apply**."
        )
        return (gr.update(value=debug_img, visible=True),
                status,
                gr.update(value=bar_px_str),
                gr.update(), gr.update())
    else:
        status = f"⚠ **Detection failed:** {result['message']}"
        return (gr.update(value=debug_img, visible=True),
                status,
                gr.update(value=""),
                gr.update(), gr.update())


def process(folder,
            files: Optional[List[gr.File]],
            process_curvature=True,
            process_length=True,
            process_ratio=True,
            process_eye_size=True,
            use_threshold=False,
            threshold_value=0.5,
            physical_horizontal_um_str="",
            physical_vertical_um_str=""):
    work_dir, filenames = _stage_inputs(files, folder)
    # Always load eyes for overlay visualization
    original_images, segmented_images, grown_images, eyes_images = segmentation_pipeline(work_dir, include_eyes=True)
    model = _ensure_model()

    # Parse physical distances (µm) for full image width/height from user
    phys_w_um_user = _safe_float(physical_horizontal_um_str, default=None)
    phys_h_um_user = _safe_float(physical_vertical_um_str, default=None)

    if phys_w_um_user is not None and phys_h_um_user is not None:
        y_scale_info = phys_h_um_user / 256
        x_scale_info = phys_w_um_user / 256
        spacing_info_md = (
            f"**Spacing used:** custom input | "
            f"y = {y_scale_info:.4f} µm/pixel, x = {x_scale_info:.4f} µm/pixel "
            f"(from H={phys_h_um_user:g} µm, W={phys_w_um_user:g} µm over 256 px)"
        )
    else:
        y_scale_info = 5885.0 / 256
        x_scale_info = 5885.0 / 256
        spacing_info_md = (
            f"**Spacing used:** default calibration | "
            f"y = {y_scale_info:.4f} µm/pixel, x = {x_scale_info:.4f} µm/pixel "
            f"(H=W=5885 µm over 256 px)"
        )

    fish_lengths, curvatures, ratios, eye_areas, eye_diameters, previews = [], [], [], [], [], []
    for i, seg_mask in enumerate(segmented_images):
        path_points = None
        straight_line_points = None
        eye_mask_for_vis = eyes_images[i] if i < len(eyes_images) else None
        seg_mask_bin = seg_mask > 0

        # Per-image pixel scales derived from user-provided physical distances
        h, w = seg_mask.shape[:2]
        # Default to pixel units if user did not provide values
        if phys_w_um_user is not None and phys_h_um_user is not None:
            phys_w_um = phys_w_um_user
            phys_h_um = phys_h_um_user
            # Calculate spacing for the new function: (dy, dx) in physical units per pixel
            y_scale = phys_h_um / 256  # physical units per pixel in y direction
            x_scale = phys_w_um / 256  # physical units per pixel in x direction
        else:
            # Default spacing (assuming 5885 µm per 256 pixels as per the original code)
            y_scale = 5885.0 / 256
            x_scale = 5885.0 / 256
            phys_w_um = 5885.0
            phys_h_um = 5885.0

        if process_length:
            # Use the new tube_length_border2border function
            try:
                eye_mask_for_length = (eye_mask_for_vis > 0) if eye_mask_for_vis is not None else None
                spacing = (y_scale, x_scale)
                print(f"spacing:{spacing}")
                # Use eye mask when available to stabilize head-side start point.
                length, straight_length, path_points, straight_line_points = tube_length_border2border(
                    seg_mask_bin,
                    spacing=spacing,
                    return_path=True,
                    return_straight_line=True,
                    mask_eye=eye_mask_for_length,
                    return_eye_info=False,
                )
                fish_lengths.append(float(length))
                # Calculate ratio only if checkbox is enabled
                if process_ratio:
                    # Calculate ratio, avoiding division by zero
                    if straight_length > 0:
                        ratio = float(length) / float(straight_length)
                    else:
                        ratio = 0.0
                    ratios.append(ratio)
            except Exception as e:
                print(f"Error calculating length for image {i}: {e}")
                pass

        if process_eye_size:
            try:
                eye_mask_for_metrics = (eye_mask_for_vis > 0) if eye_mask_for_vis is not None else None
                eye_info = compute_eye_metrics(
                    eye_mask_for_metrics,
                    mask_fish=seg_mask_bin,
                    spacing=(y_scale, x_scale),
                )
                eye_areas.append(float(eye_info.get("eye_area", 0.0)))
                eye_diameters.append(float(eye_info.get("eye_diameter", 0.0)))
            except Exception as e:
                print(f"Error calculating eye metrics for image {i}: {e}")
                eye_areas.append(None)
                eye_diameters.append(None)

        if process_curvature:
            try:
                _, curv = classification_curvature(original_images[i], grown_images[i], model, use_threshold, threshold_value)
                curvatures.append(int(curv.item()))
            except Exception:
                pass

        try:
            overlay = _make_seg_overlay(
                original_images[i],
                seg_mask,
                path_points=path_points,
                straight_line_points=straight_line_points,
                eye_mask=eye_mask_for_vis,
            )
            original_name = filenames[i] if i < len(filenames) else f"image_{i}"
            short = _shorten_name(original_name, max_chars=22)
            # embed index into caption so selection handlers can identify images robustly
            cap = f"{i}:{short}"
            previews.append([overlay, cap])
        except Exception:
            pass

    boxplot_png = _make_boxplots_image(fish_lengths, curvatures, ratios, eye_areas, eye_diameters)
    boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))
    out_bytes = write_lengths_to_excel_bytes(
        filenames,
        fish_lengths,
        curvatures,
        ratios,
        eye_areas,
        eye_diameters,
        use_threshold,
        threshold_value,
        boxplot_png,
    )
    tmpout = tempfile.mkdtemp(); out_xlsx = os.path.join(tmpout, "fish_data.xlsx")
    with open(out_xlsx, "wb") as f: f.write(out_bytes.getvalue())
    # Prepare state for interactive filtering
    # Keep a copy of the original previews so crosses can be added/removed reversibly
    original_previews = []
    for img, cap in previews:
        try:
            original_previews.append([img.copy(), cap])
        except Exception:
            original_previews.append([img, cap])

    data_state = {
        'filenames': filenames,
        'fish_lengths': fish_lengths,
        'curvatures': curvatures,
        'ratios': ratios,
        'eye_areas': eye_areas,
        'eye_diameters': eye_diameters,
        'boxplot_png': boxplot_png,
        'threshold_used': use_threshold,
        'threshold_value': threshold_value,
        'previews': previews,
        'original_previews': original_previews,
        'original_images': original_images,
        'segmented_images': segmented_images,
        'eyes_images': eyes_images,
        'spacing': (y_scale, x_scale),
        'manual_points': {},
    }
    shown_names = [_shorten_name(n, max_chars=22) for n in filenames[:5]]
    more_note = f" … and {len(filenames) - 5} more" if len(filenames) > 5 else ""
    filenames_md = "**Uploaded:** " + ", ".join(shown_names) + more_note
    
    return out_xlsx, boxplot_np, previews, filenames_md, data_state, spacing_info_md

def summarize_files(files):
    if not files: return "No files uploaded."
    names = [os.path.basename(f.name) for f in files[:5]]
    short = [_shorten_name(n, max_chars=22) for n in names]
    more = f" … and {len(files) - 5} more" if len(files) > 5 else ""
    return "**Uploaded:** " + ", ".join(short) + more


def _generate_corrected_excel(data):
    """Generate a fresh Excel export from the current in-memory results, including manual corrections."""
    if not data:
        return None

    out_bytes = write_lengths_to_excel_bytes(
        data.get('filenames', []),
        data.get('fish_lengths', []),
        data.get('curvatures', []),
        data.get('ratios', []),
        data.get('eye_areas', []),
        data.get('eye_diameters', []),
        data.get('threshold_used', False),
        data.get('threshold_value', 0.0),
        data.get('boxplot_png', None),
    )
    tmpout = tempfile.mkdtemp()
    out_xlsx = os.path.join(tmpout, "fish_data_corrected.xlsx")
    with open(out_xlsx, "wb") as f:
        f.write(out_bytes.getvalue())
    return out_xlsx



def _compute_manual_length(seg_mask, point1, point2, spacing):
    """
    Compute length from manually selected points using a smooth path through the center of the fish.
    Ensures the path always stays inside the segmented region.
    point1, point2: (row, col) tuples in mask coordinates
    spacing: (dy, dx) physical units per pixel
    """
    try:
        seg_mask_bin = seg_mask > 0
        dy, dx = spacing
        
        # Convert points to numpy arrays
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)
        
        from scipy.ndimage import distance_transform_edt
        from skimage.graph import route_through_array
        from scipy.ndimage import gaussian_filter
        
        # Compute distance transform - distance from each pixel to nearest background
        dist_transform = distance_transform_edt(seg_mask_bin)
        
        if dist_transform.max() == 0:
            # Fallback: straight line
            diff = p2 - p1
            straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
            path = np.array([p1, p2], dtype=int)
            return straight_length, straight_length, path, (tuple(p1.astype(int)), tuple(p2.astype(int)))
        
        # Create cost map: lower cost in the center (high distance), higher cost at edges
        # Invert distance: paths prefer to go through the thickest/central parts
        max_dist = dist_transform.max()
        cost_map = np.where(seg_mask_bin, max_dist - dist_transform + 0.1, 1e10)
        
        # Smooth the cost map to encourage smooth paths
        cost_map = gaussian_filter(cost_map, sigma=2.0)
        
        # Ensure manual points are inside the mask
        p1_int = np.clip(np.round(p1).astype(int), [0, 0], [seg_mask_bin.shape[0]-1, seg_mask_bin.shape[1]-1])
        p2_int = np.clip(np.round(p2).astype(int), [0, 0], [seg_mask_bin.shape[0]-1, seg_mask_bin.shape[1]-1])
        
        # If points are outside, find nearest inside point
        if not seg_mask_bin[p1_int[0], p1_int[1]]:
            mask_coords = np.argwhere(seg_mask_bin)
            if len(mask_coords) > 0:
                from scipy.spatial.distance import cdist
                dist_to_p1 = cdist([p1], mask_coords)[0]
                p1_int = mask_coords[np.argmin(dist_to_p1)]
        
        if not seg_mask_bin[p2_int[0], p2_int[1]]:
            mask_coords = np.argwhere(seg_mask_bin)
            if len(mask_coords) > 0:
                from scipy.spatial.distance import cdist
                dist_to_p2 = cdist([p2], mask_coords)[0]
                p2_int = mask_coords[np.argmin(dist_to_p2)]
        
        # Find path with minimum cost through the center
        try:
            indices, weight = route_through_array(
                cost_map, 
                start=tuple(p1_int), 
                end=tuple(p2_int),
                fully_connected=True,
                geometric=True
            )
            path = np.array(indices, dtype=int)
        except Exception as e:
            print(f"Route finding failed: {e}, using straight line")
            # Fallback: interpolate straight line
            n_points = int(np.ceil(np.linalg.norm(p2_int - p1_int))) + 1
            t = np.linspace(0, 1, n_points)
            path = p1_int[None, :] * (1 - t[:, None]) + p2_int[None, :] * t[:, None]
            path = np.round(path).astype(int)
        
        if len(path) < 2:
            # Fallback: straight line
            diff = p2 - p1
            straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
            path = np.array([p1, p2], dtype=int)
            return straight_length, straight_length, path, (tuple(p1.astype(int)), tuple(p2.astype(int)))
        
        # Apply smoothing while keeping points inside the mask
        def smooth_path_constrained(path, mask, dist_map, iterations=8):
            """
            Smooth path while ensuring all points stay inside the mask.
            Uses distance transform to weight smoothing - more in thick regions.
            """
            if len(path) < 5:
                return path
            
            path_smooth = path.astype(float).copy()
            n = len(path)
            
            for iteration in range(iterations):
                prev = path_smooth.copy()
                
                for i in range(1, n - 1):  # Don't smooth endpoints
                    # Weighted average with neighbors
                    window = 7
                    half_w = window // 2
                    start_idx = max(0, i - half_w)
                    end_idx = min(n, i + half_w + 1)
                    
                    # Gaussian weights
                    indices = np.arange(start_idx, end_idx)
                    weights = np.exp(-0.5 * ((indices - i) / 2.5) ** 2)
                    weights /= weights.sum()
                    
                    # Smooth
                    local_points = prev[start_idx:end_idx]
                    smoothed = (weights[:, None] * local_points).sum(axis=0)
                    
                    # High smoothing factor
                    alpha = 0.75
                    path_smooth[i] = alpha * smoothed + (1 - alpha) * prev[i]
                
                # Project points back to mask if they went outside
                for i in range(1, n - 1):
                    pi = np.round(path_smooth[i]).astype(int)
                    pi = np.clip(pi, [0, 0], [mask.shape[0]-1, mask.shape[1]-1])
                    
                    # If point is outside mask, find nearest valid point
                    if not mask[pi[0], pi[1]]:
                        # Search in small neighborhood for nearest valid point
                        search_radius = 5
                        found = False
                        for r in range(1, search_radius + 1):
                            y_min, y_max = max(0, pi[0]-r), min(mask.shape[0], pi[0]+r+1)
                            x_min, x_max = max(0, pi[1]-r), min(mask.shape[1], pi[1]+r+1)
                            local_mask = mask[y_min:y_max, x_min:x_max]
                            
                            if local_mask.any():
                                local_coords = np.argwhere(local_mask)
                                local_coords[:, 0] += y_min
                                local_coords[:, 1] += x_min
                                
                                # Find nearest valid point
                                dists = np.sum((local_coords - path_smooth[i]) ** 2, axis=1)
                                nearest = local_coords[np.argmin(dists)]
                                path_smooth[i] = nearest.astype(float)
                                found = True
                                break
                        
                        if not found:
                            # Keep previous valid position
                            path_smooth[i] = prev[i]
            
            # Final round to integers and clipping
            path_smooth = np.round(path_smooth).astype(int)
            path_smooth[:, 0] = np.clip(path_smooth[:, 0], 0, mask.shape[0] - 1)
            path_smooth[:, 1] = np.clip(path_smooth[:, 1], 0, mask.shape[1] - 1)
            
            return path_smooth
        
        # Apply constrained smoothing
        path = smooth_path_constrained(path, seg_mask_bin, dist_transform, iterations=10)
        
        # Remove duplicate consecutive points
        if len(path) >= 2:
            mask_diff = np.any(np.diff(path, axis=0) != 0, axis=1)
            keep_indices = np.concatenate([[True], mask_diff])
            path = path[keep_indices]
        
        # Compute length along path
        pf = path.astype(float)
        dxy = np.diff(pf, axis=0)
        seg = np.sqrt((dxy[:, 0] * dy) ** 2 + (dxy[:, 1] * dx) ** 2)
        length = float(seg.sum())
        
        # Compute straight-line distance
        diff = p2 - p1
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
        
        straight_line_points = (tuple(path[0]), tuple(path[-1]))
        
        return length, straight_length, path, straight_line_points
        
    except Exception as e:
        print(f"Error in manual length computation: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: straight line between points
        diff = p2 - p1
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
        path = np.array([p1, p2], dtype=int)
        return straight_length, straight_length, path, (tuple(p1.astype(int)), tuple(p2.astype(int)))
        seg = np.sqrt((dxy[:, 0] * dy) ** 2 + (dxy[:, 1] * dx) ** 2)
        length = float(seg.sum())
        
        # Compute straight-line distance
        diff = p2 - p1
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
        
        straight_line_points = (tuple(path[0]), tuple(path[-1]))
        
        return length, straight_length, path, straight_line_points
        
    except Exception as e:
        print(f"Error in manual length computation: {e}")
        # Fallback: straight line between points
        diff = p2 - p1
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
        path = np.array([p1, p2], dtype=int)
        return straight_length, straight_length, path, (tuple(p1.astype(int)), tuple(p2.astype(int)))
def _enter_manual_mode(evt: gr.SelectData, data):
    """Enter manual editing mode for selected image"""
    if data is None:
        return None, -1, "No data available", gr.update(visible=False)
    
    idx = evt.index
    if idx < 0 or idx >= len(data.get('original_images', [])):
        return None, -1, "Invalid image selection", gr.update(visible=False)
    
    # Get the original image for display
    original_img = data['original_images'][idx]
    seg_mask = data['segmented_images'][idx]
    
    # Create a composite showing original + segmentation overlay
    display_img = _make_seg_overlay(
        original_img,
        seg_mask,
        path_points=None,
        straight_line_points=None,
        mask_alpha=MANUAL_MASK_ALPHA,
    )
    
    filename = data['filenames'][idx] if idx < len(data['filenames']) else f"Image {idx}"
    instructions = f"**Editing: {filename}**\n\nClick on the image to set points:\n1. First click = HEAD (start point)\n2. Second click = TAIL (end point)\n\nAfter setting both points, click 'Apply Manual Points' to recalculate length."
    
    return display_img, idx, instructions, gr.update(visible=True)


def _record_manual_click(evt: gr.SelectData, current_img, edit_idx, manual_points_temp):
    """Record a click on the image for manual point selection"""
    if current_img is None or edit_idx < 0:
        return manual_points_temp, current_img, "Please select an image from the gallery first"
    
    # Get click coordinates from Gradio SelectData
    # For Image component, evt.index gives (x, y) coordinates
    if hasattr(evt, 'index') and evt.index is not None:
        if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
            click_x, click_y = int(evt.index[0]), int(evt.index[1])
        else:
            return manual_points_temp, current_img, "Invalid click coordinates"
    else:
        return manual_points_temp, current_img, "No click coordinates received"
    
    # Initialize manual points storage
    if manual_points_temp is None:
        manual_points_temp = {}
    
    if edit_idx not in manual_points_temp:
        manual_points_temp[edit_idx] = []
    
    points_list = manual_points_temp[edit_idx]
    
    # Add the new point (store as row, col)
    if len(points_list) < 2:
        points_list.append((click_y, click_x))  # Store as (row, col)
        manual_points_temp[edit_idx] = points_list
        
        # Draw the points on the image
        img_with_points = current_img.copy()
        
        # Draw existing points
        for i, (py, px) in enumerate(points_list):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for head, red for tail
            # Draw filled circle
            cv2.circle(img_with_points, (int(px), int(py)), 8, color, -1)
            # Draw white border
            cv2.circle(img_with_points, (int(px), int(py)), 10, (255, 255, 255), 2)
            # Add label
            label = "HEAD" if i == 0 else "TAIL"
            cv2.putText(img_with_points, label, (int(px) + 15, int(py) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if len(points_list) == 1:
            status = "✓ HEAD point set (green). Now click to set TAIL point (will be red)."
        else:
            status = "✓ Both points set! Click 'Apply Manual Points' to recalculate length."
        
        return manual_points_temp, img_with_points, status
    else:
        return manual_points_temp, current_img, "⚠ Both points already set. Click 'Reset Points' to start over, or 'Apply Manual Points' to use these."


def _reset_manual_points(edit_idx, manual_points_temp, data):
    """Reset manual points for current image"""
    if manual_points_temp and edit_idx in manual_points_temp:
        del manual_points_temp[edit_idx]
    
    if data and edit_idx >= 0 and edit_idx < len(data.get('original_images', [])):
        original_img = data['original_images'][edit_idx]
        seg_mask = data['segmented_images'][edit_idx]
        display_img = _make_seg_overlay(
            original_img,
            seg_mask,
            path_points=None,
            straight_line_points=None,
            mask_alpha=MANUAL_MASK_ALPHA,
        )
        return manual_points_temp, display_img, "Points reset. Click to set HEAD point."
    
    return manual_points_temp, None, "No image selected"


def _apply_manual_points(edit_idx, manual_points_temp, data):
    """Apply manual points and recalculate length for the selected image"""
    if data is None or edit_idx < 0:
        return data, gr.update(), gr.update(), "No data or image selected", gr.update(), gr.update()
    
    if manual_points_temp is None or edit_idx not in manual_points_temp:
        return data, gr.update(), gr.update(), "No manual points set for this image", gr.update(), gr.update()
    
    points_list = manual_points_temp[edit_idx]
    if len(points_list) != 2:
        return data, gr.update(), gr.update(), "Need exactly 2 points (head and tail)", gr.update(), gr.update()
    
    # Get the image data
    seg_mask = data['segmented_images'][edit_idx]
    h_seg, w_seg = seg_mask.shape[:2]
    
    # Convert click coordinates to mask coordinates (256x256)
    # Points are stored as (row, col) in display space
    # Need to scale to mask space
    point1_display = points_list[0]  # (row, col) in display
    point2_display = points_list[1]
    
    # Get display image size (now full resolution)
    original_img = data['original_images'][edit_idx]
    display_overlay = _make_seg_overlay(original_img, seg_mask, mask_alpha=MANUAL_MASK_ALPHA)
    h_display, w_display = display_overlay.shape[:2]
    
    # Scale points from display to mask coordinates
    scale_y = h_seg / h_display
    scale_x = w_seg / w_display
    
    point1_mask = (int(point1_display[0] * scale_y), int(point1_display[1] * scale_x))
    point2_mask = (int(point2_display[0] * scale_y), int(point2_display[1] * scale_x))
    
    # Ensure points are within bounds
    point1_mask = (np.clip(point1_mask[0], 0, h_seg-1), np.clip(point1_mask[1], 0, w_seg-1))
    point2_mask = (np.clip(point2_mask[0], 0, h_seg-1), np.clip(point2_mask[1], 0, w_seg-1))
    
    # Get spacing from data
    spacing = data.get('spacing', (5885.0/256, 5885.0/256))
    
    # Recalculate length with manual points
    try:
        length, straight_length, path, straight_line_points = _compute_manual_length(
            seg_mask, point1_mask, point2_mask, spacing
        )
        
        # Update data
        if 'fish_lengths' in data and edit_idx < len(data['fish_lengths']):
            data['fish_lengths'][edit_idx] = length
        
        if 'ratios' in data and edit_idx < len(data['ratios']):
            if straight_length > 0:
                data['ratios'][edit_idx] = length / straight_length
            else:
                data['ratios'][edit_idx] = 0.0
        
        # Store manual points in data for persistence
        if 'manual_points' not in data:
            data['manual_points'] = {}
        data['manual_points'][edit_idx] = (point1_mask, point2_mask)
        
        # Regenerate preview for this image
        eye_mask = data.get('eyes_images', [None]*(edit_idx+1))[edit_idx] if edit_idx < len(data.get('eyes_images', [])) else None
        new_overlay_gallery = _make_seg_overlay(
            original_img,
            seg_mask,
            path_points=path,
            straight_line_points=straight_line_points,
            eye_mask=eye_mask,
            mask_alpha=GALLERY_MASK_ALPHA,
        )

        new_overlay_manual = _make_seg_overlay(
            original_img,
            seg_mask,
            path_points=path,
            straight_line_points=straight_line_points,
            eye_mask=eye_mask,
            mask_alpha=MANUAL_MASK_ALPHA,
        )
        
        # Update the specific preview
        if 'original_previews' in data and edit_idx < len(data['original_previews']):
            original_name = data['filenames'][edit_idx] if edit_idx < len(data['filenames']) else f"image_{edit_idx}"
            short = _shorten_name(original_name, max_chars=22)
            cap = f"{edit_idx}:{short} (manual)"
            data['original_previews'][edit_idx] = [new_overlay_gallery, cap]
        
        # Rebuild all previews with updated one
        previews = []
        originals = data.get('original_previews', data.get('previews', []))
        for i, (orig_img, cap) in enumerate(originals):
            short_cap = cap
            if isinstance(short_cap, str) and ':' in short_cap:
                parts = short_cap.split(':', 1)
                if len(parts) > 1:
                    short_cap = parts[1]
            
            previews.append([orig_img, f"{i}:{short_cap}"])
        
        data['previews'] = previews
        
        # Regenerate boxplot with updated data
        boxplot_png = _make_boxplots_image(
            data.get('fish_lengths', []),
            data.get('curvatures', []),
            data.get('ratios', []),
            data.get('eye_areas', []),
            data.get('eye_diameters', []),
        )
        boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))
        data['boxplot_png'] = boxplot_png
        
        status = f"✓ Manual points applied! Length: {length:.2f} µm, Straight: {straight_length:.2f} µm, Ratio: {length/straight_length:.3f}"

        # Return updated overlay to manual edit window (user can see lines before accordion closes)
        return data, previews, boxplot_np, status, gr.update(), new_overlay_manual
        
    except Exception as e:
        return data, gr.update(), gr.update(), f"Error applying manual points: {str(e)}", gr.update(), gr.update()

with gr.Blocks() as demo:
    gr.Markdown("# Zebrafish Analyzer")
    gr.Markdown("""
    📖 **Documentation:** For detailed instructions and usage examples, please visit the [GitHub repository](https://github.com/MarkDanielArndt/Zebrafish_webapp).
    
    This webapp is provided freely to the research community. If you find it useful, please consider giving the repository a ⭐ (it's free!). 
    If you use this tool in your research, please cite: *[Paper - soon to be published]*.
    """
    )
    # Left: folder upload + compact upload button
    with gr.Row():
        folder = gr.File(label="Upload a folder (preferred)", file_count="directory", type="filepath")
        with gr.Column(scale=1):
            upload_btn = gr.UploadButton("Upload individual images", file_types=["image"], file_count="multiple")
            files_summary = gr.Markdown("No files uploaded yet.")

    # A hidden state to keep the uploaded files
    files_state = gr.State([])

    # When user uploads via button, store them and update the compact summary
    _upload_event = upload_btn.upload(
        fn=lambda f: (f, summarize_files(f)),
        inputs=upload_btn,
        outputs=[files_state, files_summary])

    # Hidden states for interactive results (populated after Run)
    data_state = gr.State(None)
    manual_points_temp = gr.State({})
    edit_image_idx = gr.State(-1)

    # --- Scale bar auto-detection ---
    with gr.Accordion("📏 Scale Bar Calibration", open=True):
        gr.Markdown(
            "Automatically detects the scale bar line in the **first uploaded image** "
            "and shows how many pixels it spans.  "
            "Enter the physical length printed above the bar (e.g. `500`) and its unit, "
            "then click **Apply** to compute the µm/px calibration. "
            "The image width/height fields below will be filled automatically. "
            "You can also skip this and enter the distances manually."
        )
        with gr.Row():
            detect_scalebar_btn = gr.Button(
                "🔍 Detect Scale Bar from First Image", variant="secondary")
        scalebar_preview = gr.Image(
            label="First image – detected scale bar highlighted in green",
            type="numpy", visible=False)
        scalebar_status_md = gr.Markdown(
            "Upload images, then click **Detect Scale Bar**.")
        with gr.Row():
            bar_px_display = gr.Textbox(
                label="Detected bar length (px) – read-only",
                interactive=False, placeholder="—")
            bar_label_um_input = gr.Textbox(
                label="Physical length of scale bar (µm)",
                placeholder="e.g. 500")
        with gr.Row():
            apply_scalebar_btn = gr.Button("Apply", variant="primary")

    with gr.Row():
        chk_curv = gr.Checkbox(value=True, label="Process Curvature")
        chk_len  = gr.Checkbox(value=True, label="Process Length")
        chk_ratio = gr.Checkbox(value=True, label="Process Length/Straight Line Ratio")
        chk_eye = gr.Checkbox(value=True, label="Process Eye Size")
        chk_thr  = gr.Checkbox(value=False, label="Use Threshold", visible=False)
        thr_val  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Threshold Value", visible=False)

    # --- Physical distance inputs – auto-filled by scale bar detection, or enter manually ---
    with gr.Row():
        phys_w_um = gr.Textbox(
            label="Physical horizontal distance (µm) – auto-filled or enter manually",
            placeholder="e.g. 5885 (DKFZ E041)")
        phys_h_um = gr.Textbox(
            label="Physical vertical distance (µm) – auto-filled or enter manually",
            placeholder="e.g. 5885 (DKFZ E041)")

    spacing_used_md = gr.Markdown("**Spacing used:** not calculated yet. Click Run.")

    run = gr.Button("Run")

    with gr.Row():
        out_file = gr.File(label="Download results (.xlsx)")

    with gr.Accordion("Results previews", open=True):
        with gr.Row():
            out_box = gr.Image(label="Box plots", type="numpy")
        with gr.Row():
            gallery = gr.Gallery(label="Segmentations (click to select for manual editing)", columns=5, height="auto", object_fit="contain")
        
        # Manual point editing section
        with gr.Accordion("🔧 Manual Point Adjustment", open=False) as manual_edit_accordion:
            gr.Markdown("""
            **Use this tool to manually set head and tail points when automatic detection fails.**
            
            1. Click an image in the gallery above to select it for manual editing
            2. Click on the large image below to set HEAD (green) and TAIL (red) points
            3. Click 'Apply Manual Points' to recalculate the length
            
            **Note:** To exclude images from results, use the "Exclude images" checkboxes below.
            """)
            manual_edit_instructions = gr.Markdown("Select an image from the gallery above to begin manual editing.")
            
            with gr.Row():
                manual_edit_image = gr.Image(label="Click to set points: HEAD (1st click) → TAIL (2nd click)", type="numpy", interactive=False)
            
            manual_status = gr.Markdown("")
            
            with gr.Row():
                reset_points_btn = gr.Button("Reset Points", variant="secondary")
                apply_manual_btn = gr.Button("Apply Manual Points", variant="primary")
        
        filenames_list = gr.Markdown("")
        
        with gr.Accordion("📄 Corrected Excel Export", open=False):
            gr.Markdown("""
            Create a fresh Excel export from the current results in memory.

            If you adjusted manual points, this export will include the updated length and ratio values.
            """)

            with gr.Row():
                gen_corrected_btn = gr.Button("Generate Corrected Excel", variant="primary")

            with gr.Row():
                out_file_corrected = gr.File(label="Download corrected results (.xlsx)")

    # Use files from state, not a giant Files list
    run.click(
        fn=process,
        inputs=[folder, files_state, chk_curv, chk_len, chk_ratio, chk_eye, chk_thr, thr_val, phys_w_um, phys_h_um],
        outputs=[out_file, out_box, gallery, filenames_list, data_state, spacing_used_md]
    )

    # --- Scale bar detection event wiring ---
    _scalebar_outputs = [scalebar_preview, scalebar_status_md, bar_px_display, phys_w_um, phys_h_um]
    _scalebar_inputs_detect = [folder, files_state]          # no label yet on initial detect
    _scalebar_inputs_apply  = [folder, files_state, bar_label_um_input]

    # Detect button – find the bar, show px length (no label yet)
    detect_scalebar_btn.click(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_detect,
        outputs=_scalebar_outputs,
    )

    # Apply button – re-run detection with the user-supplied label
    apply_scalebar_btn.click(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_apply,
        outputs=_scalebar_outputs,
    )

    # Auto-trigger detect (no label) when folder upload changes
    folder.change(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_detect,
        outputs=_scalebar_outputs,
    )

    # Auto-trigger detect after individual file upload (chain after state update)
    _upload_event.then(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_detect,
        outputs=_scalebar_outputs,
    )

    # Gallery click handler - only prepares for manual editing
    def _on_gallery_click(evt: gr.SelectData, data):
        """Handle gallery click: prepare for manual editing"""
        # Prepare manual editing view
        if data is None:
            manual_img = None
            manual_idx = -1
            manual_instr = "No data available"
        else:
            idx = evt.index
            if idx < 0 or idx >= len(data.get('original_images', [])):
                manual_img = None
                manual_idx = -1
                manual_instr = "Invalid image selection"
            else:
                original_img = data['original_images'][idx]
                seg_mask = data['segmented_images'][idx]
                manual_img = _make_seg_overlay(
                    original_img,
                    seg_mask,
                    path_points=None,
                    straight_line_points=None,
                    mask_alpha=MANUAL_MASK_ALPHA,
                )
                manual_idx = idx
                filename = data['filenames'][idx] if idx < len(data['filenames']) else f"Image {idx}"
                manual_instr = f"**Selected: {filename}**\n\nClick on the image below to set points:\n- **First click** = HEAD (start point) - shown in GREEN\n- **Second click** = TAIL (end point) - shown in RED\n\nAfter setting both points, click 'Apply Manual Points' to recalculate length."
        
        return manual_img, manual_idx, manual_instr
    
    # When a gallery image is clicked, prepare for manual editing
    gallery.select(
        fn=_on_gallery_click,
        inputs=[data_state],
        outputs=[manual_edit_image, edit_image_idx, manual_edit_instructions]
    )

    gen_corrected_btn.click(
        fn=_generate_corrected_excel,
        inputs=[data_state],
        outputs=[out_file_corrected]
    )
    
    # When manual edit image is clicked, record the point
    manual_edit_image.select(
        fn=_record_manual_click,
        inputs=[manual_edit_image, edit_image_idx, manual_points_temp],
        outputs=[manual_points_temp, manual_edit_image, manual_status]
    )
    
    # Reset points button
    reset_points_btn.click(
        fn=_reset_manual_points,
        inputs=[edit_image_idx, manual_points_temp, data_state],
        outputs=[manual_points_temp, manual_edit_image, manual_status]
    )
    
    # Apply manual points button
    apply_manual_btn.click(
        fn=_apply_manual_points,
        inputs=[edit_image_idx, manual_points_temp, data_state],
        outputs=[data_state, gallery, out_box, manual_status, manual_edit_accordion, manual_edit_image]
    )

if __name__ == "__main__":
    demo.launch(share=True)

