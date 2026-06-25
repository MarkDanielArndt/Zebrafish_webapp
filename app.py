import gradio as gr
import tempfile, os, shutil
from typing import List, Optional, Tuple
from seg import segmentation_pipeline
from length import load_model, classification_curvature, tube_length_border2border, compute_eye_metrics, compute_eye_diameters
import openpyxl, io
from openpyxl.drawing.image import Image as ExcelImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.graph import route_through_array
import time

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from scalebar import (detect_scalebar as _detect_scalebar,
                          draw_scalebar_endpoints as _draw_scalebar_endpoints,
                          calibrate_from_endpoints as _calibrate_from_endpoints)
    _HAS_SCALEBAR = True
except Exception:
    _HAS_SCALEBAR = False

MODEL_CACHE = {}  # lazy-loaded cache keyed by model filename

# Registry of available segmentation models
# Each entry: display name -> (body_hf_filename, body_encoder_name, eye_hf_filename, target_size, edema_hf_filename)
# None for eye/edema filenames means use the pipeline default
SEG_MODEL_OPTIONS = {
    "Fast & Easy (256 px, ~2s/image)": ("best_model_body_3400_vgg19.pth", "vgg19", None, 256, None),
    "Complex & Slower (512 px, ~7s/image)": ("best_model_body_512.pth", "vgg19", "best_model_eye_512.pth", 512, None),
    "Fine-tuned DESY": ("desy_body_512_finetuned.pth", "vgg19", "desy_eye_512_finetuned.pth", 512, "desy_edema_512_finetuned.pth"),
}

def _ensure_model():
    global MODEL_CACHE
    key = "classification"
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = load_model()
    return MODEL_CACHE[key]

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

def _make_boxplots_image(fish_lengths, curvatures, ratios, eye_areas=None, edema_areas=None):
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
    edema_areas_clean = _clean_numeric(edema_areas)

    # Count how many plots we need
    num_plots = sum([
        bool(fish_lengths_clean),
        bool(curvatures_clean),
        bool(ratios_clean),
        bool(eye_areas_clean),
        bool(edema_areas_clean),
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

    if edema_areas_clean:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(edema_areas_clean, vert=True, patch_artist=True)
        plt.title("Edema Areas"); plt.ylabel("Area (µm²)")
    
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close(fig)
    img_bytes.seek(0)
    return img_bytes.getvalue()

_EXCEL_FORBIDDEN = str.maketrans('', '', r'/\?*[]:'+"'")
_EXCEL_MAX_SHEET_NAME = 31
_FILENAME_FORBIDDEN = str.maketrans('', '', r'/\?*[]:<>|"')

def _sanitize_sheet_name(name: str, default: str = "Fish Data") -> str:
    name = (name or "").strip().translate(_EXCEL_FORBIDDEN)
    return name[:_EXCEL_MAX_SHEET_NAME] if name else default

def _sanitize_filename(name: str, default: str = "Fish Data") -> str:
    name = (name or "").strip().translate(_FILENAME_FORBIDDEN)
    return name if name else default

def write_lengths_to_excel_bytes(
    filenames,
    fish_lengths,
    curvatures,
    ratios,
    eye_areas,
    edema_areas,
    threshold_used,
    threshold_value,
    boxplot_png_bytes,
    sheet_name: str = "Fish Data",
    exclusions=None,
    eye_widths=None,
    eye_heights=None,
):
    EXCLUDED = "Excluded"
    exclusions = exclusions or {}

    def _is_included(idx, metric):
        return exclusions.get(idx, {}).get(metric, True)

    wb = openpyxl.Workbook()
    sh = wb.active
    sh.title = _sanitize_sheet_name(sheet_name)

    header = ["Filename"]
    if fish_lengths: header.append("Fish Length (µm)")
    if curvatures: header.append("Curvature")
    if ratios: header.append("Length/Straight Line Ratio")
    if eye_areas: header.append("Eye Area (µm²)")
    if eye_widths: header.append("Eye Width / Horizontal Ø (µm)")
    if eye_heights: header.append("Eye Height / Vertical Ø (µm)")
    if edema_areas: header.append("Edema Area (µm²)")
    sh.append(header)

    for i, fname in enumerate(filenames):
        row = [fname]
        if fish_lengths:
            L = fish_lengths[i] if i < len(fish_lengths) and fish_lengths[i] is not None else "N/A"
            row.append(L if _is_included(i, 'fish_length') else EXCLUDED)
        if curvatures:
            c = curvatures[i] if i < len(curvatures) else None
            if c is None:
                c = "N/A"
            elif c == 5:
                c = "Not Classified"
            row.append(c if _is_included(i, 'curvature') else EXCLUDED)
        if ratios:
            r = ratios[i] if i < len(ratios) and ratios[i] is not None else "N/A"
            row.append(r if _is_included(i, 'ratio') else EXCLUDED)
        if eye_areas:
            ea = eye_areas[i] if i < len(eye_areas) and eye_areas[i] is not None else "N/A"
            row.append(ea if _is_included(i, 'eye_area') else EXCLUDED)
        if eye_widths:
            ew = eye_widths[i] if i < len(eye_widths) and eye_widths[i] is not None else "N/A"
            row.append(ew if _is_included(i, 'eye_area') else EXCLUDED)
        if eye_heights:
            eh = eye_heights[i] if i < len(eye_heights) and eye_heights[i] is not None else "N/A"
            row.append(eh if _is_included(i, 'eye_area') else EXCLUDED)
        if edema_areas:
            eda = edema_areas[i] if i < len(edema_areas) and edema_areas[i] is not None else "N/A"
            row.append(eda if _is_included(i, 'edema_area') else EXCLUDED)
        sh.append(row)

    def _stats(vals, metric_key):
        clean_vals = np.array([
            float(v) for idx, v in enumerate(vals or [])
            if _is_included(idx, metric_key)
            and isinstance(v, (int, float)) and np.isfinite(v)
        ])
        if len(clean_vals) == 0:
            return ("N/A",) * 5
        return (
            np.median(clean_vals),
            np.percentile(clean_vals, 25),
            np.percentile(clean_vals, 75),
            np.mean(clean_vals),
            np.std(clean_vals),
        )

    sh.append([])
    if threshold_used:
        sh.append([f"Threshold used; statistics may be unreliable (threshold: {threshold_value})"])

    # Note on excluded metrics
    excluded_counts = {}
    for metric in ('fish_length', 'curvature', 'ratio', 'eye_area', 'edema_area'):
        excluded_counts[metric] = sum(
            1 for i in range(len(filenames)) if not _is_included(i, metric)
        )
    excl_note_parts = [f"{k.replace('_', ' ')}: {v}" for k, v in excluded_counts.items() if v > 0]
    if excl_note_parts:
        sh.append(["Excluded from statistics — " + ", ".join(excl_note_parts)])

    sh.append(["Statistics (excluded values not counted)"])

    if fish_lengths:
        medL,p25L,p75L,meanL,stdL = _stats(fish_lengths, 'fish_length')
        sh.append(["Median Length (µm)", medL]); sh.append(["25th Percentile Length (µm)", p25L])
        sh.append(["75th Percentile Length (µm)", p75L]); sh.append(["Mean Length (µm)", meanL])
        sh.append(["Standard Deviation Length (µm)", stdL])

    if curvatures:
        medC,p25C,p75C,meanC,stdC = _stats(curvatures, 'curvature')
        sh.append(["Median Curvature", medC]); sh.append(["25th Percentile Curvature", p25C])
        sh.append(["75th Percentile Curvature", p75C]); sh.append(["Mean Curvature", meanC])
        sh.append(["Standard Deviation Curvature", stdC])

    if ratios:
        medR,p25R,p75R,meanR,stdR = _stats(ratios, 'ratio')
        sh.append(["Median Ratio", medR]); sh.append(["25th Percentile Ratio", p25R])
        sh.append(["75th Percentile Ratio", p75R]); sh.append(["Mean Ratio", meanR])
        sh.append(["Standard Deviation Ratio", stdR])

    if eye_areas:
        medEA,p25EA,p75EA,meanEA,stdEA = _stats(eye_areas, 'eye_area')
        sh.append(["Median Eye Area (µm²)", medEA]); sh.append(["25th Percentile Eye Area (µm²)", p25EA])
        sh.append(["75th Percentile Eye Area (µm²)", p75EA]); sh.append(["Mean Eye Area (µm²)", meanEA])
        sh.append(["Standard Deviation Eye Area (µm²)", stdEA])

    if eye_widths:
        medEW,p25EW,p75EW,meanEW,stdEW = _stats(eye_widths, 'eye_area')
        sh.append(["Median Eye Width (µm)", medEW]); sh.append(["25th Percentile Eye Width (µm)", p25EW])
        sh.append(["75th Percentile Eye Width (µm)", p75EW]); sh.append(["Mean Eye Width (µm)", meanEW])
        sh.append(["Standard Deviation Eye Width (µm)", stdEW])

    if eye_heights:
        medEH,p25EH,p75EH,meanEH,stdEH = _stats(eye_heights, 'eye_area')
        sh.append(["Median Eye Height (µm)", medEH]); sh.append(["25th Percentile Eye Height (µm)", p25EH])
        sh.append(["75th Percentile Eye Height (µm)", p75EH]); sh.append(["Mean Eye Height (µm)", meanEH])
        sh.append(["Standard Deviation Eye Height (µm)", stdEH])

    if edema_areas:
        medEDA,p25EDA,p75EDA,meanEDA,stdEDA = _stats(edema_areas, 'edema_area')
        sh.append(["Median Edema Area (µm²)", medEDA]); sh.append(["25th Percentile Edema Area (µm²)", p25EDA])
        sh.append(["75th Percentile Edema Area (µm²)", p75EDA]); sh.append(["Mean Edema Area (µm²)", meanEDA])
        sh.append(["Standard Deviation Edema Area (µm²)", stdEDA])

    sh.append([]); sh.append(["Class Distribution"])
    cls_counts = [0,0,0,0,0]
    for idx, c in enumerate(curvatures):
        if not _is_included(idx, 'curvature') or c is None:
            continue
        i_cls = 4 if c == 5 else int(c)-1
        if 0 <= i_cls < 5:
            cls_counts[i_cls] += 1
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
MAX_EDITOR_PX = 800  # max display dimension for the mask editor (memory optimisation)
GALLERY_MAX_PX = 900  # max display dimension for gallery thumbnails (browser memory optimisation)

def _make_seg_overlay(original_img, seg_mask, path_points=None, straight_line_points=None, eye_mask=None, edema_mask=None, mask_alpha=GALLERY_MASK_ALPHA, draw_eye_diameters=True, max_px=None) -> np.ndarray:
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

    if edema_mask is not None:
        edema_norm = _normalize_mask(edema_mask)
        if edema_norm.shape[:2] != base.shape[:2]:
            edema_norm = np.array(PILImage.fromarray(edema_norm).resize((base.shape[1], base.shape[0]), resample=PILImage.NEAREST))
        blue = np.zeros_like(overlay)
        blue[..., 2] = 255
        edm = (edema_norm > 0)[..., None].astype(np.float32)
        overlay = overlay * (1 - 0.4 * edm) + blue * (0.4 * edm)

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
                # dark outline for contrast, then bright cyan on top
                cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_AA)
                cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
        except Exception:
            pass

    if straight_line_points is not None:
        try:
            (r1, c1), (r2, c2) = straight_line_points
            p1 = (int(np.clip(round(c1 * sx), 0, w_base - 1)), int(np.clip(round(r1 * sy), 0, h_base - 1)))
            p2 = (int(np.clip(round(c2 * sx), 0, w_base - 1)), int(np.clip(round(r2 * sy), 0, h_base - 1)))
            # dark outline for contrast, then bright magenta on top
            cv2.line(overlay, p1, p2, (0, 0, 0), 6, lineType=cv2.LINE_AA)
            cv2.line(overlay, p1, p2, (255, 0, 255), 3, lineType=cv2.LINE_AA)
        except Exception:
            pass
    if eye_mask is not None and draw_eye_diameters:
        try:
            eye_norm = _normalize_mask(eye_mask)
            if eye_norm.shape[:2] != overlay.shape[:2]:
                eye_norm = np.array(PILImage.fromarray(eye_norm).resize(
                    (overlay.shape[1], overlay.shape[0]), resample=PILImage.NEAREST))
            em = eye_norm > 0
            if em.any():
                num, labels, stats, _ = cv2.connectedComponentsWithStats(
                    em.astype(np.uint8), connectivity=8)
                if num > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    em = labels == largest
                ys, xs = np.where(em)
                ymin, ymax = int(ys.min()), int(ys.max())
                xmin, xmax = int(xs.min()), int(xs.max())
                cy = (ymin + ymax) // 2
                cx = (xmin + xmax) // 2
                # horizontal diameter (dark outline + green line)
                cv2.line(overlay, (xmin, cy), (xmax, cy), (0, 0, 0), 4, lineType=cv2.LINE_AA)
                cv2.line(overlay, (xmin, cy), (xmax, cy), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                # vertical diameter
                cv2.line(overlay, (cx, ymin), (cx, ymax), (0, 0, 0), 4, lineType=cv2.LINE_AA)
                cv2.line(overlay, (cx, ymin), (cx, ymax), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    if max_px is not None:
        h, w = overlay.shape[:2]
        if max(h, w) > max_px:
            scale = max_px / max(h, w)
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return overlay  # Full resolution unless max_px is given

def _shorten_name(name: str, max_chars: int = 22) -> str:
    base = os.path.basename(name)
    if len(base) <= max_chars: return base
    root, ext = os.path.splitext(base)
    keep = max_chars - len(ext) - 3
    if keep <= 0: return base[:max(1, max_chars-3)] + '...'
    head = keep // 2; tail = keep - head
    return f"{root[:head]}...{root[-tail:]}{ext}"

def _stage_inputs(files: Optional[List[gr.File]], folder_input) -> Tuple[str, list, Optional[str]]:
    """
    Normalize inputs into a working directory with all images inside, and a
    sorted list of filenames (basenames) that match what will be processed.
    - If `folder_input` is a list/tuple of paths (Gradio folder upload), copy ALL
      of them into a temp dir and return that dir + filenames.
    - If `folder_input` is a string path to a directory, enumerate it.
    - Otherwise, fall back to `files` (individual uploads) and copy into a temp dir.

    Returns (work_dir, filenames, tmpdir_to_clean): the third element is the temp
    directory the caller should delete after use, or None when work_dir belongs to
    the user (Case 2) and must not be removed.
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
            return tmpdir, basenames, tmpdir

    # Case 2: Folder upload as a single directory path (less common)
    if isinstance(folder_input, str) and os.path.isdir(folder_input):
        names = [n for n in os.listdir(folder_input)
                 if os.path.splitext(n)[1].lower() in exts]
        names.sort()
        return folder_input, names, None  # user's own folder — do not delete

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
    return tmpdir, filenames, tmpdir


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


def _load_manual_scalebar_image(folder_input, files):
    """Load the first uploaded image for manual scale bar endpoint selection."""
    first_path = _get_first_image_path(folder_input, files)
    if first_path is None:
        return None, [], "Upload images first."
    try:
        img_bgr = cv2.imread(first_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil = PILImage.open(first_path).convert('RGB')
            img_rgb = np.array(pil)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None, [], f"Could not load image: {e}"
    return img_rgb, [], "Click to set **START** point (one end of scale bar)."


def _record_scalebar_click(evt: gr.SelectData, current_img, sb_points):
    """Record a click for manual scale bar endpoint selection."""
    if current_img is None:
        return sb_points, current_img, "Click **Load Image** first."
    if not (hasattr(evt, 'index') and evt.index is not None):
        return sb_points, current_img, "No click coordinates received."
    if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
        click_x, click_y = int(evt.index[0]), int(evt.index[1])
    else:
        return sb_points, current_img, "Invalid click coordinates."
    if sb_points is None:
        sb_points = []
    sb_points = list(sb_points)
    if len(sb_points) >= 2:
        return sb_points, current_img, "⚠ Both endpoints already set. Click **Reset Points** to start over."
    sb_points.append((click_x, click_y))
    img_with_points = _draw_scalebar_endpoints(current_img, sb_points) if _HAS_SCALEBAR else np.array(current_img).copy()
    if len(sb_points) == 2:
        cal = _calibrate_from_endpoints(sb_points[0], sb_points[1], np.array(current_img).shape) if _HAS_SCALEBAR else {}
        dist_px = cal.get('bar_length_px', 0.0) or 0.0
        status = (
            f"✓ Both endpoints set ({dist_px:.1f} px apart). "
            "Enter the physical length in **Physical length of scale bar (µm)** above, "
            "then click **Apply Manual Points**."
        )
    else:
        status = "✓ START point set (green). Now click the other end of the scale bar (END, red)."
    return sb_points, img_with_points, status


def _reset_scalebar_points(folder_input, files):
    """Reset manual scale bar points and reload the original image."""
    first_path = _get_first_image_path(folder_input, files)
    if first_path is None:
        return [], None, "Upload images first."
    try:
        img_bgr = cv2.imread(first_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil = PILImage.open(first_path).convert('RGB')
            img_rgb = np.array(pil)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return [], None, f"Could not reload image: {e}"
    return [], img_rgb, "Points reset. Click to set START point."


def _apply_scalebar_points(sb_points, bar_label_um_str, folder_input, files):
    """Compute µm/px calibration from two manually placed scale bar endpoints."""
    if sb_points is None or len(sb_points) != 2:
        return gr.update(), "⚠ Need exactly 2 points. Click on the image to set START and END.", gr.update(), gr.update()
    # Retrieve image shape for physical size computation
    img_shape = (0, 0)
    first_path = _get_first_image_path(folder_input, files)
    if first_path:
        try:
            img_bgr = cv2.imread(first_path, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_shape = img_bgr.shape[:2]
            else:
                pil = PILImage.open(first_path)
                img_shape = (pil.size[1], pil.size[0])
        except Exception:
            pass
    label_um = _safe_float(bar_label_um_str, default=None)
    if not _HAS_SCALEBAR:
        return gr.update(), "⚠ scalebar module unavailable.", gr.update(), gr.update()
    result = _calibrate_from_endpoints(sb_points[0], sb_points[1], img_shape, label_um=label_um)
    bar_px_str = f"{result['bar_length_px']:.1f}" if result.get('bar_length_px') is not None else ""
    if not result['bar_found']:
        return gr.update(), f"⚠ {result['message']}", gr.update(), gr.update()
    if result['success']:
        phys_w = f"{result['phys_width_um']:.1f}"
        phys_h = f"{result['phys_height_um']:.1f}"
        return gr.update(value=bar_px_str), f"✅ {result['message']}", gr.update(value=phys_w), gr.update(value=phys_h)
    else:
        return gr.update(value=bar_px_str), f"📏 {result['message']}", gr.update(), gr.update()


def process(folder,
            files: Optional[List[gr.File]],
            seg_model_choice="General Model",
            use_finetuned_desy=False,
            process_curvature=True,
            process_length=True,
            process_ratio=True,
            process_eye_size=True,
            process_edema=True,
            use_threshold=False,
            threshold_value=0.5,
            physical_horizontal_um_str="",
            physical_vertical_um_str=""):
    t0 = time.perf_counter()
    work_dir, filenames, _tmpdir_to_clean = _stage_inputs(files, folder)
    # Resolve chosen segmentation model (fine-tuned DESY checkbox overrides the preset radio)
    if use_finetuned_desy:
        seg_model_choice = "Fine-tuned DESY"
    seg_filename, seg_encoder, eye_filename, model_target_size, edema_filename = SEG_MODEL_OPTIONS.get(
        seg_model_choice, SEG_MODEL_OPTIONS["Fast & Easy (256 px, ~2s/image)"]
    )
    # Build kwargs for eye/edema models (use pipeline defaults when filename is None)
    eye_kwargs = {} if eye_filename is None else {"eye_model_filename": eye_filename}
    edema_kwargs = {} if edema_filename is None else {"edema_model_filename": edema_filename}
    # Pass sorted file paths so segmentation results match the sorted filenames list
    file_paths_sorted = [os.path.join(work_dir, fn) for fn in filenames]
    # Always load eyes for overlay visualization; load edema if requested
    try:
        if process_edema:
            original_images, segmented_images, grown_images, eyes_images, edema_images = segmentation_pipeline(
                file_list=file_paths_sorted,
                target_size=(model_target_size, model_target_size),
                include_eyes=True,
                include_edema=True,
                body_model_filename=seg_filename,
                body_encoder_name=seg_encoder,
                **eye_kwargs,
                **edema_kwargs,
            )
        else:
            original_images, segmented_images, grown_images, eyes_images = segmentation_pipeline(
                file_list=file_paths_sorted,
                target_size=(model_target_size, model_target_size),
                include_eyes=True,
                include_edema=False,
                body_model_filename=seg_filename,
                body_encoder_name=seg_encoder,
                **eye_kwargs,
            )
            edema_images = [None] * len(original_images)
    finally:
        if _tmpdir_to_clean:
            shutil.rmtree(_tmpdir_to_clean, ignore_errors=True)
    model = _ensure_model()

    # Parse physical distances (µm) for full image width/height from user
    phys_w_um_user = _safe_float(physical_horizontal_um_str, default=None)
    phys_h_um_user = _safe_float(physical_vertical_um_str, default=None)

    if phys_w_um_user is not None and phys_h_um_user is not None:
        y_scale_info = phys_h_um_user / model_target_size
        x_scale_info = phys_w_um_user / model_target_size
        spacing_info_md = (
            f"**Spacing used:** custom input | "
            f"y = {y_scale_info:.4f} µm/pixel, x = {x_scale_info:.4f} µm/pixel "
            f"(from H={phys_h_um_user:g} µm, W={phys_w_um_user:g} µm over {model_target_size} px)"
        )
    else:
        y_scale_info = 5885.0 / model_target_size
        x_scale_info = 5885.0 / model_target_size
        spacing_info_md = (
            f"**Spacing used:** default calibration | "
            f"y = {y_scale_info:.4f} µm/pixel, x = {x_scale_info:.4f} µm/pixel "
            f"(H=W=5885 µm over {model_target_size} px)"
        )

    fish_lengths, curvatures, ratios, eye_areas, edema_areas, previews = [], [], [], [], [], []
    eye_widths, eye_heights = [], []
    paths, straight_lines = [], []  # stored per-image for gallery overlay regeneration
    for i, seg_mask in enumerate(segmented_images):
        path_points = None
        straight_line_points = None
        eye_mask_for_vis = eyes_images[i] if i < len(eyes_images) else None
        edema_mask_for_vis = edema_images[i] if i < len(edema_images) else None
        seg_mask_bin = seg_mask > 0

        # Per-image pixel scales derived from user-provided physical distances
        h, w = seg_mask.shape[:2]
        # Default to pixel units if user did not provide values
        if phys_w_um_user is not None and phys_h_um_user is not None:
            phys_w_um = phys_w_um_user
            phys_h_um = phys_h_um_user
            # Calculate spacing for the new function: (dy, dx) in physical units per pixel
            y_scale = phys_h_um / model_target_size  # physical units per pixel in y direction
            x_scale = phys_w_um / model_target_size  # physical units per pixel in x direction
        else:
            # Default spacing (assuming 5885 µm per model_target_size pixels as per the original code)
            y_scale = 5885.0 / model_target_size
            x_scale = 5885.0 / model_target_size
            phys_w_um = 5885.0
            phys_h_um = 5885.0

        if process_length:
            # Use the new tube_length_border2border function
            try:
                eye_mask_for_length = (eye_mask_for_vis > 0) if eye_mask_for_vis is not None else None
                spacing = (y_scale, x_scale)
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
                fish_lengths.append(None)
                if process_ratio:
                    ratios.append(None)

        if process_eye_size:
                    try:
                        eye_mask_for_metrics = (eye_mask_for_vis > 0) if eye_mask_for_vis is not None else None
                        eye_info = compute_eye_metrics(
                            eye_mask_for_metrics,
                            mask_fish=seg_mask_bin,
                            spacing=(y_scale, x_scale),
                        )
                        eye_areas.append(float(eye_info.get("eye_area", 0.0)))
                        dia = compute_eye_diameters(eye_mask_for_metrics, spacing=(y_scale, x_scale))
                        eye_widths.append(float(dia.get("eye_width_um", 0.0)))
                        eye_heights.append(float(dia.get("eye_height_um", 0.0)))
                    except Exception as e:
                        print(f"Error calculating eye metrics for image {i}: {e}")
                        eye_areas.append(None)
                        eye_widths.append(None)
                        eye_heights.append(None)
        if process_edema:
            try:
                edema_mask_bin = (edema_mask_for_vis > 0) if edema_mask_for_vis is not None else None
                edema_info = compute_eye_metrics(
                    edema_mask_bin,
                    mask_fish=None,
                    spacing=(y_scale, x_scale),
                )
                edema_areas.append(float(edema_info.get("eye_area", 0.0)))
            except Exception as e:
                print(f"Error calculating edema area for image {i}: {e}")
                edema_areas.append(None)

        if process_curvature:
            try:
                _, curv = classification_curvature(original_images[i], grown_images[i], model, use_threshold, threshold_value)
                curvatures.append(int(curv.item()))
            except Exception as e:
                print(f"Error calculating curvature for image {i}: {e}")
                curvatures.append(None)

        paths.append(path_points)
        straight_lines.append(straight_line_points)
        try:
            overlay = _make_seg_overlay(
                original_images[i],
                seg_mask,
                path_points=path_points,
                straight_line_points=straight_line_points,
                eye_mask=eye_mask_for_vis,
                edema_mask=edema_mask_for_vis,
                max_px=GALLERY_MAX_PX,
            )
            original_name = filenames[i] if i < len(filenames) else f"image_{i}"
            short = _shorten_name(original_name, max_chars=22)
            # embed index into caption so selection handlers can identify images robustly
            cap = f"{i}:{short}"
            previews.append([overlay, cap])
        except Exception:
            pass

    boxplot_png = _make_boxplots_image(fish_lengths, curvatures, ratios, eye_areas, edema_areas)
    boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))
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
        'eye_widths': eye_widths,
        'eye_heights': eye_heights,
        'edema_areas': edema_areas,
        'boxplot_png': boxplot_png,
        'threshold_used': use_threshold,
        'threshold_value': threshold_value,
        'previews': previews,
        'original_previews': original_previews,
        'original_images': original_images,
        'segmented_images': segmented_images,
        'eyes_images': eyes_images,
        'edema_images': edema_images,
        'spacing': (y_scale, x_scale),
        'paths': paths,
        'straight_lines': straight_lines,
        'manual_points': {},
        'exclusions': {},
    }
    shown_names = [_shorten_name(n, max_chars=22) for n in filenames[:5]]
    more_note = f" … and {len(filenames) - 5} more" if len(filenames) > 5 else ""
    filenames_md = "**Uploaded:** " + ", ".join(shown_names) + more_note
    elapsed = time.perf_counter() - t0
    n_imgs = len(filenames)
    print(f"[PROCESS TIMING] {seg_model_choice}: {elapsed:.2f}s ({n_imgs} images)")
    return boxplot_np, previews, filenames_md, data_state, spacing_info_md + f"\n\n⏱ **{seg_model_choice} processing time:** {elapsed:.2f} s ({n_imgs} image{'s' if n_imgs != 1 else ''})"

def summarize_files(files):
    if not files: return "No files uploaded."
    names = [os.path.basename(f.name) for f in files[:5]]
    short = [_shorten_name(n, max_chars=22) for n in names]
    more = f" … and {len(files) - 5} more" if len(files) > 5 else ""
    return "**Uploaded:** " + ", ".join(short) + more


def _generate_corrected_excel(data, sheet_name="Fish Data"):
    if not data:
        return None

    out_bytes = write_lengths_to_excel_bytes(
        data.get('filenames', []),
        data.get('fish_lengths', []),
        data.get('curvatures', []),
        data.get('ratios', []),
        data.get('eye_areas', []),
        data.get('edema_areas', []),
        data.get('threshold_used', False),
        data.get('threshold_value', 0.0),
        data.get('boxplot_png', None),
        sheet_name=sheet_name,
        exclusions=data.get('exclusions', {}),
        eye_widths=data.get('eye_widths', []),
        eye_heights=data.get('eye_heights', []),
    )
    out_dir = tempfile.mkdtemp(prefix='fish_data_')
    out_xlsx = os.path.join(out_dir, f"{_sanitize_filename(sheet_name)}.xlsx")
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
        
        # Clip points to image bounds
        p1_int = np.clip(np.round(p1).astype(int), [0, 0], [seg_mask_bin.shape[0]-1, seg_mask_bin.shape[1]-1])
        p2_int = np.clip(np.round(p2).astype(int), [0, 0], [seg_mask_bin.shape[0]-1, seg_mask_bin.shape[1]-1])

        # Track whether clicked points were outside the mask so we can extend the path later
        p1_outside = not seg_mask_bin[p1_int[0], p1_int[1]]
        p2_outside = not seg_mask_bin[p2_int[0], p2_int[1]]
        p1_anchor = p1_int.copy()  # actual clicked position (clamped to image bounds)
        p2_anchor = p2_int.copy()

        # If points are outside the mask, find nearest inside point for internal routing
        if p1_outside:
            mask_coords = np.argwhere(seg_mask_bin)
            if len(mask_coords) > 0:
                from scipy.spatial.distance import cdist
                dist_to_p1 = cdist([p1], mask_coords)[0]
                p1_int = mask_coords[np.argmin(dist_to_p1)]

        if p2_outside:
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
            
            for _ in range(iterations):
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

        # If clicked points were outside the mask, extend path with straight-line segments
        # from the actual clicked position to the mask boundary entry/exit point.
        if p1_outside:
            n_ext = max(2, int(np.ceil(np.linalg.norm(p1_anchor.astype(float) - p1_int.astype(float)))) + 1)
            t = np.linspace(0, 1, n_ext)[:-1]  # exclude p1_int — already path[0]
            ext = np.round(p1_anchor[None, :] * (1 - t[:, None]) + p1_int[None, :] * t[:, None]).astype(int)
            path = np.vstack([ext, path])

        if p2_outside:
            n_ext = max(2, int(np.ceil(np.linalg.norm(p2_anchor.astype(float) - p2_int.astype(float)))) + 1)
            t = np.linspace(0, 1, n_ext)[1:]  # exclude p2_int — already path[-1]
            ext = np.round(p2_int[None, :] * (1 - t[:, None]) + p2_anchor[None, :] * t[:, None]).astype(int)
            path = np.vstack([path, ext])

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
        edema_mask = data.get('edema_images', [None]*(edit_idx+1))[edit_idx] if edit_idx < len(data.get('edema_images', [])) else None
        new_overlay_gallery = _make_seg_overlay(
            original_img,
            seg_mask,
            path_points=path,
            straight_line_points=straight_line_points,
            eye_mask=eye_mask,
            edema_mask=edema_mask,
            mask_alpha=GALLERY_MASK_ALPHA,
            max_px=GALLERY_MAX_PX,
        )

        new_overlay_manual = _make_seg_overlay(
            original_img,
            seg_mask,
            path_points=path,
            straight_line_points=straight_line_points,
            eye_mask=eye_mask,
            edema_mask=edema_mask,
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
            data.get('edema_areas', []),
        )
        boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))
        data['boxplot_png'] = boxplot_png
        
        status = f"✓ Manual points applied! Length: {length:.2f} µm, Straight: {straight_length:.2f} µm, Ratio: {length/straight_length:.3f}"

        # Return updated overlay to manual edit window (user can see lines before accordion closes)
        return data, previews, boxplot_np, status, gr.update(), new_overlay_manual
        
    except Exception as e:
        return data, gr.update(), gr.update(), f"Error applying manual points: {str(e)}", gr.update(), gr.update()

# ── Manual Mask Editor helpers ───────────────────────────────────────────────

_MASK_KEYS = {
    'Body':  'segmented_images',
    'Eye':   'eyes_images',
    'Edema': 'edema_images',
}


def _prepare_editor_value(edit_idx, data, mask_type):
    """Return an ImageEditor value dict built from the stored mask.

    The existing mask is baked into the background image (avoids Gradio's
    layer/background CSS misalignment). The layer starts empty so user
    strokes are always at the correct canvas coordinates.
    """
    if data is None or edit_idx < 0:
        return None
    imgs = data.get('original_images', [])
    if edit_idx >= len(imgs):
        return None

    orig = _to_numpy(imgs[edit_idx])
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=-1)
    elif orig.shape[2] == 4:
        orig = orig[:, :, :3]

    # Downscale for display – keeps browser memory low
    h, w = orig.shape[:2]
    if max(h, w) > MAX_EDITOR_PX:
        scale = MAX_EDITOR_PX / max(h, w)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        bg = cv2.resize(orig, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        bg = orig.copy()
    dh, dw = bg.shape[:2]

    # Bake the current mask into the background (pixel-perfect alignment)
    key = _MASK_KEYS.get(mask_type, 'segmented_images')
    masks = data.get(key, [])
    if edit_idx < len(masks) and masks[edit_idx] is not None:
        m = _normalize_mask(masks[edit_idx])
        if m.shape[:2] != (dh, dw):
            m = np.array(PILImage.fromarray(m).resize((dw, dh), resample=PILImage.NEAREST))
        alpha = 0.45
        m_f = (m > 127).astype(np.float32)[:, :, None]
        yellow = np.array([[[255, 200, 0]]], dtype=np.float32)
        bg = (bg.astype(np.float32) * (1 - alpha * m_f) +
              yellow * (alpha * m_f)).clip(0, 255).astype(np.uint8)

    # Empty layer – user strokes go here; no pre-existing content means no offset
    empty_layer = np.zeros((dh, dw, 4), dtype=np.uint8)
    return {"background": bg, "layers": [empty_layer], "composite": None}


def _apply_mask_edit(editor_data, edit_idx, mask_type, data):
    """Apply user strokes from the layer to the stored mask, then recompute metrics.

    Yellow strokes (#FFC800) → add pixels to mask.
    Blue strokes (#0044FF)   → remove pixels from mask.
    Eraser removes strokes from the layer (those pixels keep their original value).
    Returns updated (data_state, gallery, out_box, status, mask_editor).
    """
    if data is None or edit_idx < 0 or editor_data is None:
        return data, gr.update(), gr.update(), "⚠ No image selected.", gr.update()
    layers = editor_data.get("layers") or []
    if not layers or layers[0] is None:
        return data, gr.update(), gr.update(), "⚠ No layer data found.", gr.update()
    layer = np.asarray(layers[0])
    if layer.ndim != 3 or layer.shape[2] < 4:
        return data, gr.update(), gr.update(), "⚠ Unexpected layer format.", gr.update()

    dh, dw = layer.shape[:2]
    a = layer[:, :, 3] > 64
    additions = a & (layer[:, :, 0] > 180) & (layer[:, :, 2] < 80)   # yellow
    removals  = a & (layer[:, :, 2] > 180) & (layer[:, :, 0] < 80)   # blue

    key = _MASK_KEYS.get(mask_type, 'segmented_images')
    orig_masks = data.get(key, [])
    if edit_idx >= len(orig_masks):
        return data, gr.update(), gr.update(), f"⚠ {mask_type} mask not found for image {edit_idx}.", gr.update()

    # Retrieve current mask and resize to display resolution for editing
    if orig_masks[edit_idx] is not None:
        current = _normalize_mask(orig_masks[edit_idx])
        oh, ow = current.shape[:2]
        cur_disp = (
            np.array(PILImage.fromarray(current).resize((dw, dh), resample=PILImage.NEAREST))
            if current.shape[:2] != (dh, dw) else current.copy()
        )
    else:
        # No mask exists yet for this image — use the resolution of the body
        # segmentation mask (all stored masks share that resolution), not the
        # editor's display canvas size, so newly-drawn masks stay aligned with
        # spacing-dependent metrics (area/length calculations).
        seg_ref = data.get('segmented_images', [])
        if edit_idx < len(seg_ref) and seg_ref[edit_idx] is not None:
            oh, ow = _normalize_mask(seg_ref[edit_idx]).shape[:2]
        else:
            oh, ow = dh, dw
        cur_disp = np.zeros((dh, dw), dtype=np.uint8)

    new_mask_disp = cur_disp.copy()
    new_mask_disp[additions] = 255
    new_mask_disp[removals]  = 0

    # Resize back to original mask resolution
    new_mask = (
        np.array(PILImage.fromarray(new_mask_disp).resize((ow, oh), resample=PILImage.NEAREST))
        if (dh, dw) != (oh, ow) else new_mask_disp
    )
    data[key][edit_idx] = new_mask

    # ── Recompute metrics that depend on the changed mask ────────────────────
    spacing = data.get('spacing', (1.0, 1.0))
    n_seg   = len(data.get('segmented_images', []))
    n_eye   = len(data.get('eyes_images',      []))
    n_edm   = len(data.get('edema_images',     []))

    seg_mask  = data['segmented_images'][edit_idx] if edit_idx < n_seg else None
    eye_mask  = data['eyes_images'][edit_idx]      if edit_idx < n_eye else None
    edm_mask  = data['edema_images'][edit_idx]     if edit_idx < n_edm else None

    seg_bin = (seg_mask > 0) if seg_mask is not None else None

    new_path_pts    = None  # will be set for Body edit, used in preview
    new_straight_pts = None

    if mask_type == 'Body' and seg_bin is not None:
        try:
            eye_bin = (eye_mask > 0) if eye_mask is not None else None
            length, straight, new_path_pts, new_straight_pts = tube_length_border2border(
                seg_bin,
                spacing=spacing,
                return_path=True,
                return_straight_line=True,
                mask_eye=eye_bin,
                return_eye_info=False,
            )
            if edit_idx < len(data.get('fish_lengths', [])):
                data['fish_lengths'][edit_idx] = float(length)
            if edit_idx < len(data.get('ratios', [])):
                data['ratios'][edit_idx] = float(length / straight) if straight > 0 else 0.0
            # Keep stored paths in sync
            if 'paths' in data and edit_idx < len(data['paths']):
                data['paths'][edit_idx] = new_path_pts
            if 'straight_lines' in data and edit_idx < len(data['straight_lines']):
                data['straight_lines'][edit_idx] = new_straight_pts
        except Exception:
            pass

    if mask_type == 'Eye' and eye_mask is not None:
        try:
            eye_bin = eye_mask > 0
            eye_info = compute_eye_metrics(eye_bin, mask_fish=seg_bin, spacing=spacing)
            if edit_idx < len(data.get('eye_areas', [])):
                data['eye_areas'][edit_idx] = float(eye_info.get('eye_area', 0.0))
            dia = compute_eye_diameters(eye_bin, spacing=spacing)
            if edit_idx < len(data.get('eye_widths', [])):
                data['eye_widths'][edit_idx]  = float(dia.get('eye_width_um',  0.0))
            if edit_idx < len(data.get('eye_heights', [])):
                data['eye_heights'][edit_idx] = float(dia.get('eye_height_um', 0.0))
        except Exception:
            pass

    if mask_type == 'Edema' and edm_mask is not None:
        try:
            edm_bin  = edm_mask > 0
            edm_info = compute_eye_metrics(edm_bin, mask_fish=None, spacing=spacing)
            if edit_idx < len(data.get('edema_areas', [])):
                data['edema_areas'][edit_idx] = float(edm_info.get('eye_area', 0.0))
        except Exception:
            pass

    # Regenerate boxplot
    boxplot_out = gr.update()
    try:
        bp_png = _make_boxplots_image(
            data.get('fish_lengths', []),
            data.get('curvatures',   []),
            data.get('ratios',       []),
            data.get('eye_areas',    []),
            data.get('edema_areas',  []),
        )
        data['boxplot_png'] = bp_png
        boxplot_out = np.array(PILImage.open(io.BytesIO(bp_png)))
    except Exception:
        pass

    # Regenerate preview overlay, preserving path lines
    try:
        orig = data['original_images'][edit_idx]
        # For Body edit use freshly computed paths; for Eye/Edema reuse stored paths
        stored_paths    = data.get('paths',        [])
        stored_straights = data.get('straight_lines', [])
        path_pts    = new_path_pts    if mask_type == 'Body' else (stored_paths[edit_idx]    if edit_idx < len(stored_paths)    else None)
        straight_pts = new_straight_pts if mask_type == 'Body' else (stored_straights[edit_idx] if edit_idx < len(stored_straights) else None)
        new_overlay = _make_seg_overlay(orig, seg_mask, path_pts, straight_pts, eye_mask, edm_mask, max_px=GALLERY_MAX_PX)
        fname = data['filenames'][edit_idx] if edit_idx < len(data.get('filenames', [])) else f"Image {edit_idx}"
        cap = f"{edit_idx}:{_shorten_name(fname, max_chars=22)}"
        if 'original_previews' in data and edit_idx < len(data['original_previews']):
            data['original_previews'][edit_idx] = [new_overlay, cap]
        if 'previews' in data and edit_idx < len(data['previews']):
            data['previews'][edit_idx] = [new_overlay, cap]
    except Exception:
        pass

    return (
        data,
        data.get('previews', []),
        boxplot_out,
        f"✅ {mask_type} mask saved, metrics recalculated for image {edit_idx}.",
        None,
    )


with gr.Blocks() as demo:
    gr.Markdown("# Zebrafish Analyzer")
    gr.Markdown("""
    📖 **Documentation:** For detailed instructions and usage examples, please visit the [GitHub repository](https://github.com/MarkDanielArndt/Zebrafish_webapp).
    
    This webapp is provided freely to the research community. If you find it useful, please consider giving the repository a ⭐ (it's free!). 
    If you use this tool in your research, please cite: *[Paper - soon to be published]*.
    """
    )

    gr.Markdown("## 1. Choose Model")
    # --- Model selection ---
    with gr.Group():
        gr.Markdown("### 🔬 Segmentation Model")
        gr.Markdown(
            "Select the body segmentation model to use. "
            "**Fast & Easy** is quicker but less accurate. "
            "**Complex & Slower** takes longer but is more accurate."
        )
        model_choice = gr.Radio(
            choices=["Fast & Easy (256 px, ~2s/image)", "Complex & Slower (512 px, ~7s/image)"],
            value="Fast & Easy (256 px, ~2s/image)",
            label="Model",
        )
        last_preset_choice = gr.State("Fast & Easy (256 px, ~2s/image)")
        with gr.Accordion("Other models", open=False):
            finetuned_choice = gr.Checkbox(
                value=False,
                label="Fine-tuned DESY (only use if you know what this is)",
            )
        # Picking a preset directly always wins: uncheck DESY and remember the preset.
        model_choice.select(
            lambda choice: (gr.update(value=False), choice),
            inputs=model_choice,
            outputs=[finetuned_choice, last_preset_choice],
        )
        # Checking DESY clears the preset selection; unchecking restores the last preset.
        finetuned_choice.change(
            lambda checked, preset: gr.update(value=None) if checked else gr.update(value=preset),
            inputs=[finetuned_choice, last_preset_choice],
            outputs=model_choice,
        )

    gr.Markdown("## 2. Upload Images")
    # Left: folder upload + compact upload button
    with gr.Row():
        folder = gr.File(label="Upload a folder", file_count="directory", type="filepath")
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
    manual_scalebar_points = gr.State([])

    gr.Markdown("## 3. Calibrate Scale Bar")
    # --- Scale bar auto-detection ---
    with gr.Accordion("📏 Scale Bar Calibration", open=True):
        gr.Markdown(
            "Automatically detects the scale bar line in the **first uploaded image** "
            "and shows how many pixels it spans.  "
            "Enter only the number printed above the bar (e.g. `500`). The unit is µm. "
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
            bar_label_um_auto = gr.Textbox(
                label="Physical length of scale bar (µm)",
                placeholder="e.g. 500")
        with gr.Row():
            apply_scalebar_btn = gr.Button("Apply", variant="primary")

        # --- Manual scale bar entry ---
        with gr.Accordion("📐 Manual Scale Bar Entry (if auto-detection fails)", open=False):
            gr.Markdown(
                "The image loads automatically below. Click on it to mark the "
                "**two endpoints** of the scale bar line:\n"
                "- **1st click** → START (one end, shown in **green**)\n"
                "- **2nd click** → END (other end, shown in **red**)\n\n"
                "After both endpoints are set, enter the physical length below and click "
                "**Apply Manual Points** to fill in the calibration automatically."
            )
            manual_sb_status = gr.Markdown("Upload images to begin.")
            manual_sb_image = gr.Image(
                label="Click to mark endpoints: START (green, 1st click) → END (red, 2nd click)",
                type="numpy", interactive=False)
            bar_label_um_input = gr.Textbox(
                label="Physical length of scale bar (µm)",
                placeholder="e.g. 500")
            with gr.Row():
                load_sb_image_btn = gr.Button("Reload Image", variant="secondary")
                reset_sb_points_btn = gr.Button("Reset Points", variant="secondary")
                apply_sb_points_btn = gr.Button("Apply Manual Points", variant="primary")

    gr.Markdown("## 4. Select Analyses")
    with gr.Row():
        chk_curv = gr.Checkbox(value=True, label="Process Curvature")
        chk_len  = gr.Checkbox(value=True, label="Process Length")
        chk_ratio = gr.Checkbox(value=True, label="Process Length/Straight Line Ratio")
        chk_eye = gr.Checkbox(value=True, label="Process Eye Size")
        chk_edema = gr.Checkbox(value=True, label="Process Edema")
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

    gr.Markdown("## 5. Run")
    run = gr.Button("Run")

    gr.Markdown("## 6. Review, Edit, and Export Final Excel")
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

        # ── Manual Mask Editor ───────────────────────────────────────────────
        with gr.Accordion("✏️ Manual Mask Editor", open=False):
            gr.Markdown(
                "**🟡 Yellow brush** — add to mask · "
                "**🔵 Blue brush** — remove from mask · "
                "**Eraser** — undo your strokes  \n"
                "Select an image in the gallery → choose mask type → click **Load Image into Editor** → draw → click **Apply**."
            )
            mask_type_radio = gr.Radio(
                choices=["Body", "Eye", "Edema"],
                value="Body",
                label="Mask type",
            )
            load_mask_btn = gr.Button("📥 Load Image into Editor", variant="secondary")
            mask_editor = gr.ImageEditor(
                type="numpy",
                sources=[],
                layers=True,
                brush=gr.Brush(
                    default_size=20,
                    colors=["#FFC800", "#0044FF"],
                    default_color="#FFC800",
                    color_mode="fixed",
                ),
                eraser=gr.Eraser(default_size=20),
                transforms=[],
                label="🟡 yellow = add  ·  🔵 blue = remove",
                height=460,
                value=None,
            )
            with gr.Row():
                apply_mask_btn = gr.Button("✅ Apply", variant="primary")
                reset_mask_btn = gr.Button("↺ Reset",  variant="secondary")
            mask_edit_status = gr.Markdown("")

        filenames_list = gr.Markdown("")
        
        with gr.Accordion("📄 Final Excel Export", open=True):
            gr.Markdown("""
            Create a single final Excel export from the current results in memory.

            If you adjusted manual points, this export will include the updated length and ratio values.
            """)

            with gr.Group():
                gr.Markdown("### 🚫 Exclude Measurements for This Image")
                gr.Markdown(
                    "Uncheck any metric you want to mark as **Excluded** in the final Excel "
                    "(the row will still appear but the cell will say *Excluded* and it won't "
                    "count toward statistics)."
                )
                with gr.Row():
                    excl_length = gr.Checkbox(value=True, label="Fish Length")
                    excl_curv   = gr.Checkbox(value=True, label="Curvature")
                    excl_ratio  = gr.Checkbox(value=True, label="Ratio")
                    excl_eye    = gr.Checkbox(value=True, label="Eye Area")
                    excl_edema  = gr.Checkbox(value=True, label="Edema Area")
                with gr.Row():
                    save_excl_btn = gr.Button("💾 Save Exclusions for This Image", variant="primary")
                excl_status = gr.Markdown("")

            with gr.Row():
                excel_sheet_name = gr.Textbox(
                    label="Sheet name",
                    value="Fish Data",
                    placeholder="Fish Data",
                    max_lines=1,
                )

            with gr.Row():
                gen_corrected_btn = gr.DownloadButton("Generate Final Excel", variant="primary")


    def _load_exclusions_for_image(evt: gr.SelectData, data):
        idx = evt.index
        if data is None or idx < 0:
            return True, True, True, True, True
        excl = data.get('exclusions', {}).get(idx, {})
        return (
            excl.get('fish_length', True),
            excl.get('curvature',   True),
            excl.get('ratio',       True),
            excl.get('eye_area',    True),
            excl.get('edema_area',  True),
        )

    def _save_exclusions(edit_idx, inc_length, inc_curv, inc_ratio, inc_eye, inc_edema, data):
        if data is None or edit_idx < 0:
            return data, "⚠ No image selected."
        if 'exclusions' not in data:
            data['exclusions'] = {}
        data['exclusions'][edit_idx] = {
            'fish_length': inc_length,
            'curvature':   inc_curv,
            'ratio':       inc_ratio,
            'eye_area':    inc_eye,
            'edema_area':  inc_edema,
        }
        fname = (data['filenames'][edit_idx]
                 if edit_idx < len(data.get('filenames', []))
                 else f"Image {edit_idx}")
        excluded = [k for k, v in data['exclusions'][edit_idx].items() if not v]
        msg = (f"✅ **{fname}** — excluded: {', '.join(excluded)}"
               if excluded else f"✅ **{fname}** — all metrics included.")
        return data, msg


    # Use files from state, not a giant Files list
    run.click(
        fn=process,
        inputs=[folder, files_state, model_choice, finetuned_choice, chk_curv, chk_len, chk_ratio, chk_eye, chk_edema, chk_thr, thr_val, phys_w_um, phys_h_um],
        outputs=[out_box, gallery, filenames_list, data_state, spacing_used_md]
    )

    # --- Scale bar detection event wiring ---
    _scalebar_outputs = [scalebar_preview, scalebar_status_md, bar_px_display, phys_w_um, phys_h_um]
    _scalebar_inputs_detect = [folder, files_state]          # no label yet on initial detect
    _scalebar_inputs_apply  = [folder, files_state, bar_label_um_auto]

    # Detect button – find the bar, show px length (no label yet)
    detect_scalebar_btn.click(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_detect,
        outputs=_scalebar_outputs,
    ).then(
        fn=_load_manual_scalebar_image,
        inputs=[folder, files_state],
        outputs=[manual_sb_image, manual_scalebar_points, manual_sb_status],
    )

    # Apply button – re-run detection with the user-supplied label
    apply_scalebar_btn.click(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_apply,
        outputs=_scalebar_outputs,
    ).then(
        fn=_load_manual_scalebar_image,
        inputs=[folder, files_state],
        outputs=[manual_sb_image, manual_scalebar_points, manual_sb_status],
    )

    # Auto-trigger detect (no label) when folder upload changes
    folder.change(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_detect,
        outputs=_scalebar_outputs,
    ).then(
        fn=_load_manual_scalebar_image,
        inputs=[folder, files_state],
        outputs=[manual_sb_image, manual_scalebar_points, manual_sb_status],
    )

    # Auto-trigger detect after individual file upload (chain after state update)
    _upload_event.then(
        fn=_run_scalebar_detection,
        inputs=_scalebar_inputs_detect,
        outputs=_scalebar_outputs,
    ).then(
        fn=_load_manual_scalebar_image,
        inputs=[folder, files_state],
        outputs=[manual_sb_image, manual_scalebar_points, manual_sb_status],
    )

    # --- Manual scale bar event wiring ---
    load_sb_image_btn.click(
        fn=_load_manual_scalebar_image,
        inputs=[folder, files_state],
        outputs=[manual_sb_image, manual_scalebar_points, manual_sb_status],
    )

    manual_sb_image.select(
        fn=_record_scalebar_click,
        inputs=[manual_sb_image, manual_scalebar_points],
        outputs=[manual_scalebar_points, manual_sb_image, manual_sb_status],
    )

    reset_sb_points_btn.click(
        fn=_reset_scalebar_points,
        inputs=[folder, files_state],
        outputs=[manual_scalebar_points, manual_sb_image, manual_sb_status],
    )

    apply_sb_points_btn.click(
        fn=_apply_scalebar_points,
        inputs=[manual_scalebar_points, bar_label_um_input, folder, files_state],
        outputs=[bar_px_display, scalebar_status_md, phys_w_um, phys_h_um],
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

    # Load exclusion checkboxes when a gallery image is selected
    gallery.select(
        fn=_load_exclusions_for_image,
        inputs=[data_state],
        outputs=[excl_length, excl_curv, excl_ratio, excl_eye, excl_edema],
    )

    # Save exclusions
    save_excl_btn.click(
        fn=_save_exclusions,
        inputs=[edit_image_idx, excl_length, excl_curv, excl_ratio, excl_eye, excl_edema, data_state],
        outputs=[data_state, excl_status],
    )

    gen_corrected_btn.click(
        fn=_generate_corrected_excel,
        inputs=[data_state, excel_sheet_name],
        outputs=[gen_corrected_btn]
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

    # ── Mask editor event wiring ─────────────────────────────────────────────
    # Load mask into editor only when the user explicitly clicks the button
    load_mask_btn.click(
        fn=_prepare_editor_value,
        inputs=[edit_image_idx, data_state, mask_type_radio],
        outputs=[mask_editor],
    )

    # Save edits back to state, then clear the canvas
    apply_mask_btn.click(
        fn=_apply_mask_edit,
        inputs=[mask_editor, edit_image_idx, mask_type_radio, data_state],
        outputs=[data_state, gallery, out_box, mask_edit_status, mask_editor],
    )

    # Discard edits and reload original mask
    reset_mask_btn.click(
        fn=_prepare_editor_value,
        inputs=[edit_image_idx, data_state, mask_type_radio],
        outputs=[mask_editor],
    )

if __name__ == "__main__":
    demo.launch(share=True)
