import gradio as gr
import tempfile, os, shutil
from typing import List, Optional, Tuple
from seg import segmentation_pipeline
from length import load_model, get_fish_length_circles_fixed, classification_curvature, tube_length_border2border
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

def _make_boxplots_image(fish_lengths, curvatures, ratios):
    # Count how many plots we need
    num_plots = sum([bool(fish_lengths), bool(curvatures), bool(ratios)])
    if num_plots == 0:
        num_plots = 1  # At least one subplot
    
    fig = plt.figure(figsize=(5*num_plots, 5))
    plot_idx = 1
    
    if fish_lengths:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(fish_lengths, vert=True, patch_artist=True)
        plt.title("Fish Lengths"); plt.ylabel("Length (µm)")
        plot_idx += 1
    
    if curvatures:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(curvatures, vert=True, patch_artist=True)
        plt.title("Curvatures"); plt.ylabel("Curvature")
        plot_idx += 1
    
    if ratios:
        plt.subplot(1, num_plots, plot_idx)
        plt.boxplot(ratios, vert=True, patch_artist=True)
        plt.title("Length/Straight Line Ratio"); plt.ylabel("Ratio")
    
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close(fig)
    img_bytes.seek(0)
    return img_bytes.getvalue()

def write_lengths_to_excel_bytes(filenames, fish_lengths, curvatures, ratios, threshold_used, threshold_value, boxplot_png_bytes):
    wb = openpyxl.Workbook()
    sh = wb.active
    sh.title = "Fish Data"
    # Build header dynamically based on what data we have
    header = ["Filename"]
    if fish_lengths: header.append("Fish Length (µm)")
    if curvatures: header.append("Curvature")
    if ratios: header.append("Length/Straight Line Ratio")
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
        sh.append(row)

    def _stats(vals):
        if not vals: return ("N/A",)*5
        vals_sorted = sorted(vals); n = len(vals_sorted)
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

def _resize_max(img: np.ndarray, max_w: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w: return img
    scale = max_w / float(w); new_w, new_h = int(w*scale), int(h*scale)
    return np.array(PILImage.fromarray(img).resize((new_w, new_h), resample=PILImage.BILINEAR))

def _make_seg_overlay(original_img, seg_mask, path_points=None, straight_line_points=None, eye_mask=None) -> np.ndarray:
    base = _to_numpy(original_img); mask = _normalize_mask(seg_mask)
    if base.ndim == 2: base = np.stack([base]*3, axis=-1)
    if mask.shape[:2] != base.shape[:2]:
        mask = np.array(PILImage.fromarray(mask).resize((base.shape[1], base.shape[0]), resample=PILImage.NEAREST))
    overlay = base.copy().astype(np.float32)
    # fish mask overlay in yellow
    alpha = 0.35
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

    return _resize_max(overlay)

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
        s = str(s).strip().replace(",", ".")
        return float(s)
    except Exception:
        return default

def process(folder,
            files: Optional[List[gr.File]],
            process_curvature=True,
            process_length=True,
            process_ratio=True,
            use_threshold=False,
            threshold_value=0.5,
            physical_horizontal_um_str="",
            physical_vertical_um_str=""):
    work_dir, filenames = _stage_inputs(files, folder)
    # Always enable eye segmentation for visualization
    original_images, segmented_images, grown_images, eyes_images = segmentation_pipeline(work_dir, include_eyes=True)
    model = _ensure_model()

    # Parse physical distances (µm) for full image width/height from user
    phys_w_um_user = _safe_float(physical_horizontal_um_str, default=None)
    phys_h_um_user = _safe_float(physical_vertical_um_str, default=None)

    fish_lengths, curvatures, ratios, previews = [], [], [], []
    for i, seg_mask in enumerate(segmented_images):
        path_points = None
        straight_line_points = None
        eye_mask_for_vis = eyes_images[i] if i < len(eyes_images) else None

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
                seg_mask_bin = seg_mask > 0
                eye_mask_bin = None
                if eye_mask_for_vis is not None:
                    eye_mask_bin = _normalize_mask(eye_mask_for_vis) > 0
                spacing = (y_scale, x_scale)
                length, straight_length, path_points, straight_line_points, eye_info = tube_length_border2border(
                    seg_mask_bin,
                    spacing=spacing,
                    return_path=True,
                    return_straight_line=True,
                    mask_eye=eye_mask_bin,
                    return_eye_info=True,
                )
                if eye_info is not None and eye_info.get("eye_mask") is not None:
                    eye_mask_for_vis = eye_info.get("eye_mask")
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
                # Fallback to old method if needed
                try:
                    L, _ = get_fish_length_circles_fixed(seg_mask, phys_w_um, phys_h_um, circle_dia=15)
                    fish_lengths.append(float(L))
                    if process_ratio:
                        ratios.append(0.0)  # No ratio available for fallback method
                except Exception:
                    pass

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

    boxplot_png = _make_boxplots_image(fish_lengths, curvatures, ratios)
    boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))
    out_bytes = write_lengths_to_excel_bytes(filenames, fish_lengths, curvatures, ratios, use_threshold, threshold_value, boxplot_png)
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
        'boxplot_png': boxplot_png,
        'threshold_used': use_threshold,
        'threshold_value': threshold_value,
        'previews': previews,
        'original_previews': original_previews,
    }
    excluded_state = [False] * len(filenames)
    shown_names = [_shorten_name(n, max_chars=22) for n in filenames[:5]]
    more_note = f" … and {len(filenames) - 5} more" if len(filenames) > 5 else ""
    filenames_md = "**Uploaded:** " + ", ".join(shown_names) + more_note
    # also prepare checkbox choices (index:shortname) for exclusion UI
    exclude_choices = [f"{i}:{_shorten_name(n, max_chars=22)}" for i, n in enumerate(filenames)]
    # return a gradio update for choices so the CheckboxGroup can be populated
    return out_xlsx, boxplot_np, previews, filenames_md, data_state, excluded_state, gr.update(choices=exclude_choices, value=[])

def summarize_files(files):
    if not files: return "No files uploaded."
    names = [os.path.basename(f.name) for f in files[:5]]
    short = [_shorten_name(n, max_chars=22) for n in names]
    more = f" … and {len(files) - 5} more" if len(files) > 5 else ""
    return "**Uploaded:** " + ", ".join(short) + more


def _draw_cross(img_np):
    try:
        img = img_np.copy()
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        h, w = img.shape[:2]
        thickness = max(2, int(min(h, w) * 0.03))
        # draw a less-bright red cross so it is visible but not overpowering
        cross_color = (150, 0, 0)
        cv2.line(img, (0, 0), (w - 1, h - 1), cross_color, thickness=thickness)
        cv2.line(img, (w - 1, 0), (0, h - 1), cross_color, thickness=thickness)
        return img
    except Exception:
        return img_np


def _toggle_exclusion(sel_idx, excluded, data):
    # sel_idx: index of clicked image (int or list); excluded: list of bools; data: data_state dict
    if data is None:
        return gr.update(), excluded, data
    # ensure excluded list length
    if excluded is None or len(excluded) != len(data['filenames']):
        excluded = [False] * len(data['filenames'])
    # sel_idx from gallery.select may be an int, a caption string, or a [image, caption] tuple
    idx = None
    try:
        # direct int (gradio sometimes provides index)
        idx = int(sel_idx)
    except Exception:
        pass
    if idx is None:
        # sel_idx may be like [image, caption] or caption string
        try:
            if isinstance(sel_idx, (list, tuple)) and len(sel_idx) > 1 and isinstance(sel_idx[1], str):
                caption = sel_idx[1]
            elif isinstance(sel_idx, str):
                caption = sel_idx
            else:
                caption = None
            if caption is not None and ':' in caption:
                idx = int(caption.split(':', 1)[0])
        except Exception:
            idx = None
    if idx is None:
        return gr.update(), excluded, data
    if 0 <= idx < len(excluded):
        excluded[idx] = not bool(excluded[idx])
    # rebuild previews with cross for excluded and update data state
    previews = []
    originals = data.get('original_previews', data.get('previews', []))
    for i, (orig_img, cap) in enumerate(originals):
        short_cap = cap
        if isinstance(short_cap, str) and ':' in short_cap:
            short_cap = short_cap.split(':', 1)[1]
        if i < len(excluded) and excluded[i]:
            previews.append([_draw_cross(orig_img), f"{i}:{short_cap} (excluded)"])
        else:
            previews.append([orig_img, f"{i}:{short_cap}"])
    # persist the updated previews in data so subsequent toggles use original images
    data['previews'] = previews
    return previews, excluded, data


def _generate_filtered_excel(data, excluded):
    if not data:
        return None
    excluded = excluded or []
    fnames, Ls, Cs, Rs = [], [], [], []
    # Check what data we have available
    has_lengths = 'fish_lengths' in data and data['fish_lengths']
    has_curvatures = 'curvatures' in data and data['curvatures']
    has_ratios = 'ratios' in data and data['ratios']
    
    for i, name in enumerate(data['filenames']):
        if i < len(excluded) and excluded[i]:
            continue
        fnames.append(name)
        if has_lengths:
            Ls.append(data['fish_lengths'][i] if i < len(data['fish_lengths']) else "N/A")
        if has_curvatures:
            Cs.append(data['curvatures'][i] if i < len(data['curvatures']) else "N/A")
        if has_ratios:
            Rs.append(data['ratios'][i] if i < len(data['ratios']) else "N/A")
    
    # Pass empty lists for features that weren't processed
    if not has_lengths:
        Ls = []
    if not has_curvatures:
        Cs = []
    if not has_ratios:
        Rs = []
    
    out_bytes = write_lengths_to_excel_bytes(fnames, Ls, Cs, Rs, data.get('threshold_used', False), data.get('threshold_value', 0.0), data.get('boxplot_png', None))
    tmpout = tempfile.mkdtemp(); out_xlsx = os.path.join(tmpout, "fish_data_filtered.xlsx")
    with open(out_xlsx, "wb") as f:
        f.write(out_bytes.getvalue())
    return out_xlsx


def _update_from_checkboxes(selected, data):
    # selected: list of caption strings like '3:shortname' to be excluded
    if data is None:
        return gr.update(), [], data
    # derive excluded boolean list
    n = len(data.get('filenames', []))
    excluded = [False] * n
    if selected:
        for s in selected:
            try:
                idx = int(str(s).split(':', 1)[0])
                if 0 <= idx < n:
                    excluded[idx] = True
            except Exception:
                continue
    # rebuild previews with crosses for excluded images
    previews = []
    originals = data.get('original_previews', data.get('previews', []))
    for i, (orig_img, cap) in enumerate(originals):
        short_cap = cap
        if isinstance(short_cap, str) and ':' in short_cap:
            short_cap = short_cap.split(':', 1)[1]
        if excluded[i]:
            previews.append([_draw_cross(orig_img), f"{i}:{short_cap} (excluded)"])
        else:
            previews.append([orig_img, f"{i}:{short_cap}"])
    data['previews'] = previews
    return previews, excluded, data

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
    upload_btn.upload(fn=lambda f: (f, summarize_files(f)), inputs=upload_btn, outputs=[files_state, files_summary])

    # Hidden states for interactive filtering (populated after Run)
    data_state = gr.State(None)
    excluded_state = gr.State([])

    with gr.Row():
        chk_curv = gr.Checkbox(value=True, label="Process Curvature")
        chk_len  = gr.Checkbox(value=True, label="Process Length")
        chk_ratio = gr.Checkbox(value=True, label="Process Length/Straight Line Ratio")
        chk_thr  = gr.Checkbox(value=False, label="Use Threshold")
        thr_val  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Threshold Value")

    # --- New physical distance inputs (µm), placed just above Run button ---
    with gr.Row():
        phys_w_um = gr.Textbox(label="Physical horizontal distance (µm)", placeholder="dkfz E041: 5885")
        phys_h_um = gr.Textbox(label="Physical vertical distance (µm)", placeholder="dkfz E041: 5885")

    run = gr.Button("Run")

    with gr.Row():
        out_file = gr.File(label="Download results (.xlsx)")

    with gr.Accordion("Results previews", open=True):
        with gr.Row():
            out_box = gr.Image(label="Box plots", type="numpy")
        with gr.Row():
            gallery = gr.Gallery(label="Segmentations (click to toggle exclude)", columns=5, height="auto")
        with gr.Row():
            gen_filtered_btn = gr.Button("Generate Filtered Excel")
        filenames_list = gr.Markdown("")
        # CheckboxGroup to select images to exclude (populated after Run)
        with gr.Row():
            exclude_checkboxes = gr.CheckboxGroup(choices=[], label="Exclude images (check to exclude)")
        with gr.Row():
            out_file_filtered = gr.File(label="Download filtered results (.xlsx)")

    # Use files from state, not a giant Files list
    run.click(
        fn=process,
        inputs=[folder, files_state, chk_curv, chk_len, chk_ratio, chk_thr, thr_val, phys_w_um, phys_h_um],
        outputs=[out_file, out_box, gallery, filenames_list, data_state, excluded_state, exclude_checkboxes]
    )

    # When a gallery image is clicked, toggle its exclusion state and update previews
    # Note: the handler returns previews, excluded_state and updated data_state
    gallery.select(fn=_toggle_exclusion, inputs=[excluded_state, data_state], outputs=[gallery, excluded_state, data_state])

    # Checkbox-based exclusion handler (alternative to clicking images)
    exclude_checkboxes.change(fn=_update_from_checkboxes, inputs=[exclude_checkboxes, data_state], outputs=[gallery, excluded_state, data_state])

    # Generate a filtered excel that omits excluded images (writes to separate file output)
    gen_filtered_btn.click(fn=_generate_filtered_excel, inputs=[data_state, excluded_state], outputs=[out_file_filtered])

if __name__ == "__main__":
    demo.launch(share=True)

