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

def _make_boxplots_image(fish_lengths, curvatures):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    if fish_lengths: plt.boxplot(fish_lengths, vert=True, patch_artist=True)
    plt.title("Fish Lengths"); plt.ylabel("Length (µm)")
    plt.subplot(1,2,2)
    if curvatures: plt.boxplot(curvatures, vert=True, patch_artist=True)
    plt.title("Curvatures"); plt.ylabel("Curvature")
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close(fig)
    img_bytes.seek(0)
    return img_bytes.getvalue()

def write_lengths_to_excel_bytes(filenames, fish_lengths, curvatures, threshold_used, threshold_value, boxplot_png_bytes):
    wb = openpyxl.Workbook()
    sh = wb.active
    sh.title = "Fish Data"
    sh.append(["Filename", "Fish Length (µm)", "Curvature"])
    for i, fname in enumerate(filenames):
        L = fish_lengths[i] if i < len(fish_lengths) else "N/A"
        c = curvatures[i] if i < len(curvatures) else "N/A"
        if c == 5:
            c = "Not Classified"
        sh.append([fname, L, c])

    def _stats(vals):
        if not vals: return ("N/A",)*5
        vals_sorted = sorted(vals); n = len(vals_sorted)
        median = vals_sorted[n//2]
        p25 = vals_sorted[int(n*0.25)]
        p75 = vals_sorted[int(n*0.75)]
        mean = sum(vals_sorted)/n
        std = (sum((x-mean)**2 for x in vals_sorted)/n)**0.5
        return median, p25, p75, mean, std

    medL,p25L,p75L,meanL,stdL = _stats(fish_lengths)
    medC,p25C,p75C,meanC,stdC = _stats(curvatures)
    sh.append([])
    if threshold_used:
        sh.append([f"Threshold used; statistics may be unreliable (threshold: {threshold_value})"])
    sh.append(["Statistics"])
    sh.append(["Median Length (µm)", medL]); sh.append(["25th Percentile Length (µm)", p25L])
    sh.append(["75th Percentile Length (µm)", p75L]); sh.append(["Mean Length (µm)", meanL])
    sh.append(["Standard Deviation Length (µm)", stdL])
    sh.append(["Median Curvature", medC]); sh.append(["25th Percentile Curvature", p25C])
    sh.append(["75th Percentile Curvature", p75C]); sh.append(["Mean Curvature", meanC])
    sh.append(["Standard Deviation Curvature", stdC])
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

def _make_seg_overlay(original_img, seg_mask) -> np.ndarray:
    base = _to_numpy(original_img); mask = _normalize_mask(seg_mask)
    if base.ndim == 2: base = np.stack([base]*3, axis=-1)
    if mask.shape[:2] != base.shape[:2]:
        mask = np.array(PILImage.fromarray(mask).resize((base.shape[1], base.shape[0]), resample=PILImage.NEAREST))
    overlay = base.copy().astype(np.float32)
    alpha = 0.4; red = np.zeros_like(overlay); red[..., 0] = 255
    m = (mask > 0)[..., None].astype(np.float32)
    overlay = overlay * (1 - alpha * m) + red * (alpha * m)
    overlay = overlay.clip(0,255).astype(np.uint8)
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
            use_threshold=False,
            threshold_value=0.5,
            physical_horizontal_um_str="",
            physical_vertical_um_str=""):
    work_dir, filenames = _stage_inputs(files, folder)
    original_images, segmented_images, grown_images = segmentation_pipeline(work_dir)
    model = _ensure_model()

    # Parse physical distances (µm) for full image width/height from user
    phys_w_um_user = _safe_float(physical_horizontal_um_str, default=None)
    phys_h_um_user = _safe_float(physical_vertical_um_str, default=None)

    fish_lengths, curvatures, previews = [], [], []
    for i, seg_mask in enumerate(segmented_images):
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
                spacing = (y_scale, x_scale)
                length = tube_length_border2border(seg_mask_bin, spacing=spacing, return_path=False, return_skeleton=False)
                fish_lengths.append(float(length))
            except Exception as e:
                print(f"Error calculating length for image {i}: {e}")
                # Fallback to old method if needed
                try:
                    L, _ = get_fish_length_circles_fixed(seg_mask, phys_w_um, phys_h_um, circle_dia=15)
                    fish_lengths.append(float(L))
                except Exception:
                    pass

        if process_curvature:
            try:
                _, curv = classification_curvature(original_images[i], grown_images[i], model, use_threshold, threshold_value)
                curvatures.append(int(curv.item()))
            except Exception:
                pass

        if len(previews) < 5:
            try:
                overlay = _make_seg_overlay(original_images[i], seg_mask)
                original_name = filenames[i] if i < len(filenames) else f"image_{i}"
                cap = _shorten_name(original_name, max_chars=22)
                previews.append([overlay, cap])
            except Exception:
                pass

    boxplot_png = _make_boxplots_image(fish_lengths, curvatures)
    boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))
    out_bytes = write_lengths_to_excel_bytes(filenames, fish_lengths, curvatures, use_threshold, threshold_value, boxplot_png)
    tmpout = tempfile.mkdtemp(); out_xlsx = os.path.join(tmpout, "fish_data.xlsx")
    with open(out_xlsx, "wb") as f: f.write(out_bytes.getvalue())
    shown_names = [_shorten_name(n, max_chars=22) for n in filenames[:5]]
    more_note = f" … and {len(filenames) - 5} more" if len(filenames) > 5 else ""
    filenames_md = "**Uploaded:** " + ", ".join(shown_names) + more_note
    return out_xlsx, boxplot_np, previews, filenames_md

def summarize_files(files):
    if not files: return "No files uploaded."
    names = [os.path.basename(f.name) for f in files[:5]]
    short = [_shorten_name(n, max_chars=22) for n in names]
    more = f" … and {len(files) - 5} more" if len(files) > 5 else ""
    return "**Uploaded:** " + ", ".join(short) + more

with gr.Blocks() as demo:
    gr.Markdown("# Zebrafish Analyzer")

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

    with gr.Row():
        chk_curv = gr.Checkbox(value=True, label="Process Curvature")
        chk_len  = gr.Checkbox(value=True, label="Process Length")
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
            gallery = gr.Gallery(label="Example segmentations (first 5)", columns=5, height="auto")
        filenames_list = gr.Markdown("")

    # Use files from state, not a giant Files list
    run.click(
        fn=process,
        inputs=[folder, files_state, chk_curv, chk_len, chk_thr, thr_val, phys_w_um, phys_h_um],
        outputs=[out_file, out_box, gallery, filenames_list]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

