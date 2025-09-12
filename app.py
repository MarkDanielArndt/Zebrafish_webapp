import gradio as gr
import tempfile, os, shutil
from typing import List, Optional, Tuple
from seg import segmentation_pipeline
from length import load_model, get_fish_length_circles_fixed, classification_curvature
import openpyxl, io
from openpyxl.drawing.image import Image as ExcelImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

# Optional: try import torch only to safely convert tensors to numpy
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
        img_min, img_max = float(img.min()), float(img.max()) if img.size else (0.0,1.0)
        if img_max <= 1.0 and img_min >= 0.0:
            img = (img * 255.0).clip(0,255).astype(np.uint8)
        else:
            denom = (img_max - img_min) if (img_max - img_min) != 0 else 1.0
            img = ((img - img_min) / denom * 255.0).clip(0,255).astype(np.uint8)
    return img

def _make_boxplots_image(fish_lengths, curvatures):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); 
    if fish_lengths: plt.boxplot(fish_lengths, vert=True, patch_artist=True)
    plt.title("Fish Lengths"); plt.ylabel("Length (units)")

    plt.subplot(1,2,2);
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
    sh.append(["Filename", "Fish Length (units)", "Curvature"])

    for i, fname in enumerate(filenames):
        L = fish_lengths[i] if i < len(fish_lengths) else "N/A"
        c = curvatures[i] if i < len(curvatures) else "N/A"
        if c == 5:
            c = "Not Classified"
        sh.append([fname, L, c])

    def _stats(vals):
        if not vals: return ("N/A",)*5
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
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
    sh.append(["Median Length", medL]); sh.append(["25th Percentile Length", p25L])
    sh.append(["75th Percentile Length", p75L]); sh.append(["Mean Length", meanL])
    sh.append(["Standard Deviation Length", stdL])
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
        img = ExcelImage(img_stream)
        sh.add_image(img, "E2")

    buf = io.BytesIO()
    wb.save(buf); buf.seek(0)
    return buf

def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    m = _to_numpy(mask).astype(np.float32)
    if m.ndim == 3 and m.shape[-1] == 3:
        m = m[...,0]
    if m.max() <= 1.0:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255
    return m

def _resize_max(img: np.ndarray, max_w: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    new_w, new_h = int(w*scale), int(h*scale)
    return np.array(PILImage.fromarray(img).resize((new_w, new_h), resample=PILImage.BILINEAR))

def _make_seg_overlay(original_img, seg_mask) -> np.ndarray:
    base = _to_numpy(original_img)
    mask = _normalize_mask(seg_mask)

    if base.ndim == 2:
        base = np.stack([base]*3, axis=-1)

    if mask.shape[:2] != base.shape[:2]:
        mask = np.array(PILImage.fromarray(mask).resize((base.shape[1], base.shape[0]), resample=PILImage.NEAREST))

    overlay = base.copy().astype(np.float32)
    alpha = 0.4
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    m = (mask > 0)[..., None].astype(np.float32)
    overlay = overlay * (1 - alpha * m) + red * (alpha * m)
    overlay = overlay.clip(0,255).astype(np.uint8)
    return _resize_max(overlay)

def _shorten_name(name: str, max_chars: int = 22) -> str:
    """Shorten long filenames with middle ellipsis, keep extension."""
    base = os.path.basename(name)
    if len(base) <= max_chars:
        return base
    root, ext = os.path.splitext(base)
    keep = max_chars - len(ext) - 3  # 3 for '...'
    if keep <= 0:
        return base[:max(1, max_chars-3)] + '...'
    head = keep // 2
    tail = keep - head
    return f"{root[:head]}...{root[-tail:]}{ext}"

def _stage_inputs(files: Optional[List[gr.File]], folder_input) -> Tuple[str, list]:
    folder_path = None
    if folder_input:
        if isinstance(folder_input, (list, tuple)):
            if len(folder_input) > 0 and isinstance(folder_input[0], str):
                folder_path = folder_input[0]
        elif isinstance(folder_input, str):
            folder_path = folder_input

    if folder_path and os.path.isdir(folder_path):
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        names = [n for n in os.listdir(folder_path) if os.path.splitext(n)[1].lower() in exts]
        names.sort()
        return folder_path, names

    tmpdir = tempfile.mkdtemp()
    filenames = []
    if files:
        for f in files:
            dst = os.path.join(tmpdir, os.path.basename(f.name))
            shutil.copy(f.name, dst)
            filenames.append(os.path.basename(f.name))
    return tmpdir, filenames

def process(folder, files: Optional[List[gr.File]], process_curvature=True, process_length=True, use_threshold=False, threshold_value=0.5):
    work_dir, filenames = _stage_inputs(files, folder)
    original_images, segmented_images, grown_images = segmentation_pipeline(work_dir)

    model = _ensure_model()

    fish_lengths, curvatures = [], []
    previews = []  # up to 5 [(img, caption), ...]

    for i, seg_mask in enumerate(segmented_images):
        if process_length:
            L, _ = get_fish_length_circles_fixed(seg_mask)
            try:
                fish_lengths.append(float(L))
            except Exception:
                pass
        if process_curvature:
            _, curv = classification_curvature(original_images[i], grown_images[i], model, use_threshold, threshold_value)
            try:
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

    # Boxplots
    boxplot_png = _make_boxplots_image(fish_lengths, curvatures)
    boxplot_np = np.array(PILImage.open(io.BytesIO(boxplot_png)))

    # Excel
    out_bytes = write_lengths_to_excel_bytes(filenames, fish_lengths, curvatures, use_threshold, threshold_value, boxplot_png)
    tmpout = tempfile.mkdtemp()
    out_xlsx = os.path.join(tmpout, "fish_data.xlsx")
    with open(out_xlsx, "wb") as f:
        f.write(out_bytes.getvalue())

    # Filenames summary (first 5 shortened)
    shown_names = [ _shorten_name(n, max_chars=22) for n in filenames[:5] ]
    more_note = ""
    if len(segmented_images) > 5:
        more_note = f"… and {len(segmented_images) - 5} more"
    filenames_md = "**Files:** " + ",  ".join(shown_names) + ("  " + more_note if more_note else "")

    return out_xlsx, boxplot_np, previews, filenames_md

with gr.Blocks() as demo:
    gr.Markdown("# Zebrafish Analyzer")
    with gr.Row():
        folder = gr.File(label="Upload a folder (preferred)", file_count="directory", type="filepath")
        files = gr.Files(label="Or upload individual images", file_count="multiple", type="filepath")
    with gr.Row():
        chk_curv = gr.Checkbox(value=True, label="Process Curvature")
        chk_len  = gr.Checkbox(value=True, label="Process Length")
        chk_thr  = gr.Checkbox(value=False, label="Use Threshold")
        thr_val  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Threshold Value")
    run = gr.Button("Run")

    with gr.Row():
        out_file = gr.File(label="Download results (.xlsx)")

    with gr.Accordion("Results previews", open=True):
        with gr.Row():
            out_box = gr.Image(label="Box plots", type="numpy")
        with gr.Row():
            gallery = gr.Gallery(label="Example segmentations (first 5)", columns=5, height="auto")
        filenames_list = gr.Markdown("")

    run.click(fn=process, inputs=[folder, files, chk_curv, chk_len, chk_thr, thr_val], outputs=[out_file, out_box, gallery, filenames_list])

demo.launch()
