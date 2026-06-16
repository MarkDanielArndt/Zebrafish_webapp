"""
Main widget for the ZebrafishAnalysis Slicer extension.

Left panel: input, analysis toggles, model selection, scalebar, run, export.
Right panel: QTabWidget with Gallery / Detail / Results / Exclude tabs.
"""

import importlib.util
import os
import sys

# Put our lib dir first and evict any cached Slicer modules with the same name.
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
elif sys.path[0] != _LIB_DIR:
    sys.path.remove(_LIB_DIR)
    sys.path.insert(0, _LIB_DIR)

for _m in ("logic", "overlay", "export"):
    sys.modules.pop(_m, None)

import qt
import ctk
import slicer


_ZEBRAFISH_MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "zebrafish_models")


def _local_model_path(filename):
    return os.path.join(_ZEBRAFISH_MODEL_CACHE, filename)


def _run_downloads(models_with_urls, hf_headers, progress_state, cancel_event):
    """
    Stream-download each (url, filename, label) to _ZEBRAFISH_MODEL_CACHE.
    URLs and headers are pre-resolved in the main thread to avoid huggingface_hub
    circular-import issues when imported from a background thread.
    """
    # Pre-initialize huggingface_hub here (background thread, no concurrency risk).
    # segmentation_models_pytorch imports it at module level; if it hasn't been
    # initialized yet when analyse_images runs in the main thread, Slicer's
    # broken lazy-loader causes an ImportError.  Doing it here ensures
    # sys.modules['huggingface_hub'] is fully populated before analyse_images fires.
    try:
        import huggingface_hub as _hf  # noqa: F401
    except Exception:
        pass

    import requests

    os.makedirs(_ZEBRAFISH_MODEL_CACHE, exist_ok=True)
    tmp_path = None

    try:
        for url, filename, label in models_with_urls:
            if cancel_event.is_set():
                progress_state["cancelled"] = True
                return

            progress_state["label"] = label
            progress_state["done"] = 0
            progress_state["total"] = 0

            local_path = _local_model_path(filename)
            tmp_path = local_path + ".tmp"

            # HEAD request to get Content-Length before streaming starts.
            try:
                head = requests.head(url, headers=hf_headers,
                                     allow_redirects=True, timeout=15)
                total = int(head.headers.get("content-length", 0))
                if total:
                    progress_state["total"] = total
            except Exception:
                pass

            resp = requests.get(url, headers=hf_headers,
                                stream=True, allow_redirects=True, timeout=60)
            resp.raise_for_status()

            if not progress_state["total"]:
                total = int(resp.headers.get("content-length", 0))
                if total:
                    progress_state["total"] = total

            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if cancel_event.is_set():
                        progress_state["cancelled"] = True
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        return
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_state["done"] = downloaded

            os.replace(tmp_path, local_path)
            tmp_path = None

        progress_state["done_flag"] = True
    except Exception as exc:
        progress_state["error"] = str(exc)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


class ZebrafishAnalysisMainWidget:
    def __init__(self, parent_layout):
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView
        )

        _mw = slicer.util.mainWindow()

        # Bottom dock spans full width (corners go to bottom area, not left/right)
        _mw.setCorner(qt.Qt.BottomLeftCorner,  qt.Qt.BottomDockWidgetArea)
        _mw.setCorner(qt.Qt.BottomRightCorner, qt.Qt.BottomDockWidgetArea)

        # Dock Python console below everything
        _pyDock = _mw.findChild(qt.QDockWidget, "PythonConsoleDockWidget")
        if _pyDock:
            _pyDock.setFloating(False)
            _mw.addDockWidget(qt.Qt.BottomDockWidgetArea, _pyDock)
            _pyDock.setMinimumHeight(1)
            inner = _pyDock.widget()
            if inner:
                for w in [inner] + list(inner.findChildren(qt.QWidget)):
                    w.setMinimumHeight(0)
                    sp = w.sizePolicy
                    sp.setVerticalPolicy(qt.QSizePolicy.Ignored)
                    w.setSizePolicy(sp)

        # Collapse the central slice view and expand module panel to full width.
        # Deferred so the window geometry is finalised before resizing.
        qt.QTimer.singleShot(0, self._expand_panel)

        self._results = []
        self._excluded = set()
        self._image_paths = []
        self._current_detail_idx = 0

        self._build_ui(parent_layout)
        self._connect_signals()

    def _expand_panel(self):
        mw = slicer.util.mainWindow()
        central = mw.centralWidget()
        if central:
            central.setMinimumWidth(0)
            central.hide()
        panelDock = mw.findChild(qt.QDockWidget, "PanelDockWidget")
        if panelDock:
            panelDock.setMinimumHeight(300)
            mw.resizeDocks([panelDock], [mw.width], qt.Qt.Horizontal)
        _pyDock = mw.findChild(qt.QDockWidget, "PythonConsoleDockWidget")
        if _pyDock:
            mw.resizeDocks([_pyDock], [150], qt.Qt.Vertical)
        dataProbe = mw.findChild(ctk.ctkCollapsibleButton, "DataProbeCollapsibleWidget")
        if dataProbe:
            dataProbe.collapsed = True

    def _build_ui(self, layout):
        layout.setAlignment(qt.Qt.Alignment())  # clear AlignTop set by Slicer base class

        splitter = qt.QSplitter(qt.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        splitter.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        layout.addWidget(splitter, 1)  # stretch=1 → fills all available vertical space

        self._build_left_panel(splitter)
        self._build_right_panel(splitter)
        splitter.setStretchFactor(1, 1)

        # progress bar removed — run button serves as progress indicator

    def _build_left_panel(self, splitter):
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(200)
        scroll.setMaximumWidth(500)
        splitter.addWidget(scroll)

        left = qt.QWidget()
        vbox = qt.QVBoxLayout(left)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(6)
        scroll.setWidget(left)

        input_box = ctk.ctkCollapsibleButton()
        input_box.text = "Input"
        vbox.addWidget(input_box)
        input_box.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        in_layout = qt.QVBoxLayout(input_box)

        self._btn_folder = qt.QPushButton("Load Folder…")
        self._btn_files  = qt.QPushButton("Load Images…")
        _load_row = qt.QHBoxLayout()
        _load_row.addWidget(self._btn_folder)
        _load_row.addWidget(self._btn_files)
        in_layout.addLayout(_load_row)
        in_layout.addWidget(qt.QLabel("Queue:"))
        self._queue_list = qt.QListWidget()
        self._queue_list.setMaximumHeight(120)
        in_layout.addWidget(self._queue_list)
        in_layout.addStretch()

        analysis_box = ctk.ctkCollapsibleButton()
        analysis_box.text = "Analysis"
        vbox.addWidget(analysis_box)
        analysis_box.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        an_layout = qt.QVBoxLayout(analysis_box)

        self._chk_length    = qt.QCheckBox("Body length");        self._chk_length.setChecked(True)
        self._chk_curvature = qt.QCheckBox("Curvature class");    self._chk_curvature.setChecked(True)
        self._chk_ratio     = qt.QCheckBox("Length/straight ratio"); self._chk_ratio.setChecked(True)
        self._chk_eyes      = qt.QCheckBox("Eye segmentation");   self._chk_eyes.setChecked(False)
        self._chk_hitl      = qt.QCheckBox("Confidence threshold"); self._chk_hitl.setChecked(False)

        for chk in (self._chk_length, self._chk_curvature, self._chk_ratio,
                    self._chk_eyes, self._chk_hitl):
            an_layout.addWidget(chk)

        self._threshold_slider = ctk.ctkSliderWidget()
        self._threshold_slider.minimum    = 0.0
        self._threshold_slider.maximum    = 1.0
        self._threshold_slider.singleStep = 0.01
        self._threshold_slider.value      = 0.85
        self._threshold_slider.decimals   = 2
        an_layout.addWidget(self._threshold_slider)
        an_layout.addStretch()

        model_box = ctk.ctkCollapsibleButton()
        model_box.text      = "Model"
        model_box.collapsed = True
        vbox.addWidget(model_box)
        m_layout = qt.QFormLayout(model_box)

        self._model_combo = qt.QComboBox()
        self._model_combo.addItem("General Model",   ("best_model_body_3400_vgg19.pth", "vgg19", None))
        self._model_combo.addItem("Fine-tuned DESY", ("best_model_body_finetuned.pth",  "vgg19", "best_model_eye_finetuned.pth"))
        m_layout.addRow("Segmentation model:", self._model_combo)

        scale_box = ctk.ctkCollapsibleButton()
        scale_box.text      = "Scale bar"
        scale_box.collapsed = False
        vbox.addWidget(scale_box)
        sc_layout = qt.QVBoxLayout(scale_box)

        self._btn_detect_scale = qt.QPushButton("Auto-detect from first image")
        sc_layout.addWidget(self._btn_detect_scale)

        self._scale_status = qt.QLabel("Load images first.")
        self._scale_status.setWordWrap(True)
        self._scale_status.setStyleSheet("color: #888; font-size: 11px;")
        sc_layout.addWidget(self._scale_status)

        form = qt.QFormLayout()
        self._bar_um_edit = qt.QLineEdit()
        self._bar_um_edit.setPlaceholderText("e.g. 500")
        form.addRow("Physical bar length (µm):", self._bar_um_edit)
        sc_layout.addLayout(form)

        self._btn_apply_scale = qt.QPushButton("Apply")
        sc_layout.addWidget(self._btn_apply_scale)

        sep = qt.QLabel("— or enter µm/px directly —")
        sep.setStyleSheet("color: #888; font-size: 11px;")
        sep.setAlignment(qt.Qt.AlignCenter)
        sc_layout.addWidget(sep)

        direct = qt.QFormLayout()
        self._um_per_px = ctk.ctkDoubleSpinBox()
        self._um_per_px.minimum    = 0.001
        self._um_per_px.maximum    = 9999.0
        self._um_per_px.singleStep = 0.01
        self._um_per_px.value      = 22.99
        self._um_per_px.decimals   = 4
        self._um_per_px.suffix     = " µm/px"
        direct.addRow("µm per pixel:", self._um_per_px)
        sc_layout.addLayout(direct)

        vbox.addStretch(1)  # push run + export to bottom

        self._btn_run = qt.QPushButton("▶  Run Analysis")
        self._btn_run.setStyleSheet("font-weight: bold; padding: 6px;")

        self._run_progress = qt.QProgressBar()
        self._run_progress.setTextVisible(True)
        self._run_progress.setStyleSheet("""
            QProgressBar {
                font-weight: bold; border-radius: 3px; border: none;
                background: #3a3a3a; color: white; text-align: center;
                min-height: 28px;
            }
            QProgressBar::chunk { background: #2e7d32; border-radius: 2px; }
        """)

        self._run_stack = qt.QStackedWidget()
        self._run_stack.addWidget(self._btn_run)       # index 0 — idle
        self._run_stack.addWidget(self._run_progress)  # index 1 — running
        vbox.addWidget(self._run_stack)

        export_box = ctk.ctkCollapsibleButton()
        export_box.text = "Export"
        vbox.addWidget(export_box)
        ex_layout = qt.QHBoxLayout(export_box)
        self._btn_excel = qt.QPushButton("Excel")
        self._btn_csv   = qt.QPushButton("CSV")
        ex_layout.addWidget(self._btn_excel)
        ex_layout.addWidget(self._btn_csv)


    def _build_right_panel(self, splitter):
        self._tabs = qt.QTabWidget()
        splitter.addWidget(self._tabs)

        from gallery_tab import GalleryTab
        self._gallery = GalleryTab(on_select=self._on_gallery_select)
        self._tabs.addTab(self._gallery, "Gallery")

        from detail_tab import DetailTab
        self._detail = DetailTab(
            on_navigate=self._navigate_detail,
            on_back=lambda: self._tabs.setCurrentIndex(0),
        )
        self._detail._params_getter = self._get_correction_params
        self._tabs.addTab(self._detail, "Detail")

        from results_tab import ResultsTab
        self._results_tab = ResultsTab()
        self._tabs.addTab(self._results_tab, "Results")

        from exclude_tab import ExcludeTab
        self._exclude_tab = ExcludeTab(on_change=lambda exc: setattr(self, "_excluded", exc))
        self._tabs.addTab(self._exclude_tab, "Exclude")

    def _connect_signals(self):
        self._btn_folder.clicked.connect(self._on_load_folder)
        self._btn_files.clicked.connect(self._on_load_files)
        self._btn_detect_scale.clicked.connect(self._on_detect_scale)
        self._btn_apply_scale.clicked.connect(self._on_apply_scale)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_excel.clicked.connect(self._on_export_excel)
        self._btn_csv.clicked.connect(self._on_export_csv)
        self._model_combo.currentIndexChanged.connect(self._start_preload)

    def _on_load_folder(self):
        settings = qt.QSettings()
        last = settings.value("ZebrafishAnalysis/lastFolder", "")
        folder = qt.QFileDialog.getExistingDirectory(None, "Select image folder", last)
        if not folder:
            return
        settings.setValue("ZebrafishAnalysis/lastFolder", folder)
        import os
        exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
        paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        self._set_queue(paths)

    def _on_load_files(self):
        paths = qt.QFileDialog.getOpenFileNames(
            None, "Select images", "",
            "Images (*.png *.tif *.tiff *.jpg *.jpeg)"
        )
        if isinstance(paths, (list, tuple)) and paths and isinstance(paths[0], list):
            paths = paths[0]  # Slicer Qt binding wraps in extra tuple
        if paths:
            self._set_queue(sorted(paths))

    def _set_queue(self, paths):
        import os
        self._image_paths = paths
        self._queue_list.clear()

        stubs = []
        for p in paths:
            self._queue_list.addItem(os.path.basename(p))
            stubs.append({"filename": os.path.basename(p), "original": None,
                          "mask": None, "error": None, "length": None})

        self._results = stubs
        self._detail.invalidate_cache()
        self._gallery.populate(stubs)
        self._tabs.setCurrentIndex(0)

        # Read image header only (no pixel data) to get dimensions for µm/px default.
        # PIL.Image.open is lazy — reads TIFF header in milliseconds.
        if paths:
            try:
                from PIL import Image as _PIL
                with _PIL.open(paths[0]) as _img:
                    w, h = _img.size
                self._um_per_px.value = round(5885.0 / h, 4)
            except Exception:
                pass

        import sys
        if "torch" in sys.modules:
            self._start_preload()
        self._load_originals_bg(paths, stubs)

    def _load_originals_bg(self, paths, stubs):
        """Load remaining images in background; drain queue on main thread via timer."""
        import threading
        import cv2

        # Queue holds (index, thumb_rgb_150px) — thumbnail built in background thread
        # so main thread only does QPixmap creation (fast on small array).
        thumb_queue = []
        done_count = [0]  # mutable box so _work can update it

        def _work():
            from gallery_tab import THUMB_SIZE as _THUMB_SIZE
            for i, p in enumerate(paths):
                if stubs is not self._results:
                    return
                img = cv2.imread(p)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    stubs[i]["original"] = rgb
                    # Resize directly — skip make_overlay (no mask yet, avoids
                    # full-res cvtColor before resize).
                    h, w = rgb.shape[:2]
                    scale = _THUMB_SIZE / max(h, w)
                    thumb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
                    thumb_queue.append((i, thumb))
                done_count[0] += 1

        def _drain():
            if stubs is not self._results:
                return
            if thumb_queue:
                idx, thumb_rgb = thumb_queue.pop(0)
                self._gallery.update_thumb_prebuilt(idx, thumb_rgb)
            if done_count[0] < len(paths) or thumb_queue:
                qt.QTimer.singleShot(0, _drain)

        if paths:
            threading.Thread(target=_work, daemon=True).start()
            qt.QTimer.singleShot(50, _drain)

    def _start_preload(self):
        """Kick off background model preload so Run Analysis starts instantly.

        Skip if models are not yet in local cache — the download dialog will
        run first, and importing huggingface_hub from two threads concurrently
        causes circular-import errors in Slicer's Python environment.

        Returns the Thread if started, None if skipped.
        """
        import threading
        from logic import preload_models
        model_data = self._model_combo.currentData
        if not model_data:
            return None
        body_file, body_enc, eye_file = model_data
        include_eyes = self._chk_eyes.isChecked()

        # Only preload if all required models are already locally cached.
        files_needed = [body_file]
        if include_eyes and eye_file:
            files_needed.append(eye_file)
        if not all(os.path.exists(_local_model_path(f)) for f in files_needed):
            return None  # download not done yet — skip preload, avoid concurrent HF import

        params = {
            "curvature":           True,  # always preload curvature
            "eyes":                include_eyes,
            "body_model_filename": body_file,
            "body_encoder_name":   body_enc,
            "eye_model_filename":  eye_file,
            "body_model_path":     _local_model_path(body_file),
        }
        if include_eyes and eye_file:
            params["eye_model_path"] = _local_model_path(eye_file)
        t = threading.Thread(target=preload_models, args=(params,), daemon=True)
        t.start()
        return t

    @staticmethod
    def _models_to_download(body_filename, eye_filename, include_eyes):
        """Return list of (repo_id, filename, label) not yet in local zebrafish cache."""
        REPO = "markdanielarndt/Zebrafish_Segmentation"
        candidates = [(REPO, body_filename, "body segmentation model")]
        if include_eyes and eye_filename:
            candidates.append((REPO, eye_filename, "eye segmentation model"))
        return [
            (repo_id, fn, label)
            for repo_id, fn, label in candidates
            if not os.path.exists(_local_model_path(fn))
        ]

    def _check_and_download_models(self, params):
        """
        Check local model cache. If any required models are missing, show a
        QProgressDialog and stream-download them in a background thread.

        On return, injects body_model_path / eye_model_path into params so
        logic.py loads from local file instead of re-downloading.

        Returns True when all models are ready, False if the user cancelled.
        """
        import threading
        import time as _time

        body_file    = params.get("body_model_filename", "best_model_body_3400_vgg19.pth")
        eye_file     = params.get("eye_model_filename")
        include_eyes = params.get("eyes", False)

        missing = self._models_to_download(body_file, eye_file, include_eyes)
        if not missing:
            # Already cached — inject local paths so logic skips HF download.
            params["body_model_path"] = _local_model_path(body_file)
            if include_eyes and eye_file:
                params["eye_model_path"] = _local_model_path(eye_file)
            return True

        # Build URLs without importing huggingface_hub — importing it in the main
        # thread leaves it partially initialised, breaking segmentation_models_pytorch's
        # subsequent import of it when analyse_images runs.  HF's public URL scheme is
        # stable; no auth needed for the public markdanielarndt repos.
        models_with_urls = [
            (f"https://huggingface.co/{repo_id}/resolve/main/{fn}", fn, label)
            for repo_id, fn, label in missing
        ]
        hf_headers = {}

        progress_state = {"done": 0, "total": 0, "label": missing[0][2],
                          "done_flag": False, "cancelled": False, "error": None}
        cancel_event = threading.Event()

        dlg = qt.QProgressDialog(
            f"Downloading models (first run only)\n{missing[0][2]}…",
            "Cancel",
            0, 1,  # 0/1 = empty bar; switched to 0/100 once we have byte count
            slicer.util.mainWindow(),
        )
        dlg.setValue(0)
        dlg.setWindowTitle("Downloading Models")
        dlg.setWindowModality(qt.Qt.ApplicationModal)
        dlg.setMinimumWidth(420)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()
        slicer.app.processEvents()

        thread = threading.Thread(
            target=_run_downloads,
            args=(models_with_urls, hf_headers, progress_state, cancel_event),
            daemon=True,
        )
        thread.start()
        t0 = _time.time()
        _determinate = False  # track whether we've switched to percentage mode

        while thread.is_alive() and not progress_state["done_flag"] and not progress_state["cancelled"]:
            if dlg.wasCanceled:
                cancel_event.set()
                progress_state["cancelled"] = True
                break

            done  = progress_state.get("done", 0)
            total = progress_state.get("total", 0)
            label = progress_state.get("label", "")
            elapsed = _time.time() - t0

            if total > 0:
                if not _determinate:
                    dlg.setRange(0, 100)  # switch from pulsing to percentage
                    _determinate = True
                pct = int(done / total * 100)
                dlg.setValue(pct)
                if pct > 2 and elapsed > 0:
                    eta_s = elapsed / pct * (100 - pct)
                    if eta_s >= 60:
                        eta_str = f"~{int(eta_s // 60)}m {int(eta_s % 60):02d}s left"
                    else:
                        eta_str = f"~{int(eta_s)}s left"
                    dlg.setLabelText(
                        f"Downloading models (first run only)\n{label}  {pct}%  ·  {eta_str}"
                    )
                else:
                    dlg.setLabelText(
                        f"Downloading models (first run only)\n{label}  {pct}%"
                    )
            elif done > 0:
                mb = done / 1024 / 1024
                dlg.setLabelText(
                    f"Downloading models (first run only)\n{label}  "
                    f"{mb:.1f} MB  ·  {int(elapsed)}s elapsed…"
                )
            else:
                dlg.setLabelText(
                    f"Downloading models (first run only)\n{label}  "
                    f"{int(elapsed)}s elapsed…"
                )

            slicer.app.processEvents()
            _time.sleep(0.2)

        thread.join(timeout=2.0)
        dlg.close()

        if progress_state.get("cancelled"):
            return False

        if progress_state.get("error"):
            slicer.util.errorDisplay(
                f"Model download failed:\n{progress_state['error']}\n\n"
                "Check your internet connection and try again."
            )
            return False

        # Inject local paths so logic.py loads from disk, skips HF download.
        params["body_model_path"] = _local_model_path(body_file)
        if include_eyes and eye_file:
            params["eye_model_path"] = _local_model_path(eye_file)
        return True

    def _on_detect_scale(self):
        from logic import detect_scalebar
        if not self._image_paths:
            self._scale_status.setText("Load images first.")
            return
        result = detect_scalebar(self._image_paths[0])
        if result.get("bar_found"):
            um_per_px = result.get("scale_um_per_px")
            bar_px = result.get("bar_length_px")
            if um_per_px is not None:
                self._um_per_px.value = um_per_px
                label_detected = result.get("label_um_detected")
                if label_detected is not None:
                    self._bar_um_edit.text = f"{label_detected:.0f}"
                self._scale_status.setText(f"Detected: {um_per_px:.4f} µm/px")
                self._scale_status.setStyleSheet("color: #4CAF50;")
            else:
                bar_info = f"  ({bar_px:.0f} px)" if bar_px is not None else ""
                self._scale_status.setText(
                    f"Bar found{bar_info}. Enter physical length (µm) + click Apply."
                )
                self._scale_status.setStyleSheet("color: #FFC107;")
        else:
            self._scale_status.setText(
                "No scalebar detected. Enter µm/px directly."
            )
            self._scale_status.setStyleSheet("color: #F44336;")

        if result.get("debug_img") is not None:
            self._detail.show_raw_image(
                result["debug_img"],
                result.get("message", ""),
            )
            self._tabs.setCurrentIndex(1)

    def _on_apply_scale(self):
        from logic import detect_scalebar
        text = self._bar_um_edit.text.strip()
        if not text or not self._image_paths:
            return
        try:
            label_um = float(text)
        except ValueError:
            self._scale_status.setText("Invalid value — enter a number.")
            return
        result = detect_scalebar(self._image_paths[0], label_um=label_um)
        if result.get("success"):
            self._um_per_px.value = result["scale_um_per_px"]
            self._scale_status.setText(
                f"Applied: {result['scale_um_per_px']:.4f} µm/px"
            )

    def _on_run(self):
        from logic import analyse_images

        if not self._image_paths:
            slicer.util.warningDisplay("No images loaded.")
            return

        model_data = self._model_combo.currentData
        body_file, body_enc, eye_file = model_data

        params = {
            "length":              self._chk_length.isChecked(),
            "curvature":           self._chk_curvature.isChecked(),
            "ratio":               self._chk_ratio.isChecked(),
            "eyes":                self._chk_eyes.isChecked(),
            "hitl":                self._chk_hitl.isChecked(),
            "threshold":           self._threshold_slider.value,
            "um_per_px":           self._um_per_px.value,
            "body_model_filename": body_file,
            "body_encoder_name":   body_enc,
            "eye_model_filename":  eye_file,
        }

        # Download any missing models before starting analysis.
        if not self._check_and_download_models(params):
            return  # user cancelled or download failed

        n = len(self._image_paths)
        import time as _time

        # Marquee: fixed-width chunk slides left→right via stylesheet margins.
        # Value pinned at 100 so the chunk always fills the bar; margins clip it
        # to ~25% width.  Text stays visible (indeterminate mode hides it on macOS).
        _mrq_chunk = (
            "QProgressBar{font-weight:bold;border-radius:3px;border:none;"
            "background:#3a3a3a;color:white;text-align:center;min-height:28px;}"
            "QProgressBar::chunk{background:#2e7d32;border-radius:2px;"
            "margin-left:%dpx;margin-right:%dpx;}"
        )
        _mrq_base = (
            "QProgressBar{font-weight:bold;border-radius:3px;border:none;"
            "background:#3a3a3a;color:white;text-align:center;min-height:28px;}"
            "QProgressBar::chunk{background:#2e7d32;border-radius:2px;}"
        )
        self._run_progress.setRange(0, 100)
        self._run_progress.setValue(100)
        self._run_progress.setFormat("Loading models…")
        self._run_stack.setCurrentIndex(1)
        slicer.app.processEvents()

        _mrq_t0 = _time.time()
        preload_t = self._start_preload()
        # Run until preload done AND at least 0.5 s elapsed so animation is
        # always visible, even when models were already in RAM.
        while True:
            _mrq_elapsed = _time.time() - _mrq_t0
            if _mrq_elapsed >= 0.5 and (preload_t is None or not preload_t.is_alive()):
                break
            slicer.app.processEvents()
            _time.sleep(0.05)
            bar_w = max(self._run_progress.width, 100)
            chunk_px = bar_w // 4
            # pos_px: -chunk_px (fully off left) → bar_w (fully off right), then wraps
            total_range = bar_w + chunk_px
            pos_px = int(_mrq_elapsed * (total_range / 2.5) % total_range) - chunk_px
            left_px = max(0, pos_px)
            right_px = max(0, bar_w - pos_px - chunk_px)
            self._run_progress.setStyleSheet(_mrq_chunk % (left_px, right_px))
        self._run_progress.setStyleSheet(_mrq_base)

        self._run_progress.setRange(0, n)
        self._run_progress.setValue(0)
        slicer.app.processEvents()

        _t0 = _time.time()

        def _set_btn_progress(i, total):
            self._run_progress.setValue(i)
            elapsed = _time.time() - _t0
            if i > 0 and total > i:
                remaining = elapsed / i * (total - i)
                eta = (f"~{int(remaining // 60)}m {int(remaining % 60):02d}s left"
                       if remaining >= 60 else f"~{int(remaining)}s left")
                self._run_progress.setFormat(f"Image {i} / {total}  ·  {eta}")
            else:
                self._run_progress.setFormat(f"Image {i} / {total}")
            slicer.app.processEvents()

        self._results = analyse_images(self._image_paths, params, _set_btn_progress)

        self._run_stack.setCurrentIndex(0)
        self._on_results_ready()

    def _get_correction_params(self):
        """Return current hitl/threshold settings for manual correction curvature recompute."""
        return {
            "hitl": self._chk_hitl.isChecked(),
            "threshold": float(self._threshold_slider.value),
        }

    def _on_gallery_select(self, index: int):
        self._current_detail_idx = index
        self._tabs.setCurrentIndex(1)
        self._detail.show_result(index, self._results)
        self._detail.setFocus()

    def _navigate_detail(self, delta: int):
        if not self._results:
            return
        idx = max(0, min(len(self._results) - 1, self._current_detail_idx + delta))
        if idx != self._current_detail_idx:
            self._on_gallery_select(idx)

    def _on_results_ready(self):
        self._detail.invalidate_cache()
        self._gallery.populate(self._results)
        self._results_tab.populate(self._results)
        self._exclude_tab.populate(self._results)
        self._tabs.setCurrentIndex(0)
        errors = [r for r in self._results if r.get("error")]
        if errors:
            msg = "\n".join(f"• {r['filename']}: {r['error']}" for r in errors)
            slicer.util.warningDisplay(f"Errors in {len(errors)} image(s):\n\n{msg}")

    def _on_export_excel(self):
        from export import export_excel
        if not self._results:
            slicer.util.warningDisplay("No results to export. Run analysis first.")
            return
        path = qt.QFileDialog.getSaveFileName(None, "Save Excel", "", "Excel (*.xlsx)")
        if path:
            if not path.endswith(".xlsx"):
                path += ".xlsx"
            active = [r for r in self._results if r["filename"] not in self._excluded]
            try:
                export_excel(active, path)
                slicer.util.infoDisplay(f"Saved {len(active)} rows to:\n{path}")
            except Exception as e:
                slicer.util.errorDisplay(f"Export failed:\n{e}")

    def _on_export_csv(self):
        from export import export_csv
        if not self._results:
            slicer.util.warningDisplay("No results to export. Run analysis first.")
            return
        path = qt.QFileDialog.getSaveFileName(None, "Save CSV", "", "CSV (*.csv)")
        if path:
            if not path.endswith(".csv"):
                path += ".csv"
            active = [r for r in self._results if r["filename"] not in self._excluded]
            try:
                export_csv(active, path)
                slicer.util.infoDisplay(f"Saved {len(active)} rows to:\n{path}")
            except Exception as e:
                slicer.util.errorDisplay(f"Export failed:\n{e}")
