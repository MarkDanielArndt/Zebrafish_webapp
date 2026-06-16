"""
Gallery tab — scrollable grid of result thumbnails.

Click a thumbnail -> emits index via on_select callback.
populate(results)  — rebuild grid from result list.
update_thumb(index, rgb_array) — update single thumbnail in-place.
"""

import os
import sys

# Ensure ZebrafishAnalysisLib is first on sys.path so 'overlay' resolves locally.
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
elif sys.path[0] != _LIB_DIR:
    sys.path.remove(_LIB_DIR)
    sys.path.insert(0, _LIB_DIR)

import qt
import numpy as np


THUMB_SIZE   = 150
BORDER_OK    = "2px solid #4CAF50"
BORDER_WARN  = "2px solid #FFC107"
BORDER_ERROR = "2px solid #F44336"
BORDER_LOADING = "2px solid #555555"


class _ClickableLabel(qt.QLabel):
    def __init__(self, idx, on_select, loaded=True):
        super().__init__()
        self._idx = idx
        self._on_select = on_select
        self._loaded = loaded

    def mousePressEvent(self, event):
        if self._loaded:
            self._on_select(self._idx)


def _numpy_to_qpixmap(rgb_array: np.ndarray) -> "qt.QPixmap":
    from PIL import Image as PILImage
    import io
    arr = np.ascontiguousarray(rgb_array.clip(0, 255).astype("uint8"))
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="BMP")
    data = qt.QByteArray(buf.getvalue())
    pixmap = qt.QPixmap()
    pixmap.loadFromData(data)
    return pixmap


class GalleryTab(qt.QWidget):
    def __init__(self, on_select):
        super().__init__()
        self._on_select = on_select
        self._thumbnails = []
        self._cells = []
        self._n_cols = 0

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)

        self._container = qt.QWidget()
        self._grid      = qt.QGridLayout(self._container)
        self._grid.setSpacing(6)
        scroll.setWidget(self._container)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(scroll)

    def populate(self, results: list) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbnails = []
        self._cells = []
        self._n_cols = 0

        from overlay import make_overlay

        for i in range(len(results)):
            r = results[i]
            loaded = r.get("original") is not None
            thumb_rgb = make_overlay(r, thumbnail_size=THUMB_SIZE)
            pixmap    = _numpy_to_qpixmap(thumb_rgb)

            label = _ClickableLabel(i, self._on_select, loaded=loaded)
            label.setPixmap(pixmap)
            label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            label.setScaledContents(True)

            if not loaded:
                border = BORDER_LOADING
            elif r.get("error"):
                border = BORDER_ERROR
            elif r.get("length") is None:
                border = BORDER_WARN
            else:
                border = BORDER_OK
            label.setStyleSheet(f"border: {border};")

            caption = qt.QLabel()
            caption.setFixedWidth(THUMB_SIZE)
            caption.setAlignment(qt.Qt.AlignCenter)
            caption.setWordWrap(False)
            caption.setStyleSheet("font-size: 10px;")
            caption.setToolTip(r["filename"])
            _elided = caption.fontMetrics().elidedText(
                r["filename"], qt.Qt.ElideRight, THUMB_SIZE
            )
            _mparts = []
            if r.get("error"):
                _mparts.append("ERROR")
            else:
                if r.get("length")    is not None: _mparts.append(f"{r['length']:.0f} µm")
                if r.get("curvature") is not None: _mparts.append(f"Cls {r['curvature']}")
            caption.setText(_elided + ("\n" + " | ".join(_mparts) if _mparts else ""))

            cell = qt.QWidget()
            cell_layout = qt.QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(2)
            cell_layout.addWidget(label)
            cell_layout.addWidget(caption)

            self._cells.append(cell)
            self._thumbnails.append(label)

        self._reflow()

    def update_thumb(self, index: int, rgb_array: np.ndarray) -> None:
        """Update a single thumbnail — builds thumb from full-res rgb on main thread."""
        if index >= len(self._thumbnails):
            return
        from overlay import make_overlay
        stub = {"original": rgb_array, "mask": None, "length": None, "error": None}
        thumb_rgb = make_overlay(stub, thumbnail_size=THUMB_SIZE)
        self.update_thumb_prebuilt(index, thumb_rgb)

    def update_thumb_prebuilt(self, index: int, thumb_rgb: np.ndarray) -> None:
        """Update a single thumbnail from a pre-built 150px RGB array (fast, no resize)."""
        if index >= len(self._thumbnails):
            return
        pixmap = _numpy_to_qpixmap(thumb_rgb)
        label = self._thumbnails[index]
        label.setPixmap(pixmap)
        label._loaded = True
        label.setStyleSheet(f"border: {BORDER_WARN};")

    def _reflow(self):
        if not self._cells:
            return
        spacing = self._grid.spacing
        cols = max(1, self.width // (THUMB_SIZE + spacing))
        if cols == self._n_cols:
            return
        if self._n_cols > 0:
            old_rows = (len(self._cells) - 1) // self._n_cols + 1
            self._grid.setRowStretch(old_rows, 0)
        self._n_cols = cols
        for i, cell in enumerate(self._cells):
            row, col = divmod(i, cols)
            self._grid.addWidget(cell, row, col)
        rows = (len(self._cells) - 1) // cols + 1
        self._grid.setRowStretch(rows, 1)

    def resizeEvent(self, event):
        self._reflow()
