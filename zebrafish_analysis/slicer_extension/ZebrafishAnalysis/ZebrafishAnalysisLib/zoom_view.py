"""
Zoomable, pannable image viewer for the ZebrafishAnalysis detail tab.

ZoomableImageView  — QGraphicsView subclass: zoom, pan, swipe nav, manual dots
_MinimapOverlay    — picture-in-picture thumbnail with viewport rect
"""

import os
import sys
import numpy as np

# Ensure ZebrafishAnalysisLib is first on sys.path (same pattern as detail_tab.py)
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
elif sys.path[0] != _LIB_DIR:
    sys.path.remove(_LIB_DIR)
    sys.path.insert(0, _LIB_DIR)

import qt


# ---------------------------------------------------------------------------
# Pure math helpers (no Qt — used for coordinate mapping and testing)
# ---------------------------------------------------------------------------

def _zoom_factor_from_delta(dy: int) -> float:
    """Compute zoom scale factor from a wheel event vertical delta."""
    return 1.15 ** (dy / 120.0)


def _minimap_viewport_rect(
    vis_x, vis_y, vis_w, vis_h,
    scene_w, scene_h,
    thumb_x, thumb_y,
    thumb_w, thumb_h,
):
    """Map visible scene rect → pixel rect inside the minimap thumbnail. Pure math."""
    if scene_w <= 0 or scene_h <= 0:
        return (thumb_x, thumb_y, thumb_w, thumb_h)
    sx = thumb_w / scene_w
    sy = thumb_h / scene_h
    rx = int(thumb_x + vis_x * sx)
    ry = int(thumb_y + vis_y * sy)
    rw = max(2, int(vis_w * sx))
    rh = max(2, int(vis_h * sy))
    return (rx, ry, rw, rh)


# ---------------------------------------------------------------------------
# _MinimapOverlay
# ---------------------------------------------------------------------------

class _MinimapOverlay(qt.QLabel):
    """Picture-in-picture minimap overlaid on ZoomableImageView.

    Shows a thumbnail of the full image with a cyan rectangle indicating
    the currently visible viewport region.

    Placement: bottom-left corner of parent view, 8 px margin.
    # TODO: detect scalebar corner from result dict, place minimap in
    #       the opposite corner (scalebar typically bottom-right).
    """

    _MAX_SIDE = 160  # longer side of minimap in px

    def __init__(self, parent):
        super().__init__(parent)
        self._W = self._MAX_SIDE
        self._H = self._MAX_SIDE
        self.setFixedSize(self._W, self._H)
        self.setStyleSheet(
            "background: rgba(26,26,26,220); border: 1px solid #555; border-radius: 3px;"
        )
        self._thumb = None        # QPixmap, pre-computed in set_thumbnail()
        self._thumb_x = 0         # thumbnail offset inside the minimap canvas
        self._thumb_y = 0
        self._thumb_w = self._W
        self._thumb_h = self._H
        self.setVisible(False)
        self.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)

    def set_thumbnail(self, full_pixmap: "qt.QPixmap") -> None:
        """Pre-compute thumbnail once when a new image is loaded.

        Resizes the minimap widget to match the image aspect ratio so there
        is no letterboxing — the thumbnail fills the widget exactly.
        """
        if full_pixmap.isNull():
            self._thumb = None
            return
        pw = full_pixmap.width()
        ph = full_pixmap.height()
        if pw >= ph:
            self._W = self._MAX_SIDE
            self._H = max(20, int(self._MAX_SIDE * ph / pw))
        else:
            self._H = self._MAX_SIDE
            self._W = max(20, int(self._MAX_SIDE * pw / ph))
        self.setFixedSize(self._W, self._H)
        # IgnoreAspectRatio: we already computed _W/_H to match the ratio,
        # so forcing exact fill avoids grey bars from integer rounding.
        self._thumb = full_pixmap.scaled(
            self._W, self._H,
            qt.Qt.IgnoreAspectRatio,
            qt.Qt.SmoothTransformation,
        )
        self._thumb_w = self._W
        self._thumb_h = self._H
        self._thumb_x = 0
        self._thumb_y = 0

    def update_viewport(self, visible_rect, full_rect) -> None:
        """Repaint with updated viewport rectangle.

        visible_rect, full_rect: QRectF in scene (image pixel) coordinates.
        """
        if self._thumb is None or self._thumb.isNull():
            return

        canvas = qt.QPixmap(self._W, self._H)
        canvas.fill(qt.QColor(26, 26, 26))

        painter = qt.QPainter(canvas)
        painter.drawPixmap(self._thumb_x, self._thumb_y, self._thumb)

        # Map viewport to minimap coords
        rx, ry, rw, rh = _minimap_viewport_rect(
            visible_rect.x(), visible_rect.y(),
            visible_rect.width(), visible_rect.height(),
            full_rect.width(), full_rect.height(),
            self._thumb_x, self._thumb_y,
            self._thumb_w, self._thumb_h,
        )
        painter.setPen(qt.QPen(qt.QColor(0, 200, 255), 2))
        painter.setBrush(qt.QBrush(qt.QColor(0, 200, 255, 35)))
        painter.drawRect(rx, ry, rw, rh)
        painter.end()

        self.setPixmap(canvas)


# ---------------------------------------------------------------------------
# ZoomableImageView
# ---------------------------------------------------------------------------

class ZoomableImageView(qt.QGraphicsView):
    """Zoomable, pannable image viewer.

    Scene coordinate space = original image pixel space:
        scene.x() == image column,  scene.y() == image row.

    Callbacks (set after construction):
        _tap_handler  : callable(row: int, col: int) | None
                        Called on left-click tap (< 5 px movement) in manual mode.
        _on_navigate  : callable(delta: int) | None
                        Called on horizontal two-finger swipe (+1 / -1).
    """

    _DRAG_THRESHOLD = 5    # pixels of movement before classifying as drag
    _ZOOM_MIN_MULT  = 1.0  # fit-to-view = minimum zoom
    _ZOOM_MAX_MULT  = 20.0 # 20× fit-to-view = maximum zoom

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene + pixmap item
        self._scene = qt.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = self._scene.addPixmap(qt.QPixmap())

        # Render settings
        self.setRenderHint(qt.QPainter.SmoothPixmapTransform, True)
        self.setDragMode(qt.QGraphicsView.NoDrag)         # manual pan
        self.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(qt.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qt.QGraphicsView.AnchorViewCenter)
        self.setStyleSheet("background: #1a1a1a; border: none;")
        self.setMinimumHeight(300)
        self.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)
        self.setFocusPolicy(qt.Qt.StrongFocus)

        # Image state
        self._orig_size = (0, 0)    # (height, width) of original image
        self._fit_scale = 1.0       # transform.m11() at fit-to-view

        # Manual correction state
        self._manual_mode = False
        self._dot_items = []        # list of QGraphicsEllipseItem

        # Pan state
        self._drag_start = None            # QPoint viewport coords at press
        self._drag_sb_origin = (0, 0)      # (h_value, v_value) at press
        self._is_dragging = False

        # Swipe navigation state
        self._swipe_locked = False

        # Public callbacks
        self._tap_handler = None    # callable(row, col)
        self._on_navigate = None    # callable(delta)

        # Overlay widgets
        self._minimap = _MinimapOverlay(self)

        # Placeholder label (shown when no image loaded)
        self._placeholder = qt.QLabel(self)
        self._placeholder.setText("Select an image from the Gallery.")
        self._placeholder.setAlignment(qt.Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #666; font-size: 13px; background: transparent;")
        self._placeholder.setVisible(True)
        self._placeholder.adjustSize()

        # Enable pinch gesture (best-effort; falls back gracefully if not supported)
        self.grabGesture(qt.Qt.PinchGesture)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pixmap(self, pixmap: "qt.QPixmap", reset_zoom: bool = True) -> None:
        """Load a new image. reset_zoom=True on navigation, False on overlay rebuild.

        Only pass reset_zoom=False when the image pixel dimensions are identical
        to the currently loaded image (e.g. rebuilding overlay after manual correction).
        Passing False for a differently-sized image leaves zoom/minimap scale stale.
        """
        self._pix_item.setPixmap(pixmap)
        rect = self._pix_item.boundingRect()
        self._scene.setSceneRect(rect)
        self._orig_size = (int(rect.height()), int(rect.width()))
        self._placeholder.setVisible(False)
        self._minimap.set_thumbnail(pixmap)
        self._reposition_minimap()   # size may have changed to match image aspect ratio
        if reset_zoom:
            self._reset_zoom()
            self.clear_dots()
        self._update_minimap()

    def show_placeholder(self, text: str = "") -> None:
        """Show placeholder text, clear scene (used for 'Loading…' and initial state)."""
        self._pix_item.setPixmap(qt.QPixmap())
        self._scene.setSceneRect(qt.QRectF(0, 0, 1, 1))
        self._orig_size = (0, 0)
        self._placeholder.setText(text if text else "Select an image from the Gallery.")
        self._placeholder.adjustSize()
        self._placeholder.setVisible(True)
        self._minimap.setVisible(False)
        self.clear_dots()
        self._reposition_placeholder()

    def clear(self) -> None:
        """Clear scene and hide minimap."""
        self.show_placeholder("")

    def set_manual_mode(self, enabled: bool) -> None:
        """Enable/disable tap-to-place-point mode."""
        self._manual_mode = enabled
        self._update_cursor()

    def add_dot(self, row: int, col: int, color: "qt.QColor") -> None:
        """Add a correction dot at (row, col) in original image coords."""
        pen = qt.QPen(color, 3)
        brush = qt.QBrush(qt.QColor(color.red(), color.green(), color.blue(), 180))
        item = self._scene.addEllipse(col - 8, row - 8, 16, 16, pen, brush)
        self._dot_items.append(item)

    def clear_dots(self) -> None:
        """Remove all correction dots from scene."""
        for item in self._dot_items:
            self._scene.removeItem(item)
        self._dot_items = []

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _reset_zoom(self) -> None:
        """Fit image to view and record fit scale."""
        if self._orig_size[0] == 0:
            return
        self.fitInView(self._scene.sceneRect, qt.Qt.KeepAspectRatio)
        self._fit_scale = self.transform().m11()
        self._minimap.setVisible(False)

    def _apply_zoom(self, factor: float) -> None:
        """Apply a zoom factor, clamped to [fit_scale, 20×fit_scale]."""
        if self._orig_size[0] == 0 or self._fit_scale <= 0:
            return
        current = self.transform().m11()
        new_scale = np.clip(
            current * factor,
            self._fit_scale * self._ZOOM_MIN_MULT,
            self._fit_scale * self._ZOOM_MAX_MULT,
        )
        actual_factor = new_scale / current
        self.scale(actual_factor, actual_factor)
        # Show minimap only when zoomed in
        at_fit = abs(self.transform().m11() - self._fit_scale) < self._fit_scale * 0.02
        self._minimap.setVisible(not at_fit)
        if not at_fit:
            self._update_minimap()

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        dx = event.angleDelta().x()
        dy = event.angleDelta().y()

        if abs(dx) > abs(dy):
            # Horizontal swipe → navigate images
            if self._swipe_locked:
                if abs(dx) < 8:
                    self._swipe_locked = False
            else:
                if abs(dx) > 20 and self._on_navigate:
                    self._swipe_locked = True
                    self._on_navigate(-1 if dx > 0 else 1)
            event.accept()
        else:
            # Vertical scroll → zoom
            if self._orig_size[0] == 0:
                event.accept()
                return
            self._apply_zoom(_zoom_factor_from_delta(dy))
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == qt.Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_sb_origin = (
                self.horizontalScrollBar().value,
                self.verticalScrollBar().value,
            )
            self._is_dragging = False
        qt.QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and (event.buttons() & qt.Qt.LeftButton):
            delta = event.pos() - self._drag_start
            if not self._is_dragging:
                if abs(delta.x()) > self._DRAG_THRESHOLD or abs(delta.y()) > self._DRAG_THRESHOLD:
                    self._is_dragging = True
                    self.setCursor(qt.Qt.ClosedHandCursor)
            if self._is_dragging:
                self.horizontalScrollBar().setValue(self._drag_sb_origin[0] - delta.x())
                self.verticalScrollBar().setValue(self._drag_sb_origin[1] - delta.y())
                self._update_minimap()
        qt.QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == qt.Qt.LeftButton:
            if not self._is_dragging:
                self._handle_tap(event.x(), event.y())
            self._drag_start = None
            self._is_dragging = False
            self._update_cursor()
        qt.QGraphicsView.mouseReleaseEvent(self, event)

    def keyPressEvent(self, event):
        """Arrow keys → navigate images."""
        if self._on_navigate:
            if event.key() == qt.Qt.Key_Right:
                self._on_navigate(1)
                return
            if event.key() == qt.Qt.Key_Left:
                self._on_navigate(-1)
                return
        qt.QGraphicsView.keyPressEvent(self, event)

    def event(self, event):
        """Route pinch gesture events."""
        if event.type() == qt.QEvent.Gesture:
            pinch = event.gesture(qt.Qt.PinchGesture)
            if pinch:
                state = pinch.state
                if state == qt.Qt.GestureStarted:
                    self.setTransformationAnchor(qt.QGraphicsView.AnchorViewCenter)
                self._apply_zoom(pinch.scaleFactor())
                if state == qt.Qt.GestureFinished:
                    self.setTransformationAnchor(qt.QGraphicsView.AnchorUnderMouse)
            return True
        return qt.QGraphicsView.event(self, event)

    def scrollContentsBy(self, dx, dy):
        qt.QGraphicsView.scrollContentsBy(self, dx, dy)
        self._update_minimap()

    def resizeEvent(self, event):
        qt.QGraphicsView.resizeEvent(self, event)
        if self._orig_size[0] > 0:
            # Re-fit only if already at fit scale (don't reset user's zoom on resize)
            current = self.transform().m11()
            if self._fit_scale > 0 and abs(current - self._fit_scale) < self._fit_scale * 0.02:
                self._reset_zoom()
            self._reposition_minimap()
            self._update_minimap()
        self._reposition_placeholder()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_tap(self, vx: int, vy: int) -> None:
        """Convert a viewport tap to (row, col) in original image space and call handler."""
        if not self._manual_mode or self._tap_handler is None:
            return
        if self._orig_size[0] == 0:
            return
        scene_pos = self.mapToScene(vx, vy)
        h, w = self._orig_size
        row = int(np.clip(scene_pos.y(), 0, h - 1))
        col = int(np.clip(scene_pos.x(), 0, w - 1))
        self._tap_handler(row, col)

    def _update_cursor(self) -> None:
        if self._manual_mode:
            self.setCursor(qt.Qt.CrossCursor)
        else:
            self.setCursor(qt.Qt.ArrowCursor)

    def _update_minimap(self) -> None:
        if not self._minimap.isVisible():
            return
        visible = self.mapToScene(self.viewport().rect).boundingRect()
        full = self._scene.sceneRect
        self._minimap.update_viewport(visible, full)

    def _reposition_minimap(self) -> None:
        """Position minimap bottom-left with 8 px margin."""
        # TODO: detect scalebar corner from result dict, place minimap in opposite corner
        # Use _W/_H (our computed values) rather than Qt widget properties, which may be
        # stale before setFixedSize is processed by the event loop.
        mw = self._minimap._W
        mh = self._minimap._H
        self._minimap.move(8, self.height - mh - 8)
        self._minimap.raise_()

    def _reposition_placeholder(self) -> None:
        pw = self._placeholder.width
        ph = self._placeholder.height
        self._placeholder.move(
            (self.width - pw) // 2,
            (self.height - ph) // 2,
        )
        self._placeholder.raise_()
