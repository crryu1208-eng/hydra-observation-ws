"""Camera panel: the RGB and depth images of one recorded frame, with a scrubber over the trajectory."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets

from . import model

TURBO = (colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def load_depth(path) -> np.ndarray | None:
    """Depth image in meters (float32 TIFF from the observation recorder; 16-bit PNG/TIFF taken as mm)."""
    try:
        with Image.open(path) as im:
            d = np.asarray(im)
    except (OSError, ValueError):
        return None
    if d.ndim == 3:
        d = d[..., 0]
    if np.issubdtype(d.dtype, np.integer):
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32)


def colorize_depth(d: np.ndarray, max_m: float) -> QtGui.QImage:
    valid = np.isfinite(d) & (d > 0)
    idx = np.clip(np.where(valid, d, 0) / max(max_m, 1e-3) * 255, 0, 255).astype(np.uint8)
    rgb = TURBO[idx]
    rgb[~valid] = 0
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class ImagePane(QtWidgets.QLabel):
    """Shows a pixmap scaled to fit, keeps a crosshair and reports the hovered image pixel."""

    hovered = QtCore.Signal(int, int)  # u, v in image pixels (-1, -1 when outside)

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #1e2124; color: #9aa5ad;")
        self._pm: QtGui.QPixmap | None = None
        self._img_size = QtCore.QSize()
        self.cross: tuple[int, int] | None = None
        self.setText(f"{title}: no frame")

    def set_image(self, img: QtGui.QImage | None, missing: str = "not found"):
        self._pm = QtGui.QPixmap.fromImage(img) if img is not None and not img.isNull() else None
        self._img_size = img.size() if self._pm else QtCore.QSize()
        if self._pm is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText(f"{self.title}: {missing}")
        self.update()

    def _target(self) -> QtCore.QRect | None:
        if self._pm is None:
            return None
        s = self._img_size.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        return QtCore.QRect((self.width() - s.width()) // 2, (self.height() - s.height()) // 2, s.width(), s.height())

    def paintEvent(self, e):
        r = self._target()
        if r is None:
            return super().paintEvent(e)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1e2124"))
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.drawPixmap(r, self._pm)
        if self.cross is not None:
            sx = r.x() + (self.cross[0] + 0.5) * r.width() / self._img_size.width()
            sy = r.y() + (self.cross[1] + 0.5) * r.height() / self._img_size.height()
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
            p.drawLine(QtCore.QPointF(sx - 8, sy), QtCore.QPointF(sx + 8, sy))
            p.drawLine(QtCore.QPointF(sx, sy - 8), QtCore.QPointF(sx, sy + 8))
        p.setPen(QtGui.QColor("#e8ecef"))
        p.drawText(r.adjusted(6, 4, -6, -4), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, self.title)
        p.end()

    def mouseMoveEvent(self, e):
        r = self._target()
        pos = e.position() if hasattr(e, "position") else e.pos()
        if r is None or not r.contains(pos.toPoint()):
            self.hovered.emit(-1, -1)
            return
        u = int((pos.x() - r.x()) * self._img_size.width() / r.width())
        v = int((pos.y() - r.y()) * self._img_size.height() / r.height())
        self.hovered.emit(min(u, self._img_size.width() - 1), min(v, self._img_size.height() - 1))

    def leaveEvent(self, e):
        self.hovered.emit(-1, -1)
        super().leaveEvent(e)


class CameraPanel(QtWidgets.QWidget):
    """RGB + depth of the current frame; prev/next/play and a slider over all frames in time order."""

    frame_changed = QtCore.Signal(int)  # frame id, emitted when the user moves to another frame
    select_node = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.obs: model.Observations | None = None
        self.frame: model.Frame | None = None
        self._depth: np.ndarray | None = None
        self._t0 = 0

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        imgs = QtWidgets.QHBoxLayout()
        self.rgb = ImagePane("RGB")
        self.depth = ImagePane("Depth")
        for pane in (self.rgb, self.depth):
            pane.hovered.connect(self._on_hover)
            imgs.addWidget(pane, 1)
        lay.addLayout(imgs, 1)

        bar = QtWidgets.QHBoxLayout()
        style = self.style()
        self.prev = QtWidgets.QToolButton()
        self.prev.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward))
        self.prev.setToolTip("Previous frame (,)")
        self.play = QtWidgets.QToolButton()
        self.play.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play.setCheckable(True)
        self.play.setToolTip("Play along the trajectory (Space)")
        self.next = QtWidgets.QToolButton()
        self.next.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaSeekForward))
        self.next.setToolTip("Next frame (.)")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.pos_label = QtWidgets.QLabel("–")
        self.pos_label.setMinimumWidth(150)
        self.max_depth = QtWidgets.QDoubleSpinBox()
        self.max_depth.setRange(0.5, 20)
        self.max_depth.setSingleStep(0.5)
        self.max_depth.setValue(5.0)
        self.max_depth.setSuffix(" m")
        self.max_depth.setToolTip("Depth color scale: 0 m (blue) to this value (red); black = no depth")
        self.node_btn = QtWidgets.QPushButton("Node")
        self.node_btn.setToolTip("Select the trajectory (agent) node this frame is attached to")
        self.node_btn.setEnabled(False)
        for w in (self.prev, self.play, self.next):
            bar.addWidget(w)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.pos_label)
        bar.addWidget(QtWidgets.QLabel("Depth max"))
        bar.addWidget(self.max_depth)
        bar.addWidget(self.node_btn)
        lay.addLayout(bar)
        self.info = QtWidgets.QLabel("Click a camera position on the trajectory in the 3D view, "
                                     "or a frame in the inspector.")
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.info)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(66)  # ~15 fps, the recording rate
        self.timer.timeout.connect(lambda: self.step(1, wrap=False))
        self.prev.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))
        self.play.toggled.connect(self._on_play)
        self.slider.valueChanged.connect(self._on_slider)
        self.max_depth.valueChanged.connect(lambda _: self._render_depth())
        self.node_btn.clicked.connect(self._on_node)
        for key, fn in ((",", lambda: self.step(-1)), (".", lambda: self.step(1)),
                        ("Space", lambda: self.play.toggle())):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ---- data -----------------------------------------------------------------------------------
    def set_observations(self, obs: model.Observations | None):
        self.play.setChecked(False)
        self.obs = obs
        self.frame = None
        self._depth = None
        self.rgb.set_image(None, "no frame")
        self.depth.set_image(None, "no frame")
        n = len(obs.ordered) if obs else 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        for w in (self.slider, self.prev, self.next, self.play):
            w.setEnabled(n > 0)
        self._t0 = obs.frames[obs.ordered[0]].timestamp_ns if n else 0
        if obs is not None and obs.intrinsics and obs.intrinsics.max_range:
            self.max_depth.setValue(float(obs.intrinsics.max_range) + 1.0)
        self.pos_label.setText(f"{n} frames" if n else "no recorded frames")
        self.info.setText("Click a camera position on the trajectory in the 3D view, or a frame in the inspector."
                          if n else "No observations/ folder next to this scene graph: no RGB / depth to show.")
        self.node_btn.setEnabled(False)

    def show_frame(self, frame_id: int):
        if self.obs is None or frame_id not in self.obs.frames:
            return
        f = self.obs.frames[frame_id]
        if self.frame is not None and self.frame.frame_id == frame_id:
            return
        self.frame = f
        img = QtGui.QImage(str(f.rgb_path)) if f.rgb_path and f.rgb_path.exists() else None
        self.rgb.set_image(img, "image not found")
        self._depth = load_depth(f.depth_path) if f.depth_path and f.depth_path.exists() else None
        self._render_depth()
        idx = self.obs.ordered.index(frame_id) if frame_id in self.obs.ordered else 0
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.pos_label.setText(f"{idx + 1} / {len(self.obs.ordered)}   t = {(f.timestamp_ns - self._t0) / 1e9:.2f} s")
        self.node_btn.setEnabled(f.node_id is not None)
        self.node_btn.setText(f"Node {f.mapping.get('node_symbol', '')}".strip())
        self._update_info()

    def _render_depth(self):
        if self.frame is None:
            return
        if self._depth is None:
            self.depth.set_image(None, "depth not found")
        else:
            self.depth.set_image(colorize_depth(self._depth, self.max_depth.value()))

    def _update_info(self, u: int = -1, v: int = -1):
        f = self.frame
        if f is None:
            return
        mp = f.mapping or {}
        bits = [f"frame {f.frame_id}", f.rgb_path.name if f.rgb_path else "no rgb"]
        if mp.get("node_symbol"):
            m = f"→ {mp['node_symbol']} ({mp.get('type', '?')}"
            if isinstance(mp.get("distance_m"), (int, float)):
                m += f", {mp['distance_m']:.2f} m"
            if isinstance(mp.get("angle_rad"), (int, float)):
                m += f", {np.degrees(mp['angle_rad']):.0f}°"
            bits.append(m + ")")
        if f.position is not None:
            bits.append("body ({:.2f}, {:.2f}, {:.2f}) m".format(*f.position))
        if self._depth is not None:
            valid = self._depth[np.isfinite(self._depth) & (self._depth > 0)]
            if valid.size:
                bits.append(f"depth {valid.min():.2f}–{np.percentile(valid, 99):.2f} m, "
                            f"{100 * valid.size / self._depth.size:.0f}% valid")
        if u >= 0 and v >= 0:
            px = f"pixel ({u}, {v})"
            if self._depth is not None and v < self._depth.shape[0] and u < self._depth.shape[1]:
                z = float(self._depth[v, u])
                px += f" depth {z:.3f} m" if np.isfinite(z) and z > 0 else " no depth"
            bits.append(px)
        self.info.setText("   ·   ".join(bits))

    # ---- interaction --------------------------------------------------------------------------
    def _on_hover(self, u, v):
        cross = (u, v) if u >= 0 else None
        for pane in (self.rgb, self.depth):
            pane.cross = cross
            pane.update()
        self._update_info(u, v)

    def step(self, delta: int, wrap: bool = False):
        if self.obs is None or not self.obs.ordered:
            return
        n = len(self.obs.ordered)
        i = self.slider.value() + delta if self.frame is not None else 0
        if not 0 <= i < n:
            if not wrap:
                self.play.setChecked(False)
                return
            i %= n
        self._go(i)

    def _on_slider(self, i):
        self._go(i)

    def _go(self, i):
        fid = self.obs.ordered[i]
        self.show_frame(fid)
        self.frame_changed.emit(fid)

    def _on_play(self, on):
        self.play.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPause if on else QtWidgets.QStyle.SP_MediaPlay))
        if on and self.obs and self.frame is not None and self.slider.value() >= len(self.obs.ordered) - 1:
            self._go(0)
        (self.timer.start if on else self.timer.stop)()

    def _on_node(self):
        if self.frame is not None and self.frame.node_id is not None:
            self.select_node.emit(self.frame.node_id)
