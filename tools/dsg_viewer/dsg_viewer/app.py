"""Qt application: open a Hydra scene graph and navigate its layers, nodes, edges, mesh,
trajectory and the RGB frames the observation recorder attached to each agent node."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

if os.environ.get("WAYLAND_DISPLAY") and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # VTK renders through X11; use XWayland on Wayland sessions
os.environ.setdefault("QT_API", "pyside6")

import numpy as np  # noqa: E402
from pyvistaqt import QtInteractor  # noqa: E402
from qtpy import QtCore, QtGui, QtWidgets  # noqa: E402

from . import model  # noqa: E402
from .scene import DEFAULT_STYLE, GROUP_STYLE, SceneView, label_color  # noqa: E402

ID_ROLE = QtCore.Qt.UserRole
GROUP_ROLE = QtCore.Qt.UserRole + 1
THUMB = QtCore.QSize(128, 96)


def fmt_value(v, depth=0) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        if len(v) <= 4 and all(not isinstance(x, (list, dict)) for x in v):
            return "[" + ", ".join(fmt_value(x) for x in v) + "]"
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return "{…}" if v else "{}"
    return str(v)


def swatch(color) -> QtGui.QIcon:
    pm = QtGui.QPixmap(12, 12)
    pm.fill(QtGui.QColor(color) if isinstance(color, str) else QtGui.QColor.fromRgbF(*color))
    return QtGui.QIcon(pm)


class Inspector(QtWidgets.QWidget):
    """Attributes, connections and recorded frames of the selected node."""

    navigate = QtCore.Signal(int)
    frame_selected = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.title = QtWidgets.QLabel("Select a node in the tree or click it in the 3D view.")
        self.title.setWordWrap(True)
        f = self.title.font()
        f.setPointSizeF(f.pointSizeF() * 1.25)
        f.setBold(True)
        self.title.setFont(f)
        self.sub = QtWidgets.QLabel("")
        self.sub.setWordWrap(True)
        self.sub.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        btns = QtWidgets.QHBoxLayout()
        self.fly = QtWidgets.QPushButton("Fly to")
        self.copy = QtWidgets.QPushButton("Copy JSON")
        btns.addWidget(self.fly)
        btns.addWidget(self.copy)
        btns.addStretch(1)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)
        lay.addLayout(btns)

        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)
        self.attrs = QtWidgets.QTreeWidget()
        self.attrs.setHeaderLabels(["Attribute", "Value"])
        self.attrs.setColumnWidth(0, 170)
        self.attrs.setUniformRowHeights(True)
        tabs.addTab(self.attrs, "Attributes")

        self.links = QtWidgets.QTreeWidget()
        self.links.setHeaderLabels(["Node", "Layer", "Distance"])
        self.links.setRootIsDecorated(False)
        self.links.itemActivated.connect(self._on_link)
        self.links.itemClicked.connect(self._on_link)
        tabs.addTab(self.links, "Connections")

        frames_tab = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(frames_tab)
        fl.setContentsMargins(0, 0, 0, 0)
        self.frames_info = QtWidgets.QLabel("")
        self.frames_info.setWordWrap(True)
        self.frames = QtWidgets.QListWidget()
        self.frames.setViewMode(QtWidgets.QListView.IconMode)
        self.frames.setIconSize(THUMB)
        self.frames.setResizeMode(QtWidgets.QListView.Adjust)
        self.frames.setUniformItemSizes(True)
        self.frames.setMovement(QtWidgets.QListView.Static)
        self.frames.currentItemChanged.connect(self._on_frame)
        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.frame_meta = QtWidgets.QLabel("")
        self.frame_meta.setWordWrap(True)
        self.frame_meta.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.frames)
        pv_box = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pv_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.preview, 1)
        pl.addWidget(self.frame_meta)
        split.addWidget(pv_box)
        split.setSizes([200, 320])
        fl.addWidget(self.frames_info)
        fl.addWidget(split, 1)
        self.frames_tab_index = tabs.addTab(frames_tab, "Frames")
        self.tabs = tabs

        self._pending: list[tuple[QtWidgets.QListWidgetItem, model.Frame]] = []
        self._thumb_timer = QtCore.QTimer(self)
        self._thumb_timer.setInterval(0)
        self._thumb_timer.timeout.connect(self._load_thumbs)
        self.copy.clicked.connect(self._copy_json)
        self._node = None
        self._preview_pm = None

    # ---- content ------------------------------------------------------------------------
    def show_node(self, g: model.SceneGraph, n: model.Node | None):
        self._node = n
        self.attrs.clear()
        self.links.clear()
        self.frames.clear()
        self._pending.clear()
        self.preview.clear()
        self._preview_pm = None
        self.frame_meta.clear()
        self.fly.setEnabled(n is not None)
        self.copy.setEnabled(n is not None)
        if n is None:
            self.title.setText("Select a node in the tree or click it in the 3D view.")
            self.sub.setText("")
            self.frames_info.setText("")
            return
        self.title.setText(n.title)
        p = n.position
        part = ""
        if n.partition:
            part = f", partition '{chr(n.partition)}'" if 32 < n.partition < 127 else f", partition {n.partition}"
        bits = [f"{n.group} · layer {n.layer}{part}", n.type,
                f"id {n.id}", f"position ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m"]
        if n.timestamp_ns:
            bits.append(f"timestamp {n.timestamp_ns}")
        self.sub.setText("\n".join(bits))

        def add(parent, key, value):
            item = QtWidgets.QTreeWidgetItem([str(key), fmt_value(value)])
            item.setToolTip(1, json.dumps(value)[:2000] if not isinstance(value, str) else value[:2000])
            (parent.addChild(item) if parent is not None else self.attrs.addTopLevelItem(item))
            if isinstance(value, dict):
                for k, v in value.items():
                    add(item, k, v)
            elif isinstance(value, list) and fmt_value(value).startswith("[") and "items" in fmt_value(value):
                for i, v in enumerate(value[:200]):
                    add(item, i, v)
                if len(value) > 200:
                    item.addChild(QtWidgets.QTreeWidgetItem(["…", f"{len(value) - 200} more"]))

        for k in sorted(n.attrs):
            add(None, k, n.attrs[k])

        for m in sorted(g.neighbors.get(n.id, []), key=lambda i: (g.nodes[i].group, i)):
            o = g.nodes[m]
            d = float(np.linalg.norm(o.position - n.position))
            it = QtWidgets.QTreeWidgetItem([o.title, o.group, f"{d:.2f} m"])
            it.setData(0, ID_ROLE, o.id)
            it.setIcon(0, swatch(GROUP_STYLE.get(o.group, DEFAULT_STYLE)[0]))
            self.links.addTopLevelItem(it)
        self.links.resizeColumnToContents(0)
        self.tabs.setTabText(1, f"Connections ({self.links.topLevelItemCount()})")

        obs = g.observations
        frames = obs.frames_of(n.id) if obs else []
        self.tabs.setTabText(self.frames_tab_index, f"Frames ({len(frames)})")
        if obs is None:
            self.frames_info.setText("No observations/ folder found next to this scene graph.")
        elif n.group != "Agents":
            self.frames_info.setText("Recorded frames are attached to agent (trajectory) nodes.")
        else:
            self.frames_info.setText(f"{len(frames)} RGB frames mapped to {n.symbol}. Click one to preview it "
                                     "and mark its camera position in 3D.")
        for f in frames:
            it = QtWidgets.QListWidgetItem(f"#{f.frame_id}")
            it.setData(ID_ROLE, f.frame_id)
            it.setSizeHint(QtCore.QSize(THUMB.width() + 12, THUMB.height() + 24))
            self.frames.addItem(it)
            self._pending.append((it, f))
        if self._pending:
            self._thumb_timer.start()

    def _load_thumbs(self):
        for _ in range(6):
            if not self._pending:
                self._thumb_timer.stop()
                return
            it, f = self._pending.pop(0)
            if f.rgb_path and f.rgb_path.exists():
                reader = QtGui.QImageReader(str(f.rgb_path))
                reader.setScaledSize(THUMB)
                img = reader.read()
                if not img.isNull():
                    it.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(img)))

    def _on_frame(self, cur, _prev=None):
        if cur is None or self._node is None:
            return
        win = self.window()
        g = getattr(win, "graph", None)
        if g is None or g.observations is None:
            return
        f = g.observations.frames.get(cur.data(ID_ROLE))
        if f is None:
            return
        self._preview_pm = QtGui.QPixmap(str(f.rgb_path)) if f.rgb_path and f.rgb_path.exists() else None
        self._fit_preview()
        mp = f.mapping or {}
        t0 = self._node.timestamp_ns or f.timestamp_ns
        lines = [f"frame {f.frame_id} · {f.rgb_path.name if f.rgb_path else 'no rgb'}",
                 f"t = {f.timestamp_ns} ({(f.timestamp_ns - t0) / 1e9:+.2f} s from node)",
                 f"mapping: {mp.get('type', '?')} / {mp.get('status', '?')}"
                 + (f", {mp['distance_m']:.2f} m" if isinstance(mp.get('distance_m'), (int, float)) else "")
                 + (f", {np.degrees(mp['angle_rad']):.0f}°" if isinstance(mp.get('angle_rad'), (int, float)) else "")]
        if f.position is not None:
            lines.append(f"camera body at ({f.position[0]:.2f}, {f.position[1]:.2f}, {f.position[2]:.2f}) m")
        self.frame_meta.setText("\n".join(lines))
        self.frame_selected.emit(f)

    def _fit_preview(self):
        if self._preview_pm is None or self._preview_pm.isNull():
            self.preview.setText("image not found")
            return
        self.preview.setPixmap(self._preview_pm.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio,
                                                       QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._preview_pm is not None:
            self._fit_preview()

    def _on_link(self, item, _col=0):
        nid = item.data(0, ID_ROLE)
        if nid is not None:
            self.navigate.emit(int(nid))

    def _copy_json(self):
        if self._node is not None:
            QtWidgets.QApplication.clipboard().setText(json.dumps(
                {"id": self._node.id, "layer": self._node.layer, "partition": self._node.partition,
                 "attributes": self._node.attrs}, indent=2))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSG Viewer")
        self.resize(1500, 920)
        self.graph: model.SceneGraph | None = None
        self._tree_items: dict[int, QtWidgets.QTreeWidgetItem] = {}

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#f4f6f7", top="#dfe5ea")
        self.setCentralWidget(self.plotter)
        self.scene = SceneView(self.plotter)
        self.plotter.add_axes()

        # left: scene graph tree
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search nodes (symbol, class, id)…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Layer / node", "Count"])
        self.tree.setColumnWidth(0, 190)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._on_tree_current)
        self.tree.itemChanged.connect(self._on_tree_check)
        ll.addWidget(self.search)
        ll.addWidget(self.tree, 1)
        hint = QtWidgets.QLabel("Checkboxes show or hide a layer. Click a node to select it.")
        hint.setWordWrap(True)
        hint.setEnabled(False)  # rendered in the palette's disabled (secondary) text color
        ll.addWidget(hint)
        dock = QtWidgets.QDockWidget("Scene graph", self)
        dock.setObjectName("graph")
        dock.setWidget(left)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # right: inspector
        self.inspector = Inspector()
        self.inspector.navigate.connect(lambda nid: self.select_node(nid, fly=True))
        self.inspector.fly.clicked.connect(lambda: self.scene.selected is not None and self.scene.focus(self.scene.selected))
        self.inspector.frame_selected.connect(self._mark_frame)
        rdock = QtWidgets.QDockWidget("Inspector", self)
        rdock.setObjectName("inspector")
        rdock.setWidget(self.inspector)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, rdock)
        self.resizeDocks([dock, rdock], [300, 420], QtCore.Qt.Horizontal)

        self._build_toolbar()
        self._build_menu(dock, rdock)
        self.statusBar().showMessage("Open a scene graph: File ▸ Open (dsg.json, dsg_with_mesh.json or a session folder)")

        # click-to-pick (ignore drags)
        self._press = None
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        self.setAcceptDrops(True)

    # ---- UI construction ------------------------------------------------------------------
    def _build_menu(self, *docks):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Open scene graph…", self.open_file_dialog)
        a.setShortcut(QtGui.QKeySequence.Open)
        m.addAction("Open &session folder…", self.open_dir_dialog)
        m.addAction("&Reload", lambda: self.graph and self.open_path(self.graph.path)).setShortcut("F5")
        m.addAction("Save &screenshot…", self.save_screenshot)
        m.addSeparator()
        m.addAction("&Quit", self.close).setShortcut(QtGui.QKeySequence.Quit)
        v = self.menuBar().addMenu("&View")
        for d in docks:
            v.addAction(d.toggleViewAction())
        v.addSeparator()
        v.addAction("Reset camera", self.reset_view).setShortcut("R")
        v.addAction("Top view", self.top_view).setShortcut("T")
        v.addAction("Fly to selection", lambda: self.scene.selected is not None and self.scene.focus(self.scene.selected)).setShortcut("F")
        v.addAction("Clear selection", lambda: self.select_node(None)).setShortcut("Esc")

    def _build_toolbar(self):
        tb = self.addToolBar("Display")
        tb.setObjectName("display")
        tb.addAction("Open…", self.open_file_dialog)
        tb.addSeparator()
        tb.addAction("Reset view", self.reset_view)
        tb.addAction("Top view", self.top_view)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Mesh "))
        self.mesh_mode = QtWidgets.QComboBox()
        self.mesh_mode.addItems(["RGB", "Semantic", "Hidden"])
        self.mesh_mode.currentTextChanged.connect(lambda t: self._set_opt(mesh_mode=t.lower()))
        tb.addWidget(self.mesh_mode)
        self.mesh_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mesh_alpha.setRange(10, 100)
        self.mesh_alpha.setValue(100)
        self.mesh_alpha.setFixedWidth(80)
        self.mesh_alpha.setToolTip("Mesh opacity")
        self.mesh_alpha.sliderReleased.connect(lambda: self._set_opt(mesh_opacity=self.mesh_alpha.value() / 100))
        tb.addWidget(self.mesh_alpha)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Layer offset "))
        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setRange(0, 30)
        self.offset.setSingleStep(0.5)
        self.offset.setSuffix(" m")
        self.offset.setToolTip("Lift places, rooms and buildings above the map (Hydra-style exploded view)")
        self.offset.valueChanged.connect(lambda v: self._set_opt(layer_offset=v))
        tb.addWidget(self.offset)
        tb.addSeparator()
        self.toggles = {}
        for key, text, on, tip in [
            ("intra_edges", "Edges", True, "Edges within a layer (place graph, ...)"),
            ("inter_edges", "Inter-layer", False, "Edges between layers (room-place, place-object, place-agent)"),
            ("agent_path", "Trajectory", True, "Agent (pose graph) nodes connected in time, colored start to end"),
            ("frame_path", "Frame path", False, "Camera path of every recorded frame (observations)"),
            ("boxes", "Boxes", True, "Object bounding boxes"),
            ("object_labels", "Labels", False, "Object class labels"),
        ]:
            a = tb.addAction(text)
            a.setCheckable(True)
            a.setChecked(on)
            a.setToolTip(tip)
            a.toggled.connect(lambda checked, k=key: self._set_opt(**{k: checked}))
            self.toggles[key] = a

    # ---- file handling ----------------------------------------------------------------------
    def open_file_dialog(self):
        start = str(self.graph.path.parent) if self.graph else str(pathlib.Path("~/hydra_data").expanduser())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open scene graph", start,
                                                        "Scene graph JSON (*.json);;All files (*)")
        if path:
            self.open_path(path)

    def open_dir_dialog(self):
        start = str(pathlib.Path("~/hydra_data").expanduser())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open session folder", start)
        if path:
            self.open_path(path)

    def open_path(self, path):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            g = model.load(path)
        except Exception as e:  # show the reason instead of crashing
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Cannot open scene graph", f"{path}\n\n{e}")
            return
        self.graph = g
        self.scene.opt.hidden_groups = frozenset()
        self.scene.set_graph(g)
        self._fill_tree()
        self.inspector.show_node(g, None)
        QtWidgets.QApplication.restoreOverrideCursor()
        counts = ", ".join(f"{len(v)} {k.lower()}" for k, v in g.groups().items())
        mesh = f", mesh {len(g.mesh.points):,} vertices" if g.mesh is not None else ", no mesh"
        obs = f", {len(g.observations.frames):,} recorded frames" if g.observations else ""
        self.setWindowTitle(f"DSG Viewer — {g.path}")
        self.statusBar().showMessage(f"{counts}; {len(g.edges)} edges{mesh}{obs}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "dsg.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)

    # ---- tree ---------------------------------------------------------------------------------
    def _fill_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        self._tree_items.clear()
        for group, nodes in self.graph.groups().items():
            top = QtWidgets.QTreeWidgetItem([group, str(len(nodes))])
            top.setData(0, GROUP_ROLE, group)
            top.setFlags(top.flags() | QtCore.Qt.ItemIsUserCheckable)
            top.setCheckState(0, QtCore.Qt.Checked)
            top.setIcon(0, swatch(GROUP_STYLE.get(group, DEFAULT_STYLE)[0]))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.tree.addTopLevelItem(top)
            for n in nodes:
                it = QtWidgets.QTreeWidgetItem([n.title, ""])
                it.setData(0, ID_ROLE, n.id)
                if group == "Objects":
                    it.setIcon(0, swatch(label_color(n.label)))
                if group == "Agents" and self.graph.observations:
                    k = len(self.graph.observations.node_frames.get(n.id, []))
                    it.setText(1, str(k) if k else "")
                    it.setToolTip(1, f"{k} recorded frames")
                top.addChild(it)
                self._tree_items[n.id] = it
            top.setExpanded(group in ("Objects", "Rooms", "Buildings"))
        self.tree.blockSignals(False)

    def _filter_tree(self, text):
        text = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                it = top.child(j)
                match = not text or text in it.text(0).lower() or text in str(it.data(0, ID_ROLE))
                it.setHidden(not match)
                shown += match
            top.setText(1, f"{shown}/{top.childCount()}" if text else str(top.childCount()))
            if text:
                top.setExpanded(shown > 0)

    def _on_tree_current(self, cur, _prev):
        if cur is None:
            return
        nid = cur.data(0, ID_ROLE)
        if nid is not None:
            self.select_node(int(nid), from_tree=True)

    def _on_tree_check(self, item, col):
        group = item.data(0, GROUP_ROLE)
        if group is None or col != 0:
            return
        hidden = set(self.scene.opt.hidden_groups)
        (hidden.discard if item.checkState(0) == QtCore.Qt.Checked else hidden.add)(group)
        self._set_opt(hidden_groups=frozenset(hidden))

    # ---- selection and view -----------------------------------------------------------------
    def select_node(self, nid, from_tree=False, fly=False):
        if self.graph is None:
            return
        n = self.graph.nodes.get(nid) if nid is not None else None
        self.scene.select(n.id if n else None)
        self.inspector.show_node(self.graph, n)
        if n is not None and not from_tree:
            it = self._tree_items.get(n.id)
            if it is not None:
                self.tree.blockSignals(True)
                it.setHidden(False)
                self.tree.setCurrentItem(it)
                self.tree.scrollToItem(it)
                self.tree.blockSignals(False)
        if n is not None and n.group == "Agents" and self.graph.observations:
            self.inspector.tabs.setCurrentIndex(self.inspector.frames_tab_index)
        if fly and n is not None:
            self.scene.focus(n.id)

    def _mark_frame(self, f: model.Frame):
        self.scene._drop("frame")
        if f.position is not None:
            import pyvista as pv
            a = self.plotter.add_mesh(pv.Sphere(radius=0.12, center=f.position), color="#1b998b",
                                      pickable=False, reset_camera=False)
            self.scene._add("frame", a)
            self.plotter.render()

    def _set_opt(self, **kw):
        for k, v in kw.items():
            setattr(self.scene.opt, k, v)
        if self.graph is not None:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.scene.rebuild()
            QtWidgets.QApplication.restoreOverrideCursor()

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def top_view(self):
        self.plotter.view_xy()
        self.plotter.reset_camera()

    def _on_press(self, *_):
        self._press = self.plotter.iren.get_event_position()

    def _on_release(self, *_):
        if self._press is None or self.graph is None:
            return
        x, y = self.plotter.iren.get_event_position()
        px, py = self._press
        self._press = None
        if abs(x - px) + abs(y - py) > 4:
            return  # a drag (rotate), not a click
        nid = self.scene.pick(x, y)
        if nid is not None:
            self.select_node(nid)

    def closeEvent(self, e):
        self.plotter.close()
        super().closeEvent(e)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Navigate a Hydra 3D scene graph (dsg.json) in 3D.")
    parser.add_argument("path", nargs="?", help="dsg.json, dsg_with_mesh.json or a session folder")
    parser.add_argument("--screenshot", help=argparse.SUPPRESS)  # smoke test: render, save, quit
    parser.add_argument("--select", help=argparse.SUPPRESS)  # smoke test: select a node symbol
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("DSG Viewer")
    w = MainWindow()
    w.show()
    if args.path:
        QtCore.QTimer.singleShot(0, lambda: w.open_path(args.path))
    if args.screenshot:
        def shoot():
            if args.select and w.graph:
                nid = next((n.id for n in w.graph.nodes.values() if n.symbol == args.select), None)
                w.select_node(nid, fly=False)
                if w.inspector.frames.count():
                    w.inspector.frames.setCurrentRow(0)
            def save():
                w.grab().save(args.screenshot)
                w.plotter.screenshot(str(pathlib.Path(args.screenshot).with_suffix("")) + "_3d.png")
                app.quit()
            QtCore.QTimer.singleShot(1500, save)
        QtCore.QTimer.singleShot(1500, shoot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
