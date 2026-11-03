"""Build and update the 3D scene for a SceneGraph on a pyvista plotter."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from .model import GROUP_LEVEL, Node, SceneGraph

GROUP_STYLE = {  # color, point size (px)
    "Objects": ("#e4572e", 11),
    "Agents": ("#2e86ab", 7),
    "Places": ("#7a8b99", 7),
    "Places 2D": ("#76b041", 7),
    "Rooms": ("#9b5de5", 16),
    "Buildings": ("#f2a541", 18),
}
DEFAULT_STYLE = ("#888888", 8)
SELECT_COLOR = "#ffd166"
FRAME_COLOR = "#e63973"


def quat_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, float) / np.linalg.norm(q_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def frustum(t, q_wxyz, intr, depth: float = 0.45) -> pv.PolyData:
    """Camera pyramid in world coordinates (optical frame: x right, y down, z forward)."""
    w, h, fx, fy, cx, cy = (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy) if intr else \
        (640, 480, 600.0, 600.0, 320.0, 240.0)
    corners = np.array([[(u - cx) / fx * depth, (v - cy) / fy * depth, depth]
                        for u, v in ((0, 0), (w, 0), (w, h), (0, h))])
    pts = np.vstack([np.zeros(3), corners, [[0, -0.25 * depth, depth]]])  # last point: "up" tick
    pts = pts @ quat_matrix(q_wxyz).T + np.asarray(t, float)
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return _lines(pts, pairs)


def label_color(label: int | str) -> tuple[float, float, float]:
    """Stable, well-spread color for a class id or name (golden-ratio hue walk)."""
    if isinstance(label, str):
        h = sum((i + 1) * ord(c) for i, c in enumerate(label))
    else:
        h = int(label)
    hue = (h * 0.6180339887) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.62, 0.92)


def _lines(points: np.ndarray, pairs: list[tuple[int, int]]) -> pv.PolyData:
    poly = pv.PolyData(points)
    if pairs:
        cells = np.empty((len(pairs), 3), dtype=np.int64)
        cells[:, 0] = 2
        cells[:, 1:] = np.asarray(pairs, dtype=np.int64)
        poly.lines = cells.ravel()
    return poly


def _box_edges(center, dims) -> pv.PolyData:
    c, d = np.asarray(center, float), np.maximum(np.asarray(dims, float), 0.02) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * d + c
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return _lines(corners, pairs)


@dataclass
class Options:
    layer_offset: float = 0.0
    mesh_mode: str = "rgb"  # rgb | semantic | hidden
    mesh_opacity: float = 1.0
    intra_edges: bool = True
    inter_edges: bool = False
    agent_path: bool = True
    frame_path: bool = False
    boxes: bool = True
    object_labels: bool = False
    hidden_groups: frozenset = frozenset()


class SceneView:
    """Owns the actors of one scene graph on a plotter and maps clicks back to nodes."""

    def __init__(self, plotter: pv.Plotter):
        self.p = plotter
        self.g: SceneGraph | None = None
        self.opt = Options()
        self.actors: dict[str, list] = {}
        self.selected: int | None = None
        self._ids = np.zeros(0, dtype=np.int64)  # node ids in display order (for picking)
        self._pos = np.zeros((0, 3))
        self._frame_ids = np.zeros(0, dtype=np.int64)  # recorded frames on the trajectory (for picking)
        self._frame_pos = np.zeros((0, 3))
        self.current_frame: int | None = None

    # ---- geometry helpers -------------------------------------------------------------
    def display_pos(self, n: Node) -> np.ndarray:
        level = GROUP_LEVEL.get(n.group, 0)
        return n.position + np.array([0.0, 0.0, level * self.opt.layer_offset])

    def _drop(self, key: str):
        for a in self.actors.pop(key, []):
            self.p.remove_actor(a, render=False)

    def _add(self, key: str, actor):
        self.actors.setdefault(key, []).append(actor)

    # ---- scene construction ----------------------------------------------------------
    def set_graph(self, g: SceneGraph):
        for k in list(self.actors):
            self._drop(k)
        self.g = g
        self.selected = None
        self.current_frame = None
        self.rebuild()
        self.p.reset_camera()
        self.p.render()

    def rebuild(self):
        if self.g is None:
            return
        for k in list(self.actors):
            self._drop(k)
        self._build_mesh()
        self._build_nodes()
        self._build_edges()
        self._build_paths()
        self._build_boxes()
        self._build_selection()
        self._build_frame_marker()
        self.p.render()

    def _build_mesh(self):
        m = self.g.mesh
        if m is None or self.opt.mesh_mode == "hidden" or len(m.faces) == 0:
            return
        faces = np.hstack([np.full((len(m.faces), 1), 3, dtype=np.int64), m.faces]).ravel()
        poly = pv.PolyData(m.points.astype(float), faces)
        if self.opt.mesh_mode == "semantic" and m.labels is not None:
            lut = {int(lbl): label_color(int(lbl)) for lbl in np.unique(m.labels)}
            rgb = np.array([lut[int(lbl)] for lbl in m.labels]) * 255
            poly.point_data["rgb"] = rgb.astype(np.uint8)
        elif m.colors is not None:
            poly.point_data["rgb"] = m.colors
        if "rgb" in poly.point_data:
            a = self.p.add_mesh(poly, scalars="rgb", rgb=True, opacity=self.opt.mesh_opacity,
                                lighting=False, pickable=False, show_scalar_bar=False, reset_camera=False)
        else:
            a = self.p.add_mesh(poly, color="#c8c8c8", opacity=self.opt.mesh_opacity,
                                pickable=False, reset_camera=False)
        self._add("mesh", a)

    def _visible_nodes(self) -> list[Node]:
        return [n for n in self.g.nodes.values() if n.group not in self.opt.hidden_groups]

    def _build_nodes(self):
        ids, pos = [], []
        for group, nodes in self.g.groups().items():
            if group in self.opt.hidden_groups:
                continue
            pts = np.array([self.display_pos(n) for n in nodes])
            color, size = GROUP_STYLE.get(group, DEFAULT_STYLE)
            poly = pv.PolyData(pts)
            if group == "Objects":
                poly.point_data["rgb"] = (np.array([label_color(n.label) for n in nodes]) * 255).astype(np.uint8)
                a = self.p.add_mesh(poly, scalars="rgb", rgb=True, point_size=size, render_points_as_spheres=True,
                                    pickable=False, show_scalar_bar=False, reset_camera=False)
            else:
                a = self.p.add_mesh(poly, color=color, point_size=size, render_points_as_spheres=True,
                                    pickable=False, reset_camera=False)
            self._add("nodes", a)
            ids += [n.id for n in nodes]
            pos.append(pts)
            if group == "Objects" and self.opt.object_labels:
                a = self.p.add_point_labels(pts + np.array([0, 0, 0.25]), [n.label or n.symbol for n in nodes],
                                            font_size=11, point_size=1, shape_opacity=0.55, show_points=False,
                                            always_visible=True, pickable=False, reset_camera=False)
                self._add("nodes", a)
            if group in ("Rooms", "Buildings"):
                a = self.p.add_point_labels(pts + np.array([0, 0, 0.4]), [n.title for n in nodes], font_size=12,
                                            point_size=1, shape_opacity=0.55, show_points=False,
                                            always_visible=True, pickable=False, reset_camera=False)
                self._add("nodes", a)
        self._ids = np.array(ids, dtype=np.int64)
        self._pos = np.vstack(pos) if pos else np.zeros((0, 3))

    def _build_edges(self):
        if not (self.opt.intra_edges or self.opt.inter_edges):
            return
        index = {nid: i for i, nid in enumerate(self._ids)}
        intra: dict[str, list] = {}
        inter = []
        for e in self.g.edges:
            if e.source not in index or e.target not in index:
                continue
            a, b = self.g.nodes[e.source], self.g.nodes[e.target]
            pair = (index[e.source], index[e.target])
            if a.group == b.group:
                if self.opt.intra_edges and a.group != "Agents":  # agents: drawn as the path
                    intra.setdefault(a.group, []).append(pair)
            elif self.opt.inter_edges:
                inter.append(pair)
        for group, pairs in intra.items():
            color, _ = GROUP_STYLE.get(group, DEFAULT_STYLE)
            a = self.p.add_mesh(_lines(self._pos, pairs), color=color, line_width=1.5, opacity=0.8,
                                pickable=False, reset_camera=False)
            self._add("edges", a)
        if inter:
            a = self.p.add_mesh(_lines(self._pos, inter), color="#9aa5ad", line_width=1, opacity=0.45,
                                pickable=False, reset_camera=False)
            self._add("edges", a)

    def _build_paths(self):
        self._frame_ids = np.zeros(0, dtype=np.int64)
        self._frame_pos = np.zeros((0, 3))
        obs = self.g.observations
        path_shown = self.opt.agent_path and "Agents" not in self.opt.hidden_groups
        if obs is not None and (path_shown or self.opt.frame_path):
            ids = [i for i in obs.ordered if obs.frames[i].position is not None]
            if ids:
                self._frame_ids = np.array(ids, dtype=np.int64)
                self._frame_pos = np.array([obs.frames[i].position for i in ids])
        if self.opt.agent_path and "Agents" not in self.opt.hidden_groups:
            path = self.g.agent_path()
            if len(path) > 1:
                pts = np.array([self.display_pos(n) for n in path])
                poly = pv.lines_from_points(pts)
                poly.point_data["t"] = np.linspace(0, 1, len(pts))
                a = self.p.add_mesh(poly, scalars="t", cmap="viridis", line_width=4, show_scalar_bar=False,
                                    pickable=False, reset_camera=False)
                self._add("paths", a)
        obs = self.g.observations
        if self.opt.frame_path and obs is not None:
            frames = sorted((f for f in obs.frames.values() if f.position is not None), key=lambda f: f.timestamp_ns)
            if len(frames) > 1:
                pts = np.array([f.position for f in frames])
                a = self.p.add_mesh(pv.lines_from_points(pts), color="#1b998b", line_width=1.5, opacity=0.8,
                                    pickable=False, reset_camera=False)
                self._add("paths", a)

    def _build_boxes(self):
        if not self.opt.boxes or "Objects" in self.opt.hidden_groups:
            return
        # one actor for all boxes (colored per vertex): an actor per object made each live
        # reload re-render the scene once per object
        boxes, colors = [], []
        for n in self.g.groups().get("Objects", []):
            bb = n.attrs.get("bounding_box") or {}
            if bb.get("type") in (None, "INVALID") or "dimensions" not in bb:
                continue
            center = np.asarray(bb.get("world_P_center", n.position), float)
            boxes.append(_box_edges(center, bb["dimensions"]))
            colors.append(np.tile(np.asarray(label_color(n.label)) * 255, (8, 1)))
        if not boxes:
            return
        merged = boxes[0].merge(boxes[1:], merge_points=False) if len(boxes) > 1 else boxes[0]
        merged.point_data["rgb"] = np.vstack(colors).astype(np.uint8)
        a = self.p.add_mesh(merged, scalars="rgb", rgb=True, line_width=2, pickable=False,
                            show_scalar_bar=False, reset_camera=False)
        self._add("boxes", a)

    def _build_selection(self):
        self._drop("selection")
        if self.selected is None or self.selected not in self.g.nodes:
            return
        n = self.g.nodes[self.selected]
        p = self.display_pos(n)
        a = self.p.add_mesh(pv.PolyData(p[None]), color=SELECT_COLOR, point_size=22, render_points_as_spheres=True,
                            pickable=False, reset_camera=False)
        self._add("selection", a)
        nbrs = [m for m in self.g.neighbors.get(n.id, []) if self.g.nodes[m].group not in self.opt.hidden_groups]
        if nbrs:
            pts = np.vstack([p] + [self.display_pos(self.g.nodes[m]) for m in nbrs])
            a = self.p.add_mesh(_lines(pts, [(0, i) for i in range(1, len(pts))]), color=SELECT_COLOR,
                                line_width=3, pickable=False, reset_camera=False)
            self._add("selection", a)
        bb = n.attrs.get("bounding_box") or {}
        if n.group == "Objects" and bb.get("type") not in (None, "INVALID") and "dimensions" in bb:
            a = self.p.add_mesh(_box_edges(bb.get("world_P_center", n.position), bb["dimensions"]),
                                color=SELECT_COLOR, line_width=4, pickable=False, reset_camera=False)
            self._add("selection", a)

    def _build_frame_marker(self):
        self._drop("frame")
        obs = self.g.observations if self.g else None
        f = obs.frames.get(self.current_frame) if obs and self.current_frame is not None else None
        if f is None or f.position is None:
            return
        a = self.p.add_mesh(pv.PolyData(np.asarray(f.position, float)[None]), color=FRAME_COLOR, point_size=16,
                            render_points_as_spheres=True, pickable=False, reset_camera=False)
        self._add("frame", a)
        if f.world_T_sensor is not None:
            a = self.p.add_mesh(frustum(*f.world_T_sensor, obs.intrinsics), color=FRAME_COLOR, line_width=3,
                                pickable=False, reset_camera=False)
            self._add("frame", a)

    def show_frame(self, frame_id: int | None):
        self.current_frame = frame_id
        self._build_frame_marker()
        self.p.render()

    # ---- interaction -----------------------------------------------------------------
    def select(self, node_id: int | None):
        self.selected = node_id
        self._build_selection()
        self.p.render()

    def focus(self, node_id: int, distance: float = 4.0):
        n = self.g.nodes[node_id]
        target = self.display_pos(n)
        cam = self.p.camera
        direction = np.asarray(cam.position) - np.asarray(cam.focal_point)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)
        cam.focal_point = target
        cam.position = target + direction * distance
        self.p.reset_camera_clipping_range()
        self.p.render()

    def _nearest_on_screen(self, pos: np.ndarray, x: int, y: int, radius_px: float):
        """Index of the point nearest to display pixel (x, y; origin bottom-left) within radius
        (ties broken by depth), and its pixel distance."""
        if len(pos) == 0:
            return None, None
        ren = self.p.renderer
        w, h = ren.GetSize()
        if w == 0 or h == 0:
            return None, None
        cam = ren.GetActiveCamera()
        m = cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
        M = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        hom = np.c_[pos, np.ones(len(pos))] @ M.T
        valid = hom[:, 3] > 1e-9
        ndc = hom[:, :3] / np.where(valid, hom[:, 3], 1)[:, None]
        ox, oy = ren.GetOrigin()
        px = (ndc[:, 0] + 1) / 2 * w + ox
        py = (ndc[:, 1] + 1) / 2 * h + oy
        d = np.hypot(px - x, py - y)
        ok = valid & (np.abs(ndc[:, 2]) <= 1) & (d <= radius_px)
        if not ok.any():
            return None, None
        cand = np.flatnonzero(ok)
        best = cand[np.lexsort((ndc[cand, 2], np.round(d[cand] / 4)))][0]
        return int(best), float(d[best])

    def pick(self, x: int, y: int, radius_px: float = 12.0) -> int | None:
        """Node nearest to display pixel (x, y)."""
        i, _ = self._nearest_on_screen(self._pos, x, y, radius_px)
        return None if i is None else int(self._ids[i])

    def pick_with_distance(self, x: int, y: int, radius_px: float = 12.0):
        i, d = self._nearest_on_screen(self._pos, x, y, radius_px)
        return (None, None) if i is None else (int(self._ids[i]), d)

    def pick_frame(self, x: int, y: int, radius_px: float = 10.0):
        """Recorded frame whose camera position is nearest to display pixel (x, y), and its distance."""
        i, d = self._nearest_on_screen(self._frame_pos, x, y, radius_px)
        return (None, None) if i is None else (int(self._frame_ids[i]), d)
