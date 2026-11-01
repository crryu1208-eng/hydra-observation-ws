"""Load a Hydra scene graph (spark_dsg JSON) plus the optional mesh and observation-recorder output."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

# node attribute type -> display name of its layer group
GROUP_NAMES = {
    "ObjectNodeAttributes": "Objects",
    "KhronosObjectAttributes": "Objects",
    "AgentNodeAttributes": "Agents",
    "PlaceNodeAttributes": "Places",
    "Place2dNodeAttributes": "Places 2D",
    "RoomNodeAttributes": "Rooms",
    "SemanticNodeAttributes": "Buildings",
}
# vertical order of groups in the exploded view (multiplied by the layer offset)
GROUP_LEVEL = {"Objects": 0, "Agents": 0, "Places": 1, "Places 2D": 1, "Rooms": 2, "Buildings": 3}


def symbol_str(node_id: int) -> str:
    """gtsam-style symbol: top 8 bits are a character, the low 56 bits an index."""
    key = (node_id >> 56) & 0xFF
    index = node_id & ((1 << 56) - 1)
    if 32 < key < 127:
        return f"{chr(key)}{index}"
    return str(node_id)


@dataclass
class Node:
    id: int
    symbol: str
    layer: int
    partition: int
    type: str
    group: str
    position: np.ndarray
    attrs: dict
    label: str = ""  # semantic class name, if any

    @property
    def title(self) -> str:
        return f"{self.symbol}  {self.label}" if self.label else self.symbol

    @property
    def timestamp_ns(self) -> int | None:
        t = self.attrs.get("timestamp")
        return int(t) if t is not None else None


@dataclass
class Edge:
    source: int
    target: int
    weight: float | None


@dataclass
class Mesh:
    points: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int64
    colors: np.ndarray | None  # (N, 3) uint8
    labels: np.ndarray | None  # (N,) int64


@dataclass
class Frame:
    frame_id: int
    timestamp_ns: int
    rgb_path: pathlib.Path | None
    depth_path: pathlib.Path | None
    position: np.ndarray | None  # world_T_body translation (loop-closure corrected when available)
    mapping: dict
    world_T_sensor: tuple[np.ndarray, np.ndarray] | None = None  # (t, q_wxyz) of the camera optical frame

    @property
    def node_id(self) -> int | None:
        nid = (self.mapping or {}).get("node_id")
        return int(nid) if nid is not None else None


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    max_range: float | None = None


@dataclass
class Observations:
    root: pathlib.Path
    frames: dict[int, Frame]
    node_frames: dict[int, list[int]]  # agent node id -> frame ids
    intrinsics: Intrinsics | None = None
    ordered: list[int] = field(default_factory=list)  # frame ids in time order

    def frames_of(self, node_id: int) -> list[Frame]:
        return [self.frames[i] for i in self.node_frames.get(node_id, []) if i in self.frames]


@dataclass
class SceneGraph:
    path: pathlib.Path
    nodes: dict[int, Node]
    edges: list[Edge]
    mesh: Mesh | None = None
    mesh_path: pathlib.Path | None = None
    observations: Observations | None = None
    neighbors: dict[int, list[int]] = field(default_factory=dict)

    def groups(self) -> dict[str, list[Node]]:
        out: dict[str, list[Node]] = {}
        for n in self.nodes.values():
            out.setdefault(n.group, []).append(n)
        order = list(GROUP_LEVEL)
        for g in out.values():
            g.sort(key=lambda n: (n.symbol[:1], n.id))
        return dict(sorted(out.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))

    def agent_path(self) -> list[Node]:
        agents = [n for n in self.nodes.values() if n.group == "Agents"]
        return sorted(agents, key=lambda n: (n.timestamp_ns or 0, n.id))


def _label_names(metadata: dict) -> dict[str, dict[int, str]]:
    out = {}
    for key, entries in (metadata or {}).get("labelspaces", {}).items():
        try:
            out[key] = {int(i): str(name) for i, name in entries}
        except (TypeError, ValueError):
            continue
    return out


def _load_mesh(m: dict) -> Mesh | None:
    if not m or not m.get("points"):
        return None
    points = np.asarray(m["points"], dtype=np.float32)
    faces = np.asarray(m.get("faces") or [], dtype=np.int64).reshape(-1, 3)
    colors = None
    if m.get("has_colors") and m.get("colors"):
        colors = np.array([[c["r"], c["g"], c["b"]] for c in m["colors"]], dtype=np.uint8)
    labels = None
    if m.get("has_labels") and m.get("labels"):
        labels = np.asarray(m["labels"], dtype=np.int64)
    return Mesh(points, faces, colors, labels)


def find_session_root(dsg_path: pathlib.Path) -> pathlib.Path | None:
    """Session folder holding observations/ (dsg is usually <session>/hydra/backend/dsg.json)."""
    for p in [dsg_path.parent, *dsg_path.parents]:
        if (p / "observations" / "frames.jsonl").exists() or (p / "observations" / "frames_raw.jsonl").exists():
            return p
        if p == p.parent:
            break
    return None


def load_observations(obs_dir: pathlib.Path) -> Observations | None:
    frames_file = obs_dir / "frames.jsonl"
    if not frames_file.exists():
        frames_file = obs_dir / "frames_raw.jsonl"
    if not frames_file.exists():
        return None
    frames: dict[int, Frame] = {}
    with frames_file.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") != "written":
                continue
            corrected = d.get("corrected") or {}
            pose = corrected.get("world_T_body") or d.get("world_T_body")
            sensor = corrected.get("world_T_sensor") or d.get("world_T_sensor")
            frames[d["frame_id"]] = Frame(
                frame_id=d["frame_id"],
                timestamp_ns=int(d["timestamp_ns"]),
                rgb_path=obs_dir / d["rgb_path"] if d.get("rgb_path") else None,
                depth_path=obs_dir / d["depth_path"] if d.get("depth_path") else None,
                position=np.asarray(pose["t"], dtype=float) if pose else None,
                mapping=d.get("mapping") or {},
                world_T_sensor=(np.asarray(sensor["t"], float), np.asarray(sensor["q_wxyz"], float))
                if sensor else None,
            )
    node_frames: dict[int, list[int]] = {}
    agent_file = obs_dir / "agent_frames.json"
    if agent_file.exists():
        for key, v in json.loads(agent_file.read_text()).items():
            node_frames[int(key)] = list(v.get("frames", []))
    intrinsics = None
    calib_file = obs_dir / "camera_calib.json"
    if calib_file.exists():
        try:
            c = next(iter(json.loads(calib_file.read_text())["sensors"].values()))
            intrinsics = Intrinsics(int(c["width"]), int(c["height"]), float(c["fx"]), float(c["fy"]),
                                    float(c["cx"]), float(c["cy"]), c.get("max_range"))
        except (KeyError, StopIteration, TypeError, ValueError):
            intrinsics = None
    ordered = [f.frame_id for f in sorted(frames.values(), key=lambda f: (f.timestamp_ns, f.frame_id))]
    return Observations(obs_dir, frames, node_frames, intrinsics, ordered)


def load(path: str | pathlib.Path) -> SceneGraph:
    """Load a dsg JSON file or a session folder (<session>/hydra/backend/dsg*.json)."""
    path = pathlib.Path(path).expanduser().resolve()
    if path.is_dir():
        for cand in ("hydra/backend/dsg_with_mesh.json", "hydra/backend/dsg.json",
                     "backend/dsg_with_mesh.json", "backend/dsg.json", "dsg_with_mesh.json", "dsg.json"):
            if (path / cand).exists():
                path = path / cand
                break
        else:
            raise FileNotFoundError(f"no dsg.json or dsg_with_mesh.json under {path}")

    data = json.loads(path.read_text())
    if "nodes" not in data:
        raise ValueError(f"{path.name} is not a scene graph JSON file (no 'nodes')")
    names = _label_names(data.get("metadata", {}))

    nodes: dict[int, Node] = {}
    for n in data["nodes"]:
        a = n.get("attributes", {})
        t = a.get("type", "")
        layer, partition = int(n.get("layer", -1)), int(n.get("partition", 0))
        label = ""
        if "semantic_label" in a:
            lbl = int(a["semantic_label"])
            label = names.get(f"_l{layer}p{partition}", {}).get(lbl, "")
            if not label and t in ("ObjectNodeAttributes", "KhronosObjectAttributes"):
                label = str(lbl)
        if t == "RoomNodeAttributes" and a.get("name"):
            label = a["name"]
        nodes[int(n["id"])] = Node(
            id=int(n["id"]), symbol=symbol_str(int(n["id"])), layer=layer, partition=partition,
            type=t, group=GROUP_NAMES.get(t, f"Layer {layer}.{partition}"),
            position=np.asarray(a.get("position", [0, 0, 0]), dtype=float), attrs=a, label=label)

    edges = []
    neighbors: dict[int, list[int]] = {}
    for e in data.get("edges", []):
        s, t = int(e["source"]), int(e["target"])
        if s not in nodes or t not in nodes:
            continue
        edges.append(Edge(s, t, (e.get("info") or {}).get("weight")))
        neighbors.setdefault(s, []).append(t)
        neighbors.setdefault(t, []).append(s)

    g = SceneGraph(path=path, nodes=nodes, edges=edges, neighbors=neighbors)
    g.mesh = _load_mesh(data.get("mesh"))
    if g.mesh is not None:
        g.mesh_path = path
    else:
        sibling = path.with_name("dsg_with_mesh.json")
        if sibling.exists() and sibling != path:
            g.mesh = _load_mesh(json.loads(sibling.read_text()).get("mesh"))
            g.mesh_path = sibling if g.mesh is not None else None

    root = find_session_root(path)
    if root is not None:
        g.observations = load_observations(root / "observations")
    return g
