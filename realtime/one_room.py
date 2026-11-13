#!/usr/bin/env python3
"""Make one room of a scene graph that has only 2D (floor) places (see places_2d_only.py).

  one_room.py <in dsg.json> <out dsg.json> [--max-z 0.15] [--name NAME]

- 2D places higher than --max-z above the floor (z = 0) are not on the floor (e.g. on furniture):
  they are removed with their edges, and the objects / agent nodes that hung on them move to the
  nearest remaining 2D place (x / y).
- All rooms are merged into the first one: the other rooms and the room - room edges are removed,
  and every 2D place hangs on the kept room. Its position is the centre of the 2D places (z: the
  old rooms' median) and its bounding box spans the 2D places in x / y and the floor to the
  mesh top (99th percentile z; --top or 2.5 m without a mesh) in z.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--max-z", type=float, default=0.15)
ap.add_argument("--name", default="")
ap.add_argument("--top", type=float, help="room box top [m] (default: the mesh's 99th percentile z, or 2.5 m)")
a = ap.parse_args()
g = json.load(open(a.src))
N = {n["id"]: n for n in g["nodes"]}
key = lambda n: (n["layer"], n.get("partition", 0))
pos = lambda i: np.asarray(N[i]["attributes"]["position"], float)
P2 = [n["id"] for n in g["nodes"] if key(n) == (3, 1)]
rooms = [n["id"] for n in g["nodes"] if key(n) == (4, 0)]
if not rooms or not P2:
    raise SystemExit("needs rooms and 2D places")
if any(key(n) == (3, 0) for n in g["nodes"]):
    raise SystemExit("the graph still has 3D places: run places_2d_only.py first")
off = {i for i in P2 if pos(i)[2] > a.max_z}
keep2 = [i for i in P2 if i not in off]
xy = np.array([pos(i)[:2] for i in keep2])
nearest = lambda p: keep2[int(np.argmin(((xy - p[:2]) ** 2).sum(1)))]
room = rooms[0]
edge = lambda s, t: {"info": {"metadata": {}, "type": "EdgeAttributes", "weight": 1.0, "weighted": False},
                     "source": s, "target": t}
edges, moved = [], 0
for e in g["edges"]:
    s, t = e["source"], e["target"]
    if s in rooms or t in rooms:
        continue  # rebuilt below
    if s in off or t in off:
        o = t if s in off else s
        if o not in off and key(N[o]) != (3, 1):  # an object / agent on a removed place
            edges.append(edge(nearest(pos(o)), o)); moved += 1
        continue
    edges.append(e)
edges += [edge(room, i) for i in keep2]
# the kept room: centre, bounding box
P = np.array([pos(i) for i in keep2])
top = a.top
if top is None and isinstance(g.get("mesh"), dict) and g["mesh"].get("points"):
    top = float(np.percentile(np.asarray(g["mesh"]["points"], float)[:, 2], 99))
top = 2.5 if top is None else top
lo = np.r_[P[:, :2].min(0), 0.0]; hi = np.r_[P[:, :2].max(0), top]
r = N[room]["attributes"]
r["position"] = [*P[:, :2].mean(0).tolist(), float(np.median([pos(i)[2] for i in rooms]))]
r["bounding_box"] = {"dimensions": (hi - lo).tolist(), "type": "AABB", "world_P_center": ((hi + lo) / 2).tolist(),
                     "world_R_center": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}
if a.name:
    r["name"] = a.name
g["nodes"] = [n for n in g["nodes"] if n["id"] not in off and (key(n) != (4, 0) or n["id"] == room)]
g["edges"] = edges
json.dump(g, open(a.dst, "w"))
print(f"{a.src}: {len(rooms)} rooms -> 1; 2D places {len(P2)} -> {len(keep2)} ({len(off)} above {a.max_z} m removed, "
      f"{moved} objects / agents moved to the nearest 2D place); room box {np.round(hi - lo, 2).tolist()} m (top {top:.2f} m) -> {a.dst}")
