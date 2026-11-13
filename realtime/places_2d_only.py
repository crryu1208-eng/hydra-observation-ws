#!/usr/bin/env python3
"""Replace the 3D places of a scene graph with its 2D (floor) places.

  places_2d_only.py <in dsg.json> <out dsg.json>     (dsg_with_mesh.json works the same, mesh kept)

Hydra's 3D places (layer 3, partition 0, from the GVD) float in free space; the 2D places (layer 3,
partition 1, MESH_PLACES) lie on the floor. The 3D places and their edges are removed, and what
hung on them is attached to the 2D places instead:
  objects, agent (keyframe) nodes -> the 2D place nearest in x / y
  rooms                           -> each 2D place joins the room of the 3D place nearest to it in x / y
2D place - 2D place edges stay as they are.
"""
import json, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
g = json.load(open(src))
N = {n["id"]: n for n in g["nodes"]}
key = lambda n: (n["layer"], n.get("partition", 0))
P3 = [n for n in g["nodes"] if key(n) == (3, 0)]
P2 = [n for n in g["nodes"] if key(n) == (3, 1)]
if not P2:
    sys.exit("no 2D places")
ids3 = {n["id"] for n in P3}
xy2 = np.array([n["attributes"]["position"][:2] for n in P2]); id2 = [n["id"] for n in P2]
xy3 = np.array([n["attributes"]["position"][:2] for n in P3]) if P3 else np.zeros((0, 2))
nearest2 = lambda p: id2[int(np.argmin(((xy2 - np.asarray(p[:2])) ** 2).sum(1)))]

room_of3 = {}
kept, attach = [], {}  # attach: node id -> layer key of what it hung on
for e in g["edges"]:
    s, t = e["source"], e["target"]
    if s in ids3 and t in ids3:
        continue
    if s in ids3 or t in ids3:
        p, o = (s, t) if s in ids3 else (t, s)
        if key(N[o]) == (4, 0):
            room_of3[p] = o
        else:
            attach[o] = True
        continue
    kept.append(e)

edge = lambda a, b: {"info": {"metadata": {}, "type": "EdgeAttributes", "weight": 1.0, "weighted": False},
                     "source": a, "target": b}
new = []
for o in attach:  # objects / agents: nearest 2D place
    new.append(edge(nearest2(N[o]["attributes"]["position"]), o))
rooms = {}
if P3 and room_of3:
    for n in P2:  # 2D place -> room of the nearest 3D place that has a room
        cand = [i for i, p in enumerate(P3) if p["id"] in room_of3]
        d = ((xy3[cand] - np.asarray(n["attributes"]["position"][:2])) ** 2).sum(1)
        r = room_of3[P3[cand[int(np.argmin(d))]]["id"]]
        rooms[r] = rooms.get(r, 0) + 1
        new.append(edge(r, n["id"]))
g["nodes"] = [n for n in g["nodes"] if n["id"] not in ids3]
g["edges"] = kept + new
json.dump(g, open(dst, "w"))
kinds = {}
for o in attach:
    k = N[o]["attributes"]["type"].replace("NodeAttributes", "")
    kinds[k] = kinds.get(k, 0) + 1
print(f"{src}: removed {len(P3)} 3D places; {len(P2)} 2D places kept; re-attached {kinds}; "
      f"2D places per room {sorted(rooms.values(), reverse=True)} -> {dst}")
