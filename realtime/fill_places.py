#!/usr/bin/env python3
"""Fill gaps in the 2D (floor) places of a scene graph from the recorded data.

  fill_places.py <in dsg_with_mesh.json> <out dsg_with_mesh.json> [--also <in dsg.json> <out dsg.json>]

For graphs with 2D places only (places_2d_only.py). Where the floor places stop, new 2D places are
added from what was recorded:
  trajectory  - the camera was carried there, so there is walkable floor: along the agent node path
                (every 0.3 m), a place wherever no place is within --gap (0.55 m)
  floor mesh  - mesh vertices labelled floor / rug (0.8 m cells with enough of them) with no place
                within 0.9 m and nothing standing on them (no other vertex 0.15-1.0 m above within 0.4 m)
New places sit on the floor (median z of the floor vertices around them, or of the nearest places
when there are few or they are > 0.3 m off),
carry metadata {"filled_from": "trajectory" | "floor_mesh"} and the floor vertices within 0.4 m as
pcl_mesh_connections, and hang on the room of the nearest place. Edges (to new and old places within
1.3 m) are added only where the floor is seen all along the segment (a floor vertex within 0.25 m of
every 0.15 m step) or the camera walked along it (within 0.6 m of the path); consecutive trajectory
places are always linked. Finally separate groups of places are joined by their closest supported pair (up to 3 m);
--drop-isolated removes the groups still apart from the largest one.
"""
import argparse, copy, json, math
import numpy as np
from scipy.spatial import cKDTree

FLOOR = (3, 28)  # ADE20K floor, rug
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--also", nargs=2, metavar=("SRC", "DST"), help="apply the same places / edges to dsg.json")
ap.add_argument("--gap", type=float, default=0.55)
ap.add_argument("--drop-isolated", action="store_true", help="remove the places not joined to the largest group (their objects / agents move to the nearest kept place)")
a = ap.parse_args()

g = json.load(open(a.src))
key = lambda n: (n["layer"], n.get("partition", 0))
N = {n["id"]: n for n in g["nodes"]}
P2 = [n for n in g["nodes"] if key(n) == (3, 1)]
mesh = g["mesh"]
V = np.asarray(mesh["points"], float); L = np.asarray(mesh["labels"])
fidx = np.flatnonzero(np.isin(L, FLOOR)); F = V[fidx]
ftree = cKDTree(F[:, :2])
agents = sorted((n for n in g["nodes"] if key(n) == (2, 97)), key=lambda n: n["attributes"]["timestamp"])
T = np.array([n["attributes"]["position"] for n in agents])
# the walked path, every 0.15 m (for the "walked along" edge test)
path = [T[0, :2]]
for p, q in zip(T[:-1, :2], T[1:, :2]):
    if np.linalg.norm(q - p) > 1.5:  # not walked (e.g. between merged sessions): no path in between
        path.append(q); continue
    k = max(1, int(np.linalg.norm(q - p) / 0.15))
    path += [p + (q - p) * t for t in np.linspace(0, 1, k + 1)[1:]]
path = np.array(path); ptree = cKDTree(path)

xy = [n["attributes"]["position"][:2] for n in P2]; z = [n["attributes"]["position"][2] for n in P2]
ids = [n["id"] for n in P2]
room_of = {}
for e in g["edges"]:
    s, t = e["source"], e["target"]
    if s in N and t in N and key(N[s]) == (4, 0) and key(N[t]) == (3, 1): room_of[t] = s
    if s in N and t in N and key(N[t]) == (4, 0) and key(N[s]) == (3, 1): room_of[s] = t


def floor_z(p):
    """median z of the floor vertices around p; the nearest places' z when there are too few or
    they are more than 0.3 m off it (vertices mislabelled floor, e.g. on furniture)"""
    d = np.linalg.norm(np.array(xy) - p, axis=1)
    near = float(np.median(np.array(z)[np.argsort(d)[:3]]))
    i = ftree.query_ball_point(p, 0.4)
    if len(i) >= 5 and abs(float(np.median(F[i, 2])) - near) < 0.3:
        return float(np.median(F[i, 2]))
    return near


new = []  # (xy, z, source)
def add(p, src):
    new.append((np.asarray(p, float), floor_z(p), src)); xy.append(list(p)); z.append(new[-1][1])

# trajectory, every 0.3 m
traj_new = []
for p in path[::2]:
    if np.min(np.linalg.norm(np.array(xy) - p, axis=1)) > a.gap:
        add(p, "trajectory"); traj_new.append(len(new) - 1)
# floor mesh cells
cell = 0.8
ij = np.floor(F[:, :2] / cell).astype(int)
cells = {}
for k, c in enumerate(map(tuple, ij)):
    cells.setdefault(c, []).append(k)
otree = cKDTree(V[:, :2])
def clear(p, fz):
    """nothing stands on the floor there: no other vertex 0.15 .. 1.0 m above it within 0.4 m"""
    i = otree.query_ball_point(p, 0.4)
    h = V[i, 2] - fz
    return not ((h > 0.15) & (h < 1.0) & ~np.isin(L[i], FLOOR)).any()
for c, ks in cells.items():
    if len(ks) < 40:
        continue
    p = np.median(F[ks, :2], 0)
    if np.min(np.linalg.norm(np.array(xy) - p, axis=1)) > 0.9 and clear(p, floor_z(p)):
        add(p, "floor_mesh")

# node ids after the last place 2D id (same symbol)
base = max(ids); sym = base >> 56; nxt = (base & ((1 << 56) - 1)) + 1
tmpl = P2[0]
nodes, new_ids = [], []
for k, (p, pz, src) in enumerate(new):
    nid = (sym << 56) + nxt + k; new_ids.append(nid)
    at = copy.deepcopy(tmpl["attributes"])
    r = 0.4
    bnd = [[p[0] + r * math.cos(t), p[1] + r * math.sin(t), pz] for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
    at.update({"position": [float(p[0]), float(p[1]), pz], "boundary": bnd, "ellipse_centroid": [float(p[0]), float(p[1]), pz],
               "ellipse_matrix_compress": [1 / r ** 2, 0.0, 0.0, 1 / r ** 2], "ellipse_matrix_expand": [r, 0.0, 0.0, r],
               "pcl_mesh_connections": [int(fidx[i]) for i in ftree.query_ball_point(p, r)], "pcl_boundary_connections": [],
               "voxblox_mesh_connections": [], "mesh_vertex_labels": [], "has_active_mesh_indices": False,
               "need_cleanup_splitting": False, "last_update_time_ns": 0, "is_active": False, "semantic_label": 3,
               "metadata": {"filled_from": src}, "name": ""})
    at["bounding_box"] = copy.deepcopy(tmpl["attributes"]["bounding_box"])
    nodes.append({"attributes": at, "id": nid, "layer": 3, "partition": 1})
all_ids = ids + new_ids
XY = np.array(xy); Z = np.array(z)


def supported(i, j):
    p, q = XY[i], XY[j]
    n = max(2, int(np.linalg.norm(q - p) / 0.15) + 1)
    S = p + (q - p) * np.linspace(0, 1, n)[:, None]
    floor = (ftree.query(S, distance_upper_bound=0.25)[0] < np.inf).all()
    walked = (ptree.query(S, distance_upper_bound=0.6)[0] < np.inf).all()
    return floor or walked


edge = lambda s, t, w: {"info": {"metadata": {}, "type": "EdgeAttributes", "weight": float(w), "weighted": True}, "source": s, "target": t}
have = set()
for e in g["edges"]:
    have.add((e["source"], e["target"])); have.add((e["target"], e["source"]))
added = []
def link(i, j):
    s, t = all_ids[i], all_ids[j]
    if s != t and (s, t) not in have:
        have.add((s, t)); have.add((t, s)); added.append(edge(s, t, np.linalg.norm(XY[i] - XY[j])))
tree = cKDTree(XY)
n0 = len(ids)
for k in range(len(new)):
    i = n0 + k
    for j in tree.query_ball_point(XY[i], 1.3):
        if j != i and supported(i, j):
            link(i, j)
for k1, k2 in zip(traj_new[:-1], traj_new[1:]):  # consecutive trajectory places
    if np.linalg.norm(XY[n0 + k1] - XY[n0 + k2]) < 1.0:
        link(n0 + k1, n0 + k2)
# join separate groups: closest supported pair (up to 3 m), repeatedly
def groups():
    idx = {s: i for i, s in enumerate(all_ids)}; adj = [[] for _ in all_ids]
    for e in g["edges"] + added:
        if e["source"] in idx and e["target"] in idx:
            adj[idx[e["source"]]].append(idx[e["target"]]); adj[idx[e["target"]]].append(idx[e["source"]])
    lab = [-1] * len(all_ids); c = 0
    for s in range(len(all_ids)):
        if lab[s] >= 0: continue
        st = [s]
        while st:
            x = st.pop()
            if lab[x] >= 0: continue
            lab[x] = c; st += adj[x]
        c += 1
    return np.array(lab)
for _ in range(50):
    lab = groups(); sizes = np.bincount(lab)
    if len(sizes) == 1: break
    best = None
    for c in np.argsort(sizes)[:-1]:  # every group but the largest
        A = np.flatnonzero(lab == c); B = np.flatnonzero(lab != c)
        bt = cKDTree(XY[B])
        for i in A:
            for j in bt.query_ball_point(XY[i], 3.0):
                d = np.linalg.norm(XY[i] - XY[B[j]])
                if (best is None or d < best[0]) and supported(i, B[j]):
                    best = (d, i, B[j])
    if best is None: break
    link(best[1], best[2])
lab = groups(); sizes = sorted(np.bincount(lab).tolist(), reverse=True)

# rooms: the room of the nearest old place
oldtree = cKDTree(XY[:n0])
for k, nid in enumerate(new_ids):
    r = room_of.get(ids[oldtree.query(XY[n0 + k])[1]])
    if r is not None:
        added.append(edge(r, nid, 1.0) | {"info": {"metadata": {}, "type": "EdgeAttributes", "weight": 1.0, "weighted": False}})


drop = set()
if a.drop_isolated:
    big = np.argmax(np.bincount(lab))
    drop = {all_ids[i] for i in np.flatnonzero(lab != big)}
    keep = [i for i in range(len(all_ids)) if all_ids[i] not in drop]
    ktree = cKDTree(XY[keep])


def apply(gg, out):
    gg["nodes"] += copy.deepcopy(nodes); gg["edges"] += copy.deepcopy(added)
    if drop:
        NN = {n["id"]: n for n in gg["nodes"]}
        edges = []
        for e in gg["edges"]:
            s_, t_ = e["source"], e["target"]
            if s_ in drop or t_ in drop:
                o = t_ if s_ in drop else s_
                if o not in drop and key(NN[o]) in ((2, 0), (2, 97)):  # object / agent: nearest kept place
                    edges.append(edge(all_ids[keep[ktree.query(NN[o]["attributes"]["position"][:2])[1]]], o, 1.0))
                continue
            edges.append(e)
        gg["edges"] = edges
        gg["nodes"] = [n for n in gg["nodes"] if n["id"] not in drop]
    json.dump(gg, open(out, "w"))
apply(g, a.dst)
if a.also:
    apply(json.load(open(a.also[0])), a.also[1])
src = {}
for _, _, s in new: src[s] = src.get(s, 0) + 1
print(f"{a.src}: {n0} 2D places + {len(new)} new {src}; {len(added)} edges added; place groups now {sizes[:6]}"
      + (f"; {len(drop)} isolated places removed" if drop else ""))
