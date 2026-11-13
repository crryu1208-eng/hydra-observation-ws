#!/usr/bin/env python3
"""Register the scene graph of one session to another (same place, separate recordings).

  align_sessions.py <reference session> <session to align> [--map rebuilt/levelled]

Both maps are levelled (floor at z = 0, z up; level_session.py), so only x, y, yaw and a small z
offset remain. Coarse: top-view occupancy of the walls / shelves / furniture (mesh vertices 0.2-2.0 m
above the floor, 10 cm cells) correlated over yaw in 2 deg steps (FFT for the shift). Fine: ICP on the
3D mesh vertices (x, y, z, yaw), trimmed to the closest 80%.
Writes <session to align>/aligned_to_<reference>/:
  hydra/backend/dsg_with_mesh.json, dsg.json, mesh.ply   the aligned scene graph (spark_dsg transform)
  observations/frames.jsonl, trajectory_nodes.csv         poses in the reference frame
  transform.json                                         ref_T_session (4x4), yaw, shift, fit stats
  overlay.png                                            top view: reference grey, aligned red
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_session as ls  # noqa: E402  (PLY rewrite and pose helpers)


def read_vertices(ply):
    with open(ply) as f:
        props, n = [], 0
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[2])
            elif line.startswith("property") and not line.startswith("property list"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        v = np.loadtxt(f, max_rows=n)
    return v[:, :3], (v[:, props.index("label")].astype(int) if "label" in props else None)


def grid(points, origin, cell, shape):
    ij = np.floor((points[:, :2] - origin) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    g = np.zeros(shape)
    np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.minimum(g, 3.0)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def coarse(ref, src, cell=0.1):
    """Best (yaw, shift xy) of src onto ref from top-view occupancy, correlation by FFT."""
    lo = np.minimum(ref[:, :2].min(0), -np.abs(src[:, :2]).max() - 1.0) - 1.0
    hi = np.maximum(ref[:, :2].max(0), np.abs(src[:, :2]).max() + 1.0) + 1.0
    shape = tuple((np.ceil((hi - lo) / cell)).astype(int) + 1)
    shape = (int(2 ** math.ceil(math.log2(shape[0]))), int(2 ** math.ceil(math.log2(shape[1]))))
    G = np.fft.rfft2(grid(ref, lo, cell, shape))
    best = (-1.0, 0.0, np.zeros(2))
    for deg in np.arange(0.0, 360.0, 2.0):
        R = rot_z(math.radians(deg))
        p = src @ R.T
        # src centred in the grid; the correlation peak gives the shift
        c0 = (hi + lo) / 2 - p[:, :2].mean(0)
        S = np.fft.rfft2(grid(p + np.r_[c0, 0.0], lo, cell, shape))
        corr = np.fft.irfft2(G * np.conj(S), s=shape)
        k = np.unravel_index(np.argmax(corr), corr.shape)
        score = corr[k] / max(1.0, math.sqrt((grid(ref, lo, cell, shape) ** 2).sum()))
        d = np.array([k[0] if k[0] < shape[0] // 2 else k[0] - shape[0],
                      k[1] if k[1] < shape[1] // 2 else k[1] - shape[1]]) * cell
        if score > best[0]:
            best = (score, deg, c0 + d)
    return best


def icp4(ref, src, T, iters=40, trim=0.8):
    """x, y, z, yaw ICP of src (N x 3) onto ref, starting from T (4 x 4)."""
    tree = cKDTree(ref)
    for _ in range(iters):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, i = tree.query(p)
        keep = d <= np.quantile(d, trim)
        a, b = p[keep], ref[i[keep]]
        ma, mb = a.mean(0), b.mean(0)
        A, B = a[:, :2] - ma[:2], b[:, :2] - mb[:2]
        yaw = math.atan2((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum(), (A * B).sum())
        D = np.eye(4)
        D[:3, :3] = rot_z(yaw)
        D[:3, 3] = mb - D[:3, :3] @ ma
        T = D @ T
        if abs(yaw) < 1e-5 and np.linalg.norm(D[:3, 3]) < 1e-4:
            break
    p = src @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(p)
    return T, d


def transform_observations(src, dst, T):
    """frames.jsonl and trajectory_nodes.csv with every world pose mapped by T (as level_session.py)."""
    with open(f"{src}/frames.jsonl") as f, open(f"{dst}/frames.jsonl", "w") as g:
        for line in f:
            fr = json.loads(line)
            for k in ("world_T_body", "world_T_sensor"):
                if k in fr:
                    fr[k] = ls.tf_pose_dict(T, fr[k])
                if isinstance(fr.get("corrected"), dict) and k in fr["corrected"]:
                    fr["corrected"][k] = ls.tf_pose_dict(T, fr["corrected"][k])
            g.write(json.dumps(fr) + "\n")
    with open(f"{src}/trajectory_nodes.csv") as f, open(f"{dst}/trajectory_nodes.csv", "w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, r.fieldnames)
        w.writeheader()
        for row in r:
            for p in ("", "orig_"):
                t, q = ls.tf_pose(T, [float(row[p + k]) for k in "xyz"],
                                  [float(row[p + k]) for k in ("qw", "qx", "qy", "qz")])
                for k, v in zip(["x", "y", "z", "qw", "qx", "qy", "qz"], list(t) + q):
                    row[p + k] = repr(float(v))
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("session")
    ap.add_argument("--map", default="rebuilt/levelled", help="map inside each session (default rebuilt/levelled)")
    ap.add_argument("--ref-map", default=None,
                    help="map inside the reference if different (e.g. . for a merge_sessions.sh output)")
    args = ap.parse_args()
    R_, S_ = args.reference.rstrip("/"), args.session.rstrip("/")
    mr = f"{R_}/{args.ref_map if args.ref_map is not None else args.map}"
    ms = f"{S_}/{args.map}"
    out = f"{S_}/aligned_to_{os.path.basename(R_)}"
    if os.path.exists(out):
        sys.exit(f"{out} exists")
    ref_all, ref_lab = read_vertices(f"{mr}/hydra/backend/mesh.ply")
    src_all, src_lab = read_vertices(f"{ms}/hydra/backend/mesh.ply")
    band = lambda v: v[(v[:, 2] > 0.2) & (v[:, 2] < 2.0)]
    score, deg, shift = coarse(band(ref_all), band(src_all))
    T = np.eye(4)
    T[:3, :3] = rot_z(math.radians(deg))
    T[:2, 3] = shift
    print(f"coarse: yaw {deg:.0f} deg, shift {shift.round(2).tolist()} m (score {score:.3f})")
    rng = np.random.default_rng(0)
    sub = lambda v, n: v[rng.choice(len(v), min(n, len(v)), replace=False)]
    T, d = icp4(sub(ref_all, 200000), sub(src_all, 60000), T)
    yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
    stats = {"yaw_deg": round(yaw, 2), "shift_m": T[:3, 3].round(3).tolist(),
             "median_dist_m": round(float(np.median(d)), 3), "p90_dist_m": round(float(np.quantile(d, 0.9)), 3),
             "overlap_10cm": round(float((d < 0.10).mean()), 3), "coarse_score": round(float(score), 3)}
    print(f"fine: yaw {yaw:.2f} deg, shift {stats['shift_m']} m; vertex distance median "
          f"{stats['median_dist_m']} m, p90 {stats['p90_dist_m']} m, within 10 cm {stats['overlap_10cm']:.0%}")

    # aligned copy of the scene graph and the observations
    import spark_dsg as dsg
    os.makedirs(f"{out}/hydra/backend")
    os.makedirs(f"{out}/observations")
    G = dsg.DynamicSceneGraph.load(f"{ms}/hydra/backend/dsg_with_mesh.json")
    G.transform(T)
    G.save(f"{out}/hydra/backend/dsg_with_mesh.json", include_mesh=True)
    G.save(f"{out}/hydra/backend/dsg.json", include_mesh=False)
    ls.level_ply(f"{ms}/hydra/backend/mesh.ply", f"{out}/hydra/backend/mesh.ply", T)
    transform_observations(f"{ms}/observations", f"{out}/observations", T)
    B = f"{ms}/hydra/backend"
    if os.path.exists(f"{B}/trajectory.csv"):
        with open(f"{B}/trajectory.csv") as f, open(f"{out}/hydra/backend/trajectory.csv", "w") as g:
            g.write(f.readline())
            for row in csv.reader(f):
                t, q = ls.tf_pose(T, [float(v) for v in row[1:4]], [float(v) for v in row[4:8]])
                g.write(",".join([row[0]] + [repr(float(v)) for v in list(t) + q]) + "\n")
    if os.path.exists(f"{B}/loop_closures.csv"):
        shutil.copy(f"{B}/loop_closures.csv", f"{out}/hydra/backend/")  # relative poses
    for name in ("camera_calib.json", "agent_frames.json"):
        if os.path.exists(f"{ms}/observations/{name}"):
            shutil.copy(f"{ms}/observations/{name}", f"{out}/observations/{name}")
    for name in ("rgb", "depth"):
        src = os.path.realpath(f"{ms}/observations/{name}")
        if os.path.isdir(src):
            os.symlink(os.path.relpath(src, f"{out}/observations"), f"{out}/observations/{name}")
    json.dump({"reference": os.path.abspath(R_), "session": os.path.abspath(S_), "map": args.map,
               "ref_T_session": T.round(6).tolist(), **stats}, open(f"{out}/transform.json", "w"), indent=2)

    # top view overlay
    try:
        import cv2
        a = band(ref_all)[:, :2]
        b = band(src_all) @ T[:3, :3].T + T[:3, 3]
        b = b[:, :2]
        lo = np.minimum(a.min(0), b.min(0)) - 0.5
        hi = np.maximum(a.max(0), b.max(0)) + 0.5
        px = 0.05
        W, H = (np.ceil((hi - lo) / px)).astype(int) + 1
        img = np.full((H, W, 3), 255, np.uint8)
        for pts, col in ((a, (150, 150, 150)), (b, (40, 40, 220))):
            ij = ((pts - lo) / px).astype(int)
            img[H - 1 - ij[:, 1], ij[:, 0]] = col
        cv2.imwrite(f"{out}/overlay.png", img)
    except Exception as e:  # the overlay is optional
        print(f"no overlay: {e}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
