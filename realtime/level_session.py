#!/usr/bin/env python3
"""Level a finished session: rotate its world so the floor is horizontal and shift it so the floor is
at z = 0. For sessions recorded before the odometry IMU fix (rtabmap_ros fork, floor_init.py), whose
map stayed in the camera's start orientation (office_rt6: floor tilted 35 deg).

The floor plane is fitted (RANSAC, 3 cm) to the floor-labelled vertices of the corrected mesh; the
rotation is the smallest one taking its normal to +z (no yaw change). One rigid transform W' = L * W
is applied to every world pose and position; relative poses (loop closures, node_T_body,
body_T_sensor) do not change. The result is written next to the original, which is not touched:

  <session>/levelled/hydra/backend/  dsg.json dsg_with_mesh.json mesh.ply trajectory.csv loop_closures.csv
  <session>/levelled/observations/   frames.jsonl trajectory_nodes.csv + copies / links of the rest
  <session>/levelled/level_transform.json

When too little floor was seen (office_rt1 looked mostly at the ceiling), --ceiling levels with the
ceiling plane instead; the height then cannot be known and z is left as it was (rotation only).

usage: level_session.py <session_dir> [--floor-label 3] [--ceiling] [--force]
"""
import argparse, csv, json, os, shutil, sys
import numpy as np
import spark_dsg as dsg
from scipy.spatial.transform import Rotation


def fit_floor(points, rng, iters=1000, tol=0.03):
    best = None
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.5:  # the floor is within 60 deg of the world z axis
            continue
        inl = np.abs((points - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    p = points[best]  # least-squares refinement on the inliers
    c = p.mean(0)
    n = np.linalg.svd(p - c)[2][2]
    if n[2] < 0:
        n = -n
    return n, c, best.mean()


def level_transform(n, c):
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, z)
    ang = np.arccos(np.clip(n @ z, -1, 1))
    R = Rotation.from_rotvec(axis / np.linalg.norm(axis) * ang).as_matrix() if ang > 1e-9 else np.eye(3)
    L = np.eye(4)
    L[:3, :3] = R
    L[2, 3] = -(R @ c)[2]
    return L, np.degrees(ang)


def tf_pose(L, t, q_wxyz):
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    t2 = L[:3, :3] @ np.asarray(t) + L[:3, 3]
    x, y, z, w = Rotation.from_matrix(L[:3, :3] @ R).as_quat()
    return t2, [w, x, y, z]


def tf_pose_dict(L, p):
    if not isinstance(p, dict) or "t" not in p:
        return p
    t, q = tf_pose(L, p["t"], p["q_wxyz"])
    return {**p, "t": t.tolist(), "q_wxyz": q}


def level_ply(src, dst, L):
    with open(src) as f, open(dst, "w") as g:
        nv, ascii_ = None, False
        for line in f:
            g.write(line)
            ascii_ |= line.startswith("format ascii")
            if line.startswith("element vertex"):
                nv = int(line.split()[2])
            if line.strip() == "end_header":
                break
        if nv is None or not ascii_:
            raise SystemExit(f"{src}: expected an ascii PLY with a vertex element")
        for _ in range(nv):
            tok = f.readline().split()
            p = L[:3, :3] @ np.array(tok[:3], float) + L[:3, 3]
            g.write(" ".join([f"{v:.6g}" for v in p] + tok[3:]) + "\n")
        shutil.copyfileobj(f, g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--floor-label", type=int, default=3, help="ADE20K floor = 3")
    ap.add_argument("--ceiling", action="store_true",
                    help="level with the ceiling (ADE20K 5) when the floor was not seen; rotation only")
    ap.add_argument("--force", action="store_true", help="replace an existing levelled/")
    a = ap.parse_args()
    S = os.path.abspath(a.session)
    B, O, OUT = f"{S}/hydra/backend", f"{S}/observations", f"{S}/levelled"
    if os.path.exists(OUT):
        if not a.force:
            sys.exit(f"{OUT} exists (--force replaces it)")

    G = dsg.DynamicSceneGraph.load(f"{B}/dsg_with_mesh.json")
    V = np.asarray(G.mesh.get_vertices())[:3].T
    label = 5 if a.ceiling else a.floor_label
    F = V[np.asarray(G.mesh.get_labels()) == label]
    if len(F) < 500:
        sys.exit(f"only {len(F)} vertices labelled {label}" + ("" if a.ceiling else " (floor); try --ceiling"))
    n, c, frac = fit_floor(F, np.random.default_rng(0))
    L, tilt = level_transform(n, c)
    if a.ceiling:
        L[2, 3] = 0.0
    Fl = F @ L[:3, :3].T + L[:3, 3]
    print(f"{os.path.basename(S)}: {'ceiling' if a.ceiling else 'floor'} normal {np.round(n, 4)}, tilt {tilt:.2f} deg, height {c @ n:.3f} m "
          f"along the normal, plane inliers {100 * frac:.0f}% of {len(F)} plane-labelled vertices; "
          f"after: plane z p5/50/95 {np.round(np.percentile(Fl[:, 2], [5, 50, 95]), 3)} m")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/hydra/backend")
    os.makedirs(f"{OUT}/observations")

    # scene graphs (positions, bounding boxes, agent poses and the mesh)
    G.transform(L)
    G.save(f"{OUT}/hydra/backend/dsg_with_mesh.json", include_mesh=True)
    G2 = dsg.DynamicSceneGraph.load(f"{B}/dsg.json")
    G2.transform(L)
    G2.save(f"{OUT}/hydra/backend/dsg.json", include_mesh=False)
    level_ply(f"{B}/mesh.ply", f"{OUT}/hydra/backend/mesh.ply", L)
    shutil.copy(f"{B}/loop_closures.csv", f"{OUT}/hydra/backend/")  # relative poses

    with open(f"{B}/trajectory.csv") as f, open(f"{OUT}/hydra/backend/trajectory.csv", "w") as g:
        g.write(f.readline())
        for row in csv.reader(f):
            t, q = tf_pose(L, [float(v) for v in row[1:4]], [float(v) for v in row[4:8]])
            g.write(",".join([row[0]] + [repr(v) for v in list(t) + q]) + "\n")

    with open(f"{O}/trajectory_nodes.csv") as f, open(f"{OUT}/observations/trajectory_nodes.csv", "w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, r.fieldnames)
        w.writeheader()
        for row in r:
            for p in ("", "orig_"):
                t, q = tf_pose(L, [float(row[p + k]) for k in "xyz"], [float(row[p + k]) for k in ("qw", "qx", "qy", "qz")])
                for k, v in zip(["x", "y", "z", "qw", "qx", "qy", "qz"], list(t) + q):
                    row[p + k] = repr(float(v))
            w.writerow(row)

    with open(f"{O}/frames.jsonl") as f, open(f"{OUT}/observations/frames.jsonl", "w") as g:
        for line in f:
            fr = json.loads(line)
            for k in ("world_T_body", "world_T_sensor"):
                if k in fr:
                    fr[k] = tf_pose_dict(L, fr[k])
                if isinstance(fr.get("corrected"), dict) and k in fr["corrected"]:
                    fr["corrected"][k] = tf_pose_dict(L, fr["corrected"][k])
            g.write(json.dumps(fr) + "\n")

    for k in ("agent_frames.json", "camera_calib.json", "mapping_stats.json"):
        shutil.copy(f"{O}/{k}", f"{OUT}/observations/")
    for k in ("rgb", "depth"):
        os.symlink(f"../../observations/{k}", f"{OUT}/observations/{k}")
    json.dump({"levelled_T_original": L.tolist(), "floor_normal_original": n.tolist(),
               "plane": "ceiling (rotation only, z not shifted)" if a.ceiling else "floor (floor at z = 0)",
               "floor_tilt_deg": tilt, "floor_plane_inlier_fraction": frac,
               "note": "W_levelled = levelled_T_original * W_original for every world pose / point"},
              open(f"{OUT}/level_transform.json", "w"), indent=2)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
