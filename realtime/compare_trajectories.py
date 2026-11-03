#!/usr/bin/env python3
"""Check that Hydra applied RTAB-Map's loop closures: compare Hydra's optimized agent trajectory
(hydra/backend/trajectory.csv) with RTAB-Map's optimized and odometry keyframe poses at the same
stamps (rtabmap-export --poses [--poses_odom] --poses_format 11).

  compare_trajectories.py <run_dir>    (expects <run_dir>/rtabmap_export/{slam,odom}_poses.txt)
"""
import math
import pathlib
import sys

import numpy as np


def read_rtabmap(path):
    poses = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        poses[float(v[0])] = (np.array([float(x) for x in v[1:4]]),
                              np.array([float(v[7]), float(v[4]), float(v[5]), float(v[6])]))
    return poses


def read_hydra(path):
    poses = {}
    for line in pathlib.Path(path).read_text().splitlines()[1:]:
        v = line.split(",")
        poses[int(v[0]) / 1e9] = (np.array([float(x) for x in v[1:4]]),
                                  np.array([float(x) for x in v[4:8]]))
    return poses


def align(src, dst):
    """Rigid (rotation + translation) least-squares alignment of src onto dst (Umeyama, no scale);
    RTAB-Map rotates its whole graph to gravity (IMU links), so frames differ globally."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return (R @ (src - ms).T).T + md


def angle_deg(q1, q2):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))


def main():
    run = pathlib.Path(sys.argv[1])
    slam = read_rtabmap(run / "rtabmap_export/slam_poses.txt")
    odom = read_rtabmap(run / "rtabmap_export/odom_poses.txt")
    hydra = read_hydra(run / "hydra/backend/trajectory.csv")
    keys = sorted(slam)
    matched = []
    for t, (p, q) in sorted(hydra.items()):
        k = min(keys, key=lambda s: abs(s - t))
        if abs(k - t) < 0.002 and k in odom:
            matched.append((t, p, q, *slam[k], *odom[k]))
    if not matched:
        print("no matching stamps")
        return 1
    P_h = np.array([m[1] for m in matched])
    P_s = np.array([m[3] for m in matched])
    P_o = np.array([m[5] for m in matched])
    d_hs = np.linalg.norm(align(P_h, P_s) - P_s, axis=1)
    d_os = np.linalg.norm(align(P_o, P_s) - P_s, axis=1)
    print(f"{len(matched)} of {len(hydra)} Hydra agent poses matched RTAB-Map keyframes")
    print(f"position vs RTAB-Map SLAM after rigid alignment: Hydra median {np.median(d_hs):.3f} m "
          f"max {np.max(d_hs):.3f} m | odometry median {np.median(d_os):.3f} m max {np.max(d_os):.3f} m")
    first, last = matched[0], matched[-1]
    for name, i in (("Hydra", 1), ("RTAB-Map SLAM", 3), ("odometry", 5)):
        print(f"start-end distance {name}: {np.linalg.norm(last[i] - first[i]):.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
