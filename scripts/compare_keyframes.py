#!/usr/bin/env python3
"""Match Hydra agent nodes to RTAB-Map keyframes by stamp and compare their poses.

Needs an RTAB-Map run with a keyframe at every frame (loop_close_d435i_bag.sh with KEYFRAMES=all),
so each agent node has a keyframe at the same image stamp. For every agent node it reports:
  - odom:   the raw visual(-inertial) odometry pose (RTAB-Map node pose, before optimization)
  - dense:  the loop-closure optimized pose from the dense run
  - agent:  the pose Hydra stored in the scene graph (from the corrected bag)
  - sparse: the optimized pose of the 2 Hz run, when that run has a keyframe at the stamp
and the loop closures of the dense run, mapped to the agent nodes nearest in time.

usage: compare_keyframes.py <dsg.json> <dense rtabmap.db> <dense poses.txt> <out_dir>
                            [--sparse-poses poses.txt]
Writes <out_dir>/agent_keyframes.csv, loop_closures.csv and summary.json.
"""

import argparse
import bisect
import csv
import json
import pathlib
import sqlite3

import numpy as np
from scipy.spatial.transform import Rotation

MATCH_TOL_NS = 1_000_000  # agent nodes and keyframes come from the same image stamps
LINK_TYPES = {1: "global_loop_closure", 2: "local_space_closure", 3: "local_time_closure", 4: "user_closure"}


def symbol(node_id):
    key, index = (node_id >> 56) & 0xFF, node_id & ((1 << 56) - 1)
    return f"{chr(key)}{index}" if 32 < key < 127 else str(node_id)


def pose(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def diff(A, B):
    """Translation (m) and rotation (deg) between two poses."""
    d = np.linalg.inv(A) @ B
    return float(np.linalg.norm(B[:3, 3] - A[:3, 3])), float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()))


def read_agents(dsg_path):
    data = json.loads(pathlib.Path(dsg_path).read_text())
    out = []
    for n in data["nodes"]:
        a = n.get("attributes", {})
        if a.get("type") != "AgentNodeAttributes":
            continue
        q = a["world_R_body"]
        out.append({"id": int(n["id"]), "symbol": symbol(int(n["id"])), "stamp": int(a["timestamp"]),
                    "T": pose(a["position"], [q["x"], q["y"], q["z"], q["w"]])})
    return sorted(out, key=lambda x: x["stamp"])


def read_db(db_path):
    """RTAB-Map nodes (stamp, raw odometry pose) and loop-closure links."""
    con = sqlite3.connect(db_path)
    nodes = {}
    for nid, stamp, blob in con.execute("select id, stamp, pose from Node"):
        if blob is None or len(blob) < 48:
            continue
        m = np.frombuffer(blob[:48], dtype=np.float32).reshape(3, 4).astype(float)
        T = np.eye(4)
        T[:3, :] = m
        nodes[nid] = {"stamp": int(round(stamp * 1e9)), "odom": T}
    links = []
    seen = set()
    for a, b, typ, blob in con.execute("select from_id, to_id, type, transform from Link"):
        if typ not in LINK_TYPES or (b, a, typ) in seen:
            continue  # each link is stored in both directions
        seen.add((a, b, typ))
        m = np.frombuffer(blob[:48], dtype=np.float32).reshape(3, 4) if blob else None
        links.append({"from": a, "to": b, "type": LINK_TYPES[typ],
                      "length_m": float(np.linalg.norm(m[:, 3])) if m is not None else None})
    return nodes, links


def read_poses(path):
    """rtabmap-export format 11: stamp x y z qx qy qz qw id -> {id: (stamp_ns, T)}."""
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        v = line.split()
        if len(v) < 9 or line.startswith("#"):
            continue
        out[int(v[8])] = (int(round(float(v[0]) * 1e9)), pose([float(x) for x in v[1:4]], [float(x) for x in v[4:8]]))
    return out


def nearest(keys, t, tol=None):
    i = bisect.bisect_left(keys, t)
    cands = [keys[j] for j in (i - 1, i) if 0 <= j < len(keys)]
    if not cands:
        return None
    best = min(cands, key=lambda k: abs(k - t))
    return best if tol is None or abs(best - t) <= tol else None


def stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dsg")
    ap.add_argument("db")
    ap.add_argument("poses")
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--sparse-poses")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    agents = read_agents(args.dsg)
    nodes, links = read_db(args.db)
    optimized = read_poses(args.poses)
    by_stamp = {n["stamp"]: nid for nid, n in nodes.items()}
    stamps = sorted(by_stamp)
    sparse = {}
    if args.sparse_poses:
        sparse = {s: T for s, T in read_poses(args.sparse_poses).values()}
    sparse_stamps = sorted(sparse)
    t0 = agents[0]["stamp"] if agents else 0

    rows = []
    for a in agents:
        s = nearest(stamps, a["stamp"], MATCH_TOL_NS)
        row = {"agent": a["symbol"], "agent_id": a["id"], "t_s": round((a["stamp"] - t0) / 1e9, 3),
               "x": round(a["T"][0, 3], 4), "y": round(a["T"][1, 3], 4), "z": round(a["T"][2, 3], 4),
               "keyframe_id": None, "in_optimized_graph": False}
        if s is not None:
            nid = by_stamp[s]
            row["keyframe_id"] = nid
            odom = nodes[nid]["odom"]
            row["loop_correction_m"], row["loop_correction_deg"] = diff(odom, a["T"])
            if nid in optimized:
                row["in_optimized_graph"] = True
                dense = optimized[nid][1]
                row["agent_vs_dense_m"], row["agent_vs_dense_deg"] = diff(dense, a["T"])
                row["dense_correction_m"], _ = diff(odom, dense)
        ks = nearest(sparse_stamps, a["stamp"], MATCH_TOL_NS)
        if ks is not None and row["keyframe_id"] in optimized:
            row["sparse_vs_dense_m"], row["sparse_vs_dense_deg"] = diff(optimized[row["keyframe_id"]][1], sparse[ks])
        rows.append(row)

    agent_stamps = [a["stamp"] for a in agents]
    closures = []
    for ln in links:
        ends = []
        for nid in (ln["from"], ln["to"]):
            n = nodes.get(nid)
            if n is None:
                ends.append((None, None))
                continue
            k = nearest(agent_stamps, n["stamp"])
            ag = agents[agent_stamps.index(k)]
            ends.append((round((n["stamp"] - t0) / 1e9, 2), ag["symbol"]))
        closures.append({"type": ln["type"], "from_keyframe": ln["from"], "to_keyframe": ln["to"],
                         "from_t_s": ends[0][0], "from_agent": ends[0][1], "to_t_s": ends[1][0], "to_agent": ends[1][1],
                         "link_length_m": round(ln["length_m"], 3) if ln["length_m"] is not None else None})
    closures.sort(key=lambda c: (c["from_t_s"] or 0))

    fields = ["agent", "agent_id", "t_s", "x", "y", "z", "keyframe_id", "in_optimized_graph",
              "loop_correction_m", "loop_correction_deg", "dense_correction_m",
              "agent_vs_dense_m", "agent_vs_dense_deg", "sparse_vs_dense_m", "sparse_vs_dense_deg"]
    with (args.out_dir / "agent_keyframes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with (args.out_dir / "loop_closures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(closures[0]) if closures else ["type"])
        w.writeheader()
        w.writerows(closures)

    col = lambda k: [r.get(k) for r in rows]  # noqa: E731
    summary = {
        "agent_nodes": len(rows),
        "rtabmap_keyframes": len(nodes),
        "rtabmap_keyframes_optimized": len(optimized),
        "agents_matched_to_keyframe": sum(r["keyframe_id"] is not None for r in rows),
        "agents_in_optimized_graph": sum(r["in_optimized_graph"] for r in rows),
        "agents_unmatched": [r["agent"] for r in rows if r["keyframe_id"] is None],
        "agents_matched_to_sparse_keyframe": sum("sparse_vs_dense_m" in r for r in rows),
        "loop_closures": {t: sum(c["type"] == t for c in closures) for t in sorted({c["type"] for c in closures})},
        "loop_correction_m (odometry -> Hydra agent)": stats(col("loop_correction_m")),
        "agent_vs_dense_m (Hydra agent vs dense optimized)": stats(col("agent_vs_dense_m")),
        "agent_vs_dense_deg": stats(col("agent_vs_dense_deg")),
        "sparse_vs_dense_m (2 Hz vs dense optimized, shared stamps)": stats(col("sparse_vs_dense_m")),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
