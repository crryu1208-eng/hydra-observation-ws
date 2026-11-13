#!/usr/bin/env python3
"""Export a scene graph with the RGB and depth of every keyframe (agent node).

  export_keyframes.py <map dir with hydra/backend> <frames.jsonl> <out dir>

frames.jsonl: the frames with their poses in the map's frame, e.g. <session>/rebuilt/levelled/
observations/frames.jsonl, or <merged>/merged_poses/frames.jsonl from merge_sessions.sh (each frame
then names its source_observations). Every agent node of the scene graph is matched to the frame
with its timestamp (keyframes are recorded frames).
Out:
  hydra/backend/   dsg_with_mesh.json dsg.json mesh.ply trajectory.csv   (copied)
  keyframes/keyframes.jsonl   one line per agent node: node, time, source session / frame,
                              world_T_body / world_T_sensor (map frame, of that frame), image
                              sharpness; frame_dt_ms != 0 when the node's own image was not
                              recorded and the nearest frame (within 0.3 s) is used
  keyframes/rgb/<node>.png    color image (BGR PNG as recorded)
  keyframes/depth/<node>.png  aligned depth, uint16 mm, same size as the color image
  camera_calib.json           intrinsics, body_T_sensor
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

MAP, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists")
frames_dir = os.path.dirname(os.path.abspath(FRAMES))
frames = {}
for line in open(FRAMES):
    f = json.loads(line)
    frames[int(f["timestamp_ns"])] = f
stamps = np.array(sorted(frames))
g = json.load(open(f"{MAP}/hydra/backend/dsg.json"))
agents = sorted((n for n in g["nodes"] if n["attributes"].get("type") == "AgentNodeAttributes"),
                key=lambda n: (n["attributes"].get("timestamp", 0), n["id"]))

os.makedirs(f"{OUT}/hydra/backend")
os.makedirs(f"{OUT}/keyframes/rgb")
os.makedirs(f"{OUT}/keyframes/depth")
for name in ("dsg_with_mesh.json", "dsg.json", "mesh.ply", "trajectory.csv"):
    if os.path.exists(f"{MAP}/hydra/backend/{name}"):
        shutil.copy(f"{MAP}/hydra/backend/{name}", f"{OUT}/hydra/backend/{name}")


def symbol(n):
    s = n.get("symbol") or n["attributes"].get("name")
    if s:
        return s
    i = n["id"]
    return f"{chr(i >> 56)}{i & ((1 << 56) - 1)}"


missing, calib, rows = 0, None, []
for n in agents:
    t = int(n["attributes"].get("timestamp", 0))
    f = frames.get(t)
    sym = symbol(n)
    dt_ms = 0.0
    if f is None:  # its own image was not recorded: the nearest frame within 0.3 s
        k = int(np.argmin(np.abs(stamps - t)))
        dt_ms = (int(stamps[k]) - t) / 1e6
        if abs(dt_ms) > 300:
            missing += 1
            print(f"no frame within 0.3 s for agent node {sym} (t {t})")
            continue
        f = frames[int(stamps[k])]
        print(f"agent node {sym}: nearest frame {dt_ms:+.0f} ms")
    src = f.get("source_observations", frames_dir)
    calib = calib or f"{src}/camera_calib.json"
    rgb = cv2.imread(f"{src}/{f['rgb_path']}", cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f"{OUT}/keyframes/rgb/{sym}.png", rgb)
    shutil.copy(f"{src}/{f['depth_path']}", f"{OUT}/keyframes/depth/{sym}.png")
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
    c = f.get("corrected") or {}
    rows.append({
        "node": sym, "node_id": n["id"], "timestamp_ns": t,
        "session": os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(f"{src}/rgb")))),
        "source_frame": f["rgb_path"], "frame_id": f.get("frame_id"), "frame_dt_ms": round(dt_ms, 1),
        "rgb": f"keyframes/rgb/{sym}.png", "depth": f"keyframes/depth/{sym}.png",
        "world_T_body": c.get("world_T_body", f.get("world_T_body")),
        "world_T_sensor": c.get("world_T_sensor", f.get("world_T_sensor")),
        "sharpness": round(float(cv2.Laplacian(cv2.resize(gray, (640, 360)), cv2.CV_64F).var()), 1),
    })
with open(f"{OUT}/keyframes/keyframes.jsonl", "w") as fo:
    fo.writelines(json.dumps(r) + "\n" for r in rows)
if calib:
    shutil.copy(calib, f"{OUT}/camera_calib.json")
sessions = sorted({r["session"] for r in rows})
print(f"{len(rows)} keyframes exported ({missing} agent nodes without a frame) from {sessions} -> {OUT}")
