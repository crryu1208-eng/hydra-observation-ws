#!/usr/bin/env python3
"""Quick health check of a finished session, printed right after recording:
odometry and recorder rates, what the bag holds for a later replay, frames without aligned depth.
  session_check.py <session dir>"""
import json
import re
import sys

import yaml

S = sys.argv[1]
warn = []
m = yaml.safe_load(open(f"{S}/bag/metadata.yaml"))["rosbag2_bagfile_information"]
dur = m["duration"]["nanoseconds"] / 1e9
c = {t["topic_metadata"]["name"]: t["message_count"] for t in m["topics_with_message_count"]}
color = c.get("/camera/camera/color/image_raw", 0)
depth = c.get("/camera/camera/aligned_depth_to_color/image_raw", 0)
raw = c.get("/rtabmap/odom_raw", 0)
odom_t = []
try:
    for line in open(f"{S}/logs/launch_odom.log"):
        mm = re.search(r"\[(\d{10}\.\d+)\].*Odom: quality=\d+", line)
        if mm:
            odom_t.append(float(mm.group(1)))
except FileNotFoundError:
    pass
odom_hz = (len(odom_t) - 1) / (odom_t[-1] - odom_t[0]) if len(odom_t) > 1 else 0.0
frames = []
try:
    frames = [int(json.loads(l)["timestamp_ns"]) for l in open(f"{S}/observations/frames_raw.jsonl")]
except FileNotFoundError:
    pass
fps = (len(frames) - 1) / ((frames[-1] - frames[0]) / 1e9) if len(frames) > 1 else 0.0
cam_fps = color / dur if dur else 0.0
print(f"session check: bag {dur:.0f} s, color {cam_fps:.1f} Hz, aligned depth missing "
      f"{100 * max(0, color - depth) / max(color, 1):.1f}% | odometry {odom_hz:.1f} Hz | "
      f"recorded {len(frames)} frames at {fps:.1f} fps | bag odom_raw {raw}")
if raw == 0:
    warn.append("the bag has no /rtabmap/odom_raw: this session cannot be replayed with its live odometry")
if odom_hz and odom_hz < 0.75 * cam_fps:
    warn.append(f"odometry ran at {odom_hz:.1f} Hz for {cam_fps:.1f} Hz camera frames (overload?)")
if len(frames) > 1 and fps < 0.75 * cam_fps:
    warn.append(f"Hydra recorded {fps:.1f} fps for {cam_fps:.1f} Hz camera frames")
if len(frames) < 0.5 * max(1, color):
    warn.append(f"only {len(frames)} of {color} camera frames were mapped (waiting for the floor?)")
for w in warn:
    print(f"=== WARNING: {w} ===")
sys.exit(1 if warn else 0)
