#!/usr/bin/env python3
"""Per-stage throughput and latency of the D435i -> segmentation -> Hydra -> recorder pipeline.

Subscribes to the image topics *without deserializing them* (raw CDR, only the header stamp is
parsed), so the monitor itself adds almost no load even at 1280x720. The recorder output is
followed by tailing frames_raw.jsonl. Every stage is keyed by the color image stamp, so the
latency of a stage is its wall-clock arrival time minus the arrival time of the same color frame.

  stage_monitor.py --out summary.json [--frames-raw <obs>/frames_raw.jsonl] [--duration s]
Stops on SIGINT/SIGTERM or after --duration and writes the summary.
"""
import argparse
import json
import os
import signal
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String

CAM = "/camera/camera"
TOPICS = {
    "rgb": (f"{CAM}/color/image_raw", Image),
    "depth": (f"{CAM}/aligned_depth_to_color/image_raw", Image),
    "labels": (f"{CAM}/semantic/image_raw", Image),
    "odom": ("/rtabmap/odom", Odometry),
}


def stamp_ns(raw: bytes) -> int:
    # CDR: 4-byte encapsulation header, then std_msgs/Header: int32 sec, uint32 nanosec
    sec, nsec = struct.unpack_from("<iI", raw, 4)
    return sec * 1_000_000_000 + nsec


class Monitor(Node):
    def __init__(self, frames_raw, save_labels=None):
        super().__init__("stage_monitor")
        self.lock = threading.Lock()
        self.arrivals = {k: {} for k in list(TOPICS) + ["recorded"]}
        qos = QoSProfile(depth=200, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        for key, (topic, typ) in TOPICS.items():
            self.create_subscription(typ, topic, lambda m, k=key: self.on_msg(k, m), qos, raw=True)
        self.save_labels = save_labels
        if save_labels:
            os.makedirs(save_labels, exist_ok=True)
        self.seg_elapsed = []
        self.create_subscription(String, f"{CAM}/semantic_inference/status", self.on_status, 10)
        self.frames_raw = frames_raw
        self.offset = 0
        if frames_raw:
            self.create_timer(0.05, self.poll_recorder)

    def on_msg(self, key, raw):
        t = time.monotonic()
        stamp = stamp_ns(raw)
        with self.lock:
            self.arrivals[key].setdefault(stamp, t)
        # keep every 10th label image (by stamp, so runs keep the same frames)
        if key == "labels" and self.save_labels and (stamp // 1_000_000) % 10 == 0:
            msg = deserialize_message(raw, Image)
            img = np.frombuffer(msg.data, dtype=np.int16).reshape(msg.height, msg.width)
            np.save(os.path.join(self.save_labels, f"{stamp}.npy"), img)

    def on_status(self, msg):
        note = json.loads(msg.data).get("note", "")
        if note.startswith("Average elapsed time:"):
            self.seg_elapsed.append(float(note.split(":")[1].split()[0]) * 1000)

    def poll_recorder(self):
        if not os.path.exists(self.frames_raw):
            return
        t = time.monotonic()
        with open(self.frames_raw) as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind("\n") + 1
        self.offset += len(data[:end].encode())
        with self.lock:
            for line in data[:end].splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self.arrivals["recorded"].setdefault(int(rec["timestamp_ns_str"]), t)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def summarize(arr):
    rgb = arr["rgb"]
    out = {}
    if not rgb:
        return {"error": "no color frames"}
    t0, t1 = min(rgb.values()), max(rgb.values())
    out["window_s"] = round(t1 - t0, 2)
    for key, a in arr.items():
        if not a:
            out[key] = {"count": 0}
            continue
        ts = sorted(a.values())
        span = ts[-1] - ts[0]
        lat = [(a[s] - rgb[s]) * 1000 for s in a if s in rgb]
        gaps = [(b - c) * 1000 for c, b in zip(ts, ts[1:])]
        out[key] = {
            "count": len(a),
            "fraction_of_rgb": round(sum(1 for s in a if s in rgb) / len(rgb), 3),
            "fps": round((len(ts) - 1) / span, 2) if span > 0 else None,
            "latency_ms_p50": round(pct(lat, 0.5), 1) if lat else None,
            "latency_ms_p90": round(pct(lat, 0.9), 1) if lat else None,
            "latency_ms_max": round(max(lat), 1) if lat else None,
            "gap_ms_p50": round(pct(gaps, 0.5), 1) if gaps else None,
            "gap_ms_max": round(max(gaps), 1) if gaps else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-raw")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--save-labels")
    args = ap.parse_args()
    rclpy.init()
    node = Monitor(args.frames_raw, args.save_labels)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()
    t_end = time.monotonic() + args.duration if args.duration else None
    while not stop.is_set() and (t_end is None or time.monotonic() < t_end):
        time.sleep(0.2)
    node.poll_recorder() if args.frames_raw else None
    with node.lock:
        summary = summarize(node.arrivals)
        summary["seg_node_elapsed_ms_p50"] = round(pct(node.seg_elapsed, 0.5), 1) if node.seg_elapsed else None
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
