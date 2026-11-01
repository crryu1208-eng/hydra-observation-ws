#!/usr/bin/env python3
"""Write a copy of a D435i session bag whose odom -> camera_link TF follows the RTAB-Map SLAM
(loop-closure optimized) trajectory.

RTAB-Map optimizes only its keyframes. For each keyframe k the correction C_k = opt_k * odom_k^-1
is computed (odom_k from the bag's /rtabmap/odom at the same stamp). Every odom -> camera_link
transform at time t is replaced by C(t) * odom(t), with C(t) interpolated between the
surrounding keyframes (linear translation, slerp rotation) and held constant outside them.
Frame names stay the same, so Hydra consumes the corrected bag unchanged.

usage: apply_loop_closures.py <src_bag_dir> <poses.txt (rtabmap-export format 11)> <out_bag_dir>
       apply_loop_closures.py <src_bag_dir> <out_bag_dir> --filter-slam-input <topic>...
"""

import argparse
import bisect
import json
import pathlib

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions
from scipy.spatial.transform import Rotation, Slerp

ODOM_TOPIC = "/rtabmap/odom"
ODOM_FRAME, BODY_FRAME = "odom", "camera_link"


def mat(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def open_reader(path):
    r = SequentialReader()
    r.open(StorageOptions(uri=str(path), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    return r


def read_keyframes(poses_file):
    """stamp_ns -> optimized 4x4 pose (format 11: stamp x y z qx qy qz qw id)."""
    out = {}
    for line in pathlib.Path(poses_file).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split()
        stamp = int(round(float(v[0]) * 1e9))
        out[stamp] = mat([float(x) for x in v[1:4]], [float(x) for x in v[4:8]])
    return out


def is_valid(odom_msg):
    q = odom_msg.pose.pose.orientation
    return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 0.5


def filter_slam_input(src_bag, out_bag, topics):
    """Copy the SLAM input topics, dropping null odometry: rtabmap_slam treats a null pose as an
    odometry reset and starts a new map, while the live odometry continued from its last pose."""
    from nav_msgs.msg import Odometry
    reader = open_reader(src_bag)
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(out_bag), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    for meta in reader.get_all_topics_and_types():
        if meta.name in topics:
            writer.create_topic(meta)
    dropped = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in topics:
            continue
        if topic == ODOM_TOPIC and not is_valid(deserialize_message(data, Odometry)):
            dropped += 1
            continue
        writer.write(topic, data, t)
    del writer
    print(json.dumps({"null_odometry_dropped": dropped}))


def read_odom(bag):
    from nav_msgs.msg import Odometry
    r = open_reader(bag)
    odom = {}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic != ODOM_TOPIC:
            continue
        m = deserialize_message(data, Odometry)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        if not is_valid(m):
            continue  # null odometry published while tracking was lost
        odom[stamp_ns(m.header.stamp)] = mat([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
    return odom


def nearest(sorted_keys, t, tol_ns):
    i = bisect.bisect_left(sorted_keys, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_keys) and abs(sorted_keys[j] - t) <= tol_ns:
            if best is None or abs(sorted_keys[j] - t) < abs(best - t):
                best = sorted_keys[j]
    return best


class Correction:
    def __init__(self, stamps, corrections):
        self.t = np.array(stamps, dtype=np.int64)
        self.trans = np.array([C[:3, 3] for C in corrections])
        rots = Rotation.from_matrix(np.array([C[:3, :3] for C in corrections]))
        self.rots = rots
        self.slerp = Slerp(self.t.astype(float), rots) if len(stamps) > 1 else None

    def at(self, t):
        C = np.eye(4)
        if t <= self.t[0] or self.slerp is None:
            C[:3, :3] = self.rots[0].as_matrix()
            C[:3, 3] = self.trans[0]
        elif t >= self.t[-1]:
            C[:3, :3] = self.rots[-1].as_matrix()
            C[:3, 3] = self.trans[-1]
        else:
            i = np.searchsorted(self.t, t) - 1
            a = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
            C[:3, 3] = (1 - a) * self.trans[i] + a * self.trans[i + 1]
            C[:3, :3] = self.slerp([float(t)]).as_matrix()[0]
        return C


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src_bag", type=pathlib.Path)
    ap.add_argument("poses", type=pathlib.Path, nargs="?")
    ap.add_argument("out_bag", type=pathlib.Path)
    ap.add_argument("--filter-slam-input", nargs="+", metavar="TOPIC",
                    help="only write these topics of src_bag to out_bag, without null odometry")
    args = ap.parse_args()
    if args.filter_slam_input:
        filter_slam_input(args.src_bag, args.out_bag, set(args.filter_slam_input))
        return
    from tf2_msgs.msg import TFMessage

    keyframes = read_keyframes(args.poses)
    odom = read_odom(args.src_bag)
    odom_keys = sorted(odom)
    stamps, corrections, unmatched = [], [], 0
    for t in sorted(keyframes):
        k = nearest(odom_keys, t, tol_ns=2_000_000)  # export prints stamps with limited precision
        if k is None:
            unmatched += 1
            continue
        stamps.append(t)
        corrections.append(keyframes[t] @ np.linalg.inv(odom[k]))
    if not stamps:
        raise SystemExit("no keyframe stamp matched /rtabmap/odom; wrong poses file?")
    corr = Correction(stamps, corrections)

    reader = open_reader(args.src_bag)
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(args.out_bag), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    for meta in reader.get_all_topics_and_types():
        writer.create_topic(meta)

    shifts, n_tf = [], 0
    while reader.has_next():
        topic, data, t_rec = reader.read_next()
        if topic == "/tf":
            msg = deserialize_message(data, TFMessage)
            changed = False
            for tr in msg.transforms:
                if tr.header.frame_id != ODOM_FRAME or tr.child_frame_id != BODY_FRAME:
                    continue
                p, q = tr.transform.translation, tr.transform.rotation
                T = mat([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
                Tc = corr.at(stamp_ns(tr.header.stamp)) @ T
                qc = Rotation.from_matrix(Tc[:3, :3]).as_quat()
                p.x, p.y, p.z = (float(v) for v in Tc[:3, 3])
                q.x, q.y, q.z, q.w = (float(v) for v in qc)
                shifts.append(float(np.linalg.norm(Tc[:3, 3] - T[:3, 3])))
                changed = True
                n_tf += 1
            if changed:
                data = serialize_message(msg)
        writer.write(topic, data, t_rec)
    del writer

    trans = [np.linalg.norm(C[:3, 3]) for C in corrections]
    angles = [np.degrees(Rotation.from_matrix(C[:3, :3]).magnitude()) for C in corrections]
    report = {
        "keyframes": len(keyframes), "keyframes_matched": len(stamps), "keyframes_unmatched": unmatched,
        "tf_corrected": n_tf,
        "correction_at_end_m": round(float(trans[-1]), 3), "correction_at_end_deg": round(float(angles[-1]), 2),
        "max_shift_m": round(max(shifts), 3) if shifts else 0.0,
        "mean_shift_m": round(float(np.mean(shifts)), 3) if shifts else 0.0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
