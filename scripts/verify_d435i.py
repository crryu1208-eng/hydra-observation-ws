#!/usr/bin/env python3
"""Check a D435i session from run_d435i_obs.sh: bag frames vs recorded frames vs odometry.

usage (in the hydra-dev container): python3 verify_d435i.py /root/data/d435i/<session>

A frame Hydra recorded while RTAB-Map had lost tracking still gets a pose (TF interpolates
between the last pose before and the first after the loss), so such frames are listed in
<session>/logs/tracking_lost_frames.txt as suspect.
"""

import argparse
import collections
import json
import pathlib
import sqlite3
import sys

from verify_observations import header_stamps

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
ODOM_INFO_TOPIC = "/rtabmap/odom_info"


def odom_lost_stamps(db):
    """Header stamps of odometry updates that report lost tracking (OdomInfo.lost)."""
    from rclpy.serialization import deserialize_message
    from rtabmap_msgs.msg import OdomInfo

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute("select id from topics where name=?", (ODOM_INFO_TOPIC,)).fetchone()
    if row is None:
        return None, set()
    total, lost = 0, set()
    for (data,) in con.execute("select data from messages where topic_id=?", row):
        msg = deserialize_message(data, OdomInfo)
        total += 1
        if msg.lost:
            lost.add(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
    return total, lost


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=pathlib.Path)
    args = parser.parse_args()
    obs = args.session / "observations"
    logs = args.session / "logs"

    db = next((args.session / "bag").glob("*.db3"))
    rgb = header_stamps(db, RGB_TOPIC)
    depth = header_stamps(db, DEPTH_TOPIC)
    odom_total, lost = odom_lost_stamps(db)

    frames = [json.loads(x) for x in (obs / "frames_raw.jsonl").open()]
    written = [f for f in frames if f["status"] == "written"]
    rec = {f["timestamp_ns"] for f in written}
    suspect = sorted(f["frame_id"] for f in written if f["timestamp_ns"] in lost)
    (logs / "tracking_lost_frames.txt").write_text("".join(f"{x}\n" for x in suspect))

    report = {
        "duration_s": (max(rgb) - min(rgb)) / 1e9 if rgb else 0,
        "bag_rgb_frames": len(rgb),
        "bag_depth_frames": len(depth),
        "odom_messages": odom_total,
        "odom_tracking_lost": len(lost),
        "recorded_status": dict(collections.Counter(f["status"] for f in frames)),
        "recorded_written": len(written),
        "recorded_with_depth": sum(f["has_depth"] for f in written),
        "bag_rgb_not_recorded": len(set(rgb) - rec),
        "recorded_while_tracking_lost": len(suspect),
    }
    for name in ("recorder_stats.json", "mapping_stats.json"):
        if (obs / name).is_file():
            report[name.removesuffix(".json")] = json.loads((obs / name).read_text())
    launch_log = logs / "launch.log"
    if launch_log.is_file():
        log = launch_log.read_text(errors="replace")
        report["hydra_log"] = {
            "dropping_input_missing_pose": log.count("due to missing pose"),
            "error_lines": sum("[ERROR]" in x for x in log.splitlines()),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
