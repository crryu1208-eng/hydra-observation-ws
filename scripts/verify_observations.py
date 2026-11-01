#!/usr/bin/env python3
"""Check that every RGB/depth frame of a ROS 2 (sqlite3) bag was recorded and mapped.

usage (in the hydra-dev container, after map_observations.py has run):
  python3 verify_observations.py /root/data/office \
      /root/data/hydra_observations/uhumans2_office [--launch-log <run>/launch.log]

Header stamps are read straight from the CDR bytes (encapsulation header, then
builtin_interfaces/Time sec:int32 nanosec:uint32), so no message deserialization is needed.
"""

import argparse
import collections
import json
import pathlib
import sqlite3
import struct
import sys

RGB_TOPIC = "/tesse/left_cam/rgb/image_raw"
DEPTH_TOPIC = "/tesse/depth_cam/mono/image_raw"


def header_stamps(db, topic):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    (tid,) = con.execute("select id from topics where name=?", (topic,)).fetchone()
    stamps = []
    for (head,) in con.execute(
        "select substr(data, 1, 12) from messages where topic_id=? order by timestamp", (tid,)
    ):
        little = head[1] == 1  # CDR_LE representation id 0x0001
        sec, nsec = struct.unpack("<iI" if little else ">iI", head[4:12])
        stamps.append(sec * 1_000_000_000 + nsec)
    return stamps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", type=pathlib.Path)
    parser.add_argument("observations", type=pathlib.Path)
    parser.add_argument("--launch-log", type=pathlib.Path, default=None)
    args = parser.parse_args()

    db = next(args.bag.glob("*.db3"))
    rgb = header_stamps(db, RGB_TOPIC)
    depth = header_stamps(db, DEPTH_TOPIC)

    frames = [json.loads(x) for x in (args.observations / "frames_raw.jsonl").open()]
    written = [f for f in frames if f["status"] == "written"]
    rec = [f["timestamp_ns"] for f in written]

    missing_files = [
        f["frame_id"]
        for f in written
        if not (args.observations / f["rgb_path"]).is_file()
        or (f["has_depth"] and not (args.observations / f["depth_path"]).is_file())
    ]

    report = {
        "bag_rgb_frames": len(rgb),
        "bag_depth_frames": len(depth),
        "bag_rgb_unique_stamps": len(set(rgb)),
        "bag_rgb_depth_same_stamps": set(rgb) == set(depth),
        "recorded_status": dict(collections.Counter(f["status"] for f in frames)),
        "recorded_written": len(written),
        "recorded_with_depth": sum(f["has_depth"] for f in written),
        "recorded_duplicate_stamps": len(rec) - len(set(rec)),
        "bag_rgb_not_recorded": len(set(rgb) - set(rec)),
        "recorded_not_in_bag": len(set(rec) - set(rgb)),
        "written_frames_missing_files": len(missing_files),
    }
    not_recorded = sorted(set(rgb) - set(rec))
    if not_recorded:
        report["first_not_recorded_ns"] = not_recorded[:10]

    stats_path = args.observations / "recorder_stats.json"
    if stats_path.is_file():
        report["recorder_stats"] = json.loads(stats_path.read_text())

    mapping_path = args.observations / "mapping_stats.json"
    if mapping_path.is_file():
        report["mapping_stats"] = json.loads(mapping_path.read_text())

    if args.launch_log and args.launch_log.is_file():
        log = args.launch_log.read_text(errors="replace")
        report["hydra_log"] = {
            "dropping_input_missing_pose": log.count("due to missing pose"),
            "clearing_input_queues": log.count("Clearing input queues"),
            "error_lines": sum("[ERROR]" in x for x in log.splitlines()),
            "fatal_lines": sum("[FATAL]" in x for x in log.splitlines()),
        }

    print(json.dumps(report, indent=2))
    ok = (
        report["bag_rgb_not_recorded"] == 0
        and report["recorded_not_in_bag"] == 0
        and report["recorded_duplicate_stamps"] == 0
        and report["written_frames_missing_files"] == 0
        and report["recorded_with_depth"] == report["recorded_written"]
    )
    print("RESULT:", "every bag RGB/depth frame recorded" if ok else "FRAMES MISSING")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
