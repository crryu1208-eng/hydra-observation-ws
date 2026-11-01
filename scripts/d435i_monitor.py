#!/usr/bin/env python3
"""Live status for a D435i session: RTAB-Map odometry rate/tracking and recorded frame count.

usage (in the hydra-dev container):
  python3 d435i_monitor.py --wait-tracking <timeout_s>   exit 0 once tracking is stable
  python3 d435i_monitor.py --watch <frames_raw.jsonl>    print a status line every 2 s

--wait-tracking calls /rtabmap/reset_odom when odometry stays lost (e.g. it initialized on a
frame taken while auto exposure was still settling, after which it never recovers).
"""

import argparse
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rtabmap_msgs.msg import OdomInfo
from std_srvs.srv import Empty

STABLE_S = 2.0
RESET_AFTER_S = 3.0


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wait-tracking", type=float, metavar="TIMEOUT_S")
    group.add_argument("--watch", metavar="FRAMES_RAW")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("d435i_monitor")
    infos = []  # (receive time, lost)
    node.create_subscription(OdomInfo, "/rtabmap/odom_info", lambda m: infos.append((time.time(), m.lost)), 50)
    reset = node.create_client(Empty, "/rtabmap/reset_odom")

    start = time.time()
    if args.wait_tracking is not None:
        ok_since = lost_since = None
        while time.time() - start < args.wait_tracking:
            rclpy.spin_once(node, timeout_sec=0.1)
            if not infos:
                continue
            now, lost = infos[-1]
            if lost:
                ok_since = None
                lost_since = lost_since or now
                if now - lost_since > RESET_AFTER_S and reset.service_is_ready():
                    print("odometry lost, resetting (point the camera at a textured scene)", flush=True)
                    reset.call_async(Empty.Request())
                    lost_since = None
            else:
                lost_since = None
                ok_since = ok_since or now
                if now - ok_since > STABLE_S:
                    print("odometry tracking", flush=True)
                    return 0
        print("odometry did not start tracking", flush=True)
        return 1

    next_print = time.time() + 2.0
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.time()
        if now < next_print:
            continue
        next_print = now + 2.0
        recent = [lost for t, lost in infos if now - t < 2.0]
        del infos[: max(0, len(infos) - 100)]
        state = "NO ODOMETRY" if not recent else ("TRACKING LOST - move back slowly" if recent[-1] else "tracking ok")
        print(
            f"[{now - start:6.0f}s] odom {len(recent) / 2.0:4.1f} Hz | {state:32s} | "
            f"frames recorded {count_lines(args.watch)}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
