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
from rclpy.serialization import deserialize_message
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


def spin_once(node):
    """spin_once that survives a message rclpy cannot take (logged, skipped)."""
    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except RuntimeError as e:
        print(f"monitor: skipped a message ({e})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wait-tracking", type=float, metavar="TIMEOUT_S")
    group.add_argument("--watch", metavar="FRAMES_RAW")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("d435i_monitor")
    infos = []  # (receive time, lost)
    # raw subscription, decoded here: rclpy's own message conversion occasionally raised
    # "RuntimeError: Unable to convert call argument '0' to Python object" inside spin_once and
    # killed the monitor (office_rt3); a message that fails to decode is now skipped
    bad = [0]

    def on_info(data):
        try:
            infos.append((time.time(), deserialize_message(data, OdomInfo).lost))
        except Exception:  # noqa: BLE001
            bad[0] += 1

    node.create_subscription(OdomInfo, "/rtabmap/odom_info", on_info, 50, raw=True)
    reset = node.create_client(Empty, "/rtabmap/reset_odom")

    start = time.time()
    if args.wait_tracking is not None:
        ok_since = lost_since = None
        while time.time() - start < args.wait_tracking:
            spin_once(node)
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
        spin_once(node)
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
