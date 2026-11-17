#!/usr/bin/env python3
"""Pass every n-th camera frame on to the mapping pipeline: the camera (and the raw bag) run at 30 fps, the
segmentation, odometry, Hydra and open-vocabulary objects at 15 Hz (lab_03 at 30 fps: Hydra keeps ~24 fps and
falls further behind every second, recording latency 0.2 -> 30 s in 2 minutes).

Color, aligned depth and their camera_info are republished under `out_ns` with the same names. Whether a frame
is kept depends only on its stamp, so the four topics agree whatever order they arrive in: kept when the stamp
falls in the first `1 / in_rate` s of a `1 / rate` s slot (one frame of every in_rate / rate). Messages are
passed on serialized (no copy into Python objects).

  frame_decimator.py --ros-args -p in_ns:=/camera/camera -p out_ns:=/camera/decimated -p in_rate:=30.0 -p rate:=15.0
"""
import struct

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

TOPICS = (("color/image_raw", Image), ("color/camera_info", CameraInfo),
          ("aligned_depth_to_color/image_raw", Image), ("aligned_depth_to_color/camera_info", CameraInfo))


def stamp_ns(raw):
    """header stamp of a serialized message whose first field is std_msgs/Header (4 bytes CDR encapsulation)"""
    sec, nsec = struct.unpack_from("<iI", raw, 4)
    return sec * 1_000_000_000 + nsec


class FrameDecimator(Node):
    def __init__(self):
        super().__init__("frame_decimator")
        # dynamic typing: "-p rate:=15" arrives as an integer, "15.0" as a float
        p = lambda n, v: self.declare_parameter(n, v, ParameterDescriptor(dynamic_typing=True)).value
        in_ns, out_ns = p("in_ns", "/camera/camera").rstrip("/"), p("out_ns", "/camera/decimated").rstrip("/")
        self.in_period = int(1e9 / float(p("in_rate", 30.0))); self.period = int(1e9 / float(p("rate", 15.0)))
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.counts = {}
        for name, typ in TOPICS:
            pub = self.create_publisher(typ, f"{out_ns}/{name}", qos)
            self.create_subscription(typ, f"{in_ns}/{name}", lambda raw, pub=pub, n=name: self.on_msg(raw, pub, n), qos, raw=True)
            self.counts[name] = [0, 0]
        self.create_timer(30.0, self.report)
        self.get_logger().info(f"{in_ns} -> {out_ns}: one frame per {self.period / 1e6:.1f} ms slot")

    def on_msg(self, raw, pub, name):
        c = self.counts[name]; c[0] += 1
        if stamp_ns(raw) % self.period < self.in_period:
            pub.publish(raw); c[1] += 1

    def report(self):
        self.get_logger().info("kept / received: " + ", ".join(f"{n} {k}/{r}" for n, (r, k) in self.counts.items()))


def main():
    rclpy.init()
    node = FrameDecimator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
