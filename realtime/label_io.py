#!/usr/bin/env python3
"""Save or play label images by frame time (for swapping the segmentation, see seg_labels.sh).

  label_io.py save <dir>   subscribes <camera_ns>/semantic/image_raw and writes <dir>/<stamp_ns>.png
  label_io.py play <dir>   for every color image, publishes <dir>/<stamp_ns>.png (same stamp) on
                           <camera_ns>/semantic/image_raw, in place of the segmentation node
Labels are written / read as 16-bit PNGs; published with the encoding the closed-set node uses
(saved in <dir>/encoding.txt by 'save', mono16 / 16UC1 otherwise).
"""
import os, sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

CAM = "/camera/camera"
mode, d = sys.argv[1], sys.argv[2]
os.makedirs(d, exist_ok=True)
DT = {"mono8": np.uint8, "8UC1": np.uint8, "mono16": np.uint16, "16UC1": np.uint16, "16SC1": np.int16, "32SC1": np.int32}


class Saver(Node):
    def __init__(self):
        super().__init__("label_saver")
        q = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.n = 0
        self.create_subscription(Image, f"{CAM}/semantic/image_raw", self.cb, q)

    def cb(self, m):
        a = np.frombuffer(m.data, DT[m.encoding]).reshape(m.height, m.width)
        if self.n == 0:
            open(f"{d}/encoding.txt", "w").write(m.encoding)
        cv2.imwrite(f"{d}/{m.header.stamp.sec * 10**9 + m.header.stamp.nanosec}.png", a.astype(np.uint16))
        self.n += 1
        if self.n % 500 == 0:
            self.get_logger().info(f"{self.n} label images saved")


class Player(Node):
    def __init__(self):
        super().__init__("label_player")
        self.enc = open(f"{d}/encoding.txt").read().strip() if os.path.exists(f"{d}/encoding.txt") else "16UC1"
        q = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Image, f"{CAM}/semantic/image_raw", q)
        self.create_subscription(Image, f"{CAM}/color/image_raw", self.cb, q)
        self.n = self.miss = 0

    def cb(self, m):
        stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        a = cv2.imread(f"{d}/{stamp}.png", cv2.IMREAD_UNCHANGED)
        if a is None:
            self.miss += 1
            if self.miss % 100 == 1:
                self.get_logger().warn(f"no labels for {stamp} ({self.miss} missing)")
            return
        a = a.astype(DT.get(self.enc, np.uint16))
        out = Image(header=m.header, height=a.shape[0], width=a.shape[1], encoding=self.enc,
                    is_bigendian=0, step=a.shape[1] * a.itemsize, data=a.tobytes())
        self.pub.publish(out)
        self.n += 1
        if self.n % 500 == 0:
            self.get_logger().info(f"{self.n} label images published ({self.miss} missing)")


rclpy.init()
node = Saver() if mode == "save" else Player()
try:
    rclpy.spin(node)
except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
    pass
