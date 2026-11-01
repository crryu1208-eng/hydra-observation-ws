#!/usr/bin/env python3
"""Count messages per topic for N seconds (C-level rate check without `ros2 topic hz`).

usage: python3 count_topics.py <seconds> <topic>...   (sensor_msgs/Image topics)
"""
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import Image

rclpy.init()
node = rclpy.create_node("count_topics")
dur = float(sys.argv[1])
counts = {t: 0 for t in sys.argv[2:]}
bright = {}


def cb(topic):
    def f(msg):
        counts[topic] += 1
        if msg.encoding in ("rgb8", "bgr8"):
            bright[topic] = float(np.frombuffer(msg.data, np.uint8).mean())
    return f


subs = [node.create_subscription(Image, t, cb(t), 50) for t in counts]
end = time.time() + dur
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
for t, n in counts.items():
    print(f"{t}: {n / dur:.1f} Hz", f"(mean brightness {bright[t]:.0f})" if t in bright else "")
