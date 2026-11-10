#!/usr/bin/env python3
"""Publish a session's corrected per-frame poses as TF (odom -> camera_link) while its bag plays.

Used to rebuild the map after a live session: Hydra runs again on the recorded images with the
loop-closed poses as odometry (no loop closures), so the mesh, objects and places are made with the
same poses as observations/frames.jsonl (corrected.world_T_body) and cannot disagree with them.
After a live session the backend mesh did (office_rt7-rt11: 15-54% of the keyframes 10-30 cm
off the mesh in loop-closed parts, kimera_pgmo deforms the mesh only approximately).

Each pose is published once the sim clock is within lead_s of its stamp, so Hydra finds it when
the image with that stamp arrives. Run with use_sim_time (bag played with --clock).
With nodes:=<session>/observations/trajectory_nodes.csv the session's keyframes (the live agent
nodes a0, a1, ... with their loop-closed poses) also go to /hydra/pose_graph as the bridge sends
them (previous + new node, odometry edge; no loop closures), for Hydra with the RosPoseGraphs
tracker: the rebuilt scene graph then has the same agent nodes as frames.jsonl refers to.
  pose_tf_player.py --ros-args -p frames:=<session>/observations/frames.jsonl -p use_sim_time:=true
"""
import csv
import json

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, TransformStamped
from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge, PoseGraphNode
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster


class PoseTfPlayer(Node):
    def __init__(self):
        super().__init__("pose_tf_player")
        path = self.declare_parameter("frames", "").value
        self.lead = self.declare_parameter("lead_s", 0.5).value
        self.parent = self.declare_parameter("parent_frame", "odom").value
        self.child = self.declare_parameter("child_frame", "camera_link").value
        self.poses = []
        for line in open(path):
            f = json.loads(line)
            c = f.get("corrected") or {}
            if c.get("status") != "ok":
                continue
            self.poses.append((int(f["timestamp_ns"]), c["world_T_body"]))
        self.poses.sort(key=lambda x: x[0])
        self.next = 0
        self.br = TransformBroadcaster(self)
        nodes = self.declare_parameter("nodes", "").value
        self.robot_id = self.declare_parameter("robot_id", 0).value
        self.nodes = []
        if nodes:
            for r in csv.DictReader(open(nodes)):
                q = [float(r[k]) for k in ("qx", "qy", "qz", "qw")]
                self.nodes.append((int(r["timestamp_ns"]), to_mat([float(r[k]) for k in "xyz"], q)))
            self.nodes.sort(key=lambda x: x[0])
            self.pg_pub = self.create_publisher(PoseGraph, "/hydra/pose_graph", QoSProfile(
                depth=100, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE))
            self.get_logger().info(f"{len(self.nodes)} keyframes from {nodes} -> /hydra/pose_graph")
        self.next_node = 0
        # ros2 bag play publishes /clock best effort
        self.create_subscription(Clock, "/clock", self.on_clock,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f"{len(self.poses)} corrected poses from {path}")

    def on_clock(self, msg):
        now = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec
        horizon = now + int(self.lead * 1e9)
        out = []
        while self.next < len(self.poses) and self.poses[self.next][0] <= horizon:
            stamp, p = self.poses[self.next]
            t = TransformStamped()
            t.header.stamp.sec, t.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            t.header.frame_id, t.child_frame_id = self.parent, self.child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p["t"])
            w, x, y, z = map(float, p["q_wxyz"])
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
            out.append(t)
            self.next += 1
        if out:
            self.br.sendTransform(out)
        while self.nodes and self.next_node < len(self.nodes) and self.nodes[self.next_node][0] <= horizon:
            self.send_node(self.next_node)
            self.next_node += 1

    def send_node(self, i):
        g = PoseGraph()
        stamp, T = self.nodes[i]
        g.header.stamp = stamp_msg(stamp)
        g.header.frame_id = self.parent
        for k in ([i] if i == 0 else [i - 1, i]):
            n = PoseGraphNode()
            n.header.stamp = stamp_msg(self.nodes[k][0])
            n.header.frame_id = self.parent
            n.robot_id = self.robot_id
            n.key = k
            n.pose = to_pose(self.nodes[k][1])
            g.nodes.append(n)
        if i > 0:
            e = PoseGraphEdge()
            e.header.stamp = stamp_msg(stamp)
            e.header.frame_id = self.parent
            e.key_from, e.key_to = i - 1, i
            e.robot_from = e.robot_to = self.robot_id
            e.type = PoseGraphEdge.ODOM
            e.pose = to_pose(np.linalg.inv(self.nodes[i - 1][1]) @ T)
            g.edges.append(e)
        self.pg_pub.publish(g)


def stamp_msg(ns):
    from builtin_interfaces.msg import Time
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def to_mat(t, q_xyzw):
    x, y, z, w = q_xyzw
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = t
    return T


def to_pose(T):
    R = T[:3, :3]
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = float(x), float(y), float(z), float(w)
    return p


def main():
    rclpy.init()
    node = PoseTfPlayer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
