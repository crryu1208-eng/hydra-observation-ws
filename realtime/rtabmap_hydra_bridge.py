#!/usr/bin/env python3
"""Live RTAB-Map -> Hydra bridge: RTAB-Map keyframes become Hydra's agent pose graph and
RTAB-Map loop closures become Hydra external loop closures, so Hydra's backend deforms the
mesh and scene graph while the session is running.

Inputs (RTAB-Map SLAM node, namespace /rtabmap):
  mapData (rtabmap_msgs/MapData)  each update carries the new keyframe (id, stamp, odometry pose)
                                  and the graph links, including loop closures
  odom (nav_msgs/Odometry)        exact image stamps, used to snap the keyframe stamp (MapData
                                  stores it as float seconds, ~0.2 us resolution)
Outputs (Hydra node /hydra, frontend.pose_graph_tracker type RosPoseGraphs):
  /hydra/pose_graph               one PoseGraph per keyframe: previous + new node, odometry edge
                                  (key = keyframe index, pose = odometry pose, edge pose =
                                  prev_T_curr as in Hydra's PoseGraphFromOdom)
  /hydra/external_loop_closures   one PoseGraph per batch of new loop closures: edge keys are the
                                  two keyframe stamps (Hydra looks up the agent node with the
                                  nearest stamp, here an exact match), pose = to_T_from

Tracking resets: RTAB-Map starts a new map (map_id) when odometry was reset after tracking loss;
the odometry across that gap is unknown (office_walk8: a ~60 deg turn while lost). The edge into
the first keyframe of a new map gets the variance reset_variance so loop closures can bend it
(needs kimera_pgmo honouring edge covariances, branch feature/realtime); with the default
odometry variance Hydra's GNC rejected those closures as outliers.

RTAB-Map link convention: Link(from, to, T) with T = from_T_to (pose of 'to' in 'from').
Hydra's external loop closures expect to_T_from (BackendModule: src = to, dest = from).
"""
import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge, PoseGraphNode
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rtabmap_msgs.msg import MapData

# rtabmap Link::Type
LINK_NAMES = {0: "neighbor", 1: "global_closure", 2: "local_space_closure",
              3: "local_time_closure", 4: "user_closure", 5: "virtual_closure",
              6: "neighbor_merged", 7: "pose_prior", 8: "landmark", 9: "gravity"}


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return (q[0], q[1], q[2], (R[k, j] - R[j, k]) / s)


def pose_to_mat(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    T[:3, 3] = (p.position.x, p.position.y, p.position.z)
    return T


def transform_to_mat(t):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
    return T


def mat_to_pose(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    q = rot_to_quat(T[:3, :3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
    return p


def rot_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def stamp_msg(ns):
    from builtin_interfaces.msg import Time
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


class Bridge(Node):
    def __init__(self):
        super().__init__("rtabmap_hydra_bridge")
        self.robot_id = self.declare_parameter("robot_id", 0).value
        self.closure_types = set(self.declare_parameter("closure_types", [1, 2]).value)
        self.frame_id = self.declare_parameter("frame_id", "odom").value
        self.log_path = self.declare_parameter("log_path", "").value
        self.reset_variance = self.declare_parameter("reset_variance", 1.0e4).value
        # odometry edges get a covariance that is stiff in roll / pitch / height (the odometry is
        # levelled with gravity and its height held at the floor, realtime/gravity_odom.py) and
        # keeps Hydra's odometry variance (backend covariance odom; rotation x 1e-2 as in
        # kimera_pgmo) for x / y / yaw. With one isotropic variance a 38 deg yaw loop closure
        # (office_rt10) was spread partly as roll / pitch: nodes tilted 13 deg and rose 0.5 m.
        # office_rt10 with its live odometry: backend floor vertices raised > 0.2 m 7383 -> 2113
        # (1e-3) -> 1527 (1e-5), floor plane tilt 20.6 -> 0.04 deg; the rest is near the loop
        # closure (mesh nodes still tilt). level_stiffness 0 leaves the edges without covariance
        any_number = ParameterDescriptor(dynamic_typing=True)  # "0" from the launch file is an int
        self.odom_variance = float(self.declare_parameter("odom_variance", 1.0e-2, any_number).value)
        self.level_stiffness = float(self.declare_parameter("level_stiffness", 1.0e-5, any_number).value)
        reliable = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST,
                              reliability=ReliabilityPolicy.RELIABLE)
        self.pg_pub = self.create_publisher(PoseGraph, "/hydra/pose_graph", reliable)
        self.lc_pub = self.create_publisher(PoseGraph, "/hydra/external_loop_closures", reliable)
        self.create_subscription(MapData, "/rtabmap/mapData", self.on_map_data, reliable)
        self.create_subscription(Odometry, "/rtabmap/odom", self.on_odom, QoSProfile(depth=200))
        self.odom_stamps = deque(maxlen=600)
        # rtabmap id -> (keyframe index, stamp_ns, odom pose 4x4)
        self.keyframes = {}
        self.last_map_id = None
        self.last_id = None
        self.sent_links = set()
        self.log = open(self.log_path, "a") if self.log_path not in ("", "none") else None
        self.num_closures = 0
        self.get_logger().info(f"bridging rtabmap keyframes and closures {sorted(self.closure_types)}"
                               " -> /hydra/pose_graph, /hydra/external_loop_closures")

    def write_log(self, record):
        if self.log:
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()

    def on_odom(self, msg):
        self.odom_stamps.append(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)

    def snap_stamp(self, stamp_s):
        ns = int(round(stamp_s * 1e9))
        best = min(self.odom_stamps, key=lambda s: abs(s - ns), default=None)
        if best is not None and abs(best - ns) < 2_000_000:
            return best
        return ns

    def on_map_data(self, msg):
        for node in sorted(msg.nodes, key=lambda n: n.id):
            if node.id in self.keyframes or node.stamp <= 0.0:
                continue
            self.add_keyframe(node)
        self.add_closures(msg.graph.links)

    def add_keyframe(self, node):
        index = len(self.keyframes)
        stamp_ns = self.snap_stamp(node.stamp)
        T = pose_to_mat(node.pose)
        prev = self.keyframes.get(self.last_id) if self.last_id is not None else None
        if prev is not None and stamp_ns <= prev[1]:
            self.get_logger().warn(f"keyframe {node.id} not newer than the previous one, skipped")
            return
        self.keyframes[node.id] = (index, stamp_ns, T)
        graph = PoseGraph()
        graph.header.stamp = stamp_msg(stamp_ns)
        graph.header.frame_id = self.frame_id
        nodes = [(index, stamp_ns, T)] if prev is None else [prev, (index, stamp_ns, T)]
        for k, s, P in nodes:
            n = PoseGraphNode()
            n.header.stamp = stamp_msg(s)
            n.header.frame_id = self.frame_id
            n.robot_id = self.robot_id
            n.key = k
            n.pose = mat_to_pose(P)
            graph.nodes.append(n)
        if prev is not None:
            e = PoseGraphEdge()
            e.header.stamp = stamp_msg(stamp_ns)
            e.header.frame_id = self.frame_id
            e.key_from, e.key_to = prev[0], index
            e.robot_from = e.robot_to = self.robot_id
            e.type = PoseGraphEdge.ODOM
            e.pose = mat_to_pose(np.linalg.inv(prev[2]) @ T)  # prev_T_curr
            if node.map_id != self.last_map_id:
                v = self.reset_variance  # rotation x 1e-2 as kimera_pgmo's scalar variance
                e.covariance = list(np.diag([v, v, v, 1e-2 * v, 1e-2 * v, 1e-2 * v]).ravel())
                self.get_logger().warn(
                    f"keyframe {node.id} starts RTAB-Map map {node.map_id} (tracking reset): "
                    f"odometry edge variance {self.reset_variance}")
            elif self.level_stiffness > 0:
                e.covariance = list(self.level_covariance(T[:3, :3]).ravel())
            graph.edges.append(e)
        self.pg_pub.publish(graph)

        reset = prev is not None and node.map_id != self.last_map_id
        self.last_id = node.id
        self.last_map_id = node.map_id
        self.write_log({"event": "keyframe", "rtabmap_id": node.id, "key": index,
                        "stamp_ns": stamp_ns, "map_id": node.map_id, "reset_edge": reset,
                        "wall": time.time()})

    def level_covariance(self, world_R_body):
        """6x6 covariance (x y z rx ry rz) in the body frame of the edge's to-node: Hydra's
        odometry variance for x / y / yaw, level_stiffness times that for z / roll / pitch
        (world axes)."""
        v, k = self.odom_variance, self.level_stiffness
        Rt = world_R_body.T
        cov = np.zeros((6, 6))
        cov[:3, :3] = Rt @ np.diag([v, v, k * v]) @ Rt.T
        cov[3:, 3:] = Rt @ np.diag([1e-2 * k * v, 1e-2 * k * v, 1e-2 * v]) @ Rt.T
        return cov

    def add_closures(self, links):
        graph = PoseGraph()
        for link in links:
            if link.type not in self.closure_types:
                continue
            key = (min(link.from_id, link.to_id), max(link.from_id, link.to_id))
            if key in self.sent_links:
                continue
            a, b = self.keyframes.get(link.from_id), self.keyframes.get(link.to_id)
            if a is None or b is None:
                continue  # an endpoint was never published as a keyframe (yet)
            self.sent_links.add(key)
            from_T_to = transform_to_mat(link.transform)
            # how far the closure moves the endpoints relative to odometry (diagnostic)
            odom_from_T_to = np.linalg.inv(a[2]) @ b[2]
            corr = np.linalg.inv(odom_from_T_to) @ from_T_to
            e = PoseGraphEdge()
            e.header.stamp = stamp_msg(max(a[1], b[1]))
            e.header.frame_id = self.frame_id
            e.key_from, e.key_to = a[1], b[1]  # stamps: Hydra resolves them to agent nodes
            e.robot_from = e.robot_to = self.robot_id
            e.type = PoseGraphEdge.LOOPCLOSE
            e.pose = mat_to_pose(np.linalg.inv(from_T_to))  # to_T_from
            graph.edges.append(e)
            self.num_closures += 1
            rec = {"event": "closure", "type": LINK_NAMES.get(link.type, link.type),
                   "from_id": link.from_id, "to_id": link.to_id,
                   "from_stamp_ns": a[1], "to_stamp_ns": b[1],
                   "dt_s": round((a[1] - b[1]) / 1e9, 2),
                   "correction_m": round(float(np.linalg.norm(corr[:3, 3])), 3),
                   "correction_deg": round(rot_angle_deg(corr[:3, :3]), 2), "wall": time.time()}
            self.write_log(rec)
            self.get_logger().info(
                f"closure #{self.num_closures} {rec['type']} {link.from_id}->{link.to_id} "
                f"(dt {rec['dt_s']} s, correction {rec['correction_m']} m / "
                f"{rec['correction_deg']} deg)")
        if graph.edges:
            graph.header.stamp = graph.edges[-1].header.stamp
            graph.header.frame_id = self.frame_id
            self.lc_pub.publish(graph)


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
