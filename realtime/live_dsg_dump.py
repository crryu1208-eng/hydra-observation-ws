#!/usr/bin/env python3
"""Write Hydra's live scene graph to disk for the DSG viewer (tools/dsg_viewer, --live).

Subscribes /hydra/backend/dsg (the loop-closure corrected backend graph with its mesh; Hydra
only serializes it while someone subscribes, at most every backend.min_dsg_separation_s) and
every `period` seconds writes the newest one to <out_dir>:
  mesh.npz   points (N,3) float32, colors (N,3) uint8, labels (N,) int32, faces (M,3) int32
  dsg.json   the graph without mesh (spark_dsg JSON), written last: the viewer reloads when
             its modification time changes
Both are replaced atomically (write to a temporary file, then rename).
Open-vocabulary objects (realtime/ov_objects.py, topic /ov_objects/objects) are added to the written graph as
layer 2 partition 1 nodes (realtime/ov_dsg.py), only those flagged confirmed (parameter ov_all: all), moved
with Hydra's loop-closure correction: the odometry pose of each agent node comes from TF (odom <- robot_frame
at the node's stamp, looked up once per node while TF still holds it).
Needs the spark_dsg python bindings in the container (realtime/install_live_viewer_deps.sh).
"""
import os
import pathlib
import time

import numpy as np
import json

import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import String
import spark_dsg as dsg
from hydra_msgs.msg import DsgUpdate
from rclpy.node import Node

import ov_dsg


class LiveDsgDump(Node):
    def __init__(self):
        super().__init__("live_dsg_dump")
        self.out = pathlib.Path(self.declare_parameter("out_dir", "/tmp/hydra_live").value)
        period = float(self.declare_parameter("period", 2.0).value)
        self.out.mkdir(parents=True, exist_ok=True)
        self.latest = None
        self.written = 0
        self.create_subscription(DsgUpdate, "/hydra/backend/dsg", self.on_dsg, 1)
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.robot_frame = self.declare_parameter("robot_frame", "camera_link").value
        self.ov_all = bool(self.declare_parameter("ov_all", False).value)
        # closed-set objects this much covered by open-vocabulary objects are dropped (0: keep all)
        self.ov_dedup = float(self.declare_parameter("ov_dedup", 0.5).value)
        self.tf = tf2_ros.Buffer(cache_time=Duration(seconds=120))
        self.tfl = tf2_ros.TransformListener(self.tf, self)
        self.misses = {}
        self.agent_odom = {}  # agent stamp ns -> 4x4 odometry pose (None: TF no longer has it)
        self.ov = None
        self.create_subscription(String, "/ov_objects/objects", self.on_ov, 1)
        self.create_timer(period, self.write)
        self.get_logger().info(f"live scene graph -> {self.out} every {period} s")

    def on_dsg(self, msg):
        self.latest = msg

    def on_ov(self, msg):
        self.ov = json.loads(msg.data)

    def odom_pose(self, stamp):
        if stamp not in self.agent_odom:
            try:
                t = self.tf.lookup_transform(self.odom_frame, self.robot_frame, Time(nanoseconds=stamp)).transform
                T = np.eye(4); q = t.rotation
                T[:3, :3] = ov_dsg.quat_R(q.w, q.x, q.y, q.z); T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
                self.agent_odom[stamp] = T
            except tf2_ros.TransformException:
                # not yet / no longer in TF: retried at the next writes, given up after 3 misses
                self.misses[stamp] = self.misses.get(stamp, 0) + 1
                if self.misses[stamp] >= 3:
                    self.agent_odom[stamp] = None
                return None
        return self.agent_odom[stamp]

    def _replace(self, name, writer):
        tmp = self.out / f".{name}.tmp{pathlib.Path(name).suffix}"
        writer(tmp)
        os.replace(tmp, self.out / name)

    def write(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        t0 = time.perf_counter()
        G = dsg.DynamicSceneGraph.from_binary(msg.layer_contents.tobytes())
        nverts = 0
        if G.has_mesh() and not G.mesh.empty():
            v = G.mesh.get_vertices()  # (6, N): xyz, rgb in [0, 1]
            nverts = v.shape[1]
            arrays = {
                "points": v[:3].T.astype(np.float32),
                "colors": np.clip(v[3:6].T * 255.0 + 0.5, 0, 255).astype(np.uint8),
                "labels": np.asarray(G.mesh.get_labels(), dtype=np.int32),
                "faces": G.mesh.get_faces().T.astype(np.int32),
            }
            self._replace("mesh.npz", lambda p: np.savez(p, **arrays))
        ov = ""
        if self.ov is not None:
            r = ov_dsg.add_objects(G, self.ov["objects"], self.ov["vocab"], self.odom_pose, not self.ov_all, self.ov_dedup)
            ov = (f", {r['objects']} open-vocab objects (shift max {r['shift_max']:.2f} m), "
                  f"{r['replaced']} closed-set objects replaced")
        self._replace("dsg.json", lambda p: G.save(str(p), include_mesh=False))
        self.written += 1
        if self.written % 10 == 1:
            self.get_logger().info(
                f"wrote #{self.written}: {G.num_nodes()} nodes, {nverts} mesh vertices "
                f"{ov} ({(time.perf_counter() - t0) * 1e3:.0f} ms)")


def main():
    rclpy.init()
    node = LiveDsgDump()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
