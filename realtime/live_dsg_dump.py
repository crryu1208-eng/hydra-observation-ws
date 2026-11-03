#!/usr/bin/env python3
"""Write Hydra's live scene graph to disk for the DSG viewer (tools/dsg_viewer, --live).

Subscribes /hydra/backend/dsg (the loop-closure corrected backend graph with its mesh; Hydra
only serializes it while someone subscribes, at most every backend.min_dsg_separation_s) and
every `period` seconds writes the newest one to <out_dir>:
  mesh.npz   points (N,3) float32, colors (N,3) uint8, labels (N,) int32, faces (M,3) int32
  dsg.json   the graph without mesh (spark_dsg JSON), written last: the viewer reloads when
             its modification time changes
Both are replaced atomically (write to a temporary file, then rename).
Needs the spark_dsg python bindings in the container (realtime/install_live_viewer_deps.sh).
"""
import os
import pathlib
import time

import numpy as np
import rclpy
import spark_dsg as dsg
from hydra_msgs.msg import DsgUpdate
from rclpy.node import Node


class LiveDsgDump(Node):
    def __init__(self):
        super().__init__("live_dsg_dump")
        self.out = pathlib.Path(self.declare_parameter("out_dir", "/tmp/hydra_live").value)
        period = float(self.declare_parameter("period", 2.0).value)
        self.out.mkdir(parents=True, exist_ok=True)
        self.latest = None
        self.written = 0
        self.create_subscription(DsgUpdate, "/hydra/backend/dsg", self.on_dsg, 1)
        self.create_timer(period, self.write)
        self.get_logger().info(f"live scene graph -> {self.out} every {period} s")

    def on_dsg(self, msg):
        self.latest = msg

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
        self._replace("dsg.json", lambda p: G.save(str(p), include_mesh=False))
        self.written += 1
        if self.written % 10 == 1:
            self.get_logger().info(
                f"wrote #{self.written}: {G.num_nodes()} nodes, {nverts} mesh vertices "
                f"({(time.perf_counter() - t0) * 1e3:.0f} ms)")


def main():
    rclpy.init()
    node = LiveDsgDump()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
