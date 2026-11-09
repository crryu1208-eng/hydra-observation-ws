#!/usr/bin/env python3
"""Keep RTAB-Map odometry level with the accelerometer (complementary filter on roll / pitch).

rgbd_odometry uses the IMU only for its start orientation and as a weak constraint in its local
bundle adjustment; its roll / pitch still drift while walking (office_rt8: 0.4 deg at the start,
5-12 deg after 50 s; the camera height sank ~0.9 m and the floor ended up 0.3 m low after loop
closing; Optimizer/GravitySigma 0.03 instead of 0.3 only took the median error from 5.0 to
3.9 deg). Hydra's backend has no gravity factor, so the odometry is levelled here, before
Hydra, RTAB-Map SLAM and the bridge see it.

  world' = C * world,  C a rotation about a horizontal axis only (yaw unchanged)
  orientation' = C R_odom,  position' += C (p_odom - p_odom_prev)
  C turns toward the measured gravity with time constant tau_s (accelerometer mean over the
  last accel_window_s, skipped while |a| is more than accel_tolerance off 9.81)

Offline on the office_rt8 odometry: tilt error vs gravity median 5.4 -> 1.1 / 1.3 deg (tau 1 / 3 s),
camera height drift over the walk 0.5 -> 0.2 m.

Height: levelling alone left the office_rt8 replay floor flat but 13 cm low (odometry height still
sank 0.3 m over the walk, now from translation drift). With floor_height true the floor itself is
the height reference: the session starts with the floor at z = 0 (floor_init.py), and whenever the
floor is in view (every floor_period_s, aligned depth + the pose published for that image) the
lowest dense horizontal layer within floor_gate_m of z = 0 is taken as the floor and the height
offset moves toward it (gain floor_gain per measurement); the gate widens while no floor is seen
(floor_gate_growth per s, up to floor_gate_max). Surfaces further than the gate (desks,
chairs) are never taken; for a session that moves between floors (stairs) turn it off. With
floor_auto (no floor in view at the start, the height was a guess) the first stable floor below
the camera defines z = 0 instead.

office_rt8 full-pipeline replay (floor tilt / mean floor height / floor p2..p98):
  without 6.1 deg / -0.31 m / -63..+13 cm; levelling only 0.35 deg / -0.16 m / -19..-13 cm;
  levelling + floor height 1.0 deg / 0.00 m / -7..+8 cm (per-frame floor -0.08..+0.02 m over 80 s);
  start height 0 with floor_auto: floor found after ~4 s, then 0 +- 0.08 m (map -13..+21 cm:
  the first seconds were integrated at the wrong height)
office_rt9 (live, no floor in view at the start, floor_auto jumped the height 1.32 m after 17 s):
  loop closures to the keyframes before the jump pulled the backend floor to -1.55 m. Replay with
  the odometry held until the floor-labelled floor is found (after ~20 s): backend floor median
  -0.10 m, map floor p2..p98 -19..+1 cm, tilt 0.7 deg

Inputs:  odom_in (nav_msgs/Odometry, rgbd_odometry with publish_tf false), imu (raw D435i, the
         accelerometer is enough), /tf_static (camera_link -> IMU / depth frames),
         depth + depth_info (aligned depth, for floor_height)
Outputs: odom_out (nav_msgs/Odometry), TF odom -> camera_link
"""
import math
import os
import sys
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_init import StaticTf, quat_to_rot  # noqa: E402

G = 9.81


def rot_to_quat(R):  # x, y, z, w
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


def axis_angle(axis, angle):
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class GravityOdom(Node):
    def __init__(self):
        super().__init__("gravity_odom")
        self.tau = self.declare_parameter("tau_s", 2.0).value
        self.window = self.declare_parameter("accel_window_s", 0.2).value
        self.tolerance = self.declare_parameter("accel_tolerance", 1.5).value
        self.jump_m = self.declare_parameter("reset_jump_m", 0.5).value
        self.publish_tf = self.declare_parameter("publish_tf", True).value
        self.floor_on = self.declare_parameter("floor_height", True).value
        self.floor_period = self.declare_parameter("floor_period_s", 0.3).value
        self.floor_gate = self.declare_parameter("floor_gate_m", 0.3).value
        self.floor_gain = self.declare_parameter("floor_gain", 0.3).value
        # while the floor is out of view the height can drift past the gate: it widens by
        # floor_gate_growth per second without a floor fix, up to floor_gate_max (below chair seats)
        self.floor_gate_growth = self.declare_parameter("floor_gate_growth", 0.01).value
        self.floor_gate_max = self.declare_parameter("floor_gate_max", 0.4).value
        self.last_fix_t = None
        self.floor_auto = self.declare_parameter("floor_auto", False).value
        # floor_auto: nothing is published until the floor is found (a later height jump would
        # contradict the keyframes taken before it; office_rt9: loop closures to those keyframes
        # pulled the whole backend map 1.5 m down); after floor_auto_timeout_s without a floor the
        # guessed height is used and only the gated floor lock remains
        # (the camera is then put at floor_auto_height, a typical handheld height, so the floor
        # comes within the gate)
        self.floor_auto_timeout = self.declare_parameter("floor_auto_timeout_s", 30.0).value
        self.floor_auto_height = self.declare_parameter("floor_auto_height", 1.3).value
        # floor_auto takes only pixels the segmentation calls floor (semantic image, labelspace
        # ids; ade20k_full floor = 3): by height alone a desk or chair seat looked like a floor
        # (office_rt9 replay: "floor" 0.36 m below the camera)
        self.floor_label = self.declare_parameter("floor_label", 3).value
        self.sem = deque(maxlen=30)  # (stamp_ns, Image) semantic images
        self.t_first = None
        self.n_held = 0
        self.zoff = 0.0  # height correction added to the levelled odometry
        self.floor_ready = not self.floor_auto
        self.auto_samples = []
        self.poses = deque(maxlen=150)  # (stamp_ns, R_out, p_out) as published
        self.K = None
        self.last_floor_t = 0.0
        self.n_floor = [0, 0]  # accepted, tried
        self.pending = None  # depth image waiting for its odometry pose
        self.why = {}  # skipped depth frames by reason (status line)
        self.tf = StaticTf()
        self.R_imu = None  # camera_link <- IMU frame
        self.accel = deque(maxlen=400)  # (stamp_s, a in camera_link)
        self.C = np.eye(3)
        self.last = None  # (stamp_s, p_odom, p_out)
        self.pub = self.create_publisher(Odometry, "odom_out", QoSProfile(depth=100))
        self.br = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Odometry, "odom_in", self.on_odom, QoSProfile(depth=100))
        self.create_subscription(Imu, "imu", self.on_imu,
                                 QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(TFMessage, "/tf_static", self.tf.add,
                                 QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if self.floor_on:
            self.create_subscription(CameraInfo, "depth_info", self.on_info, QoSProfile(depth=5))
            if self.floor_auto:
                # reliable: best effort lost ~1/3 of the label images while the floor was searched
                self.create_subscription(Image, "semantic", self.on_semantic, QoSProfile(depth=10), raw=True)
            self.create_subscription(Image, "depth", self.on_depth,
                                     QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT), raw=True)
        self.n_out = 0
        self.create_timer(10.0, self.status)
        self.get_logger().info(f"levelling odom_in -> odom_out with the accelerometer (tau {self.tau} s)")

    def status(self):
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, self.C[2, 2]))))
        held = "" if self.floor_ready else f", held until the floor is found: {self.n_held}"
        self.get_logger().info(f"{self.n_out} odometry messages{held}, correction {tilt:.2f} deg, height "
                               f"{self.zoff:+.3f} m (floor seen {self.n_floor[0]} / {self.n_floor[1]}; skipped {self.why})")

    def on_imu(self, msg):
        if self.R_imu is None:
            T = self.tf.lookup(msg.header.frame_id)
            if T is None:
                return
            self.R_imu = T[0]
        a = msg.linear_acceleration
        self.accel.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           self.R_imu @ np.array([a.x, a.y, a.z])))

    def up(self, t):
        a = [v for s, v in self.accel if t - self.window < s <= t + 0.005]
        if len(a) < 5:
            return None
        m = np.mean(a, axis=0)
        n = np.linalg.norm(m)
        return m / n if abs(n - G) < self.tolerance else None

    def on_info(self, msg):
        self.K = msg.k

    def on_depth(self, data):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.K is None:
            self.why["no camera_info"] = self.why.get("no camera_info", 0) + 1
            return
        if now - self.last_floor_t < self.floor_period:
            return
        self.last_floor_t = now
        self.pending = deserialize_message(data, Image)  # its pose comes ~50-100 ms later
        self.try_floor()

    def on_semantic(self, data):
        if self.floor_ready:
            return
        msg = deserialize_message(data, Image)
        self.sem.append((msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, msg))
        if self.pending is not None:
            self.try_floor()

    def try_floor(self):
        msg = self.pending
        if msg is None or not self.poses:
            return
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        pose = next((p for p in reversed(self.poses) if p[0] == stamp), None)
        if pose is None:
            if self.poses[-1][0] > stamp:  # odometry is past this image: no pose for it (lost)
                self.why["no pose for the image stamp"] = self.why.get("no pose for the image stamp", 0) + 1
                self.pending = None
            return
        sem = None
        if not self.floor_ready:
            sem = next((m for s, m in reversed(self.sem) if s == stamp), None)
            if sem is None:
                if not self.sem or self.sem[-1][0] <= stamp:
                    return  # its labels have not arrived yet
                self.why["no labels for the image stamp"] = self.why.get("no labels for the image stamp", 0) + 1
                self.pending = None
                return
            dt = {"8UC1": np.uint8, "mono8": np.uint8, "16UC1": np.uint16, "mono16": np.uint16,
                  "16SC1": np.int16, "32SC1": np.int32}.get(sem.encoding)
            if dt is None or (sem.height, sem.width) != (msg.height, msg.width):
                why = f"labels {sem.encoding} {sem.width}x{sem.height}"
                self.why[why] = self.why.get(why, 0) + 1
                self.pending = None
                return
        self.pending = None
        T = self.tf.lookup(msg.header.frame_id)
        why = (f"no tf for {msg.header.frame_id}" if T is None else f"encoding {msg.encoding}"
               if msg.encoding not in ("16UC1", "mono16") else None)
        if why:
            self.why[why] = self.why.get(why, 0) + 1
            return
        self.n_floor[1] += 1
        d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[::6, :msg.width:6]
        v, u = np.mgrid[0:msg.height:6, 0:msg.width:6]
        d = d.astype(np.float64) / 1000.0
        ok = (d > 0.3) & (d < 4.0)
        if sem is not None:
            isz = np.dtype(dt).itemsize
            lab = np.frombuffer(sem.data, dt).reshape(sem.height, sem.step // isz)[::6, :sem.width:6]
            n_ok = int(ok.sum())
            ok &= lab == self.floor_label
            if ok.sum() < max(300, 0.05 * n_ok):  # floor less than 5% of the view
                return
        d, u, v = d[ok], u[ok], v[ok]
        if len(d) < 500:
            return
        K = self.K
        pc = np.stack([(u - K[2]) * d / K[0], (v - K[5]) * d / K[4], d], 1)
        _, R, p = pose
        Rw = R @ T[0]
        zw = pc @ Rw[2] + (R @ T[1])[2] + p[2]
        t_img = stamp * 1e-9
        if self.last_fix_t is None:
            self.last_fix_t = t_img
        gate = min(self.floor_gate_max,
                   self.floor_gate + self.floor_gate_growth * max(0.0, t_img - self.last_fix_t))
        if self.floor_ready:
            lo, hi = -gate, gate
        else:  # floor_auto: floor-labelled points 0.2 - 2.5 m below the camera
            lo, hi = p[2] - 2.5, p[2] - 0.2
        z = self.floor_layer(zw, lo, hi)
        if z is None:
            return
        self.n_floor[0] += 1
        self.last_fix_t = t_img
        if not self.floor_ready:
            # the floor: the lowest height that 3 of the last 10 floor-labelled samples agree on
            # (within 5 cm)
            self.auto_samples.append(z)
            a = np.sort(self.auto_samples[-10:])
            for z0 in a:
                near = a[np.abs(a - z0) < 0.025]
                if len(near) >= 3:
                    self.zoff -= float(np.median(near))
                    self.floor_ready = True
                    self.get_logger().info(f"floor_auto: floor found {-np.median(near):.3f} m below z = 0, "
                                           "it is z = 0 from now on")
                    break
            return
        self.zoff -= self.floor_gain * z

    @staticmethod
    def floor_layer(zw, lo, hi):
        """Lowest dense horizontal layer of the points within [lo, hi] (world z), or None."""
        n = len(zw)
        cand = zw[(zw > lo) & (zw < hi)]
        if len(cand) < 0.03 * n:
            return None
        hist, edges = np.histogram(cand, bins=np.arange(lo, hi + 1e-9, 0.02))
        dense = np.nonzero((hist >= 0.3 * hist.max()) & (hist >= 0.02 * n))[0]
        if len(dense) == 0:
            return None
        z0 = edges[dense[0]]
        layer = cand[(cand > z0 - 0.03) & (cand < z0 + 0.05)]
        return float(np.median(layer))

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        if q.x == q.y == q.z == q.w == 0.0:  # lost (publish_null_when_lost): pass through
            self.pub.publish(msg)
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        R = quat_to_rot((q.x, q.y, q.z, q.w))
        p = msg.pose.pose.position
        p = np.array([p.x, p.y, p.z])
        if self.last is None or np.linalg.norm(p - self.last[1]) > self.jump_m:
            # first message or an odometry reset to a given pose: restart from it
            if self.last is not None:
                self.get_logger().warn("odometry jumped (reset): correction restarted")
            self.C = np.eye(3)
            self.zoff = 0.0 if self.floor_ready and not self.floor_auto else self.zoff
            p_out = p
        else:
            dt = max(0.0, t - self.last[0])
            u = self.up(t)
            if u is not None:
                up_w = self.C @ R @ u
                e = np.cross(up_w, (0.0, 0.0, 1.0))  # horizontal axis taking up_w to +z
                s = np.linalg.norm(e)
                if s > 1e-9:
                    ang = math.asin(min(s, 1.0)) * min(1.0, dt / self.tau)
                    self.C = axis_angle(e / s, ang) @ self.C
            p_out = self.last[2] + self.C @ (p - self.last[1])
        self.last = (t, p, p_out)
        p_out = p_out + np.array([0.0, 0.0, self.zoff])
        Ro = self.C @ R
        self.poses.append((msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, Ro, p_out))
        if self.pending is not None:
            self.try_floor()
        if self.t_first is None:
            self.t_first = t
        if not self.floor_ready:
            if t - self.t_first < self.floor_auto_timeout:
                self.n_held += 1
                return
            self.floor_ready = True
            self.zoff = self.floor_auto_height - self.last[2][2]
            self.get_logger().warn(f"floor_auto: no floor after {self.floor_auto_timeout:.0f} s, starting "
                                   f"with the camera {self.floor_auto_height:.2f} m above the floor "
                                   "(floor lock within the gate only)")
        p_out = self.last[2] + np.array([0.0, 0.0, self.zoff])  # zoff may have changed in try_floor
        qo = rot_to_quat(Ro)
        out = msg
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = map(float, p_out)
        (out.pose.pose.orientation.x, out.pose.pose.orientation.y,
         out.pose.pose.orientation.z, out.pose.pose.orientation.w) = map(float, qo)
        self.pub.publish(out)  # twist is in the child frame: unchanged
        self.n_out += 1
        if self.br:
            tr = TransformStamped()
            tr.header = msg.header
            tr.child_frame_id = msg.child_frame_id
            tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = map(float, p_out)
            (tr.transform.rotation.x, tr.transform.rotation.y,
             tr.transform.rotation.z, tr.transform.rotation.w) = map(float, qo)
            self.br.sendTransform(tr)


def main():
    rclpy.init()
    node = GravityOdom()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
