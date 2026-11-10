#!/usr/bin/env python3
"""Gravity and floor height at the start of a session, so the map has z up and the floor at z = 0.

RTAB-Map's odometry takes its initial roll / pitch from the IMU orientation (wait_imu_to_init),
which levels the map. In live sessions it mostly did not: rtabmap_ros 0.23.13 skips the first
buffered IMU message, and live the first IMU messages it gets are newer than the first image
(which arrives ~0.1 s late after alignment), so the first frame was processed without any IMU
and the map stayed in the tilted camera frame (office_rt6: floor 35 deg off, the camera's pitch
at the start; office_rt1/3/4/5 too, only rt2 was levelled). Fixed in the rtabmap_odom of the
workspace (fork, branch feature/realtime). Giving roll / pitch through initial_pose instead
levelled the start too, but the map tilted again over the walk (office_rt6 replay: odometry
z vs gravity up to 8 deg, floor 5.4 deg; IMU initialization: <= 4 deg, 3.5 deg), so only the
height goes through initial_pose: its rotation stays identity and the IMU initialization still
applies (Odometry::process checks the rotation only). Measured while the camera is held still:

  floor_init.py                prints "x y z roll pitch yaw" for rgbd_odometry's initial_pose:
                               0 0 <camera height above the floor> 0 0 0, from the
                               accelerometer (gravity) and one aligned depth frame. Without a
                               floor in view the height is --default-height. Exit code 1 when
                               there is no camera data or the camera kept moving.
  floor_init.py --check        after odometry started: compares its first pose's roll / pitch
                               with gravity from the accelerometer, prints the tilt error,
                               exit code 1 when it exceeds --max-tilt-deg (map not level).
  floor_init.py --bag <dir> [--start s]   the measurement on a recorded session bag

The floor is the lowest dense horizontal layer of depth points below the camera (in the
gravity-aligned frame), between MIN_BELOW (0.6) and 2.5 m down: a handheld camera is not closer to
the floor than that, and with 0.2 m a desk top 0.3 m below the camera was taken (office_rt1 / rt3 /
rt4: map 0.9-1.0 m low). Diagnostics go to stderr.
"""
import argparse
import math
import sys
import time

import numpy as np

CAM = "/camera/camera/"
IMU = CAM + "imu"
DEPTH = CAM + "aligned_depth_to_color/image_raw"
INFO = CAM + "color/camera_info"
BASE = "camera_link"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def quat_to_rot(q):  # x, y, z, w
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class StaticTf:
    """camera_link -> frame from /tf_static messages (the camera's static frames form a tree)."""

    def __init__(self):
        self.edges = {}  # child -> (parent, R, t)

    def add(self, msg):
        for tr in msg.transforms:
            r, t = tr.transform.rotation, tr.transform.translation
            self.edges[tr.child_frame_id] = (tr.header.frame_id, quat_to_rot((r.x, r.y, r.z, r.w)),
                                             np.array([t.x, t.y, t.z]))

    def lookup(self, frame):  # returns R, t with p_base = R p_frame + t
        R, t = np.eye(3), np.zeros(3)
        while frame != BASE:
            if frame not in self.edges:
                return None
            parent, Rp, tp = self.edges[frame]
            R, t = Rp @ R, Rp @ t + tp
            frame = parent
        return R, t


def level_rotation(up):
    """Rotation (roll, pitch, yaw 0) taking camera_link to a gravity-aligned frame, from the
    'up' direction in camera_link."""
    up = up / np.linalg.norm(up)
    pitch = math.asin(max(-1.0, min(1.0, -up[0])))
    roll = math.atan2(up[1], up[2])
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx, roll, pitch


def up_in_base(accels, imu_frame, tf):
    a = np.asarray(accels)
    mean = a.mean(0)
    jitter = np.linalg.norm(a - mean, axis=1).std()
    T = tf.lookup(imu_frame)
    if T is None:
        raise RuntimeError(f"no static transform {BASE} -> {imu_frame}")
    # the accelerometer measures the reaction to gravity, i.e. it points up
    return T[0] @ mean, jitter, np.linalg.norm(mean)


MIN_BELOW = 0.6


def floor_height(depth_mm, info_k, depth_frame, R_level, tf):
    T = tf.lookup(depth_frame)
    if T is None:
        raise RuntimeError(f"no static transform {BASE} -> {depth_frame}")
    fx, fy, cx, cy = info_k[0], info_k[4], info_k[2], info_k[5]
    d = depth_mm[::4, ::4].astype(np.float64) / 1000.0
    v, u = np.mgrid[0:depth_mm.shape[0]:4, 0:depth_mm.shape[1]:4]
    ok = (d > 0.2) & (d < 5.0)
    d, u, v = d[ok], u[ok], v[ok]
    if len(d) < 1000:
        return None, "too few depth pixels"
    p = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], 1)
    z = (p @ (R_level @ T[0]).T)[:, 2] + (R_level @ T[1])[2]
    cand = z[(z > -2.5) & (z < -MIN_BELOW)]
    if len(cand) < 0.015 * len(z):
        return None, (f"only {100 * len(cand) / len(z):.1f}% of the depth points are "
                      f"{MIN_BELOW}-2.5 m below the camera")
    hist, edges = np.histogram(cand, bins=np.arange(-2.5, -MIN_BELOW + 1e-9, 0.02))
    dense = np.nonzero((hist >= 0.3 * hist.max()) & (hist >= 0.015 * len(z)))[0]
    if len(dense) == 0:
        return None, "no dense horizontal layer below the camera"
    lo = edges[dense[0]]
    layer = cand[(cand > lo - 0.04) & (cand < lo + 0.06)]
    h = -float(np.median(layer))
    return h, f"floor {h:.3f} m below the camera ({100 * len(layer) / len(z):.0f}% of the depth points)"


def report(up, jitter, g, h_msg):
    _, roll, pitch = level_rotation(up)
    log(f"gravity |a| {g:.2f} m/s^2, jitter {jitter:.2f}; camera roll {math.degrees(roll):.1f} deg, "
        f"pitch {math.degrees(pitch):.1f} deg; {h_msg}")


def from_bag(args):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from tf2_msgs.msg import TFMessage
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=args.bag), rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[IMU, DEPTH, INFO, "/tf_static"]))
    tf, acc, imu_frame, info, depth, t0 = StaticTf(), [], None, None, None, None
    while r.has_next():
        topic, data, t = r.read_next()
        t0 = t0 or t
        rel = (t - t0) / 1e9
        if topic == "/tf_static":
            tf.add(deserialize_message(data, TFMessage))
        elif rel < args.start:
            continue
        elif topic == IMU and rel < args.start + args.duration:
            m = deserialize_message(data, Imu)
            imu_frame = m.header.frame_id
            acc.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        elif topic == INFO and info is None:
            info = deserialize_message(data, CameraInfo)
        elif topic == DEPTH and depth is None and rel >= args.start + args.duration / 2:
            depth = deserialize_message(data, Image)
        if rel > args.start + args.duration and depth is not None and info is not None and len(tf.edges) > 3:
            break
    return tf, acc, imu_frame, info, depth


class Live:
    def __init__(self, check):
        import rclpy
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image, Imu
        from tf2_msgs.msg import TFMessage
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("floor_init")
        self.tf, self.acc, self.imu_frame, self.info, self.depth, self.odom = StaticTf(), [], None, None, None, None
        self.want_depth = False
        # best effort accepts reliable and best-effort publishers (the camera's imu is best effort)
        rel = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(TFMessage, "/tf_static", self.tf.add, latched)
        self.node.create_subscription(Imu, IMU, self.on_imu, QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.node.create_subscription(CameraInfo, INFO, self.on_info, rel)
        self.node.create_subscription(Image, DEPTH, self.on_depth, rel)
        if check:
            from nav_msgs.msg import Odometry
            self.node.create_subscription(Odometry, "/rtabmap/odom", self.on_odom, rel)

    def on_imu(self, m):
        self.imu_frame = m.header.frame_id
        self.acc.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])

    def on_info(self, m):
        self.info = m

    def on_depth(self, m):
        if self.want_depth and self.depth is None:
            self.depth = m

    def on_odom(self, m):
        q = m.pose.pose.orientation
        if self.odom is None and (q.w, q.x, q.y, q.z) != (0, 0, 0, 0):
            self.odom = quat_to_rot((q.x, q.y, q.z, q.w))

    def spin(self, until, timeout):
        end = time.time() + timeout
        while time.time() < end and not until():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return until()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bag")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--max-tilt-deg", type=float, default=3.0)
    ap.add_argument("--default-height", type=float, default=0.0)
    ap.add_argument("--max-jitter", type=float, default=0.3, help="m/s^2, larger: camera moving")
    ap.add_argument("--attempts", type=int, default=10)
    args = ap.parse_args()

    if args.bag:
        tf, acc, imu_frame, info, depth = from_bag(args)
    else:
        live = Live(args.check)
        if not live.spin(lambda: len(live.tf.edges) > 3 and live.info is not None and len(live.acc) > 0, args.timeout):
            log("no camera data (tf_static / camera_info / imu)")
            return 1
        if args.check:
            live.acc.clear()
            if not live.spin(lambda: live.odom is not None, args.timeout):
                log("no odometry pose on /rtabmap/odom")
                return 1
            live.spin(lambda: False, args.duration)
            up, jitter, g = up_in_base(live.acc, live.imu_frame, live.tf)
            up /= np.linalg.norm(up)
            # odometry's world z expressed in camera_link: R^T e_z
            err = math.degrees(math.acos(max(-1.0, min(1.0, float(live.odom[2] @ up)))))
            ok = err <= args.max_tilt_deg
            log(f"map level check: odometry z axis vs gravity {err:.1f} deg "
                f"({'ok' if ok else 'NOT LEVEL'}; accelerometer jitter {jitter:.2f} m/s^2)")
            return 0 if ok else 1
        for attempt in range(args.attempts):
            live.acc.clear(); live.depth = None; live.want_depth = False
            live.spin(lambda: False, args.duration / 2)
            live.want_depth = True
            live.spin(lambda: False, args.duration / 2)
            live.spin(lambda: live.depth is not None, args.timeout)
            if live.acc and up_in_base(live.acc, live.imu_frame, live.tf)[1] <= args.max_jitter:
                break
            log("the camera is moving, measuring again (hold it still)")
        else:
            return 1
        tf, acc, imu_frame, info, depth = live.tf, live.acc, live.imu_frame, live.info, live.depth

    if not acc or depth is None or info is None:
        log("missing imu / depth / camera_info")
        return 1
    up, jitter, g = up_in_base(acc, imu_frame, tf)
    R_level, _, _ = level_rotation(up)
    d = np.frombuffer(bytes(depth.data), dtype=np.uint16).reshape(depth.height, depth.width)
    h, msg = floor_height(d, list(info.k), depth.header.frame_id, R_level, tf)
    if h is None:
        h = args.default_height
        msg += f"; camera height {h:.2f} m used"
    report(up, jitter, g, msg)
    if jitter > args.max_jitter:
        log("the camera moved during the measurement")
        if not args.bag:
            return 1
    print(f"0 0 {h:.3f} 0 0 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
