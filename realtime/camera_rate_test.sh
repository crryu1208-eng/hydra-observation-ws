#!/usr/bin/env bash
# Where do aligned depth frames get lost? Starts only the D435i driver (same settings as
# run_realtime.sh) under a few conditions and counts, for DUR seconds each, the frames on
#   color/image_raw, depth/image_rect_raw (raw depth), aligned_depth_to_color/image_raw
# Conditions: sync+align idle, sync+align with CPU load (LOAD busy processes, roughly the
# pipeline's ~6 cores), align without sync idle and loaded.
# Run inside the container with no other session using the camera; keep the camera still,
# facing a textured scene:  bash camera_rate_test.sh   (Env: DUR=40, LOAD=7, RES=1280x720, FPS=15)
set -o pipefail
RT="$(cd "$(dirname "$0")" && pwd)"
DUR="${DUR:-40}"; LOAD="${LOAD:-7}"; RES="${RES:-1280x720}"; FPS="${FPS:-15}"
W="${RES%x*}"; H="${RES#*x}"
OUT=/root/data/realtime/camera_rate_test/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
set -m
if pgrep -f realsense2_camera_node > /dev/null; then echo "the camera driver is already running"; exit 1; fi

count() {  # count <label>
  python3 - "$DUR" "$1" <<'EOF'
import sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from std_msgs.msg import Header
from sensor_msgs.msg import Image
dur, label = float(sys.argv[1]), sys.argv[2]
C = "/camera/camera/"
topics = {"color": C + "color/image_raw", "raw_depth": C + "depth/image_rect_raw",
          "aligned": C + "aligned_depth_to_color/image_raw"}
rclpy.init(); node = rclpy.create_node("camera_rate_test")
stamps = {k: set() for k in topics}
def cb(k):
    def f(data):  # raw subscription: rclpy's own conversion has failed on large messages before
        h = deserialize_message(data, Image).header
        stamps[k].add(h.stamp.sec * 10**9 + h.stamp.nanosec)
    return f
q = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
for k, t in topics.items():
    node.create_subscription(Image, t, cb(k), q, raw=True)
end = time.time() + 5  # settle
while time.time() < end: rclpy.spin_once(node, timeout_sec=0.1)
for s in stamps.values(): s.clear()
end = time.time() + dur
while time.time() < end: rclpy.spin_once(node, timeout_sec=0.1)
color = stamps["color"]; al = stamps["aligned"]
print(f"{label:28s} color {len(color)/dur:5.2f} Hz | raw depth {len(stamps['raw_depth'])/dur:5.2f} Hz | "
      f"aligned {len(al)/dur:5.2f} Hz | color frames without aligned depth "
      f"{len(color - al)} of {len(color)} ({100*len(color - al)/max(len(color),1):.1f}%)", flush=True)
EOF
}

run() {  # run <label> <sync true|false> <load processes>
  local label=$1 sync=$2 load=$3 lp=()
  nice -n -10 ros2 launch realsense2_camera rs_launch.py \
    rgb_camera.color_profile:=$W,$H,$FPS depth_module.depth_profile:=$W,$H,$FPS \
    align_depth.enable:=true enable_sync:=$sync \
    enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
    < /dev/null > "$OUT/camera_$label.log" 2>&1 &
  local cam=$!
  sleep 8
  for i in $(seq 1 "$load"); do (while :; do :; done) & lp+=($!); done
  count "$label"
  for p in "${lp[@]}"; do kill $p 2>/dev/null; done
  kill -INT -- -$cam 2>/dev/null; kill -INT $cam 2>/dev/null
  for i in $(seq 1 15); do kill -0 $cam 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$cam 2>/dev/null; wait $cam 2>/dev/null
  echo "  incomplete-frame warnings: $(grep -c 'Incomplete video frame' "$OUT/camera_$label.log")"
  sleep 3
}

echo "camera rate test ${W}x${H}@${FPS}, $DUR s per condition, load $LOAD busy processes -> $OUT"
run sync_align_idle true 0
run sync_align_load true "$LOAD"
run nosync_align_idle false 0
run nosync_align_load false "$LOAD"
