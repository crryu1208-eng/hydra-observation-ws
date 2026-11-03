#!/usr/bin/env bash
# Replay a recorded D435i session bag at real-time speed into the pipeline and measure each
# stage (color in -> segmentation labels -> Hydra recorder) with realtime/stage_monitor.py.
# Run inside the hydra-dev container:
#   docker exec hydra-dev bash /root/hydra_ws/src/hydra-realtime/realtime/profile_pipeline.sh <tag>
# Env: SESSION (office_walk8), RATE (1.0), START (5 s offset), DUR (40 s of bag),
#      HYDRA=0 (segmentation only), RECORDER=0, SEG=0, QOS_RELIABLE=1 (bag publishes with
#      depth 500 like reprocessing; default depth 5, closer to a live camera driver),
#      ODOM=1 runs RTAB-Map RGB-D odometry live on the bag images (the bag's /tf, which has
#      the recorded odometry, is not played; IMU is),
#      SLAM=1 adds live RTAB-Map SLAM (loop closures) on that odometry,
#      REC=1 also records the raw bag like run_realtime.sh (its CPU and disk load),
#      LAUNCH_ARGS (extra ros2 launch args), SAVE_LABELS=1 (keep every 10th label image for
#      accuracy comparisons, see compare_labels.py)
# Output: /root/data/realtime/profile/<tag>/{summary.json,logs,observations,hydra}
set -eo pipefail
# job control: background jobs keep default SIGINT handling, so kill -INT stops ros2 launch
set -m
TAG="${1:?usage: profile_pipeline.sh <tag>}"
SESSION="${SESSION:-office_walk8}"
RATE="${RATE:-1.0}"; START="${START:-5}"; DUR="${DUR:-40}"
HYDRA="${HYDRA:-1}"; RECORDER="${RECORDER:-1}"; SEG="${SEG:-1}"; ODOM="${ODOM:-0}"; SLAM="${SLAM:-0}"
RT="$(cd "$(dirname "$0")" && pwd)"
WS=/root/hydra_ws
SRC="${SRC:-/root/data/d435i/$SESSION}"  # or a realtime session: SRC=/root/data/realtime/<name>
OUT=/root/data/realtime/profile/$TAG
CAM=/camera/camera
[ -d "$SRC/bag" ] || { echo "no bag in $SRC"; exit 1; }
if pgrep -f "closed_set_node|hydra_ros_node|rgbd_odometry" > /dev/null; then
  echo "pipeline nodes from an earlier run are still running:"; pgrep -af "closed_set_node|hydra_ros_node|rgbd_odometry"; exit 1
fi
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$OUT/logs"
LOGS=$OUT/logs
b() { [ "$1" = 1 ] && echo true || echo false; }

ros2 launch $RT/rt.launch.yaml odometry:=$(b $ODOM) slam:=$(b $SLAM) slam_db:=$OUT/rtabmap.db bridge_log:=$OUT/logs/bridge.jsonl use_sim_time:=true odom_image_decimation:=2 \
  hydra:=$(b $HYDRA) recorder:=$(b $RECORDER) segmentation:=$(b $SEG) \
  log_path:=$OUT/hydra observation_path:=$OUT/observations $LAUNCH_ARGS \
  < /dev/null > "$LOGS/launch.log" 2>&1 &
LAUNCH_PID=$!
MON_PID=
trap 'kill -INT $MON_PID 2>/dev/null; kill -INT -- -$LAUNCH_PID 2>/dev/null; sleep 5; kill -KILL -- -$LAUNCH_PID 2>/dev/null || true' EXIT

wait_log() {  # wait_log <pattern> <timeout_s>
  for i in $(seq 1 "$2"); do
    kill -0 $LAUNCH_PID 2>/dev/null || { echo "launch exited early, see $LOGS"; exit 1; }
    grep -q "$1" "$LOGS/launch.log" && return 0
    sleep 1
  done
  echo "timeout waiting for: $1"; exit 1
}

QOS=$LOGS/bag_qos_overrides.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in $CAM/color/image_raw $CAM/aligned_depth_to_color/image_raw \
           $CAM/color/camera_info $CAM/aligned_depth_to_color/camera_info; do
    if [ "${QOS_RELIABLE:-0}" = 1 ]; then
      echo "$t: {history: keep_last, depth: 500, reliability: reliable, durability: volatile}"
    else
      echo "$t: {history: keep_last, depth: 5, reliability: reliable, durability: volatile}"
    fi
  done
} > $QOS

[ "$SEG" = 1 ] && wait_log "IExecutionContext creation" 600
if [ "$HYDRA" = 1 ]; then
  # 8 s: a live session bag may get the camera's /tf_static a few seconds after its first message
  for i in $(seq 1 30); do
    ros2 bag play "$SRC/bag" --topics $CAM/color/camera_info /tf_static \
      --playback-duration 8 --disable-keyboard-controls --qos-profile-overrides-path $QOS \
      < /dev/null > "$LOGS/preplay.log" 2>&1
    grep -q "\[hydra\]: Running..." "$LOGS/launch.log" && break
  done
  wait_log "\[hydra\]: Running..." 60
fi
sleep 2

python3 $RT/stage_monitor.py --out "$OUT/summary.json" ${SAVE_LABELS:+--save-labels $OUT/labels} --frames-raw "$OUT/observations/frames_raw.jsonl" \
  < /dev/null > "$LOGS/monitor.log" 2>&1 &
MON_PID=$!
sleep 1
nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm --format=csv,noheader -l 1 \
  > "$LOGS/gpu.csv" 2>/dev/null &
GPU_PID=$!
top -b -d 1 -w 200 > "$LOGS/top.log" 2>/dev/null &
TOP_PID=$!
REC_PID=
if [ "${REC:-0}" = 1 ]; then  # same topics as run_realtime.sh
  ros2 bag record -s sqlite3 --disable-keyboard-controls -o "$OUT/bag_rec" \
    $CAM/color/image_raw $CAM/color/camera_info \
    $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info \
    $CAM/imu /tf /tf_static /rtabmap/odom /rtabmap/odom_info \
    < /dev/null > "$LOGS/bag_record.log" 2>&1 &
  REC_PID=$!
  sleep 2
fi
echo "playing $SESSION [$START, +$DUR] s at $RATE x -> $OUT"
ros2 bag play "$SRC/bag" --clock -r "$RATE" --disable-keyboard-controls \
  --start-offset "$START" --playback-duration "$DUR" \
  --qos-profile-overrides-path $QOS \
  --topics $CAM/color/image_raw $CAM/color/camera_info \
  $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info /tf_static \
  $( [ "$ODOM" = 1 ] && echo $CAM/imu || echo /tf ) \
  < /dev/null > "$LOGS/bag_play.log" 2>&1
sleep 5   # let the tail of the pipeline arrive
kill $GPU_PID $TOP_PID 2>/dev/null || true
[ -n "$REC_PID" ] && { kill -INT $REC_PID; wait $REC_PID || true; }
kill -INT $MON_PID; wait $MON_PID || true
if [ "$HYDRA" = 1 ]; then
  timeout 30 ros2 service call /hydra/shutdown std_srvs/srv/Empty > /dev/null || true
  for i in $(seq 1 60); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
fi
# ros2 launch runs in its own process group (set -m): stop the whole group, escalate if needed,
# so no node survives into the next profiling run (a leftover segmentation node would share the
# GPU and publish labels on the same topic)
kill -INT -- -$LAUNCH_PID 2>/dev/null || true
for i in $(seq 1 20); do kill -0 -- -$LAUNCH_PID 2>/dev/null || break; sleep 1; done
kill -KILL -- -$LAUNCH_PID 2>/dev/null || true
wait $LAUNCH_PID 2>/dev/null || true
trap - EXIT
grep -o '"note": *"Average elapsed[^"]*"' "$LOGS/launch.log" | tail -1 || true
python3 - "$OUT/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"window {s.get('window_s')} s")
print("seg node elapsed p50 ms", s.get("seg_node_elapsed_ms_p50"))
for k in ("rgb", "depth", "labels", "recorded"):
    v = s.get(k, {})
    print(f"{k:9s} n={v.get('count')} frac={v.get('fraction_of_rgb')} fps={v.get('fps')} "
          f"lat p50/p90/max={v.get('latency_ms_p50')}/{v.get('latency_ms_p90')}/{v.get('latency_ms_max')} ms")
EOF
