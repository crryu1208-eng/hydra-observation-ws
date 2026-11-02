#!/usr/bin/env bash
# Handheld D435i session: camera + RTAB-Map RGB-D odometry + GPU segmentation + Hydra with the
# observation recorder, plus a raw rosbag (sqlite3) of the camera for offline re-runs.
# Run interactively inside the hydra-dev container:
#   docker exec -it hydra-dev bash /root/hydra_ws/scripts/run_d435i_obs.sh [session_name]
# Press Enter (or Ctrl-C) to stop. Env: FPS (default 15), RES (color and depth size, default
# 1280x720, the D435i depth maximum; 640x480 for the earlier sessions), VIS=1 starts the Hydra visualizer,
# STOP_AFTER=<s> stops automatically (non-interactive runs).
#
# Stop order: camera first (no new frames), then wait until the recorder output stops growing
# (Hydra's stop() does not drain its input queues), then /hydra/shutdown, then map frames to
# trajectory nodes.
set -eo pipefail
# job control: background jobs get their own process groups and keep default SIGINT handling
# (a non-interactive shell otherwise starts them with SIGINT ignored, so kill -INT would not stop
# the camera or finalize the bag); a terminal Ctrl-C then only reaches this script
set -m
# with job control a background job reading the terminal is stopped (SIGTTIN), so every
# background job below takes stdin from /dev/null
NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
FPS="${FPS:-15}"
RES="${RES:-1280x720}"
W="${RES%x*}"; H="${RES#*x}"
# RTAB-Map odometry detects features on a decimated image so its cost stays at the 640-wide level
ODOM_DECIMATION=1; [ "$W" -ge 1280 ] && ODOM_DECIMATION=2
VIS="${VIS:-0}"
STOP_AFTER="${STOP_AFTER:-}"
WS=/root/hydra_ws
OUT=/root/data/d435i/$NAME
OBS=$OUT/observations
LOG_PATH=$OUT/hydra
RUN_LOGS=$OUT/logs
BAG=$OUT/bag
IDLE_S=20
CAM=/camera/camera

if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
# keep DDS on this machine: the container uses host networking, and a handheld session
# unplugs the laptop's network, which must not affect node-to-node traffic
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$RUN_LOGS"
: > "$RUN_LOGS/launch.log"  # wait_log greps it before hydra is launched

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=$W,$H,$FPS depth_module.depth_profile:=$W,$H,$FPS \
  align_depth.enable:=true enable_sync:=true \
  enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
  < /dev/null > "$RUN_LOGS/camera.log" 2>&1 &
CAM_PID=$!; PIDS+=($CAM_PID)

LAUNCH_PID=
echo "session $OUT (${W}x${H} @ $FPS fps)"

stop_pid() {  # stop_pid <pid> <timeout_s>: SIGINT, then SIGTERM if it does not exit in time
  kill -INT "$1" 2>/dev/null || return 0
  for i in $(seq 1 "$2"); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  kill -TERM "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

wait_log() {  # wait_log <file> <pattern> <timeout_s>
  for i in $(seq 1 "$3"); do
    for p in $CAM_PID $LAUNCH_PID; do  # LAUNCH_PID is empty until hydra is launched
      if ! kill -0 $p 2>/dev/null; then echo "a launch exited early, see $RUN_LOGS"; exit 1; fi
    done
    if grep -qE "\[(ERROR|FATAL)\].*(semantic_inference|engine)|failed to load or build engine" \
        "$RUN_LOGS/launch.log"; then
      echo "segmentation failed to start"; exit 1
    fi
    grep -q "$2" "$1" && return 0
    sleep 1
  done
  echo "timeout waiting for: $2"; exit 1
}

wait_log "$RUN_LOGS/camera.log" "RealSense Node Is Up" 60
echo "camera up"
# record before Hydra is ready so the bag contains every frame Hydra sees
ros2 bag record -s sqlite3 --disable-keyboard-controls -o "$BAG" \
  $CAM/color/image_raw $CAM/color/camera_info \
  $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info \
  $CAM/imu /tf /tf_static /rtabmap/odom /rtabmap/odom_info \
  < /dev/null > "$RUN_LOGS/bag_record.log" 2>&1 &
BAG_PID=$!; PIDS+=($BAG_PID)

# let auto exposure settle so odometry does not initialize on a bad first frame
sleep 3
vis=false; [ "$VIS" = 1 ] && vis=true
ros2 launch $WS/launch/d435i_obs.launch.yaml \
  log_path:=$LOG_PATH observation_path:=$OBS start_visualizer:=$vis \
  odom_image_decimation:=$ODOM_DECIMATION \
  < /dev/null > "$RUN_LOGS/launch.log" 2>&1 &
LAUNCH_PID=$!; PIDS+=($LAUNCH_PID)

wait_log "$RUN_LOGS/launch.log" "IExecutionContext creation" 600
echo "segmentation engine ready"
wait_log "$RUN_LOGS/launch.log" "\[hydra\]: Running..." 60
echo "waiting for odometry to track (hold the camera still, facing a textured scene)"
python3 $WS/scripts/d435i_monitor.py --wait-tracking 60 || exit 1
python3 $WS/scripts/d435i_monitor.py --watch "$OBS/frames_raw.jsonl" < /dev/null &
MON_PID=$!; PIDS+=($MON_PID)
echo
echo "=== RECORDING. Move the camera slowly; avoid blank walls and fast turns. ==="
echo "=== Press Enter (or Ctrl-C) to stop. ==="
trap 'echo' INT
if [ -n "$STOP_AFTER" ]; then sleep "$STOP_AFTER" || true; else read -r || true; fi
# shutting down must not be interrupted: a second Ctrl-C would kill Hydra before it saves
trap 'echo "stopping, please wait (Hydra is saving)"' INT
kill -INT $MON_PID 2>/dev/null || true

echo "stopping camera"
stop_pid $CAM_PID 20
echo "camera stopped"
stop_pid $BAG_PID 30
echo "bag closed"

# drain: wait until the recorder output has not grown for IDLE_S seconds
last=-1; idle=0
while [ $idle -lt $IDLE_S ]; do
  n=$(wc -l < "$OBS/frames_raw.jsonl" 2>/dev/null || echo 0)
  if [ "$n" = "$last" ]; then idle=$((idle + 2)); else idle=0; last=$n; fi
  sleep 2
done
echo "recorder idle at $last frames, shutting down hydra"
timeout 30 ros2 service call /hydra/shutdown std_srvs/srv/Empty > /dev/null || true
for i in $(seq 1 300); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
stop_pid $LAUNCH_PID 30
trap - EXIT INT

# handheld: the camera may pause for a while between trajectory nodes (a node needs ~0.5 m of
# motion), so the anchor time limit is wide and the distance limit (1 m) decides. Turning in
# place adds no node either (office_walk3: 232 frames within 0.26 m of a node but turned
# 45-180 deg), and rotation relative to the node comes from IMU-aided odometry, so no angle limit.
python3 $WS/src/hydra/scripts/map_observations.py "$OBS" "$LOG_PATH" --max-anchor-dt-s 60.0 \
  --max-anchor-angle-deg 180 \
  > "$RUN_LOGS/map_observations.log" 2>&1 || echo "map_observations failed, see $RUN_LOGS"
python3 $WS/scripts/verify_d435i.py "$OUT" | tee "$RUN_LOGS/verify.log" || true
echo "done: $OUT"
