#!/usr/bin/env bash
# Realtime handheld D435i session (realtime branch): camera + RTAB-Map RGB-D odometry + live
# RTAB-Map SLAM whose loop closures reach Hydra through realtime/rtabmap_hydra_bridge.py + FP16
# segmentation + Hydra (realtime/hydra_rt.yaml) with the observation recorder, plus a raw rosbag.
# The scene graph is loop-closed while recording; no offline loop_close / reprocess step.
# Run interactively inside the hydra-dev container:
#   docker exec -it hydra-dev bash /root/hydra_ws/src/hydra-realtime/realtime/run_realtime.sh [session_name]
# Output: /root/data/realtime/<session>/{bag,observations,hydra,rtabmap,logs}
# Press Enter (or Ctrl-C) to stop. Env: FPS (default 15), RES (color and depth size, default
# 1280x720, the D435i depth maximum; 640x480 for the earlier sessions), VIS=1 starts the Hydra visualizer,
# CAM_NICE (camera driver nice value, default -10), CAM_HEIGHT (camera height when the floor is not in view at the start, default 0), FLOOR_LOCK=0 (no floor height lock, for sessions that change floors), CAM_CPUS / PIPE_CPUS (CPU pinning, PIN=0 off), STOP_AFTER=<s> stops automatically (non-interactive runs), LIVE=0 turns off the live scene graph
# files for the DSG viewer (<session>/hydra/live; on the host:
#   tools/dsg_viewer/run.sh --live ~/hydra_data/realtime/<session>).
#
# Startup: camera, bag record, then segmentation + Hydra (rt.launch.yaml part:=mapping, logs/launch.log)
# while the exposure settles and floor_init.py measures the start height, then the odometry part
# (rgbd_odometry, gravity_odom, RTAB-Map SLAM, bridge; part:=odometry, logs/launch_odom.log) with
# that start pose. logs/startup.log has the timeline (office_rt10, one launch: ~15-18 s to RECORDING).
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
LIVE="${LIVE:-1}"
# camera driver priority: realsense2_camera aligns depth to color on the CPU; on office_rt3 its
# aligned depth fell from 14 to 11.6 Hz once Hydra's load grew (+2 cores after loop closures),
# and every frame without aligned depth is a frame not recorded. Negative = higher priority
# (the container runs privileged); CAM_NICE=0 restores the default.
CAM_NICE="${CAM_NICE:--10}"
# map frame: z up and the floor at z = 0. Odometry levels the map from the IMU (wait_imu_to_init,
# needs the workspace's patched rtabmap_odom); realtime/floor_init.py measures the camera's height
# above the floor (depth, gravity-aligned) while the camera is held still at the start, and
# odometry starts at that height. CAM_HEIGHT (m) is used when no floor is in view.
CAM_HEIGHT="${CAM_HEIGHT:-0}"
# the height stays on the floor while walking: realtime/gravity_odom.py keeps the odometry level
# with the accelerometer and its height on the floor seen in the depth (floor at z = 0). Without a
# floor at the start the first stable floor below the camera becomes z = 0. FLOOR_LOCK=0 turns the
# floor height off (sessions that change floors, e.g. stairs); levelling stays on.
FLOOR_LOCK="${FLOOR_LOCK:-1}"
STOP_AFTER="${STOP_AFTER:-}"
# CPU pinning: the camera driver gets its own P-cores, everything else (Hydra, segmentation,
# odometry, SLAM, bag, monitors) the rest. Under the mapping load the driver lost raw depth
# frames (camera + office_rt5 replay load: 5-23% of color frames without aligned depth, raw depth
# down to 11 Hz); camera on 0-3: 1.0-4.1%, on 0-5 (pipeline on 6-19): 0.2-2.7%, with the same
# pipeline throughput (office_rt6 live walk on 0-3: 3.8%, office_rt5 unpinned 6.5%). i9-12900H: CPUs 0-11 are
# P-cores (hyperthread pairs 0/1, 2/3, ...), 12-19 E-cores.
PIN="${PIN:-1}"
CAM_CPUS="${CAM_CPUS:-0-5}"
PIPE_CPUS="${PIPE_CPUS:-6-19}"
WS=/root/hydra_ws
RT="$(cd "$(dirname "$0")" && pwd)"
OUT=/root/data/realtime/$NAME
OBS=$OUT/observations
LOG_PATH=$OUT/hydra
RUN_LOGS=$OUT/logs
BAG=$OUT/bag
IDLE_S=20
CAM=/camera/camera

if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
if [ "$PIN" = 1 ]; then
  # this script and every process it starts inherit PIPE_CPUS; only the camera is moved to CAM_CPUS
  taskset -cp "$PIPE_CPUS" $$ > /dev/null || { echo "cannot pin to CPUs $PIPE_CPUS (PIN=0 to skip)"; exit 1; }
  echo "CPU pinning: camera driver on $CAM_CPUS, pipeline on $PIPE_CPUS"
fi
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
if [ "$(ros2 pkg prefix rtabmap_odom)" = /opt/ros/jazzy ]; then
  echo "WARNING: rtabmap_odom is the ROS package, not the workspace's patched one: the map may not"
  echo "         be level (setup_workspace.sh, then colcon build --packages-select rtabmap_odom)"
fi
# keep DDS on this machine: the container uses host networking, and a handheld session
# unplugs the laptop's network, which must not affect node-to-node traffic
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$RUN_LOGS"
: > "$RUN_LOGS/launch.log"  # wait_log greps it before hydra is launched
T0=$(date +%s.%N)
stamp() {  # startup timeline: logs/startup.log, seconds since the script started
  awk -v t="$(date +%s.%N)" -v t0="$T0" -v m="$*" 'BEGIN { printf "%6.1f s  %s\n", t - t0, m }' \
    >> "$RUN_LOGS/startup.log"
}
stamp "start"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

cam_pin=(); [ "$PIN" = 1 ] && cam_pin=(taskset -c "$CAM_CPUS")
"${cam_pin[@]}" nice -n "$CAM_NICE" ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=$W,$H,$FPS depth_module.depth_profile:=$W,$H,$FPS \
  align_depth.enable:=true enable_sync:=true \
  enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
  < /dev/null > "$RUN_LOGS/camera.log" 2>&1 &
CAM_PID=$!; PIDS+=($CAM_PID)

LAUNCH_PID=; ODOM_PID=
echo "session $OUT (${W}x${H} @ $FPS fps)"
echo "=== Before RECORDING appears (about 12 s): ==="
echo "===   1. hold the camera still                                                          ==="
echo "===   2. point it at the floor (1-2 m away, more than 0.6 m below the camera): the floor  ==="
echo "===      height becomes z = 0; without a floor in view mapping waits until one is seen  ==="
echo "===   3. keep a textured scene in view (no blank wall) so odometry starts tracking       ==="

stop_pid() {  # stop_pid <pid> <timeout_s>: SIGINT, then SIGTERM if it does not exit in time
  kill -INT "$1" 2>/dev/null || return 0
  for i in $(seq 1 "$2"); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  kill -TERM "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

wait_log() {  # wait_log <file> <pattern> <timeout_s>
  for i in $(seq 1 $(( $3 * 5 ))); do
    for p in $CAM_PID $LAUNCH_PID $ODOM_PID; do  # empty until launched
      if ! kill -0 $p 2>/dev/null; then echo "a launch exited early, see $RUN_LOGS"; exit 1; fi
    done
    if grep -qE "\[(ERROR|FATAL)\].*(semantic_inference|engine)|failed to load or build engine" \
        "$RUN_LOGS/launch.log"; then
      echo "segmentation failed to start"; exit 1
    fi
    grep -q "$2" "$1" && return 0
    sleep 0.2
  done
  echo "timeout waiting for: $2"; exit 1
}

wait_log "$RUN_LOGS/camera.log" "RealSense Node Is Up" 60
echo "camera up"; stamp "camera up"
# record before Hydra is ready so the bag contains every frame Hydra sees
ros2 bag record -s sqlite3 --disable-keyboard-controls -o "$BAG" \
  $CAM/color/image_raw $CAM/color/camera_info \
  $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info \
  $CAM/imu /tf /tf_static /rtabmap/odom /rtabmap/odom_raw /rtabmap/odom_info \
  < /dev/null > "$RUN_LOGS/bag_record.log" 2>&1 &
BAG_PID=$!; PIDS+=($BAG_PID)

# segmentation and Hydra do not need the start pose: start them now (TensorRT engine load ~3 s)
# while the exposure settles and the floor is measured; the odometry part follows with the pose
live=false; [ "$LIVE" = 1 ] && live=true
if $live && ! python3 -c "import spark_dsg" 2>/dev/null; then
  echo "LIVE=1 needs the spark_dsg python bindings: bash $RT/install_live_viewer_deps.sh (or LIVE=0)"
  exit 1
fi
vis=false; [ "$VIS" = 1 ] && vis=true
mkdir -p "$OUT/rtabmap"
ros2 launch $RT/rt.launch.yaml part:=mapping \
  log_path:=$LOG_PATH observation_path:=$OBS start_visualizer:=$vis \
  seg_fp16:=true seg_debug_images:=false hydra_config:=$RT/hydra_rt.yaml \
  slam:=true live_dsg:=$live live_dsg_dir:=$LOG_PATH/live \
  < /dev/null > "$RUN_LOGS/launch.log" 2>&1 &
LAUNCH_PID=$!; PIDS+=($LAUNCH_PID); stamp "mapping launched"

# let auto exposure settle so odometry does not initialize on a bad first frame
sleep 3
echo "measuring the floor height (hold the camera still, the floor in view)"
POSE=$(python3 $RT/floor_init.py --default-height "$CAM_HEIGHT" 2> >(tee "$RUN_LOGS/floor_init.log" >&2)) \
  || { echo "could not measure the start pose, see $RUN_LOGS/floor_init.log"; exit 1; }
echo "odometry start pose (x y z roll pitch yaw): $POSE"; stamp "floor measured"
floor_lock=false; [ "$FLOOR_LOCK" = 1 ] && floor_lock=true
floor_auto=false
if $floor_lock && ! grep -q "m below the camera (" "$RUN_LOGS/floor_init.log"; then
  floor_auto=true; echo "no floor at the start: the first floor seen while walking becomes z = 0"
fi
ros2 launch $RT/rt.launch.yaml part:=odometry odom_initial_pose:="$POSE" \
  gravity_floor:=$floor_lock gravity_floor_auto:=$floor_auto \
  odom_image_decimation:=$ODOM_DECIMATION \
  slam:=true slam_db:=$OUT/rtabmap/rtabmap.db bridge_log:=$RUN_LOGS/bridge.jsonl \
  < /dev/null > "$RUN_LOGS/launch_odom.log" 2>&1 &
ODOM_PID=$!; PIDS+=($ODOM_PID); stamp "odometry launched"

wait_log "$RUN_LOGS/launch.log" "IExecutionContext creation" 600
echo "segmentation engine ready"; stamp "segmentation engine ready"
wait_log "$RUN_LOGS/launch.log" "\[hydra\]: Running..." 60
stamp "hydra running"
echo "waiting for odometry to track (hold the camera still, facing a textured scene)"
python3 $WS/src/hydra-realtime/scripts/d435i_monitor.py --wait-tracking 60 --reset-pose "$POSE" || exit 1
stamp "odometry tracking"
# level check of the odometry start (diagnostic only): in the background, recording starts now
{ python3 $RT/floor_init.py --check --odom-topic /rtabmap/odom_raw 2>&1 | tee -a "$RUN_LOGS/floor_init.log" || true; } < /dev/null &
python3 $WS/src/hydra-realtime/scripts/d435i_monitor.py --watch "$OBS/frames_raw.jsonl" < /dev/null &
MON_PID=$!; PIDS+=($MON_PID)
echo
stamp "recording"
echo "=== RECORDING (startup $(tail -1 "$RUN_LOGS/startup.log" | awk '{print $1}') s). Move the camera slowly; avoid blank walls and fast turns. ==="
$live && echo "=== Live map on the host: tools/dsg_viewer/run.sh --live ~/hydra_data/realtime/$NAME ==="
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
stop_pid $ODOM_PID 30
trap - EXIT INT

# handheld: the camera may pause for a while between trajectory nodes (a node needs ~0.5 m of
# motion), so the anchor time limit is wide and the distance limit (1 m) decides. Turning in
# place adds no node either (office_walk3: 232 frames within 0.26 m of a node but turned
# 45-180 deg), and rotation relative to the node comes from IMU-aided odometry, so no angle limit.
python3 $WS/src/hydra/scripts/map_observations.py "$OBS" "$LOG_PATH" --max-anchor-dt-s 60.0 \
  --max-anchor-angle-deg 180 \
  > "$RUN_LOGS/map_observations.log" 2>&1 || echo "map_observations failed, see $RUN_LOGS"
python3 $WS/src/hydra-realtime/scripts/verify_d435i.py "$OUT" | tee "$RUN_LOGS/verify.log" || true
# last check of the finished map: floor plane fit, levelled copy in <session>/levelled
python3 $RT/level_session.py "$OUT" 2>&1 | tee "$RUN_LOGS/level_session.log" || true
echo "done: $OUT"
