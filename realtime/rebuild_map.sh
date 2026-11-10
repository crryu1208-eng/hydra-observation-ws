#!/usr/bin/env bash
# Rebuild a session's map with its loop-closed per-frame poses (after run_realtime.sh).
#   bash rebuild_map.sh /root/data/realtime/<session>
# The session bag's images are played again; realtime/pose_tf_player.py publishes the corrected
# poses of <session>/observations/frames.jsonl (map_observations.py output) as odom -> camera_link,
# and segmentation + Hydra run without odometry, SLAM or loop closures. The mesh, objects and
# places are then made with exactly the poses the frames carry. The live backend mesh is deformed
# only approximately after loop closures (office_rt11: 37% of the keyframes 10-58 cm off it; this
# rebuild with the bridge odometry variance 1e-4: 7%, p95 8.9 cm).
# Output <session>/rebuilt/{hydra,observations -> ../observations,logs}, laid out like a session
# so level_session.py works on it. Takes about the session's length (the bag plays at 1x).
set -eo pipefail
set -m  # background launches get their own process group (stopped as a group below)
S="${1:?session directory}"
WS=/root/hydra_ws
RT=$WS/src/hydra-realtime/realtime
CAM=/camera/camera
OUT=$S/rebuilt
LOGS=$OUT/logs
[ -f "$S/observations/frames.jsonl" ] || { echo "no $S/observations/frames.jsonl (run map_observations.py first)"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT already exists"; exit 1; }
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$LOGS"
ln -s ../observations "$OUT/observations"

# part:=mapping with slam:=true: Hydra with the RosPoseGraphs tracker (agent nodes from
# pose_tf_player.py, the live keyframes) and no RTAB-Map / bridge / odometry
ros2 launch $RT/rt.launch.yaml use_sim_time:=true part:=mapping slam:=true \
  recorder:=false live_dsg:=false seg_fp16:=true seg_debug_images:=false \
  hydra_config:=$RT/hydra_rt.yaml log_path:=$OUT/hydra observation_path:=$OUT/unused \
  < /dev/null > "$LOGS/launch.log" 2>&1 &
LAUNCH_PID=$!
PTF_PID=
stop_all() {
  kill -INT $PTF_PID 2>/dev/null || true
  kill -INT -- -$LAUNCH_PID 2>/dev/null || true
  for i in $(seq 1 20); do kill -0 -- -$LAUNCH_PID 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$LAUNCH_PID 2>/dev/null || true
}
trap stop_all EXIT

wait_log() {  # wait_log <pattern> <timeout_s>
  for i in $(seq 1 $(( $2 * 5 ))); do
    kill -0 $LAUNCH_PID 2>/dev/null || { echo "launch exited early, see $LOGS"; exit 1; }
    grep -q "$1" "$LOGS/launch.log" && return 0
    sleep 0.2
  done
  echo "timeout waiting for: $1"; exit 1
}
QOS=$LOGS/bag_qos_overrides.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in $CAM/color/image_raw $CAM/aligned_depth_to_color/image_raw \
           $CAM/color/camera_info $CAM/aligned_depth_to_color/camera_info; do
    echo "$t: {history: keep_last, depth: 5, reliability: reliable, durability: volatile}"
  done
} > $QOS

wait_log "IExecutionContext creation" 600
# Hydra waits for the camera info and static transforms before it reports Running
for i in $(seq 1 30); do
  ros2 bag play "$S/bag" --topics $CAM/color/camera_info /tf_static --playback-duration 8 \
    --disable-keyboard-controls --qos-profile-overrides-path $QOS < /dev/null > "$LOGS/preplay.log" 2>&1
  grep -q "\[hydra\]: Running..." "$LOGS/launch.log" && break
done
wait_log "\[hydra\]: Running..." 60

python3 $RT/pose_tf_player.py --ros-args -p frames:="$S/observations/frames.jsonl" \
  -p nodes:="$S/observations/trajectory_nodes.csv" -p use_sim_time:=true \
  < /dev/null > "$LOGS/pose_tf_player.log" 2>&1 &
PTF_PID=$!
sleep 1
echo "rebuilding the map of $S with its corrected poses (bag at 1x)"
ros2 bag play "$S/bag" --clock --disable-keyboard-controls --qos-profile-overrides-path $QOS \
  --topics $CAM/color/image_raw $CAM/color/camera_info \
  $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info /tf_static \
  < /dev/null > "$LOGS/bag_play.log" 2>&1
sleep 5  # let the last frames through segmentation and Hydra
timeout 30 ros2 service call /hydra/shutdown std_srvs/srv/Empty > /dev/null || true
for i in $(seq 1 120); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
stop_all
trap - EXIT
rmdir "$OUT/unused" 2>/dev/null || true
[ -f "$OUT/hydra/backend/dsg_with_mesh.json" ] || { echo "rebuild produced no backend map, see $LOGS"; exit 1; }
echo "rebuilt map: $OUT/hydra"
