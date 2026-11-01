#!/usr/bin/env bash
# Re-run segmentation + Hydra (with the observation recorder) on the raw bag of a D435i session,
# without dropping frames. Live sessions can lose frames when segmentation falls behind the
# camera (e.g. GPU power-capped on battery); the bag has every camera frame plus the live
# RTAB-Map odometry in /tf, so this replays it slowly with reliable, deep queues end to end.
# Run inside the hydra-dev container:
#   docker exec hydra-dev bash /root/hydra_ws/scripts/reprocess_d435i_bag.sh <session> [play_rate]
# Output: /root/data/d435i/<session>_reproc/{observations,hydra,logs}, bag -> ../<session>/bag
set -eo pipefail
SESSION="${1:?usage: reprocess_d435i_bag.sh <session> [play_rate]}"
RATE="${2:-0.5}"
WS=/root/hydra_ws
SRC=/root/data/d435i/$SESSION
OUT=${SRC}_reproc
OBS=$OUT/observations
LOG_PATH=$OUT/hydra
RUN_LOGS=$OUT/logs
CAM=/camera/camera
IDLE_S=30

[ -d "$SRC/bag" ] || { echo "no bag in $SRC"; exit 1; }
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$RUN_LOGS"
ln -s "../$SESSION/bag" "$OUT/bag"

ros2 launch $WS/launch/d435i_obs.launch.yaml odometry:=false use_sim_time:=true \
  log_path:=$LOG_PATH observation_path:=$OBS \
  < /dev/null > "$RUN_LOGS/launch.log" 2>&1 &
LAUNCH_PID=$!
trap 'kill -INT $LAUNCH_PID 2>/dev/null || true' EXIT
echo "reprocessing $SRC at rate $RATE -> $OUT"

QOS=$RUN_LOGS/bag_qos_overrides.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in $CAM/color/image_raw $CAM/aligned_depth_to_color/image_raw \
           $CAM/color/camera_info $CAM/aligned_depth_to_color/camera_info; do
    echo "$t: {history: keep_last, depth: 500, reliability: reliable, durability: volatile}"
  done
} > $QOS

wait_log() {  # wait_log <pattern> <timeout_s>
  for i in $(seq 1 "$2"); do
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then echo "launch exited early, see $RUN_LOGS"; exit 1; fi
    if grep -qE "\[(ERROR|FATAL)\].*(semantic_inference|engine)|failed to load or build engine" \
        "$RUN_LOGS/launch.log"; then
      echo "segmentation failed to start"; exit 1
    fi
    grep -q "$1" "$RUN_LOGS/launch.log" && return 0
    sleep 1
  done
  echo "timeout waiting for: $1"; exit 1
}

wait_log "IExecutionContext creation" 600
echo "segmentation engine ready"

# Hydra subscribes to images only after it has a CameraInfo: play camera_info/tf_static first
for i in $(seq 1 30); do
  ros2 bag play "$SRC/bag" --topics $CAM/color/camera_info /tf_static \
    --playback-duration 1 --disable-keyboard-controls --qos-profile-overrides-path $QOS \
    < /dev/null > "$RUN_LOGS/preplay.log" 2>&1
  grep -q "\[hydra\]: Running..." "$RUN_LOGS/launch.log" && break
done
wait_log "\[hydra\]: Running..." 60
sleep 2
echo "hydra running, playing bag"

# wait-for-all-acked: keep the player alive until reliable subscribers have every message
ros2 bag play "$SRC/bag" --clock -r "$RATE" --disable-keyboard-controls \
  --qos-profile-overrides-path $QOS --wait-for-all-acked 30000 \
  --topics $CAM/color/image_raw $CAM/color/camera_info \
  $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info \
  /tf /tf_static \
  < /dev/null > "$RUN_LOGS/bag_play.log" 2>&1
echo "bag finished"

# drain: Hydra's stop() does not drain its input queues
last=-1; idle=0
while [ $idle -lt $IDLE_S ]; do
  n=$(wc -l < "$OBS/frames_raw.jsonl" 2>/dev/null || echo 0)
  if [ "$n" = "$last" ]; then idle=$((idle + 2)); else idle=0; last=$n; fi
  sleep 2
done
echo "recorder idle at $last frames, shutting down hydra"
timeout 30 ros2 service call /hydra/shutdown std_srvs/srv/Empty > /dev/null || true
for i in $(seq 1 300); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
kill -INT $LAUNCH_PID 2>/dev/null || true
wait $LAUNCH_PID 2>/dev/null || true
trap - EXIT

# same mapping limits as run_d435i_obs.sh
python3 $WS/src/hydra/scripts/map_observations.py "$OBS" "$LOG_PATH" --max-anchor-dt-s 60.0 \
  --max-anchor-angle-deg 180 \
  > "$RUN_LOGS/map_observations.log" 2>&1 || echo "map_observations failed, see $RUN_LOGS"
python3 $WS/scripts/verify_d435i.py "$OUT" | tee "$RUN_LOGS/verify.log" || true
echo "done: $OUT"
