#!/usr/bin/env bash
# Run Hydra on the uHumans2 office bag with GPU semantic segmentation and the observation
# recorder, without dropping frames. Run inside the hydra-dev container:
#   docker exec hydra-dev bash /root/hydra_ws/scripts/run_uhumans2_obs.sh [play_rate]
#
# Hydra's stop() does not drain its input queues, so instead of exit_after_clock this waits
# until frames_raw.jsonl stops growing after the bag ends and then calls /hydra/shutdown.
# Hydra only subscribes to images after it has a CameraInfo, so camera_info/tf_static are
# played once before the real playback. Images use reliable QoS with deep queues end to end.
set -eo pipefail
RATE="${1:-0.5}"
WS=/root/hydra_ws
BAG=/root/data/office
OBS=/root/data/hydra_observations/uhumans2_office
LOG_PATH=/root/data/hydra_logs/uhumans2_office
RUN_LOGS=/root/data/hydra_run_logs/$(date +%Y%m%d_%H%M%S)
IDLE_S=60

source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
mkdir -p "$RUN_LOGS"
rm -rf "$OBS" "$LOG_PATH"

ros2 launch hydra_ros uhumans2.launch.yaml \
  use_gt_semantics:=false \
  max_image_queue_size:=0 \
  reliable_input:=true input_queue_size:=200 label_queue_size:=200 \
  hydra_config_path:=$WS/config/uhumans2_obs.yaml \
  input_config_path:=$WS/config/uhumans2_input_obs.yaml \
  log_path:=$LOG_PATH \
  start_visualizer:=false \
  > "$RUN_LOGS/launch.log" 2>&1 &
LAUNCH_PID=$!
echo "launch pid $LAUNCH_PID, logs in $RUN_LOGS"

QOS=~/.bag_qos_overrides.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in /tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw \
           /tesse/left_cam/camera_info /tesse/depth_cam/camera_info; do
    echo "$t: {history: keep_last, depth: 500, reliability: reliable, durability: volatile}"
  done
} > $QOS

wait_log() {  # wait_log <pattern> <timeout_s>
  for i in $(seq 1 "$2"); do
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then echo "launch exited early"; exit 1; fi
    if grep -qE "\[(ERROR|FATAL)\].*(semantic_inference|engine)|failed to load or build engine" \
        "$RUN_LOGS/launch.log"; then
      echo "segmentation failed to start"; exit 1
    fi
    grep -q "$1" "$RUN_LOGS/launch.log" && return 0
    sleep 1
  done
  echo "timeout waiting for: $1"; exit 1
}

# TensorRT engine is built on the first run (~1 min), loaded afterwards
wait_log "IExecutionContext creation" 600
echo "segmentation engine ready"

# let Hydra see CameraInfo (and static tf) so its image subscriptions exist before playback
for i in $(seq 1 30); do
  ros2 bag play "$BAG" --topics /tesse/left_cam/camera_info /tf_static \
    --playback-duration 1 --qos-profile-overrides-path $QOS > "$RUN_LOGS/preplay.log" 2>&1
  grep -q "\[hydra\]: Running..." "$RUN_LOGS/launch.log" && break
done
wait_log "\[hydra\]: Running..." 60
sleep 2
echo "hydra running, starting playback at rate $RATE"

# wait-for-all-acked: keep the player alive until reliable subscribers have every message;
# otherwise samples still in flight when the player exits are lost (last ~0.5 s of the bag)
ros2 bag play "$BAG" --clock -r "$RATE" --qos-profile-overrides-path $QOS \
  --wait-for-all-acked 30000 $BAG_EXTRA_ARGS \
  > "$RUN_LOGS/bag_play.log" 2>&1
echo "bag finished"

# drain: wait until the recorder output has not grown for IDLE_S seconds
last=-1; idle=0
while [ $idle -lt $IDLE_S ]; do
  n=$(wc -l < "$OBS/frames_raw.jsonl" 2>/dev/null || echo 0)
  if [ "$n" = "$last" ]; then idle=$((idle + 5)); else idle=0; last=$n; fi
  sleep 5
done
echo "recorder idle at $last lines, shutting down"

ros2 service call /hydra/shutdown std_srvs/srv/Empty > /dev/null
wait $LAUNCH_PID || true
echo "done: observations $OBS, hydra output $LOG_PATH, logs $RUN_LOGS"
