#!/usr/bin/env bash
# Save the closed-set segmentation (EfficientViT, ADE20K) of every frame of a session.
#   bash seg_labels.sh /root/data/realtime/<session> <out dir>
# The session bag's color images are played at 1x through the segmentation node only (no Hydra);
# realtime/label_io.py saves each label image as <out dir>/<stamp_ns>.png. These are the base
# labels gdsam_labels.py paints its object masks over. RATE (default 0.5) is the bag play rate:
# at 1x the segmentation drops ~12% of the frames. Takes about the session's length / RATE.
set -eo pipefail
set -m
S="${1:?session directory}"; OUT="${2:?out dir}"
WS=/root/hydra_ws; RT=$WS/src/hydra-realtime/realtime; CAM=/camera/camera
source /opt/ros/jazzy/setup.bash; source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$OUT/logs"
ros2 launch $RT/rt.launch.yaml use_sim_time:=true part:=mapping hydra:=false recorder:=false \
  live_dsg:=false seg_fp16:=true seg_debug_images:=false < /dev/null > "$OUT/logs/launch.log" 2>&1 &
LAUNCH_PID=$!
python3 $RT/label_io.py save "$OUT" < /dev/null > "$OUT/logs/saver.log" 2>&1 &
SAVER_PID=$!
stop_all() {  # the segmentation launch must not outlive this script: it would keep publishing labels
  kill -INT $SAVER_PID 2>/dev/null || true
  kill -INT -- -$LAUNCH_PID 2>/dev/null || true
  for i in $(seq 1 20); do kill -0 -- -$LAUNCH_PID 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$LAUNCH_PID 2>/dev/null || true
  sleep 1
  kill -0 -- -$LAUNCH_PID 2>/dev/null && echo "WARNING: segmentation launch (process group $LAUNCH_PID) still running"
  return 0
}
trap stop_all EXIT INT TERM HUP
for i in $(seq 1 3000); do grep -q "IExecutionContext creation" "$OUT/logs/launch.log" && break; sleep 0.2; done
sleep 2
QOS=$OUT/logs/qos.yaml
echo "$CAM/color/image_raw: {history: keep_last, depth: 5, reliability: reliable, durability: volatile}" > $QOS
ros2 bag play "$S/bag" --rate "${RATE:-0.5}" --clock --disable-keyboard-controls --qos-profile-overrides-path $QOS \
  --topics $CAM/color/image_raw < /dev/null > "$OUT/logs/bag_play.log" 2>&1
sleep 5
stop_all; trap - EXIT INT TERM HUP
echo "$(find "$OUT" -maxdepth 1 -name '*.png' | wc -l) label images in $OUT"
