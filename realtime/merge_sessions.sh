#!/usr/bin/env bash
# One scene graph from several sessions of the same place, all in the first session's frame.
#   bash merge_sessions.sh <out dir> <session>[:<observations dir>] <session>[:<observations dir>] ...
# The observations dir holds frames.jsonl / trajectory_nodes.csv with the poses to use: default
# <session>/rebuilt/levelled/observations; for a session registered to the first one with
# align_sessions.py: <session>/aligned_to_<first>/observations.
# Like rebuild_map.sh: segmentation + Hydra (no odometry, SLAM or loop closures) on the sessions'
# images played one bag after the other; realtime/pose_tf_player.py publishes the given poses as TF
# and the keyframes of all sessions, in time order, as one agent pose graph. Hydra then merges the
# same objects seen in both sessions and builds places and rooms over all of them.
# Output <out dir>/{hydra,logs,merged_poses,sources.json}. Takes about the sum of the session lengths.
# Env: YAW_DEG turns the whole map about z (e.g. to put the walls along x / y: object boxes and the
# mesh grid then follow the building), HYDRA_CONFIG another Hydra config (default realtime/hydra_rt.yaml).
set -eo pipefail
set -m
OUT="${1:?out dir}"; shift
[ $# -ge 1 ] || { echo "usage: $0 <out dir> <session>[:<observations dir>] ..."; exit 1; }
[ -e "$OUT" ] && { echo "$OUT already exists"; exit 1; }
WS=/root/hydra_ws
RT=$WS/src/hydra-realtime/realtime
CAM=/camera/camera
LOGS=$OUT/logs
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export SEMANTIC_INFERENCE_MODEL_DIR=$WS/.models
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$LOGS" "$OUT/merged_poses"

BAGS=(); OBSS=()
for a in "$@"; do
  s="${a%%:*}"; o="${a#*:}"; [ "$o" = "$a" ] && o="$s/rebuilt/levelled/observations"
  [ -f "$o/frames.jsonl" ] && [ -f "$o/trajectory_nodes.csv" ] || { echo "no poses in $o"; exit 1; }
  BAGS+=("$s/bag"); OBSS+=("$o")
done
# all frames and keyframes of the sessions, in time order (sessions must not overlap in time)
python3 - "$OUT" "${YAW_DEG:-0}" "${OBSS[@]}" <<'E'
import csv, json, math, sys
import numpy as np
sys.path.insert(0, "/root/hydra_ws/src/hydra-realtime/realtime")
import level_session as ls
out, yaw, obs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
T = np.eye(4)
c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
T[:2, :2] = [[c, -s], [s, c]]
frames, nodes, fields = [], [], None
for o in obs:
    for l in open(f"{o}/frames.jsonl"):
        fr = json.loads(l)
        fr["source_observations"] = o  # rgb_path / depth_path are relative to it
        if yaw:
            for k in ("world_T_body", "world_T_sensor"):
                if k in fr:
                    fr[k] = ls.tf_pose_dict(T, fr[k])
                if isinstance(fr.get("corrected"), dict) and k in fr["corrected"]:
                    fr["corrected"][k] = ls.tf_pose_dict(T, fr["corrected"][k])
        frames.append(fr)
    r = csv.DictReader(open(f"{o}/trajectory_nodes.csv")); fields = fields or r.fieldnames
    for row in r:
        if yaw:
            for p in ("", "orig_"):
                t, q = ls.tf_pose(T, [float(row[p + k]) for k in "xyz"], [float(row[p + k]) for k in ("qw", "qx", "qy", "qz")])
                for k, v in zip(["x", "y", "z", "qw", "qx", "qy", "qz"], list(t) + q):
                    row[p + k] = repr(float(v))
        nodes.append(row)
frames.sort(key=lambda f: int(f["timestamp_ns"])); nodes.sort(key=lambda n: int(n["timestamp_ns"]))
with open(f"{out}/merged_poses/frames.jsonl", "w") as f:
    f.writelines(json.dumps(x) + "\n" for x in frames)
with open(f"{out}/merged_poses/trajectory_nodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(nodes)
json.dump({"sources": obs, "frames": len(frames), "keyframes": len(nodes), "yaw_deg": yaw,
           "merged_T_first_session": T.tolist()}, open(f"{out}/sources.json", "w"), indent=2)
print(f"{len(frames)} frames, {len(nodes)} keyframes from {len(obs)} sessions")
E

ros2 launch $RT/rt.launch.yaml use_sim_time:=true part:=mapping slam:=true \
  recorder:=false live_dsg:=false seg_fp16:=true seg_debug_images:=false \
  hydra_config:=${HYDRA_CONFIG:-$RT/hydra_rt.yaml} log_path:=$OUT/hydra observation_path:=$OUT/unused \
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
wait_log() {
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
for i in $(seq 1 30); do
  ros2 bag play "${BAGS[0]}" --topics $CAM/color/camera_info /tf_static --playback-duration 8 \
    --disable-keyboard-controls --qos-profile-overrides-path $QOS < /dev/null > "$LOGS/preplay.log" 2>&1
  grep -q "\[hydra\]: Running..." "$LOGS/launch.log" && break
done
wait_log "\[hydra\]: Running..." 60

python3 $RT/pose_tf_player.py --ros-args -p frames:="$OUT/merged_poses/frames.jsonl" \
  -p nodes:="$OUT/merged_poses/trajectory_nodes.csv" -p use_sim_time:=true \
  < /dev/null > "$LOGS/pose_tf_player.log" 2>&1 &
PTF_PID=$!
sleep 1
for b in "${BAGS[@]}"; do
  echo "playing $b"
  ros2 bag play "$b" --clock --disable-keyboard-controls --qos-profile-overrides-path $QOS \
    --topics $CAM/color/image_raw $CAM/color/camera_info \
    $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info /tf_static \
    < /dev/null >> "$LOGS/bag_play.log" 2>&1
  sleep 3
done
sleep 3
timeout 30 ros2 service call /hydra/shutdown std_srvs/srv/Empty > /dev/null || true
for i in $(seq 1 180); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
stop_all
trap - EXIT
rmdir "$OUT/unused" 2>/dev/null || true
[ -f "$OUT/hydra/backend/dsg_with_mesh.json" ] || { echo "no merged map, see $LOGS"; exit 1; }
echo "merged map: $OUT/hydra"
