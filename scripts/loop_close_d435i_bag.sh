#!/usr/bin/env bash
# Loop-close a D435i session offline with RTAB-Map SLAM, then rebuild the Hydra scene graph on the
# corrected trajectory. Run inside the hydra-dev container:
#   docker exec hydra-dev bash /root/hydra_ws/scripts/loop_close_d435i_bag.sh <session> [play_rate]
#
# 1. play the session bag (images + the live RTAB-Map odometry) into rtabmap_slam, which detects
#    loop closures and optimizes its keyframe graph (database <session>_lc/rtabmap/rtabmap.db)
# 2. export the optimized keyframe poses and write <session>_lc/bag: the same bag with
#    odom -> camera_link following the optimized trajectory (scripts/apply_loop_closures.py)
# 3. run scripts/reprocess_d435i_bag.sh on it -> <session>_lc_reproc/{observations,hydra,logs}
# Env: SKIP_REPROCESS=1 stops after step 2.
set -eo pipefail
SESSION="${1:?usage: loop_close_d435i_bag.sh <session> [play_rate]}"
RATE="${2:-1.0}"
WS=/root/hydra_ws
SRC=/root/data/d435i/$SESSION
OUT=${SRC}_lc
CAM=/camera/camera

[ -d "$SRC/bag" ] || { echo "no bag in $SRC"; exit 1; }
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$OUT/rtabmap" "$OUT/logs"
LOGS=$OUT/logs

python3 $WS/scripts/apply_loop_closures.py "$SRC/bag" "$OUT/slam_input" --filter-slam-input \
  $CAM/color/image_raw $CAM/color/camera_info $CAM/aligned_depth_to_color/image_raw /rtabmap/odom /tf_static \
  2> "$LOGS/filter.log" | tee "$LOGS/slam_input.json"

# rtabmap_slam on the recorded odometry; keyframes at 2 Hz, the rest of RTAB-Map's defaults
# (global loop closures by bag-of-words, local proximity detection, graph optimization)
# run the binary directly (not via ros2 run) so SIGINT reaches it and it saves the database
/opt/ros/jazzy/lib/rtabmap_slam/rtabmap -d --ros-args -r __ns:=/rtabmap -p use_sim_time:=true \
  -p frame_id:=camera_link -p map_frame_id:=map -p publish_tf:=false \
  -p subscribe_depth:=true -p subscribe_rgb:=true -p approx_sync:=false \
  -p qos:=1 -p qos_image:=1 -p qos_camera_info:=1 -p qos_odom:=1 \
  -p topic_queue_size:=30 -p sync_queue_size:=30 \
  -p database_path:=$OUT/rtabmap/rtabmap.db \
  -p "Rtabmap/DetectionRate:='2'" -p "Mem/IncrementalMemory:='true'" \
  -r rgb/image:=$CAM/color/image_raw -r depth/image:=$CAM/aligned_depth_to_color/image_raw \
  -r rgb/camera_info:=$CAM/color/camera_info -r odom:=/rtabmap/odom \
  < /dev/null > "$LOGS/rtabmap.log" 2>&1 &
SLAM_PID=$!
trap 'kill -INT $SLAM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  grep -q "rtabmap): Database" "$LOGS/rtabmap.log" 2>/dev/null && break
  grep -qiE "subscribed to|Sync" "$LOGS/rtabmap.log" 2>/dev/null && break
  kill -0 $SLAM_PID 2>/dev/null || { echo "rtabmap exited early, see $LOGS/rtabmap.log"; exit 1; }
  sleep 1
done
sleep 2
echo "rtabmap up, playing $SRC/bag at rate $RATE"

QOS=$LOGS/bag_qos_overrides.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in $CAM/color/image_raw $CAM/aligned_depth_to_color/image_raw $CAM/color/camera_info /rtabmap/odom; do
    echo "$t: {history: keep_last, depth: 500, reliability: reliable, durability: volatile}"
  done
} > $QOS
ros2 bag play "$OUT/slam_input" --clock -r "$RATE" --disable-keyboard-controls \
  --qos-profile-overrides-path $QOS --wait-for-all-acked 30000 \
  < /dev/null > "$LOGS/bag_play_slam.log" 2>&1
rm -rf "$OUT/slam_input"
sleep 3
echo "bag finished, saving rtabmap database"
kill -INT $SLAM_PID
for i in $(seq 1 120); do kill -0 $SLAM_PID 2>/dev/null || break; sleep 1; done
kill -0 $SLAM_PID 2>/dev/null && { echo "rtabmap did not exit, killing"; kill -TERM $SLAM_PID; }
wait $SLAM_PID 2>/dev/null || true
trap - EXIT

rtabmap-export --poses --poses_format 11 --output_dir "$OUT/rtabmap" --output slam \
  "$OUT/rtabmap/rtabmap.db" > "$LOGS/export.log" 2>&1
POSES=$(ls "$OUT"/rtabmap/*poses*.txt | head -1)
[ -n "$POSES" ] || { echo "no poses exported, see $LOGS/export.log"; exit 1; }
python3 - "$OUT/rtabmap/rtabmap.db" <<'EOF' | tee "$LOGS/loop_closures.json"
import json, sqlite3, sys
con = sqlite3.connect(sys.argv[1])
names = {0: "neighbor", 1: "global_loop_closure", 2: "local_space_closure", 3: "local_time_closure",
         4: "user_closure", 5: "virtual_closure", 6: "neighbor_merged", 7: "pose_prior", 9: "gravity"}
links = {names.get(t, str(t)): n for t, n in con.execute("select type, count(*) from Link group by type")}
nodes = con.execute("select count(*) from Node").fetchone()[0]
maps = con.execute("select count(distinct map_id) from Node").fetchone()[0]
# each closure is stored in both directions
print(json.dumps({"nodes": nodes, "maps": maps, "links": links,
                  "loop_closures": (links.get("global_loop_closure", 0) + links.get("local_space_closure", 0)) // 2}, indent=2))
EOF
python3 $WS/scripts/apply_loop_closures.py "$SRC/bag" "$POSES" "$OUT/bag" | tee "$LOGS/apply_loop_closures.json"
echo "corrected bag: $OUT/bag"

if [ "${SKIP_REPROCESS:-0}" != 1 ]; then
  bash $WS/scripts/reprocess_d435i_bag.sh "${SESSION}_lc"
fi
