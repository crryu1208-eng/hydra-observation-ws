#!/usr/bin/env bash
# Host side: change the color camera exposure of the running session (while recording too).
#   realtime/cam_exposure.sh auto              auto exposure
#   realtime/cam_exposure.sh <ms> [gain]       fixed exposure in ms (0.1 .. 66: longer than
#                                              a 15 fps frame lowers the frame rate) and gain (0 .. 128,
#                                              default 50), e.g. 8.3 50
#   realtime/cam_exposure.sh                   show the current values
# Each change is logged with its time in <session>/logs/camera_changes.log.
set -euo pipefail
cam() {  # set name=value ... in order and print the exposure parameters (cam_params.py, one ROS call)
  docker exec hydra-dev timeout 20 bash -c "source /opt/ros/jazzy/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST && python3 /root/hydra_ws/src/hydra-realtime/realtime/cam_params.py $*"
}
# the session name: the argument after run_realtime.sh in the running script's command line
session=$(docker exec hydra-dev bash -c 'for p in $(pgrep -f "run_realtime\.sh"); do
    tr "\0" "\n" < /proc/$p/cmdline 2>/dev/null | grep -A1 -m1 "run_realtime\.sh$" | sed -n 2p; done' | grep -v "^$" | head -1)
[ -n "$session" ] || { echo "no session running"; exit 1; }
if [ $# -eq 0 ]; then echo "session $session:"; cam; exit 0; fi
if [ "$1" = auto ]; then
  steps=(rgb_camera.enable_auto_exposure=true)
else
  exp=$(awk -v m="$1" 'BEGIN { if (m + 0 < 0.1 || m + 0 > 66) exit 1; printf "%d", m * 10 + 0.5 }') \
    || { echo "exposure must be auto or 0.1 .. 66 ms"; exit 1; }
  gain="${2:-50}"
  [[ "$gain" =~ ^[0-9]+$ ]] && [ "$gain" -le 128 ] || { echo "gain must be 0 .. 128"; exit 1; }
  steps=(rgb_camera.enable_auto_exposure=false rgb_camera.exposure=$exp rgb_camera.gain=$gain)
fi
echo "session $session:"
cam "${steps[@]}" || { echo "could not set the camera"; exit 1; }
log="/root/data/realtime/$session/logs/camera_changes.log"
docker exec hydra-dev bash -c "echo \"\$(date +%s.%N) $(date +%T) color exposure $* -> ${steps[*]}\" >> $log"
