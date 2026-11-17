#!/usr/bin/env bash
# Host side: start a realtime session in the hydra-dev container (realtime/run_realtime.sh) and the live DSG
# viewer (tools/dsg_viewer --live) next to it; the viewer waits for <session>/hydra/live/dsg.json and follows it.
#   realtime/record.sh [session] [NAME=value ...]
#   realtime/record.sh lab_05                       # 848x480, camera 30 fps, mapping 15 Hz, open-vocabulary objects
#   realtime/record.sh lab_05 OV=0 STOP_AFTER=120   # extra run_realtime.sh settings (later ones win)
# VIEWER=0 records without the viewer. Stop with Enter as usual; the viewer stays open (it has its own window,
# Ctrl-C in this terminal does not close it). Viewer output: /tmp/dsg_viewer_<session>.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-$HOME/hydra_data}"  # mounted at /root/data in the container
NAME="${1:-$(date +%Y%m%d_%H%M%S)}"; shift || true
if [ -e "$DATA/realtime/$NAME" ]; then echo "$DATA/realtime/$NAME already exists: pick another session name"; exit 1; fi

ENV=(-e FPS=30 -e PIPE_FPS=15 -e RES=848x480 -e OV=1)
for a in "$@"; do
  [[ "$a" == *=* ]] || { echo "expected NAME=value, got '$a'"; exit 1; }
  ENV+=(-e "$a")
done

if [ "${VIEWER:-1}" = 1 ]; then
  log="/tmp/dsg_viewer_$NAME.log"
  # own session: Enter / Ctrl-C for the recording do not reach the viewer
  setsid "$HERE/../tools/dsg_viewer/run.sh" --live "$DATA/realtime/$NAME" < /dev/null > "$log" 2>&1 &
  echo "live viewer started (waits for the scene graph; log $log)"
fi
exec docker exec -it "${ENV[@]}" hydra-dev bash /root/hydra_ws/src/hydra-realtime/realtime/run_realtime.sh "$NAME"
