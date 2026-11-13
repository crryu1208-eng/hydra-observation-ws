#!/usr/bin/env bash
# Host side: start a realtime session (run_realtime.sh in the hydra-dev container) together with
# the live DSG viewer, which shows the scene graph and mesh being built (updated every 2 s) and
# the newest RGB / depth frame. Works without network: the container keeps DDS on localhost and
# the viewer uses its local uv environment / cache.
#   realtime/run_realtime_live.sh [session_name]
# Env passed to the container: FPS, RES, VIS, STOP_AFTER, CAM_NICE, CAM_HEIGHT, FLOOR_LOCK, PIN, CAM_CPUS, PIPE_CPUS, REBUILD, RECORD_IR, IR_EMITTER,
# RGB_EXPOSURE, RGB_GAIN (without RGB_EXPOSURE an interactive run asks for the exposure). The viewer stays open after the
# session ends (File > Open session folder shows the final, loop-closure corrected result).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${DATASETS_PATH:-$HOME/hydra_data}/realtime/$NAME"
if [ -e "$SESSION" ]; then echo "$SESSION already exists"; exit 1; fi
docker ps --format '{{.Names}}' | grep -qx hydra-dev \
  || { echo "container hydra-dev is not running (cd src/hydra_ros/docker && make up PROFILE=dev)"; exit 1; }
docker exec hydra-dev python3 -c "import spark_dsg" 2>/dev/null || {
  echo "installing the spark_dsg python bindings in the container (once, ~1 min)"
  docker exec hydra-dev bash /root/hydra_ws/src/hydra-realtime/realtime/install_live_viewer_deps.sh
}

# color camera exposure: asked here unless RGB_EXPOSURE is given (or the run is not interactive);
# it can also be changed while recording with realtime/cam_exposure.sh
if [ -z "${RGB_EXPOSURE:-}" ] && [ -t 0 ] && [ -z "${STOP_AFTER:-}" ]; then
  echo
  echo "Color camera exposure (change it while recording with realtime/cam_exposure.sh):"
  echo "  1) auto                         (default; changing light, may blur on fast turns in dim rooms)"
  echo "  2) fixed 8.3 ms, gain 50        (sharp; bright rooms, no 60 Hz banding)"
  echo "  3) fixed, enter exposure [ms] and gain"
  read -r -t 30 -p "choice [1]: " choice || choice=1
  case "${choice:-1}" in
    2) RGB_EXPOSURE=83; RGB_GAIN=50 ;;
    3) read -r -p "exposure in ms (0.1 .. 66, e.g. 8.3; longer than a frame lowers the frame rate): " ms
       read -r -p "gain (0 .. 128, e.g. 50; higher = brighter, noisier) [50]: " RGB_GAIN
       RGB_GAIN="${RGB_GAIN:-50}"
       RGB_EXPOSURE=$(awk -v m="$ms" 'BEGIN { if (m + 0 < 0.1 || m + 0 > 66) exit 1; printf "%d", m * 10 + 0.5 }') \
         || { echo "exposure must be 0.1 .. 66 ms"; exit 1; }
       [[ "$RGB_GAIN" =~ ^[0-9]+$ ]] && [ "$RGB_GAIN" -le 128 ] || { echo "gain must be 0 .. 128"; exit 1; } ;;
    *) RGB_EXPOSURE=auto ;;
  esac
  echo "color exposure: $RGB_EXPOSURE$([ "$RGB_EXPOSURE" != auto ] && echo " (0.1 ms units), gain $RGB_GAIN")"
fi

# the viewer waits until the session writes <session>/hydra/live/dsg.json
VIEWER_LOG="$(mktemp -t dsg_viewer_live.XXXXXX.log)"
# the viewer stays off the camera driver's CPUs too (run_realtime.sh pins the driver to CAM_CPUS)
viewer_pin=(); [ "${PIN:-1}" = 1 ] && command -v taskset > /dev/null && viewer_pin=(taskset -c "${PIPE_CPUS:-6-19}")
"${viewer_pin[@]}" "$HERE/../tools/dsg_viewer/run.sh" --live "$SESSION" > "$VIEWER_LOG" 2>&1 &
VIEWER_PID=$!
sleep 2
if ! kill -0 $VIEWER_PID 2>/dev/null; then
  echo "the viewer did not start:"; tail -20 "$VIEWER_LOG"; exit 1
fi
echo "live viewer started (log $VIEWER_LOG)"

env_args=()
for v in FPS RES VIS STOP_AFTER CAM_NICE CAM_HEIGHT FLOOR_LOCK PIN CAM_CPUS PIPE_CPUS REBUILD RECORD_IR IR_EMITTER RGB_EXPOSURE RGB_GAIN; do
  [ -n "${!v:-}" ] && env_args+=(-e "$v=${!v}")
done
tty=(-i); [ -t 0 ] && tty=(-it)
trap : INT  # Ctrl-C goes to the session (it stops cleanly); this script waits for it
docker exec "${tty[@]}" "${env_args[@]}" -e LIVE=1 hydra-dev \
  bash /root/hydra_ws/src/hydra-realtime/realtime/run_realtime.sh "$NAME" || true
echo "session finished: $SESSION (the viewer is still open; close it when done)"
