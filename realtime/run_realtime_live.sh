#!/usr/bin/env bash
# Host side: start a realtime session (run_realtime.sh in the hydra-dev container) together with
# the live DSG viewer, which shows the scene graph and mesh being built (updated every 2 s) and
# the newest RGB / depth frame. Works without network: the container keeps DDS on localhost and
# the viewer uses its local uv environment / cache.
#   realtime/run_realtime_live.sh [session_name]
# Env passed to the container: FPS, RES, VIS, STOP_AFTER. The viewer stays open after the
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

# the viewer waits until the session writes <session>/hydra/live/dsg.json
VIEWER_LOG="$(mktemp -t dsg_viewer_live.XXXXXX.log)"
"$HERE/../tools/dsg_viewer/run.sh" --live "$SESSION" > "$VIEWER_LOG" 2>&1 &
VIEWER_PID=$!
sleep 2
if ! kill -0 $VIEWER_PID 2>/dev/null; then
  echo "the viewer did not start:"; tail -20 "$VIEWER_LOG"; exit 1
fi
echo "live viewer started (log $VIEWER_LOG)"

env_args=()
for v in FPS RES VIS STOP_AFTER; do
  [ -n "${!v:-}" ] && env_args+=(-e "$v=${!v}")
done
tty=(-i); [ -t 0 ] && tty=(-it)
trap : INT  # Ctrl-C goes to the session (it stops cleanly); this script waits for it
docker exec "${tty[@]}" "${env_args[@]}" -e LIVE=1 hydra-dev \
  bash /root/hydra_ws/src/hydra-realtime/realtime/run_realtime.sh "$NAME" || true
echo "session finished: $SESSION (the viewer is still open; close it when done)"
