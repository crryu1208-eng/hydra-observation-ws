#!/usr/bin/env bash
# Live sessions with the IR stereo pair recorded (visual-inertial odometry tests), on the host:
#   realtime/run_ir_live.sh on|off|onoff [session]   one session with that IR projector mode
#   realtime/run_ir_live.sh all [prefix]             on, off and onoff one after the other
# Projector: on = dots in the IR images (best depth), off = clean IR (weaker depth on blank
# surfaces), onoff = every other frame. Sessions are named <prefix or ir>_<mode>_<MMDD_HHMM>
# unless a session name is given; the map rebuild after recording is skipped (REBUILD=0).
# Same as run_realtime_live.sh with RECORD_IR=1 IR_EMITTER=<mode> REBUILD=0.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-}"
case "$MODE" in
  on|off|onoff|all) ;;
  *) echo "usage: $0 on|off|onoff|all [session or prefix]"; exit 1 ;;
esac

record() {  # record <mode> <session>
  echo
  echo "=== IR projector $1: session $2 ==="
  RECORD_IR=1 IR_EMITTER="$1" REBUILD="${REBUILD:-0}" "$HERE/run_realtime_live.sh" "$2"
}

stamp=$(date +%m%d_%H%M)
if [ "$MODE" != all ]; then
  record "$MODE" "${2:-ir_${MODE}_$stamp}"
  exit 0
fi

prefix="${2:-ir}"
modes=(on off onoff)
for i in "${!modes[@]}"; do
  m=${modes[$i]}
  if [ "$i" -gt 0 ]; then
    echo
    echo ">>> Next: projector $m. Go back to the start point (same path as before),"
    read -r -p ">>> then press Enter to start (Ctrl-C to stop here): " _
  fi
  record "$m" "${prefix}_${m}_$stamp"
done
echo
echo "done: ${prefix}_on_$stamp ${prefix}_off_$stamp ${prefix}_onoff_$stamp"
