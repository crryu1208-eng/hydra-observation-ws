#!/usr/bin/env bash
# Start the DSG viewer (creates its Python 3.12 environment with uv on first run).
# usage: tools/dsg_viewer/run.sh [dsg.json | dsg_with_mesh.json | session folder]
set -euo pipefail
cd "$(dirname "$0")"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "uv not found: install it (https://docs.astral.sh/uv/) into ~/.local/bin"; exit 1; }
args=()
for a in "$@"; do  # make relative paths absolute before uv changes directory
  if [ -e "$OLDPWD/$a" ] && [[ "$a" != /* ]]; then args+=("$OLDPWD/$a"); else args+=("$a"); fi
done
exec "$UV" run --quiet dsg-viewer "${args[@]}"
