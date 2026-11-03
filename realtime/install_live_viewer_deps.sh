#!/usr/bin/env bash
# spark_dsg python bindings for realtime/live_dsg_dump.py (not part of the docker image; run
# once inside the hydra-dev container, again after the container is recreated; ~1 min)
set -euo pipefail
pip install --user --break-system-packages /root/hydra_ws/src/spark_dsg
python3 -c "import spark_dsg; print('spark_dsg', spark_dsg.__file__)"
