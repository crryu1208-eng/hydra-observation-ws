#!/usr/bin/env bash
# Clone the pinned Hydra repos into this workspace (host side), switch hydra, hydra_ros and
# semantic_inference to the observation-recorder branch of their forks, and build the
# docker "dev" image.
# Sources are mounted into the container at /root/hydra_ws; colcon runs inside.
# usage: [DATASETS_PATH=<dir>] bash setup_workspace.sh
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"
DATASETS_PATH="${DATASETS_PATH:-$HOME/hydra_data}"
cd "$WS"

# name url commit (tested on ETRI4240, 2026-10-29~30)
cat > hydra_pinned.repos.tsv <<'EOF'
config_utilities	https://github.com/MIT-SPARK/config_utilities.git	1df5f6b
hydra	https://github.com/MIT-SPARK/Hydra.git	2e58a35b
hydra_ros	https://github.com/MIT-SPARK/Hydra-ROS.git	8f3b7e3
ianvs	https://github.com/MIT-SPARK/Ianvs.git	c204dec
kimera_pgmo	https://github.com/MIT-SPARK/Kimera-PGMO.git	382b1f3
kimera_rpgo	https://github.com/MIT-SPARK/Kimera-RPGO.git	222ad5e
pose_graph_tools	https://github.com/MIT-SPARK/pose_graph_tools.git	965dfe8
semantic_inference	https://github.com/MIT-SPARK/semantic_inference.git	f4ca55c
spark_dsg	https://github.com/MIT-SPARK/Spark-DSG.git	3c40997
spatial_hash	https://github.com/MIT-SPARK/Spatial-Hash.git	8045892
teaser_plusplus	https://github.com/MIT-SPARK/TEASER-plusplus.git	52a9c52
EOF

mkdir -p src
while IFS=$'\t' read -r name url rev; do
  if [ ! -d "src/$name/.git" ]; then
    git clone --quiet "$url" "src/$name"
  fi
  git -C "src/$name" -c advice.detachedHead=false checkout --quiet "$rev"
  printf '%-20s %s\n' "$name" "$(git -C "src/$name" rev-parse --short HEAD)"
done < hydra_pinned.repos.tsv

# observation-recorder changes live on branch feature/observation-recorder of these forks
FORK_BRANCH=feature/observation-recorder
for r in hydra:Hydra hydra_ros:Hydra-ROS semantic_inference:semantic_inference; do
  name=${r%%:*}; repo=${r#*:}
  git -C "src/$name" remote get-url fork >/dev/null 2>&1 \
    || git -C "src/$name" remote add fork "https://github.com/crryu1208-eng/$repo.git"
  git -C "src/$name" fetch --quiet fork "$FORK_BRANCH"
  if git -C "src/$name" rev-parse --verify --quiet "$FORK_BRANCH" >/dev/null; then
    git -C "src/$name" checkout --quiet "$FORK_BRANCH"
    echo "$name: $FORK_BRANCH already exists locally (not reset to fork)"
  else
    git -C "src/$name" checkout --quiet -b "$FORK_BRANCH" "fork/$FORK_BRANCH"
    echo "$name: $FORK_BRANCH from fork ($(git -C "src/$name" rev-parse --short HEAD))"
  fi
done

# same build options as the original PC (container has only system OpenCV)
cat > colcon_defaults.yaml <<'EOF'
build:
  symlink-install: true
  cmake-args:
    - -DCMAKE_BUILD_TYPE=Release
    - -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
EOF

mkdir -p "$DATASETS_PATH"
echo "DATASETS_PATH=$DATASETS_PATH" > src/hydra_ros/docker/.env

(cd src/hydra_ros/docker && make build PROFILE=dev)
echo "done. next: cd $WS/src/hydra_ros/docker && make up PROFILE=dev"
