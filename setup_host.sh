#!/usr/bin/env bash
# Ubuntu 26.04 host setup for the Hydra-ROS docker "dev" profile:
# git/make, Docker Engine + compose/buildx, NVIDIA Container Toolkit.
# usage: sudo bash setup_host.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "[error] run with sudo"; exit 1; }
TARGET_USER="${SUDO_USER:?run via sudo from the target user}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git make curl gpg ca-certificates \
  docker.io docker-compose-v2 docker-buildx

# NVIDIA Container Toolkit (generic deb repo, distro independent)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

# compose file uses "runtime: nvidia"
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

usermod -aG docker "$TARGET_USER"

echo "== verify =="
docker --version
docker compose version
nvidia-ctk --version | head -1
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L
echo "done. log out/in (or run 'newgrp docker') so $TARGET_USER can use docker without sudo"
