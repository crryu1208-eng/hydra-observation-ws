# Hydra observation-recorder workspace

Workspace files for running [Hydra](https://github.com/MIT-SPARK/Hydra) in the Hydra-ROS docker
`dev` container with the **observation recorder**. The recorder saves every RGB/depth frame
Hydra integrates, then maps each frame to a trajectory node of the scene graph. Two inputs are
set up:

- **uHumans2 office bag**: offline, lossless, with GPU semantic segmentation
- **Handheld Intel RealSense D435i**: live, with RTAB-Map RGB-D odometry and the D435i IMU

The code changes live on branch `feature/observation-recorder` of three forks:

| repo | fork |
|---|---|
| Hydra | https://github.com/crryu1208-eng/Hydra |
| Hydra-ROS | https://github.com/crryu1208-eng/Hydra-ROS |
| semantic_inference | https://github.com/crryu1208-eng/semantic_inference |

Recorder output format and mapping rules: `doc/observation_recording.md` in the Hydra fork.

Tested on Ubuntu 26.04, RTX 3080 Ti Laptop, NVIDIA driver 595.99, ROS 2 Jazzy (container).

## Layout

| path | content |
|---|---|
| `setup_host.sh` | host: Docker, NVIDIA Container Toolkit, docker group |
| `setup_workspace.sh` | clones pinned repos into `src/`, switches the three forks' branch, builds the `dev` image |
| `hydra_pinned.repos.tsv` | pinned upstream commits |
| `colcon_defaults.yaml` | colcon build options (Release) |
| `launch/d435i_obs.launch.yaml` | RTAB-Map odometry + IMU filter + segmentation + Hydra for the D435i |
| `config/d435i_*.yaml` | Hydra and input config for the D435i |
| `config/uhumans2_*.yaml` | Hydra and input config for uHumans2 (deep queues, recorder on) |
| `scripts/run_d435i_obs.sh` | one handheld session: camera, bag, Hydra, drain, mapping, verification |
| `scripts/reprocess_d435i_bag.sh` | re-runs segmentation + Hydra on a session bag without dropping frames |
| `scripts/run_uhumans2_obs.sh` | lossless uHumans2 run |
| `scripts/d435i_monitor.py` | live tracking status; resets odometry if it stays lost before recording |
| `scripts/verify_d435i.py`, `scripts/verify_observations.py` | check that frames were recorded and mapped |
| `scripts/export_efficientvit_onnx.py` | exports the ADE20K segmentation model to ONNX |
| `scripts/count_topics.py` | per-topic message counts (rate check) |

Not in git: `src/` (cloned by the setup script), `build/`, `install/`, `log/` and `.models/`.

## Setup

```bash
sudo bash setup_host.sh          # once; then log out/in for the docker group
bash setup_workspace.sh          # DATASETS_PATH defaults to ~/hydra_data (mounted as /root/data)
cd src/hydra_ros/docker && make up PROFILE=dev     # container "hydra-dev"
```

Inside the container, build from `/root/hydra_ws`:

```bash
docker exec -it hydra-dev bash -c "cd /root/hydra_ws && colcon build"
```

### Segmentation model

The `semantic_inference` node needs `.models/ade20k-efficientvit_seg_l2.onnx`. The run scripts
point `SEMANTIC_INFERENCE_MODEL_DIR` there. The upstream download link is dead, so export it:

1. Download
   [efficientvit_seg_l2_ade20k.pt](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_ade20k.pt)
   to `.models/src/`.
2. Run the `docker run` command in the docstring of `scripts/export_efficientvit_onnx.py`.

The TensorRT engine (`.trt`) is built next to the ONNX on first use.

### D435i packages

The live D435i setup needs these packages. They are installed with apt inside the container and
are lost when the container is recreated:

```bash
docker exec hydra-dev bash -c "apt-get update && apt-get install -y \
  ros-jazzy-realsense2-camera ros-jazzy-rtabmap-odom ros-jazzy-imu-filter-madgwick"
```

## Handheld D435i session

```bash
docker exec -it hydra-dev bash /root/hydra_ws/scripts/run_d435i_obs.sh <session_name>
```

1. Hold the camera still, facing a textured scene 1-3 m away, until `odometry tracking` appears.
2. Walk slowly once `RECORDING` appears; turn slowly. Avoid close-ups of monitors, blank walls
   and windows.
3. Press Enter once to stop, then wait for `done:` (30-60 s while Hydra drains and saves).

Environment options: `FPS` (default 15), `VIS=1` (Hydra visualizer), `STOP_AFTER=<s>` (stop
automatically).

Output in `/root/data/d435i/<session>/` (host: `~/hydra_data/d435i/<session>/`):

| path | content |
|---|---|
| `bag/` | raw camera, IMU and odometry rosbag (sqlite3), for offline re-runs |
| `observations/` | recorded frames, `frames.jsonl` with poses and node mapping, `agent_frames.json` |
| `hydra/` | scene graph (`backend/dsg.json`) and Hydra logs |
| `logs/` | launch logs, `verify.log`, `tracking_lost_frames.txt` |

### Reprocessing from the bag

The live run can lose frames when segmentation falls behind the camera. In office_walk5, run on
battery with the GPU power-capped, Hydra got no input for 78 s. The bag has every camera frame
and the live odometry in `/tf`. Replaying it slowly gives a lossless result:

```bash
docker exec hydra-dev bash /root/hydra_ws/scripts/reprocess_d435i_bag.sh <session> [play_rate]   # default 0.5
```

Output goes to `<session>_reproc/`, with the same layout as a live session. The script launches
`d435i_obs.launch.yaml` with `odometry:=false use_sim_time:=true`. In office_walk5 it recorded
2143 of 2207 camera frames, against 961 live. The 64 missing frames are the ones without a
pose (before odometry started, or while it was lost) or without depth.

### Notes from testing

- RTAB-Map uses `Odom/ResetCountdown: 1`: after a lost frame it restarts at the last pose
  instead of staying lost. The IMU (Madgwick filter) gives it a motion guess and gravity.
- Hydra adds a trajectory node only after about 0.5 m of motion, so a static camera produces
  no nodes. Frames are mapped to nodes within 60 s and 1 m, with no angle limit
  (`--max-anchor-angle-deg 180`), because turning in place adds no node.
- `ros2 topic hz` under-reports large images; use `scripts/count_topics.py`.
- RTAB-Map needs `qos: 1` with the realsense publishers. The default reached only about 3 Hz.

## uHumans2

The ROS 1 bag `uHumans2_office_s1_00h_v2.bag` has to be converted with `rosbags-convert` to
`~/hydra_data/office` (sqlite3). Then run:

```bash
docker exec hydra-dev bash /root/hydra_ws/scripts/run_uhumans2_obs.sh [play_rate]   # default 0.5
```
