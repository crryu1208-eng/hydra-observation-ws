# DSG Viewer

Qt desktop app for navigating a Hydra 3D scene graph. It opens `dsg.json`, `dsg_with_mesh.json`
or a session folder, and shows the following in one window:

- **Scene graph tree**: layers (objects, agents, places, 2D places, rooms, buildings) and their
  nodes. Search by symbol, class or id. Checkboxes show or hide a layer.
- **3D view** (VTK): the mesh (RGB, semantic colors or hidden, with adjustable opacity), nodes
  colored by layer, intra-layer and inter-layer edges, object bounding boxes and labels, and the
  agent trajectory colored by time. Every recorded frame's camera path is optional. The layer
  offset lifts places, rooms and buildings above the map, as in Hydra's own visualizer. Click a
  node to select it; its edges and box are highlighted.
- **Inspector**: the node's attributes, its connections (click one to jump to it), and the
  frames the observation recorder mapped to an agent node.
- **Camera panel** (bottom): the RGB and depth images of one recorded frame. Click a camera
  position on the trajectory in the 3D view (or a frame thumbnail in the inspector) to show it;
  its camera pose is drawn in 3D as a frustum. Depth is colored from 0 m (blue) to the
  "Depth max" value (red), black where there is no depth. Hover over either image to read the
  depth at that pixel. The slider, `,` / `.` and Space (play at 15 fps) move along all frames in
  time order. "Node" selects the agent node the frame is attached to.

The mesh comes from `dsg_with_mesh.json` next to the opened file. Frames come from the session's
`observations/` folder (`frames.jsonl`, `agent_frames.json`, `camera_calib.json`, `rgb/`,
`depth/`), found by walking up from the file. Frame poses are the loop-closure corrected ones when
`frames.jsonl` has them.

## Run

```bash
tools/dsg_viewer/run.sh ~/hydra_data/d435i/office_walk3                  # session folder
tools/dsg_viewer/run.sh ~/hydra_data/d435i/office_walk3/hydra/backend/dsg.json
```

The first run creates a Python 3.12 environment in `.venv/` with [uv](https://docs.astral.sh/uv/),
which installs PySide6, VTK and PyVista (about 1.2 GB). You can also drag a file onto the window.

Keys: `Ctrl+O` open, `F5` reload, `R` reset camera, `T` top view, `F` fly to selection,
`Esc` clear selection. In the camera panel: `,` previous frame, `.` next frame, Space play/pause.

VTK renders through X11. On a Wayland desktop the app runs under XWayland, which needs these
packages:

```bash
sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-render-util0 libxcb-image0 libxkbcommon-x11-0
```
