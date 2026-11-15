#!/usr/bin/env python3
"""The sharpest RGB / depth frames of every door of a session (e.g. for reading door signs).

  door_frames.py <session> <out dir> [--obs rebuilt/levelled/observations] [--every 1] [--top 3]

Run with the GroundedSAM2 environment (/root/data/envs/gdsam/bin/python, HF_HOME=/root/data/models/hf).
Every recorded frame (not only keyframes) is searched with Grounding DINO for "door" and for the
signs on it ("sign", "door sign"); each door box is cropped and scored by its sharpness (variance
of the Laplacian of the crop scaled to 240 px high). Its 3D position (median depth in the box
centre, back-projected with the frame's map pose) groups the detections into doors (within 0.8 m).
Per door the --top sharpest views whose box is clear of the image border are written:
  <out>/door_<k>/<rank>_rgb.png, _depth.png (uint16 mm, aligned), _crop.png, and door_<k>/door.json
  (door centre in the map, per view: frame, timestamp, box, sharpness, distance, world_T_sensor,
  signs found in the box), <out>/doors.json, <out>/overview.jpg (best crop of every door).
"""
import argparse, json, math, os, pathlib, time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("session"); ap.add_argument("out")
ap.add_argument("--obs", default="rebuilt/levelled/observations")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--top", type=int, default=3)
ap.add_argument("--box", type=float, default=0.35)
ap.add_argument("--dino", default="IDEA-Research/grounding-dino-base")
a = ap.parse_args()
obs = pathlib.Path(a.session) / a.obs
frames = [json.loads(l) for l in open(obs / "frames.jsonl")]
frames = [f for f in frames if f.get("status") == "written"][::a.every]
cal = next(iter(json.load(open(obs / "camera_calib.json"))["sensors"].values()))
fx, fy, cx, cy = cal["fx"], cal["fy"], cal["cx"], cal["cy"]
dev = "cuda"
proc = AutoProcessor.from_pretrained(a.dino)
proc.image_processor.size = {"shortest_edge": 480, "longest_edge": 848}  # native size, no upscaling
dino = AutoModelForZeroShotObjectDetection.from_pretrained(a.dino).to(dev).eval()
PROMPT = "door. door sign. sign."


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


dets = []  # door detections
t0 = time.time()
for n, f in enumerate(frames):
    bgr = cv2.imread(str(obs / f["rgb_path"])); h, w = bgr.shape[:2]
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16):
        inp = proc(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), text=PROMPT, return_tensors="pt").to(dev)
        r = proc.post_process_grounded_object_detection(dino(**inp), inp.input_ids, threshold=a.box,
                                                        text_threshold=0.25, target_sizes=[(h, w)])[0]
    names = r.get("text_labels", r.get("labels"))
    B = r["boxes"].float().cpu().numpy(); S = r["scores"].float().cpu().numpy()
    signs = [b for b, nm in zip(B, names) if "sign" in nm]
    for b, nm, s in zip(B, names, S):
        if nm.strip() != "door":
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, w), min(y1, h)
        if (x1 - x0) * (y1 - y0) < 0.02 * w * h:
            continue
        crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (max(1, int(240 * crop.shape[1] / crop.shape[0])), 240))
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        depth = cv2.imread(str(obs / f["depth_path"]), cv2.IMREAD_UNCHANGED)
        cxb, cyb = (x0 + x1) // 2, (y0 + y1) // 2
        rw, rh = max(2, (x1 - x0) // 6), max(2, (y1 - y0) // 6)
        d = depth[cyb - rh:cyb + rh, cxb - rw:cxb + rw].astype(float) / 1000
        d = d[(d > 0.2) & (d < 8)]
        if len(d) < 20:
            continue
        z = float(np.median(d))
        pc = np.array([(cxb - cx) * z / fx, (cyb - cy) * z / fy, z])
        T = (f.get("corrected") or {}).get("world_T_sensor") or f["world_T_sensor"]
        pw = quat_R(T["q_wxyz"]) @ pc + np.array(T["t"])
        border = x0 > 4 and y0 > 4 and x1 < w - 4 and y1 < h - 4
        in_box = [[round(float(v), 1) for v in sb] for sb in signs
                  if sb[0] >= x0 - 5 and sb[2] <= x1 + 5 and sb[1] >= y0 - 5 and sb[3] <= y1 + 5]
        dets.append({"frame_id": f["frame_id"], "timestamp_ns": int(f["timestamp_ns"]), "rgb": f["rgb_path"],
                     "depth": f["depth_path"], "box": [x0, y0, x1, y1], "score": round(float(s), 3),
                     "sharpness": round(sharp, 1), "distance_m": round(z, 2), "center": pw.round(3).tolist(),
                     "clear_of_border": border, "signs": in_box, "world_T_sensor": T})
    if (n + 1) % 200 == 0:
        print(f"{n + 1}/{len(frames)} frames, {len(dets)} door views, {(n + 1) / (time.time() - t0):.1f} fps", flush=True)

# group into doors: greedy, by 3D centre (x / y within 0.8 m)
doors = []
for d in sorted(dets, key=lambda d: -d["score"]):
    c = np.array(d["center"])
    for door in doors:
        if np.linalg.norm(np.array(door["center"])[:2] - c[:2]) < 0.8:
            door["views"].append(d)
            door["center"] = np.median([v["center"] for v in door["views"]], 0).tolist()
            break
    else:
        doors.append({"center": d["center"], "views": [d]})
doors = [dd for dd in doors if len(dd["views"]) >= 3]  # seen in at least 3 frames
doors.sort(key=lambda dd: (dd["center"][0], dd["center"][1]))
os.makedirs(a.out, exist_ok=True)
summary, tiles = [], []
for k, door in enumerate(doors):
    vs = sorted(door["views"], key=lambda v: (not v["clear_of_border"], -v["sharpness"]))
    best = vs[:a.top]
    dd = pathlib.Path(a.out) / f"door_{k:02d}"; dd.mkdir(exist_ok=True)
    for r, v in enumerate(best):
        bgr = cv2.imread(str(obs / v["rgb"])); x0, y0, x1, y1 = v["box"]
        cv2.imwrite(str(dd / f"{r}_rgb.png"), bgr)
        cv2.imwrite(str(dd / f"{r}_depth.png"), cv2.imread(str(obs / v["depth"]), cv2.IMREAD_UNCHANGED))
        cv2.imwrite(str(dd / f"{r}_crop.png"), bgr[y0:y1, x0:x1])
        v["rank"] = r
    sh = [v["sharpness"] for v in door["views"]]
    info = {"door": k, "center": [round(x, 3) for x in door["center"]], "views_seen": len(door["views"]),
            "sharpness_all_views_p50_max": [round(float(np.median(sh)), 1), round(max(sh), 1)], "best_views": best}
    json.dump(info, open(dd / "door.json", "w"), indent=1)
    summary.append({k2: info[k2] for k2 in ("door", "center", "views_seen", "sharpness_all_views_p50_max")}
                   | {"best_sharpness": best[0]["sharpness"], "best_distance_m": best[0]["distance_m"],
                      "best_frame": best[0]["frame_id"], "signs_in_best": len(best[0]["signs"])})
    c = cv2.imread(str(dd / "0_crop.png")); c = cv2.resize(c, (int(300 * c.shape[1] / c.shape[0]), 300))
    c = cv2.copyMakeBorder(c, 30, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(c, f"door {k}  sharp {best[0]['sharpness']:.0f}  {best[0]['distance_m']:.1f} m", (4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    tiles.append(c)
json.dump({"session": str(a.session), "frames_searched": len(frames), "door_views": len(dets), "doors": summary},
          open(pathlib.Path(a.out) / "doors.json", "w"), indent=1)
if tiles:
    rows, row, wsum = [], [], 0
    for t in tiles:
        if wsum + t.shape[1] > 1800 and row:
            rows.append(row); row, wsum = [], 0
        row.append(t); wsum += t.shape[1]
    rows.append(row)
    W = max(sum(t.shape[1] for t in r) for r in rows)
    sheet = np.vstack([np.hstack(r + [np.full((r[0].shape[0], W - sum(t.shape[1] for t in r), 3), 255, np.uint8)]) for r in rows])
    cv2.imwrite(str(pathlib.Path(a.out) / "overview.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(json.dumps({"frames": len(frames), "door_views": len(dets), "doors": len(doors),
                  "seconds": round(time.time() - t0)}))
for s in summary:
    print(s)
