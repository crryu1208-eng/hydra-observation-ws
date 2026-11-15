#!/usr/bin/env python3
"""Object labels from GroundedSAM2 (Grounding DINO + SAM 2.1, both Apache-2.0) painted over the
closed-set labels (ADE20K ids) of a session's frames.

  gdsam_labels.py <observations dir> <base labels dir or -> <out dir> [--every N] [--frames a:b]
                  [--prompts realtime/gdsam_prompts.yaml] [--box 0.35] [--text 0.25]

Run with the GroundedSAM2 environment (/root/data/envs/gdsam/bin/python, models in
HF_HOME=/root/data/models/hf; works offline once downloaded).
For every frame of <observations dir>/frames.jsonl: Grounding DINO finds the prompted objects
(boxes), SAM 2.1 cuts each box into a mask, and the masks are painted, largest first, over the
frame's closed-set label image (<base labels dir>/<timestamp_ns>.png, saved by seg_labels.sh) with
the ADE20K id of their prompt. Floor, rug and ceiling pixels keep their closed-set label (the floor
is what places and rooms are built on). With '-' as base, unpainted pixels are 0 (wall).
Out: <out dir>/<timestamp_ns>.png (uint16 label image, as the closed-set node publishes),
<out dir>/detections.jsonl (per frame: label, score, box, mask pixels), stats.json.
"""
import argparse, json, os, pathlib, time
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("obs"); ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--prompts", default=str(HERE / "gdsam_prompts.yaml"))
ap.add_argument("--every", type=int, default=1, help="label every Nth frame (others are skipped)")
ap.add_argument("--frames", default="", help="a:b slice of the frames (tests)")
ap.add_argument("--box", type=float, default=0.35, help="Grounding DINO box threshold")
ap.add_argument("--text", type=float, default=0.25, help="Grounding DINO text threshold")
ap.add_argument("--dino", default="IDEA-Research/grounding-dino-base")
ap.add_argument("--sam", default="facebook/sam2.1-hiera-small")
a = ap.parse_args()

KEEP = {3, 28, 5}  # floor, rug, ceiling: never painted over
prompts = yaml.safe_load(open(a.prompts))["prompts"]  # {text: ade20k id}
texts = list(prompts)
text_prompt = ". ".join(texts) + "."
dev = "cuda"
dproc = AutoProcessor.from_pretrained(a.dino)
dproc.image_processor.size = {"shortest_edge": 480, "longest_edge": 848}  # native size (default upscales to 800 px)
dino = AutoModelForZeroShotObjectDetection.from_pretrained(a.dino).to(dev).eval()
sproc = Sam2Processor.from_pretrained(a.sam)
sam = Sam2Model.from_pretrained(a.sam).to(dev).eval()


def label_of(phrase):
    """Grounding DINO returns the matched prompt words; take the prompt that matches best."""
    phrase = phrase.strip().lower()
    if phrase in prompts:
        return prompts[phrase]
    best = max(texts, key=lambda t: (t in phrase or phrase in t, len(set(t.split()) & set(phrase.split()))))
    return prompts[best] if (best in phrase or phrase in best or set(best.split()) & set(phrase.split())) else None


frames = [json.loads(l) for l in open(f"{a.obs}/frames.jsonl")]
frames = [f for f in frames if f.get("status") == "written"]
if a.frames:
    i, j = (int(x) if x else None for x in a.frames.split(":"))
    frames = frames[i:j]
frames = frames[::a.every]
os.makedirs(a.out, exist_ok=True)
det_log = open(f"{a.out}/detections.jsonl", "a")
t0, n, per = time.time(), 0, {}
for f in frames:
    stamp = int(f["timestamp_ns"]); out = f"{a.out}/{stamp}.png"
    if os.path.exists(out):
        continue
    bgr = cv2.imread(f"{a.obs}/{f['rgb_path']}")
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = bgr.shape[:2]
    if a.base != "-":
        base = cv2.imread(f"{a.base}/{stamp}.png", cv2.IMREAD_UNCHANGED)
        if base is None:
            print(f"no base labels for {stamp}, skipped"); continue
        labels = base.astype(np.uint16)
    else:
        labels = np.zeros((h, w), np.uint16)
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16):
        inp = dproc(images=rgb, text=text_prompt, return_tensors="pt").to(dev)
        res = dproc.post_process_grounded_object_detection(
            dino(**inp), inp.input_ids, threshold=a.box, text_threshold=a.text, target_sizes=[(h, w)])[0]
    boxes = res["boxes"].float().cpu().numpy()
    names = res.get("text_labels", res.get("labels"))
    scores = res["scores"].float().cpu().numpy()
    dets = []
    keep = [(b, label_of(nm), s, nm) for b, nm, s in zip(boxes, names, scores)]
    keep = [k for k in keep if k[1] is not None and (k[0][2] - k[0][0]) * (k[0][3] - k[0][1]) < 0.9 * w * h]
    if keep:
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
            sin = sproc(images=rgb, input_boxes=[[k[0].tolist() for k in keep]], return_tensors="pt").to(dev)
            sout = sam(**sin, multimask_output=False)
            masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0][:, 0].numpy() > 0
        order = np.argsort([-m.sum() for m in masks])  # largest first, smaller objects on top
        for i in order:
            b, lab, s, nm = keep[i]
            m = masks[i] & ~np.isin(labels, list(KEEP))
            labels[m] = lab
            dets.append({"label": int(lab), "phrase": nm, "score": round(float(s), 3),
                         "box": [round(float(x), 1) for x in b], "pixels": int(m.sum())})
            per[nm] = per.get(nm, 0) + 1
    cv2.imwrite(out, labels)
    det_log.write(json.dumps({"timestamp_ns": stamp, "detections": dets}) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"{n}/{len(frames)} frames, {n / (time.time() - t0):.1f} fps", flush=True)
dt = time.time() - t0
stats = {"frames": n, "seconds": round(dt, 1), "fps": round(n / dt, 2) if dt else None,
         "detections": dict(sorted(per.items(), key=lambda x: -x[1])), "prompt": text_prompt,
         "box_threshold": a.box, "text_threshold": a.text}
json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)
print(json.dumps(stats))
