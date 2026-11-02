#!/usr/bin/env python3
"""Pixel agreement of segmentation labels between two profiling runs (SAVE_LABELS=1).

  compare_labels.py <reference_run>/labels <test_run>/labels
Prints per-frame agreement on the frames both runs saved and the overall mean, plus mean
IoU over the reference classes covering >= 0.5 % of the pixels.
"""
import pathlib
import sys

import numpy as np


def main():
    ref_dir, test_dir = map(pathlib.Path, sys.argv[1:3])
    ref = {p.stem: p for p in ref_dir.glob("*.npy")}
    test = {p.stem: p for p in test_dir.glob("*.npy")}
    common = sorted(set(ref) & set(test))
    if not common:
        print("no common frames")
        return 1
    agree, inter, union, total = [], {}, {}, 0
    for k in common:
        a, b = np.load(ref[k]), np.load(test[k])
        agree.append(float((a == b).mean()))
        total += a.size
        for c in np.unique(a):
            ma, mb = a == c, b == c
            inter[c] = inter.get(c, 0) + int((ma & mb).sum())
            union[c] = union.get(c, 0) + int((ma | mb).sum())
    counts = {c: 0 for c in inter}
    for k in common:
        a = np.load(ref[k])
        for c, n in zip(*np.unique(a, return_counts=True)):
            counts[c] += int(n)
    major = [c for c in inter if counts[c] / total >= 0.005]
    miou = np.mean([inter[c] / union[c] for c in major])
    print(f"frames {len(common)}  pixel agreement mean {np.mean(agree):.4f} "
          f"min {np.min(agree):.4f}  mIoU(vs ref, {len(major)} classes) {miou:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
