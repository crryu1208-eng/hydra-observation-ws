---
name: rtabmap-keyframe-analysis
description: Findings from per-frame (15 Hz) RTAB-Map keyframes vs Hydra agent nodes on office_walk6 (2026-11-01) — TF-wait pitfall, timing, accuracy
metadata:
  type: project
---
Done 2026-11-01 (commit 5ccdc46, KEYFRAMES=all / STM_SIZE / TAG in loop_close_d435i_bag.sh, scripts/compare_keyframes.py).

- Pitfall: rtabmap_slam looks up odom→camera_link TF for every frame even with odom from topic and odom_sensor_sync:=false; without /tf in the SLAM input it waits wait_for_transform (0.2 s sim) per frame → queue overflow, only ~30% frames used. Fix: keep /tf in slam_input. (Earlier 2 Hz runs also had this wait but results unaffected.)
- walk6 15 Hz: 1056 keyframes of 1078 synced frames; 98/115 agent nodes match a keyframe exactly by stamp, the other 17 are on frames with no valid odometry (tracking lost).
- Hydra agent poses (2 Hz-corrected) vs 15 Hz optimized: median 0.28 m / 3°, mostly a ~3° global rotation; after rigid alignment 0.13 m. Equals 2 Hz-vs-15 Hz at shared stamps → difference is between SLAM solutions, not interpolation. At 2 Hz keyframe stamps agent == SLAM pose exactly.
- Timing per keyframe (median first/last quarter): 2 Hz 37/42 ms; 15 Hz STM10 87/179 ms; 15 Hz STM75 37/168 ms. Late cost = proximity registration on return to start (~90 ms) — inherent; so 15 Hz needs replay at 0.3x, not real-time. RTAB-Map is sequential per frame (links OpenMP/TBB internally); can parallelize only across sessions.
- STM75 gives 389 revisit closures (vs 75 / 7) but trajectory differs ~0.3 m from the other two; no ground truth to decide. No loop closures in return corridor 30–50 s in any run.

**Why:** user wants to compare agent nodes with SLAM keyframes; conclusion was keep 2 Hz for the pipeline, 15 Hz only for analysis.
**How to apply:** if accuracy must be decided, suggest a ground-truth measurement (tape-measured distances / known markers) rather than more SLAM variants.
