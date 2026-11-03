#!/usr/bin/env python3
"""Per-5 s rates of a realtime session from its raw bag and recorder output: camera color, aligned
depth, color+depth pairs, odometry, recorded frames, plus odometry time / features. Shows where
frames were lost (a recorded rate equal to the color+depth rate means the camera driver was the
limit, not the pipeline). In the container: python3 session_rates.py /root/data/realtime/<session>
"""
import sys, rosbag2_py, json, numpy as np
from rclpy.serialization import deserialize_message
from rtabmap_msgs.msg import OdomInfo
from sensor_msgs.msg import Image
sess=sys.argv[1]
r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=f"{sess}/bag"), rosbag2_py.ConverterOptions("cdr","cdr"))
r.set_filter(rosbag2_py.StorageFilter(topics=["/camera/camera/color/image_raw","/camera/camera/aligned_depth_to_color/image_raw","/rtabmap/odom_info"]))
S={}; oi=[]
while r.has_next():
    tp,d,t=r.read_next()
    if tp.endswith("odom_info"):
        m=deserialize_message(d,OdomInfo); st=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        oi.append((st, m.time_estimation*1e3, m.features, m.local_map_size, m.lost))
    else:
        m=deserialize_message(d,Image); S.setdefault(tp,[]).append(m.header.stamp.sec+m.header.stamp.nanosec*1e-9)
fr=[json.loads(l) for l in open(f"{sess}/observations/frames_raw.jsonl")]
rec=np.array(sorted(int(f["timestamp_ns"]) for f in fr))/1e9
oi=np.array(oi); t0=rec[0]
rgb=np.array(S["/camera/camera/color/image_raw"]); dep=np.array(S["/camera/camera/aligned_depth_to_color/image_raw"])
both=np.intersect1d(np.round(rgb,4), np.round(dep,4))
def rate(ts,a,b): ts=np.asarray(ts); return ((ts>=a)&(ts<b)).sum()/(b-a)
print("stamps relative to first recorded frame; rates in Hz")
print(" win     rgb depth rgb+d  odom  rec | odom ms p50 p90  feats lmap")
for a in range(-5,int(rec[-1]-t0)+5,5):
    A,B=t0+a,t0+a+5; sel=(oi[:,0]>=A)&(oi[:,0]<B); ms=oi[sel,1]
    print(f"{a:4d}-{a+5:<3d} {rate(rgb,A,B):5.1f} {rate(dep,A,B):5.1f} {rate(both,A,B):5.1f} {rate(oi[:,0],A,B):5.1f} {rate(rec,A,B):5.1f} | {np.percentile(ms,50) if len(ms) else 0:7.0f} {np.percentile(ms,90) if len(ms) else 0:4.0f} {np.median(oi[sel,2]) if sel.any() else 0:6.0f} {np.median(oi[sel,3]) if sel.any() else 0:5.0f}")
