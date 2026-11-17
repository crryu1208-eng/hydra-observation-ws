"""Open-vocabulary object candidates at 2 Hz beside the realtime pipeline (no Ultralytics code:
EfficientViT-SAM Apache-2.0, SigLIP2 / open_clip Apache-2.0 / MIT).

Per keyframe (every `period` s):
 0. (wait_for_pose) keyframes before the odometry frame and the camera are connected in TF are skipped (no pose:
    they could not be fused); the GPU work starts with the odometry
 1. keyframe = sharpest color frame (variance of the Laplacian at 640x360) received since the previous
    keyframe whose closed-set label image has arrived (ov_bench/acc_blur*.py: blurred keyframes 12% -> 2%)
 2. EfficientViT-SAM L0 TensorRT automatic masks, `pps` x `pps` point grid (ov_bench/evsam_bench.py TrtFastAMG)
 3. SigLIP2-B/16 TensorRT embeddings of every candidate (bbox + 10% context), `max_emb` crops per batch
 4. wall / floor drop: >= `stuff_frac` of the mask carries a structural closed-set label AND the SigLIP2 top-1
    of the vocabulary is wall / floor / ceiling / pillar (ov_bench/acc_filter2.py: stuff 148 -> 44, objects 91 -> 90)
 5. best `max_emb` by mask quality; parent = smallest kept mask holding >= `contain` of it (kept, not deleted)
 6. 3D fusion (fusion:=true, separate thread, realtime/ov_fusion.py): the keyframe's aligned depth (same stamp) and
    its `world_frame` pose from TF turn the masks into voxels, matched to / merged with the objects so far
 7. every `publish_every` fused keyframes the objects go out on `objects` (std_msgs/String, JSON: vocab + objects
    with flags and the mean SigLIP2 embedding as base64 float16); realtime/live_dsg_dump.py puts the confirmed ones
    into the scene graph it writes (realtime/ov_dsg.py)
 8. Hydra's own objects (boxes in the odometry frame on `closed_objects`, from live_dsg_dump.py): in each keyframe
    where one is fully in front, >= 70% inside the image, >= 24 px (at 848 wide) and not hidden (median depth in its
    projected box >= its distance - 0.3 m - half its size), its crop (box + 10%) is embedded with SigLIP2; the mean
    goes out with the objects ('closed'), so the graph gives them the same attributes (ov_dsg.apply_closed). Ceiling
    lights: the white pixels (gray >= 235, low saturation, blobs >= 150 px at 848 wide) around the light, cut with
    the ceiling plane at their median depth, boxed (2-98%) in the keyframe with the most of them.

Heavy code and engines come from the ov_bench directory (parameter ov_bench), whose virtualenv packages must be
on PYTHONPATH (rt.launch.yaml ov_objects:=true does this). Output in out_dir (optional): kf_<stamp>.npz per
keyframe (boxes, scores, embeddings, top-1, parent, masks at 1/2 resolution bit-packed), timing.jsonl and
objects.json (fused objects seen in >= min_obs keyframes, rewritten every 20 keyframes and at exit).
"""
import os, sys, json, time, threading, collections, queue
import numpy as np, cv2
import rclpy, rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import base64
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from ov_fusion import Fusion, quat_to_R

STUFF_IDS = [0, 1, 2, 3, 5, 6, 11, 13, 42, 52, 53, 59, 121]  # ADE20K wall building sky floor ceiling road ... step
STUFF_WORDS = ('wall', 'floor', 'ceiling', 'pillar')


def sharpness(gray):
    g = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def to_array(msg):
    if msg.encoding in ('rgb8', 'bgr8'):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return a if msg.encoding == 'rgb8' else a[..., ::-1]
    dt = {'mono8': np.uint8, '8UC1': np.uint8, 'mono16': np.uint16, '16UC1': np.uint16, '16SC1': np.int16,
          '32SC1': np.int32}[msg.encoding]
    a = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // np.dtype(dt).itemsize)[:, :msg.width]
    return a.byteswap() if msg.is_bigendian else a


class OvObjects(Node):
    def __init__(self):
        super().__init__('ov_objects')
        p = lambda n, v: self.declare_parameter(n, v).value
        self.ovb = p('ov_bench', '/root/data/ov_bench'); self.out_dir = p('out_dir', '')
        if self.out_dir == 'none': self.out_dir = ''
        self.period = p('period', 0.5); self.pps = p('pps', 16); self.max_emb = p('max_emb', 32)
        self.stuff_frac = p('stuff_frac', 0.7); self.contain = p('contain', 0.8)
        self.select_sharpest = p('select_sharpest', True)
        self.fusion = p('fusion', True); self.world_frame = p('world_frame', 'odom'); self.min_obs = p('min_obs', 2)
        self.publish_every = p('publish_every', 4)  # fused keyframes between object messages (2 s at 2 Hz)
        # until the odometry frame and the camera are connected in TF (odometry starts 15-25 s after the camera in
        # run_realtime.sh), keyframes have no pose and could not be fused: skip them before the GPU work
        self.wait_for_pose = p('wait_for_pose', True); self.pose_ready = False; self.n_waited = 0
        K = p('intrinsics', [910.663, 910.838, 637.358, 381.687])  # fx fy cx cy, D435i color 1280x720 (camera_calib.json)
        self.K = tuple(K)
        if self.out_dir: os.makedirs(self.out_dir, exist_ok=True)
        if self.fusion:  # before the ~15 s model load: the camera's /tf_static is published once, early (missed in ovn_f3)
            self.tf = tf2_ros.Buffer(cache_time=Duration(seconds=10)); self.tfl = tf2_ros.TransformListener(self.tf, self)
        self.load_models()
        self.frames = collections.deque(maxlen=30)          # (stamp_ns, recv_wall, color msg); callbacks stay light
        self.labels = collections.OrderedDict()             # stamp_ns -> label image
        self.lock = threading.Lock(); self.last_stamp = -1; self.stats = []; self.misses = 0; self.n_color = self.n_labels = 0
        # reliable like the other pipeline nodes: best effort lost ~40% of the 2.7 MB color frames (ovn_r6)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image, 'color', self.on_color, qos)
        self.create_subscription(Image, 'labels', self.on_labels, qos)
        if self.fusion:
            self.depths = collections.OrderedDict()         # stamp_ns -> aligned depth msg
            self.create_subscription(Image, 'depth', self.on_depth, qos)
            self.obj_pub = self.create_publisher(String, 'objects', QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                                 history=HistoryPolicy.KEEP_LAST, depth=1))
            self.fuser = Fusion(tuple(K), min_obs=self.min_obs); self.fq = queue.Queue(); self.fstats = []; self.fmiss = collections.Counter()
            self.fworker = threading.Thread(target=self.fusion_loop, daemon=True); self.fworker.start()
            self.closed_in = None; self.closed_acc = {}; self.clock = threading.Lock()
            self.create_subscription(String, 'closed_objects', self.on_closed, 1)
        self.running = True; self.worker = threading.Thread(target=self.loop, daemon=True); self.worker.start()
        self.get_logger().info(f'ov_objects ready (pps {self.pps}, max_emb {self.max_emb}, sharpest {self.select_sharpest})')

    def load_models(self):
        os.environ['EVSAM_NO_FRAMES'] = '1'; sys.path.insert(0, self.ovb)
        import torch; import evsam_bench as eb
        self.torch, self.eb = torch, eb
        eb.MAX_EMB = 10000                                     # no cap inside AMG: the cap follows the wall / floor drop
        self.amg = eb.make_any('l0t', self.pps)
        self.emb = eb.Embedder(os.path.join(self.ovb, 'ViT-B-16-SigLIP2.engine'), 'siglip')
        v = np.load(os.path.join(self.ovb, 'siglip2_vocab.npz'))
        self.vocab = [str(s) for s in v['labels']]; self.text_np = v['text']; self.text = torch.from_numpy(v['text']).cuda()
        self.vocab_stuff = torch.tensor([w in STUFF_WORDS for w in self.vocab], device='cuda')
        self.stuff_lut = torch.zeros(65536, dtype=torch.bool, device='cuda'); self.stuff_lut[STUFF_IDS] = True
        dummy = np.full((720, 1280, 3), 128, np.uint8); dummy[200:500, 300:900] = 30
        for _ in range(3): self.process(dummy, np.zeros((720, 1280), np.uint16))  # TensorRT / cuDNN warm-up
        torch.cuda.synchronize()

    def on_color(self, msg):
        with self.lock:
            self.frames.append((msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec, time.time(), msg)); self.n_color += 1

    def on_labels(self, msg):
        st = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        with self.lock:
            self.labels[st] = msg; self.n_labels += 1
            while len(self.labels) > 60: self.labels.popitem(last=False)

    def on_depth(self, msg):
        with self.lock:
            self.depths[msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec] = msg
            while len(self.depths) > 60: self.depths.popitem(last=False)

    def on_closed(self, msg):
        self.closed_in = json.loads(msg.data).get('objects', [])

    def observe_closed(self, stamp, rgb, frame_id):
        """Hydra's objects seen in this keyframe: SigLIP2 crop embeddings (and white pixels of ceiling lights)"""
        objs = self.closed_in
        if not objs: return
        with self.lock: dm = self.depths.get(stamp)
        try:
            tr = self.tf.lookup_transform(self.world_frame, frame_id, Time(nanoseconds=stamp), timeout=Duration(seconds=0.1)).transform
        except Exception:
            return
        q, tt = tr.rotation, tr.translation; R = quat_to_R((q.w, q.x, q.y, q.z)); t = np.array([tt.x, tt.y, tt.z])
        depth = to_array(dm).astype(np.float32) / 1000.0 if dm is not None else None
        H, W = rgb.shape[:2]; s = W / 1280.0; fx, fy, cx, cy = [v * s for v in self.K]; px = W / 848.0
        corners = lambda lo, hi: np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        boxes, seen = [], []
        for o in objs:
            lo, hi = np.asarray(o['lo'], float), np.asarray(o['hi'], float)
            P = (corners(lo, hi) - t) @ R; cz = float(((lo + hi) / 2 - t) @ R[:, 2])
            if (P[:, 2] < 0.2).any() or cz > 6: continue
            u = fx * P[:, 0] / P[:, 2] + cx; v = fy * P[:, 1] / P[:, 2] + cy
            x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max(); A = (x1 - x0) * (y1 - y0)
            if max(0, min(x1, W) - max(x0, 0)) * max(0, min(y1, H) - max(y0, 0)) / max(A, 1) < 0.7: continue
            if min(x1 - x0, y1 - y0) < 24 * px: continue
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
            if depth is not None:
                zz = depth[int(y0):int(y1), int(x0):int(x1)]; zz = zz[zz > 0.2]
                if len(zz) and np.median(zz) < cz - 0.3 - (hi - lo).max() / 2: continue  # something in front
            w, h = x1 - x0, y1 - y0
            boxes.append([max(0, x0 - .1 * w), max(0, y0 - .1 * h), min(W - 1, x1 + .1 * w), min(H - 1, y1 + .1 * h)])
            seen.append((o, A, [x0, y0, w, h]))
        if not boxes: return
        torch = self.torch; bt = torch.tensor(boxes, device='cuda', dtype=torch.float32)
        E = torch.cat([self.emb(rgb, bt[c:c + self.max_emb]).float() for c in range(0, len(bt), self.max_emb)])
        E = (E / E.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64)
        with self.clock:
            for (o, A, bx), e in zip(seen, E):
                a = self.closed_acc.setdefault(o['sym'], dict(sum=np.zeros(len(e)), n=0, first=stamp, last=stamp, best=(0.0, stamp, bx), light=None))
                a['sum'] += e; a['n'] += 1; a['first'] = min(a['first'], stamp); a['last'] = max(a['last'], stamp)
                if A > a['best'][0]: a['best'] = (A, stamp, bx)
        if depth is None: return
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); sat = rgb.max(2).astype(np.int16) - rgb.min(2)
        for o, _, _ in seen:
            if o.get('name') not in ('light', 'ceiling light'): continue
            lo, hi = np.asarray(o['lo'], float), np.asarray(o['hi'], float); elo, ehi = lo - [0.8, 0.8, 0.3], hi + [0.8, 0.8, 0.3]
            P = (corners(elo, ehi) - t) @ R; P[:, 2] = np.maximum(P[:, 2], 0.1)
            u = fx * P[:, 0] / P[:, 2] + cx; v = fy * P[:, 1] / P[:, 2] + cy
            x0, x1, y0, y1 = int(np.clip(u.min(), 0, W)), int(np.clip(u.max(), 0, W)), int(np.clip(v.min(), 0, H)), int(np.clip(v.max(), 0, H))
            if (x1 - x0) * (y1 - y0) < 2000 * px * px: continue
            m = np.zeros(g.shape, np.uint8); m[y0:y1, x0:x1] = (g[y0:y1, x0:x1] >= 235) & (sat[y0:y1, x0:x1] <= 40)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            keep = [i for i in range(1, n) if st[i, 4] >= 150 * px * px]
            if not keep: continue
            blob = np.isin(lab, keep); src = blob
            if ((depth[blob] > 0.3) & (depth[blob] < 8)).sum() < 20:  # no depth on the lamp: the ceiling around it
                src = cv2.dilate(blob.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~blob
            sv, su = np.nonzero(src); sz = depth[sv, su]; k = (sz > 0.3) & (sz < 8)
            if k.sum() < 20: continue
            Z = np.stack([(su[k] - cx) * sz[k] / fx, (sv[k] - cy) * sz[k] / fy, sz[k]], 1) @ R.T + t
            zw = float(np.median(Z[:, 2]))  # ceiling height at the light
            vv, uu = np.nonzero(blob)
            d = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(len(uu))], 1) @ R.T; sc = (zw - t[2]) / d[:, 2]
            ok = sc > 0; Q = t + sc[ok, None] * d[ok]
            Q = Q[np.all((Q[:, :2] >= elo[:2]) & (Q[:, :2] <= ehi[:2]), 1)]
            if len(Q) < 50: continue
            blo, bhi = np.percentile(Q, 2, 0), np.percentile(Q, 98, 0); blo[2], bhi[2] = zw - 0.05, zw + 0.05
            with self.clock:
                a = self.closed_acc[o['sym']]
                if a['light'] is None or len(Q) > a['light'][0]: a['light'] = (len(Q), stamp, blo.tolist(), bhi.tolist())

    def closed_results(self):
        out = {}
        with self.clock:
            for s, a in self.closed_acc.items():
                m = a['sum'] / max(np.linalg.norm(a['sum']), 1e-9); sc = self.text_np @ m; j = np.argsort(-sc)
                r = dict(emb=base64.b64encode(m.astype(np.float16).tobytes()).decode(), n_obs=a['n'], first=a['first'], last=a['last'],
                         best_stamp=a['best'][1], best_box=[float(x) for x in a['best'][2]], top3=[self.vocab[i] for i in j[:3]],
                         text_sim=float(sc[j[0]]), text_margin=float(sc[j[0]] - sc[j[1]]))
                if a['light'] is not None: r['light'] = [a['light'][2], a['light'][3]]; r['light_stamp'] = a['light'][1]
                out[s] = r
        return out

    def fusion_loop(self):
        while True:
            job = self.fq.get()
            if job is None: break
            stamp, Rt, depth, out, t_enq = job; t0 = time.perf_counter()
            masks = np.unpackbits(out['masks'], axis=-1)[:, :, :depth.shape[1]].astype(bool)
            self.fuser_last_stamp = stamp
            self.fuser.add_keyframe(stamp, Rt, depth, masks, out['emb'].astype(np.float32), out['top1'], out['boxes'], out['stuff_frac'])
            t1 = time.perf_counter()
            self.fstats.append(dict(fusion_ms=(t1 - t0) * 1e3, fusion_lag_ms=(time.time() - t_enq) * 1e3, objects=len(self.fuser.objs)))
            if self.fuser.n_kf % self.publish_every == 0:
                t2 = time.perf_counter(); self.publish_objects(); self.fstats[-1]['publish_ms'] = (time.perf_counter() - t2) * 1e3
            if self.out_dir and self.fuser.n_kf % 20 == 0:
                t2 = time.perf_counter(); self.write_objects(); self.fstats[-1]['write_ms'] = (time.perf_counter() - t2) * 1e3

    def publish_objects(self):
        res = self.fuser.results(self.text_np, self.vocab)
        for r in res:
            r.pop('dets', None); r.pop('votes', None)
            r['emb'] = base64.b64encode(self.fuser.objs[r['id']].unit().astype(np.float16).tobytes()).decode()
        self.obj_pub.publish(String(data=json.dumps(dict(stamp=self.fuser_last_stamp, vocab=self.vocab, objects=res,
                                                         closed=self.closed_results()))))

    def write_objects(self):
        if not self.out_dir: return
        res = self.fuser.results(self.text_np, self.vocab)
        tmp = os.path.join(self.out_dir, 'objects.json.tmp'); json.dump(res, open(tmp, 'w')); os.replace(tmp, os.path.join(self.out_dir, 'objects.json'))

    def to_fusion(self, stamp, frame_id, out):
        """aligned depth of the keyframe (exact stamp) + pose from TF -> fusion queue"""
        with self.lock: dm = self.depths.get(stamp)
        if dm is None:
            self.fmiss['no_depth'] += 1
            if self.fmiss['no_depth'] == 1:
                with self.lock: ks = list(self.depths)[-3:]
                self.get_logger().warn(f'no aligned depth with the keyframe stamp {stamp} (latest {ks})')
            return
        try:
            tr = self.tf.lookup_transform(self.world_frame, frame_id, Time(nanoseconds=stamp), timeout=Duration(seconds=0.2)).transform
        except Exception as e:
            self.fmiss['no_pose'] += 1
            if self.fmiss['no_pose'] == 1: self.get_logger().warn(f'no {self.world_frame} <- {frame_id} pose at the keyframe: {e!r}')
            return
        q, t = tr.rotation, tr.translation
        Rt = (quat_to_R((q.w, q.x, q.y, q.z)), np.array([t.x, t.y, t.z]))
        depth = to_array(dm)[::2, ::2].astype(np.float32) / 1000.0  # mm -> m, half resolution like the masks
        self.fq.put((stamp, Rt, depth, out, time.time()))

    def loop(self):
        nxt = time.time() + self.period
        while self.running:
            time.sleep(max(0.0, nxt - time.time())); nxt = max(nxt + self.period, time.time())
            with self.lock:
                cand = [f for f in self.frames if f[0] > self.last_stamp and f[0] in self.labels]
                if not cand: self.misses += 1; continue
                lab_msgs = {c[0]: self.labels[c[0]] for c in cand}; self.last_stamp = cand[-1][0]  # window used up
                frame_ids = {c[0]: c[2].header.frame_id for c in cand}
            t0 = time.perf_counter()
            cand = [(st, rt, rgb, sharpness(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
                    for st, rt, rgb in ((c[0], c[1], np.ascontiguousarray(to_array(c[2]))) for c in cand)]
            self.t_select = time.perf_counter() - t0
            f = max(cand, key=lambda c: c[3]) if self.select_sharpest else cand[-1]
            lab = to_array(lab_msgs[f[0]])
            try:
                self.keyframe(f, lab, cand, frame_ids[f[0]])
            except Exception as e:  # keep the node alive, report
                self.get_logger().error(f'keyframe failed: {e!r}')

    def process(self, rgb, lab):
        torch, eb = self.torch, self.eb; T = {}; sync = torch.cuda.synchronize
        t0 = time.perf_counter()
        cands = eb.amg_generate(self.amg, np.ascontiguousarray(rgb)); sync(); t1 = time.perf_counter(); T['amg'] = t1 - t0
        n = len(cands)
        if n == 0: return dict(n_cands=0, kept=0, t=T), None
        H, W = rgb.shape[:2]
        segs = torch.stack([m['segmentation'] for m in cands])                       # N,H,W bool (GPU)
        st = self.stuff_lut[torch.from_numpy(lab.astype(np.int32)).cuda()]
        if st.shape != (H, W): st = torch.nn.functional.interpolate(st[None, None].float(), (H, W))[0, 0] > 0.5
        sfrac = (segs & st).flatten(1).sum(1).float() / segs.flatten(1).sum(1).clamp(min=1)
        sync(); t2 = time.perf_counter(); T['stuff'] = t2 - t1
        bb = torch.tensor([m['bbox'] for m in cands], device='cuda')                 # XYWH
        boxes = torch.stack([(bb[:, 0] - .1 * bb[:, 2]).clamp(min=0), (bb[:, 1] - .1 * bb[:, 3]).clamp(min=0),
                             (bb[:, 0] + 1.1 * bb[:, 2]).clamp(max=W - 1), (bb[:, 1] + 1.1 * bb[:, 3]).clamp(max=H - 1)], 1)
        E = torch.cat([self.emb(rgb, boxes[c:c + self.max_emb]).float() for c in range(0, n, self.max_emb)])
        E = E / E.norm(dim=-1, keepdim=True); sync(); t3 = time.perf_counter(); T['embed'] = t3 - t2
        top1 = (E @ self.text.T).argmax(1)
        drop = (sfrac >= self.stuff_frac) & self.vocab_stuff[top1]
        score = torch.tensor([m['predicted_iou'] for m in cands], device='cuda')
        idx = torch.nonzero(~drop).flatten(); idx = idx[torch.argsort(-score[idx])][:self.max_emb]
        k = segs[idx][:, ::4, ::4].flatten(1).float()                                  # containment at 1/4 resolution
        inter = k @ k.T; area = inter.diagonal()
        cont = (inter / area[:, None].clamp(min=1) >= self.contain) & (area[None, :] > area[:, None])
        if len(idx):
            par = torch.where(cont, area[None, :].expand_as(cont), torch.full_like(inter, float('inf'))).argmin(1)
            parent = torch.where(cont.any(1), par, torch.full_like(par, -1))
        else: parent = torch.zeros(0, dtype=torch.long, device='cuda')
        sync(); t4 = time.perf_counter(); T['filter'] = t4 - t3; T['total'] = t4 - t0
        out = dict(boxes=bb[idx].cpu().numpy(), score=score[idx].cpu().numpy(), stuff_frac=sfrac[idx].cpu().numpy(),
                   emb=E[idx].half().cpu().numpy(), top1=np.array([self.vocab[i] for i in top1[idx].tolist()]),
                   parent=parent.cpu().numpy(), masks=np.packbits(segs[idx][:, ::2, ::2].cpu().numpy(), axis=-1))
        return dict(n_cands=n, dropped_stuff=int(drop.sum()), kept=len(idx), n_parent=int((parent >= 0).sum()), t=T), out

    def keyframe(self, f, lab, cand, frame_id):
        stamp, recv, rgb, s = f; t_start = time.time()
        if self.fusion and self.wait_for_pose and not self.pose_ready:
            if not self.tf.can_transform(self.world_frame, frame_id, Time()):  # latest: is the tree connected yet?
                self.n_waited += 1
                if self.n_waited == 1: self.get_logger().info(f'waiting for {self.world_frame} <- {frame_id} in TF: keyframes skipped')
                return
            self.pose_ready = True
            self.get_logger().info(f'{self.world_frame} <- {frame_id} in TF after {self.n_waited} skipped keyframes')
        info, out = self.process(rgb, lab)
        t_done = time.time()
        if out is not None and self.out_dir:
            np.savez(os.path.join(self.out_dir, f'kf_{stamp}.npz'), **out)
        if out is not None and self.fusion: self.to_fusion(stamp, frame_id, out)
        if self.fusion and self.closed_in:
            try:
                self.observe_closed(stamp, rgb, frame_id)
            except Exception as e:
                self.get_logger().error(f'closed objects failed: {e!r}')
        r = dict(stamp=stamp, select_ms=self.t_select * 1e3, wait_ms=(t_start - recv) * 1e3, age_ms=(t_done - recv) * 1e3, sharp=s,
                 sharp_newest=cand[-1][3], offset=len(cand) - 1 - cand.index(f), n_window=len(cand),
                 n_cands=info['n_cands'], dropped_stuff=info.get('dropped_stuff', 0), kept=info['kept'],
                 n_parent=info.get('n_parent', 0), **{f'{k}_ms': v * 1e3 for k, v in info['t'].items()})
        self.stats.append(r)
        if self.out_dir:
            with open(os.path.join(self.out_dir, 'timing.jsonl'), 'a') as fh: fh.write(json.dumps(r) + '\n')

    def summary(self):
        S = self.stats
        if not S: return dict(n=0, misses=self.misses, waited_for_pose=self.n_waited, color_received=self.n_color, labels_received=self.n_labels)
        q = lambda k, p: float(np.percentile([r[k] for r in S], p))
        res = dict(n=len(S), misses=self.misses, waited_for_pose=self.n_waited, color_received=self.n_color, labels_received=self.n_labels,
                   window_mean=float(np.mean([r['n_window'] for r in S])), over_period=sum(r['total_ms'] > self.period * 1e3 for r in S),
                   cands_mean=float(np.mean([r['n_cands'] for r in S])), cands_max=max(r['n_cands'] for r in S),
                   kept_mean=float(np.mean([r['kept'] for r in S])), dropped_stuff_mean=float(np.mean([r['dropped_stuff'] for r in S])),
                   sharp_gain_median=float(np.median([r['sharp'] / max(r['sharp_newest'], 1e-6) for r in S])))
        for k in ('select_ms', 'amg_ms', 'stuff_ms', 'embed_ms', 'filter_ms', 'total_ms', 'wait_ms', 'age_ms'):
            if k in S[0]: res[k] = dict(p50=q(k, 50), p95=q(k, 95), max=max(r[k] for r in S))
        if self.fusion: res.update(fusion_missing=dict(self.fmiss), depths_received=len(self.depths))
        if self.fusion and self.fstats:
            F = self.fstats; res_ = self.fuser.results(self.text_np, self.vocab)
            for k in ('fusion_ms', 'fusion_lag_ms', 'write_ms', 'publish_ms'):
                v = [r[k] for r in F if k in r]
                if v: res[k] = dict(p50=float(np.median(v)), p95=float(np.percentile(v, 95)), max=max(v))
            res.update(fused_keyframes=len(F), fusion_missing=dict(self.fmiss), fusion_stats=self.fuser.stats,
                       objects_all=len(self.fuser.objs), objects=len(res_), objects_wall=sum(r['wall'] for r in res_),
                       objects_confirmed=sum(r['confirmed'] for r in res_))
        return res


def main():
    rclpy.init(); node = OvObjects()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.running = False; node.worker.join(timeout=5)
        if node.fusion:
            node.fq.put(None); node.fworker.join(timeout=30); node.fuser.merge(); node.write_objects()
        s = node.summary(); print(json.dumps(s), flush=True)
        if node.out_dir: json.dump(s, open(os.path.join(node.out_dir, 'summary.json'), 'w'), indent=1)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
