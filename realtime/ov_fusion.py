"""3D fusion of the open-vocabulary 2D candidates (ov_objects.py) into objects (numpy only).

Per keyframe detection (mask, SigLIP2 embedding, top-1 word):
  points   mask (eroded) x aligned depth -> camera points -> world (keyframe pose); the depth run around the
           median depth is kept (sorted depths split at gaps > `gap` m), so background seen through the mask edge drops
  match    candidate objects whose box overlaps; on `assoc_voxel` cells: share of the detection inside the object
           >= `min_overlap` AND cosine >= `min_sim` AND (share of the object covered by the detection >= `min_cover`
           OR cosine >= `contain_sim`): a small detection joins a big accumulated object only when it looks the same
           (part of the same thing), not merely because it lies inside it; best by overlap + cover + cosine.
           SigLIP2 crop embeddings are close for any two crops of the same room (office_rt11 consecutive keyframes:
           median cosine 0.79 for pairs that share < 10% of their cells), so the geometry decides, the cosine vetoes
  limit    no match or merge whose union box exceeds `max_extent` m on any axis
  update   voxel union (`voxel` m cells), embedding mean, word votes, best view (largest mask)
  merge    every `merge_every` keyframes: objects with >= `merge_overlap` of the smaller inside the bigger and cosine
           >= `contain_sim`, or also >= `min_cover` of the bigger and cosine >= `merge_sim`, are merged (two halves of an object seen from different keyframes)
Objects observed in fewer than `min_obs` keyframes (detections of one keyframe count once) are left out of results() (one-off junk masks).
results() flags, without deleting:
  wall       mean closed-set structural share of its detections >= `wall_stuff` AND the SigLIP2 mean embedding is
             within `wall_gap` of its best wall / floor / ceiling / pillar word (office_rt11 ovn_r7, 242 objects checked
             by hand: junk 96 -> 54, real objects with the right label 71 -> 69; the share alone: junk 41 but 71 -> 61,
             it also takes name plates, windows, fire extinguishers on walls)
  confirmed  not wall and seen in >= `confirm_obs` keyframes (real share 0.60, junk 26% of the confirmed objects)
then (join=True) join_long(): long furniture split into several objects becomes one (no `max_extent` limit).
"""
import base64
import numpy as np, cv2

STUFF_WORDS = ('wall', 'floor', 'ceiling', 'pillar')


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def keys(p, size):
    """int64 key per occupied cell of size `size` (21 bits per axis, +-10 km at 1 cm)."""
    c = np.floor(p / size).astype(np.int64) + (1 << 20)
    return np.unique((c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2])


# long furniture: kinds whose pieces touch only when they are one thing (a long desk / counter, a bench / sofa row).
# Chairs are left out on purpose: chairs side by side touch too and are separate objects.
LONG_KINDS = {'desk': 'table', 'table': 'table', 'sofa': 'seat', 'bench': 'seat'}
_NB = np.array([(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)


def _cells_xyz(k):
    k = np.asarray(k, np.int64); m = (1 << 21) - 1
    return np.stack([(k >> 42) & m, (k >> 21) & m, k & m], 1) - (1 << 20)


def join_long(res, kinds=LONG_KINDS, min_contact=8, min_sim=0.8, min_z=0.5):
    """Join the confirmed objects of one long-furniture kind (`kinds`, by label) that touch: >= `min_contact` cells
    of one within one cell (26-neighbourhood) of the other, cosine >= `min_sim`, height ranges overlapping >= `min_z`
    of the lower one (a long desk seen piece by piece: library_merged 8 'desk' objects around one counter, the bench
    row as 2 + 2 'sofa'; the fusion's merge needs half of the smaller inside the bigger and a union <= max_extent).
    Returns a new list: each group's biggest object takes the union (cells, size, centre, detections, votes,
    `joined` = the ids it took); the others stay listed with confirmed=False and `joined_into` = its id."""
    out = [dict(r) for r in res]
    sel = [r for r in out if r.get('confirmed') and r.get('cells') and r.get('label') in kinds]
    if len(sel) < 2:
        return out
    C = [np.asarray(r['cells'], np.int64) for r in sel]
    D = [np.unique((c[:, None] + _NB[None]).ravel()) for c in C]
    X = [_cells_xyz(c) for c in C]
    E = [np.frombuffer(base64.b64decode(r['emb']), np.float16).astype(np.float32) if r.get('emb') else None for r in sel]
    par = list(range(len(sel)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if kinds[sel[i]['label']] != kinds[sel[j]['label']]:
                continue
            if np.any(X[i].min(0) - X[j].max(0) > 1) or np.any(X[j].min(0) - X[i].max(0) > 1):
                continue
            if np.isin(C[j], D[i]).sum() < min_contact:
                continue
            zi, zj = (X[i][:, 2].min(), X[i][:, 2].max()), (X[j][:, 2].min(), X[j][:, 2].max())
            if min(zi[1], zj[1]) - max(zi[0], zj[0]) + 1 < min_z * (min(zi[1] - zi[0], zj[1] - zj[0]) + 1):
                continue
            if E[i] is not None and E[j] is not None:
                if float(E[i] @ E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]) + 1e-9) < min_sim:
                    continue
            par[find(j)] = find(i)
    groups = {}
    for i in range(len(sel)):
        groups.setdefault(find(i), []).append(i)
    for g in groups.values():
        if len(g) >= 2:
            union_objects([sel[i] for i in g])
    return out


def union_objects(pieces):
    """The biggest of `pieces` (result dicts) takes them all: cells, size, centre, detections, votes,
    `joined` = the others' ids; the others get confirmed=False and `joined_into`. Returns the keeper."""
    pieces = sorted(pieces, key=lambda r: -len(r['cells'])); k = pieces[0]
    cells = np.unique(np.concatenate([np.asarray(r['cells'], np.int64) for r in pieces]))
    c = (_cells_xyz(cells) + 0.5) * k['cell']; lo, hi = c.min(0) - k['cell'] / 2, c.max(0) + k['cell'] / 2
    dets = sorted({tuple(d) for r in pieces for d in r['dets']})
    votes = {}
    for r in pieces:
        for w, n in r.get('votes', {}).items():
            votes[w] = votes.get(w, 0) + n
    k.update(cells=cells.tolist(), center=((lo + hi) / 2).tolist(), size=(hi - lo).tolist(),
             dets=[list(d) for d in dets], n_obs=len({d[0] for d in dets}), n_dets=len(dets), votes=votes,
             n_voxels=sum(r['n_voxels'] for r in pieces), first=min(r['first'] for r in pieces),
             last=max(r['last'] for r in pieces), joined=k.get('joined', []) + [r['id'] for r in pieces[1:]])
    k.pop('core', None)  # a small piece's core is not the joined object's
    for key in ('box_manual', 'label_manual'):  # set by hand on any piece: the joined object keeps it
        if key not in k:
            v = next((r[key] for r in pieces[1:] if r.get(key)), None)
            if v is not None:
                k[key] = v
                if key == 'label_manual':
                    k['label'] = v
    for r in pieces[1:]:
        r.update(confirmed=False, joined_into=k['id'])
    return k


def join_manual(res, groups):
    """Join the objects of each group of fusion ids (set by hand after a check: one thing seen as several objects,
    e.g. a door from both sides). Works on `res` in place; ids not confirmed (removed) are left out."""
    by = {r['id']: r for r in res}
    for g in groups:
        pieces = [by[i] for i in g if i in by and by[i].get('confirmed')]
        if len(pieces) >= 2:
            union_objects(pieces)
    return res


class Obj:
    __slots__ = ('id', 'vox', 'akeys', 'lo', 'hi', 'emb', 'n', 'votes', 'best', 'first', 'last', 'dets', 'stuff')

    def __init__(self, oid, d, stamp):
        self.id = oid; self.vox = d['vox']; self.akeys = d['akeys']; self.lo, self.hi = d['lo'], d['hi']
        self.emb = d['emb'].astype(np.float64).copy(); self.n = 1; self.votes = {d['word']: 1}
        self.best = (d['area'], stamp, d['box']); self.first = self.last = stamp; self.dets = [(stamp, d['k'])]; self.stuff = d['stuff']

    def add(self, d, stamp, vsize, asize):
        self.vox = np.union1d(self.vox, d['vox']); self.akeys = np.union1d(self.akeys, d['akeys'])
        self.lo, self.hi = np.minimum(self.lo, d['lo']), np.maximum(self.hi, d['hi'])
        self.emb += d['emb']; self.n += 1; self.votes[d['word']] = self.votes.get(d['word'], 0) + 1
        if d['area'] > self.best[0]: self.best = (d['area'], stamp, d['box'])
        self.last = max(self.last, stamp); self.first = min(self.first, stamp); self.dets.append((stamp, d['k'])); self.stuff += d['stuff']

    def absorb(self, o):
        self.vox = np.union1d(self.vox, o.vox); self.akeys = np.union1d(self.akeys, o.akeys)
        self.lo, self.hi = np.minimum(self.lo, o.lo), np.maximum(self.hi, o.hi); self.emb += o.emb; self.n += o.n
        for w, c in o.votes.items(): self.votes[w] = self.votes.get(w, 0) + c
        if o.best[0] > self.best[0]: self.best = o.best
        self.first = min(self.first, o.first); self.last = max(self.last, o.last); self.dets += o.dets; self.stuff += o.stuff

    def unit(self):
        return self.emb / max(np.linalg.norm(self.emb), 1e-9)


class Fusion:
    def __init__(self, K, voxel=0.05, assoc_voxel=0.10, gap=0.2, depth_range=(0.2, 4.0), min_points=60,
                 min_overlap=0.4, min_cover=0.15, min_sim=0.85, contain_sim=0.88, max_extent=3.0, merge_overlap=0.5, merge_sim=0.9, merge_every=10, min_obs=2,
                 wall_stuff=0.7, wall_gap=-0.02, confirm_obs=3, join=True):
        self.K = K  # fx fy cx cy of the full-resolution color image
        self.voxel, self.asize, self.gap, self.dr, self.min_points = voxel, assoc_voxel, gap, depth_range, min_points
        self.min_overlap, self.min_cover, self.min_sim, self.merge_overlap, self.merge_sim = min_overlap, min_cover, min_sim, merge_overlap, merge_sim
        self.contain_sim, self.max_extent = contain_sim, max_extent
        self.merge_every, self.min_obs = merge_every, min_obs
        self.wall_stuff, self.wall_gap, self.confirm_obs, self.join = wall_stuff, wall_gap, confirm_obs, join
        self.objs = {}; self.dirty = set(); self.next_id = 0; self.n_kf = 0; self.stats = dict(dets=0, too_few_points=0, matched=0, new=0, merged=0)

    def detection_points(self, mask, depth_m, Rt):
        """mask: HxW bool at depth resolution (depth_m float32 metres, same size). Returns world points (N,3)."""
        H, W = depth_m.shape; s = W / 1280.0; fx, fy, cx, cy = [v * s for v in self.K]
        m = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        v, u = np.nonzero(m)
        z = depth_m[v, u]; ok = (z > self.dr[0]) & (z < self.dr[1]); u, v, z = u[ok], v[ok], z[ok]
        if len(z) < self.min_points: return None
        o = np.argsort(z); zs = z[o]; cut = np.nonzero(np.diff(zs) > self.gap)[0]
        mid = len(zs) // 2; a = cut[cut < mid]; b = cut[cut >= mid]
        lo = a[-1] + 1 if len(a) else 0; hi = b[0] + 1 if len(b) else len(zs)
        sel = o[lo:hi]
        if len(sel) < self.min_points: return None
        u, v, z = u[sel], v[sel], z[sel]
        pc = np.stack([(u + 0.5 - cx) * z / fx, (v + 0.5 - cy) * z / fy, z], 1)
        return pc @ Rt[0].T + Rt[1]

    def add_keyframe(self, stamp, Rt, depth_m, masks, embs, words, boxes, stuff=None):
        """masks: (K,H,W) bool at depth_m resolution; embs (K,D) unit; words (K,); boxes (K,4) XYWH full-res;
        stuff (K,): share of each mask with a structural closed-set label (object feature, optional)."""
        self.n_kf += 1
        dets = []
        for k in range(len(masks)):
            self.stats['dets'] += 1
            p = self.detection_points(masks[k], depth_m, Rt)
            if p is None: self.stats['too_few_points'] += 1; continue
            dets.append(dict(k=k, vox=keys(p, self.voxel), akeys=keys(p, self.asize), lo=p.min(0), hi=p.max(0),
                             emb=embs[k].astype(np.float64), word=str(words[k]), area=int(masks[k].sum()), box=boxes[k],
                             stuff=0.0 if stuff is None else float(stuff[k])))
        ids = list(self.objs); LO = np.array([self.objs[i].lo for i in ids]).reshape(-1, 3)
        HI = np.array([self.objs[i].hi for i in ids]).reshape(-1, 3); free = np.ones(len(ids), bool)
        for d in sorted(dets, key=lambda d: -len(d['akeys'])):  # big detections first
            best, bo = -1, None; e = d['emb'] / max(np.linalg.norm(d['emb']), 1e-9)
            near = free & np.all(LO <= d['hi'] + self.asize, 1) & np.all(HI >= d['lo'] - self.asize, 1)
            for j in np.nonzero(near)[0]:
                o = self.objs[ids[j]]
                if self.too_big(o.lo, o.hi, d['lo'], d['hi']): continue
                i = np.isin(d['akeys'], o.akeys, assume_unique=True).sum(); ov, cov = i / len(d['akeys']), i / len(o.akeys)
                if ov < self.min_overlap: continue
                sim = float(e @ o.unit())
                if sim < self.min_sim or (cov < self.min_cover and sim < self.contain_sim): continue
                if ov + cov + sim > best: best, bo, bj = ov + cov + sim, o, j
            if bo is None:
                o = Obj(self.next_id, d, stamp); self.objs[o.id] = o; self.next_id += 1; self.stats['new'] += 1
            else:
                bo.add(d, stamp, self.voxel, self.asize); self.stats['matched'] += 1; o = bo
                free[bj] = False; LO[bj], HI[bj] = o.lo, o.hi  # one detection per object per keyframe
            self.dirty.add(o.id)
        if self.n_kf % self.merge_every == 0: self.merge()

    def too_big(self, lo1, hi1, lo2, hi2):
        """union box beyond max_extent: similar-looking neighbours (doors, panels, cabinets along a corridor wall)
        would otherwise chain into one 10 m object (office_rt11: 11.8 m 'door')"""
        return bool(np.any(np.maximum(hi1, hi2) - np.minimum(lo1, lo2) > self.max_extent))

    def merge(self):
        """Merge pairs that involve an object changed since the last merge (largest objects absorb smaller ones)."""
        ids = list(self.objs); idx = {k: n for n, k in enumerate(ids)}
        LO = np.array([self.objs[k].lo for k in ids]).reshape(-1, 3); HI = np.array([self.objs[k].hi for k in ids]).reshape(-1, 3)
        gone = set()
        for a in sorted(self.dirty & set(ids), key=lambda k: -len(self.objs[k].akeys)):
            if a in gone: continue
            A = self.objs[a]
            near = np.all(LO <= A.hi + self.asize, 1) & np.all(HI >= A.lo - self.asize, 1)
            for j in np.nonzero(near)[0]:
                b = ids[j]
                if b == a or b in gone or a in gone: continue
                B = self.objs[b]; big, small = (A, B) if len(A.akeys) >= len(B.akeys) else (B, A)
                n = np.isin(small.akeys, big.akeys, assume_unique=True).sum()
                if n < self.merge_overlap * len(small.akeys) or self.too_big(big.lo, big.hi, small.lo, small.hi): continue
                sim = float(big.unit() @ small.unit())
                if sim >= self.contain_sim or (sim >= self.merge_sim and n >= self.min_cover * len(big.akeys)):
                    big.absorb(small); gone.add(small.id); self.stats['merged'] += 1
                    LO[idx[big.id]], HI[idx[big.id]] = big.lo, big.hi
                    if small is A: break
        for g in gone: del self.objs[g]
        self.dirty = set()

    def features(self, o, text=None, vocab=None):
        """per-object evidence for the junk filter"""
        c = (np.stack([(o.vox >> 42) & 0x1FFFFF, (o.vox >> 21) & 0x1FFFFF, o.vox & 0x1FFFFF], 1) - (1 << 20) + 0.5) * self.voxel
        lo, hi = np.percentile(c, 2, axis=0), np.percentile(c, 98, axis=0) + self.voxel
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(c.T)) if len(c) > 3 else np.zeros(3), 0))[::-1]  # spread, big -> small
        f = dict(n_obs=len({st for st, _ in o.dets}), n_dets=o.n, n_voxels=len(o.vox), center=((lo + hi) / 2).tolist(),
                 size=(hi - lo).tolist(), spread=ev.tolist(), flat=float(ev[2] / max(ev[0], 1e-6)),
                 emb_consistency=float(np.linalg.norm(o.emb) / o.n), stuff=o.stuff / o.n,
                 vote=max(o.votes, key=o.votes.get), vote_share=max(o.votes.values()) / o.n, votes=o.votes)
        if text is not None:
            s = text @ o.unit(); j = np.argsort(-s)
            f.update(label=vocab[j[0]], top3=[vocab[i] for i in j[:3]], text_sim=float(s[j[0]]), text_margin=float(s[j[0]] - s[j[1]]),
                     stuff_word=vocab[j[0]] in ('wall', 'floor', 'ceiling', 'pillar'),
                     stuff_gap=float(max(s[i] for i, w in enumerate(vocab) if w in STUFF_WORDS)
                                     - max(s[i] for i, w in enumerate(vocab) if w not in STUFF_WORDS)))
        return f

    def results(self, text=None, vocab=None):
        out = []
        for o in self.objs.values():
            kf = len({st for st, _ in o.dets})
            if kf < self.min_obs: continue
            r = dict(id=o.id, best_stamp=o.best[1], best_box=[float(x) for x in o.best[2]], first=o.first, last=o.last,
                     dets=[[int(a), int(b)] for a, b in o.dets], **self.features(o, text, vocab))
            r['wall'] = bool(r['stuff'] >= self.wall_stuff and r.get('stuff_gap', 1.0) > self.wall_gap)
            r['confirmed'] = bool(not r['wall'] and r['n_obs'] >= self.confirm_obs)
            r['cells'] = o.akeys.tolist(); r['cell'] = self.asize  # occupied assoc_voxel cells (keys()), for ov_dsg.py
            if self.join and text is not None:
                r['emb'] = base64.b64encode(o.unit().astype(np.float16).tobytes()).decode()
            out.append(r)
        return join_long(out) if self.join and text is not None else out
