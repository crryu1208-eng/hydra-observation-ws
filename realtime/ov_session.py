"""Open-vocabulary objects for a recorded session, offline: the realtime node's models, filters, fusion and scene
graph insertion (ov_objects.py, ov_fusion.py, ov_dsg.py) run on the session's recorded frames with their final
(corrected, levelled or aligned) poses, so the objects land in the session's finished scene graph.

  extract <observations dir> <labels dir> <out dir> [period]
      keyframes like the node: per `period` s window (0.5) the sharpest recorded frame that has a closed-set label
      image (<labels dir>/<stamp_ns>.png, realtime/seg_labels.sh) -> EfficientViT-SAM + SigLIP2 + wall / floor drop
      -> <out dir>/kf_<stamp>.npz (same content as the node's out_dir) and timing.jsonl
  fuse <frames.jsonl> <kf dir>[,<kf dir>...] <map hydra/backend dir or its dsg_with_mesh.json> [out dir]
      3D fusion of the keyframes with the poses in frames.jsonl (the `corrected` pose when the frame has one, depth
      relative to its folder, or to each frame's source_observations for a merged session); the confirmed objects
      go into the graph in the standard layout (ov_dsg.py: Hydra's objects layer, one parent place per object; no
      loop-closure correction: the poses are already the graph's). By default the result replaces the graph in the
      map's hydra/backend folder:
        before_ov_objects/dsg.json, dsg_with_mesh.json   the graph as it was (kept on the first run; every later
                                                          run starts again from it)
        dsg_with_mesh.json, dsg.json                     the graph with the objects
        ov_objects.json                                  all fused objects (flags, cells, embeddings)
        ov_summary.json                                  counts
      With [out dir] the same files go there instead and hydra/backend is left alone.
  remove <hydra/backend dir> <fusion ids: file (JSON list or one per line) or comma list>
      remove objects after a manual check (by fusion id, metadata ov.id; ov_objects.json `removed`, kept over later
      removals) and build the graph again from before_ov_objects: Hydra objects that were replaced only by removed
      ones come back; rewrites dsg_with_mesh.json, dsg.json, ov_objects.json and ov_summary.json in that folder
  core <hydra/backend dir> <kf dir>[,<kf dir>...]
      add the small objects' core voxels (core_cells, also done by fuse) to a hydra/backend fused without them and
      build the graph again
  label <hydra/backend dir> <fusion id>=<word>[,...] | file
      labels set by hand (`label_manual`), kept over later builds; the graph is built again
  merge <hydra/backend dir> <id>+<id>[+...][,...]
      join objects by hand (`merge_manual`; one door seen as several objects), kept over later builds
  box <hydra/backend dir> "<id>=x0,y0,z0,x1,y1,z1[;...]"
      boxes set by hand (`box_manual`, graph frame), kept over later builds
Labels come from the stored embeddings and the current siglip2_vocab.npz at every build (relabel).
Small objects (< 1 m) get their box from the voxels most of their masks agree on (core_cells), the others from
their connected mesh (ov_dsg.mesh_region).
Long furniture split into several objects (desk / table, sofa pieces that touch) is joined into one object
(ov_fusion.join_long) after the removal; ov_objects.json keeps the pieces with `joined` / `joined_into`.

Run in the container with the ov_bench virtualenv on PYTHONPATH (like rt.launch.yaml ov_objects:=true).
"""
import base64
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ov_fusion import quat_to_R  # noqa: E402

OV_BENCH = os.environ.get('OV_BENCH', '/root/data/ov_bench')


def load_frames(path):
    base = os.path.dirname(os.path.abspath(path)); fr = {}
    for line in open(path):
        d = json.loads(line)
        if d.get('status') == 'written':
            c = d.get('corrected') or {}  # loop-closure corrected pose (map_observations), in the graph's frame
            if c.get('status') == 'ok':
                d['world_T_sensor'] = c['world_T_sensor']
            d['_dir'] = d.get('source_observations') or base; fr[int(d['timestamp_ns'])] = d
    return fr


def extract(obs, labels, out, period=0.5):
    from ov_objects import OvObjects, sharpness
    os.makedirs(out, exist_ok=True)
    node = object.__new__(OvObjects)  # the node's model code without ROS
    node.ovb, node.pps, node.max_emb, node.stuff_frac, node.contain = OV_BENCH, 16, 32, 0.7, 0.8
    OvObjects.load_models(node)
    fr = load_frames(f'{obs}/frames.jsonl'); st = sorted(fr)
    have = {int(f[:-4]) for f in os.listdir(labels) if f.endswith('.png')}
    windows = {}
    for s in st:
        if s in have: windows.setdefault((s - st[0]) // int(period * 1e9), []).append(s)
    log = open(f'{out}/timing.jsonl', 'w'); n_kept = []
    for w in sorted(windows):
        cand = [(s, cv2.imread(f"{obs}/{fr[s]['rgb_path']}")) for s in windows[w]]
        sh = [sharpness(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for _, im in cand]
        s, im = cand[int(np.argmax(sh))]
        lab = cv2.imread(f'{labels}/{s}.png', cv2.IMREAD_UNCHANGED)
        info, o = node.process(np.ascontiguousarray(im[..., ::-1]), lab)
        if o is not None:
            np.savez(f'{out}/kf_{s}.npz', **o); n_kept.append(info['kept'])
        log.write(json.dumps(dict(stamp=s, n_window=len(cand), sharp=max(sh), n_cands=info['n_cands'],
                                  dropped_stuff=info.get('dropped_stuff', 0), kept=info['kept'],
                                  **{f'{k}_ms': v * 1e3 for k, v in info['t'].items()})) + '\n')
    print(json.dumps(dict(frames=len(st), labelled=len(have & set(st)), keyframes=len(windows),
                          kept_mean=float(np.mean(n_kept)) if n_kept else 0)))


BEFORE = 'before_ov_objects'


def backend_dir(path):
    return os.path.dirname(os.path.abspath(path)) if path.endswith('.json') else os.path.abspath(path)


def save_graph(G, d):
    G.save(f'{d}/dsg_with_mesh.json', include_mesh=True); G.save(f'{d}/dsg.json', include_mesh=False)


def relabel(O, text, vocab):
    """Words from the stored mean embeddings and the current vocabulary (siglip2_vocab.npz, ov_vocab.py): a new
    word needs no new fusion. `label_manual` (set by the `label` command after a check) wins over the model's word
    (`label_source` = manual / siglip2)."""
    from ov_fusion import STUFF_WORDS
    stuff = np.array([w in STUFF_WORDS for w in vocab])
    for o in O:
        if not o.get('emb'):
            continue
        e = np.frombuffer(base64.b64decode(o['emb']), np.float16).astype(np.float32); e /= np.linalg.norm(e) + 1e-9
        sc = text @ e; j = np.argsort(-sc)
        o.update(label=vocab[j[0]], top3=[vocab[i] for i in j[:3]], text_sim=float(sc[j[0]]),
                 text_margin=float(sc[j[0]] - sc[j[1]]), stuff_word=bool(stuff[j[0]]),
                 stuff_gap=float(sc[stuff].max() - sc[~stuff].max()), label_source='siglip2')
        if o.get('label_manual'):
            o['label'] = o['label_manual']; o['label_source'] = 'manual'


def build(backend, O, src, out, extra=None):
    """The graph from `src` (the graph without open-vocabulary objects) with the objects of O that are confirmed and
    not removed after the manual check, long furniture joined (ov_fusion.join_long). O (the fusion's objects, kept
    unjoined so a later removal can join again) gets `in_graph`, `joined`, `joined_into`; writes the graph,
    ov_objects.json and ov_summary.json into `out`."""
    import spark_dsg as dsg
    import ov_dsg
    from ov_fusion import join_long, join_manual
    v = np.load(f'{OV_BENCH}/siglip2_vocab.npz'); vocab = [str(w) for w in v['labels']]
    for o in O:
        for k in ('in_graph', 'joined', 'joined_into'):
            o.pop(k, None)
    relabel(O, v['text'], vocab)
    cand = [dict(o, confirmed=bool(o['confirmed'] and not o.get('removed'))) for o in O]
    join_manual(cand, [[o['id']] + o['merge_manual'] for o in O if o.get('merge_manual')])
    J = join_long(cand); byid = {o['id']: o for o in O}
    for j in J:
        o = byid[j['id']]; o['in_graph'] = bool(j['confirmed'])
        for k in ('joined', 'joined_into'):
            if k in j:
                o[k] = j[k]
    G = dsg.DynamicSceneGraph.load(src)
    n_before = G.get_layer(ov_dsg.LAYER).num_nodes() if G.has_layer(ov_dsg.LAYER) else 0
    # Hydra objects relabelled by hand after a check: closed_labels.json, {symbol: label}
    relab = json.load(open(f'{out}/closed_labels.json')) if os.path.exists(f'{out}/closed_labels.json') else {}
    relab_orig = {}
    ls0 = G.get_labelspace(ov_dsg.LAYER, 0); ls0 = dict(ls0.labels_to_names) if ls0 is not None else {}
    if G.has_layer(ov_dsg.LAYER):
        for n in G.get_layer(ov_dsg.LAYER).nodes:
            relab_orig[n.id.str()] = n.attributes.name or ls0.get(n.attributes.semantic_label, '')
    for s, w in relab.items():
        nid = dsg.NodeSymbol(s[0], int(s[1:])).value
        if G.has_node(nid):
            G.get_node(nid).attributes.name = w
    r = ov_dsg.add_objects(G, J, vocab, None, True, 0.5)
    # Hydra objects removed by hand after a check (duplicates of open-vocabulary ones): closed_removed.json, symbols
    gone = json.load(open(f'{out}/closed_removed.json')) if os.path.exists(f'{out}/closed_removed.json') else []
    for s in gone:
        nid = dsg.NodeSymbol(s[0], int(s[1:])).value
        if G.has_node(nid) and 'ov' not in (G.get_node(nid).attributes.metadata.get() or {}):
            G.remove_node(nid)
    r['closed_set_removed'] = [s for s in gone]
    # Hydra objects' boxes set by hand: closed_boxes.json, {symbol: [x0, y0, z0, x1, y1, z1]} (world, as in the graph)
    cbox = json.load(open(f'{out}/closed_boxes.json')) if os.path.exists(f'{out}/closed_boxes.json') else {}
    for s, b in cbox.items():
        nid = dsg.NodeSymbol(s[0], int(s[1:])).value
        if G.has_node(nid):
            a = G.get_node(nid).attributes; lo, hi = np.array(b[:3]), np.array(b[3:])
            a.bounding_box = dsg.BoundingBox((hi - lo).astype(np.float32), ((lo + hi) / 2).astype(np.float32))
            a.position = (lo + hi) / 2
    r['places_2d_removed'], r['extra_parents_removed'] = ov_dsg.standardize(G)
    # Hydra objects kept in the graph, given the open-vocabulary attributes (closed_ov.json, from hydra_ov.py):
    # semantic_feature = mean SigLIP2 embedding over the keyframes that see them, metadata.ov with source 'hydra'
    cov = json.load(open(f'{out}/closed_ov.json')) if os.path.exists(f'{out}/closed_ov.json') else {}
    r['closed_set_as_ov'] = 0
    for s, o in cov.items():
        nid = dsg.NodeSymbol(s[0], int(s[1:])).value
        if not G.has_node(nid) or ov_dsg.is_ov(G.get_node(nid)):
            continue
        a = G.get_node(nid).attributes
        e = np.frombuffer(base64.b64decode(o['feature']), np.float16).astype(np.float32)
        a.semantic_feature = e.reshape(-1, 1)
        a.metadata.set({'ov': dict(id=None, source='hydra', hydra_symbol=s, hydra_label=relab_orig.get(s),
                                   **{k: o.get(k) for k in ('n_obs', 'top3', 'text_sim', 'text_margin', 'first', 'last',
                                                            'best_stamp', 'best_box')})})
        r['closed_set_as_ov'] += 1
    save_graph(G, out)
    json.dump(O, open(f'{out}/ov_objects.json', 'w'))
    summary = dict(dsg=src, **(extra or {}), objects_seen2=len(O), wall=sum(x['wall'] for x in O),
                   confirmed=sum(x['confirmed'] for x in O),
                   removed_ov_ids=sorted(o['id'] for o in O if o.get('removed')),
                   joined={str(o['id']): o['joined'] for o in O if o.get('joined')},
                   joined_away=sum('joined_into' in o for o in O), closed_set_before=n_before,
                   objects_in_graph=G.get_layer(ov_dsg.LAYER).num_nodes(),
                   ov_objects_in_graph=sum(o['in_graph'] for o in O),
                   **{k: v for k, v in r.items() if k != 'replaced_ids'}, replaced_ids=r['replaced_ids'])
    summary['removed_after_check'] = len(summary['removed_ov_ids'])
    json.dump(summary, open(f'{out}/ov_summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('replaced_ids', 'removed_ov_ids', 'joined')}))
    return summary


def core_cells(res, frames, kf_dirs, frac=0.25, max_size=1.0, voxel=0.05):
    """Small objects (largest fused side < `max_size` m): the 5 cm voxels that `frac` of the object's detections
    (>= 2) put a mask point in. The fused cells are the union of every detection, so mask edges (background seen
    past the outline) and the pose offsets between the merged sessions widen a small object (library trash can:
    0.8 x 0.9 m fused, the mesh there ~10 cm apart and dark around it); the voxels most of the masks agree on are
    the object itself. Adds `core` (voxel keys) and `core_voxel` to the objects (used by ov_dsg.add_objects)."""
    from ov_fusion import Fusion, keys
    fr = frames if isinstance(frames, dict) else load_frames(frames)
    c = json.load(open(f"{next(iter(fr.values()))['_dir']}/camera_calib.json"))['sensors']['camera']
    s = 1280.0 / c['width']; F = Fusion((c['fx'] * s, c['fy'] * s, c['cx'] * s, c['cy'] * s))
    files = {int(os.path.basename(f)[3:-4]): f for d in kf_dirs for f in glob.glob(f'{d}/kf_*.npz')}
    sel = [r for r in res if r.get('confirmed') and max(r['size']) < max_size]
    by_stamp = {}
    for r in sel:
        r['_cnt'] = {}
        for st, k in r['dets']:
            by_stamp.setdefault(st, []).append((r, k))
    for st, lst in by_stamp.items():
        f = fr.get(st)
        if f is None or st not in files:
            continue
        d = np.load(files[st])
        depth = cv2.imread(f"{f['_dir']}/{f['depth_path']}", cv2.IMREAD_UNCHANGED)[::2, ::2].astype(np.float32) / 1000
        Rt = (quat_to_R(f['world_T_sensor']['q_wxyz']), np.array(f['world_T_sensor']['t']))
        for r, k in lst:
            p = F.detection_points(np.unpackbits(d['masks'][k], axis=-1)[:, :depth.shape[1]].astype(bool), depth, Rt)
            if p is not None:
                r['_n'] = r.get('_n', 0) + 1
                for kk in keys(p, voxel).tolist():
                    r['_cnt'][kk] = r['_cnt'].get(kk, 0) + 1
    n = 0
    for r in sel:
        cnt = r.pop('_cnt'); m = r.pop('_n', 0)
        th = max(2, int(np.ceil(frac * m)))
        core = sorted(k for k, v in cnt.items() if v >= th)
        if len(core) >= 8:
            r['core'] = core; r['core_voxel'] = voxel; n += 1
    return n


def fuse(frames, kf_dirs, dsg_path, out=None):
    import shutil
    from ov_fusion import Fusion
    backend = backend_dir(dsg_path); src = f'{backend}/{BEFORE}/dsg_with_mesh.json'
    if not os.path.exists(src):
        src = f'{backend}/dsg_with_mesh.json'
        if out is None:  # first run into hydra/backend: keep the graph as it was
            os.makedirs(f'{backend}/{BEFORE}', exist_ok=True)
            for f in ('dsg_with_mesh.json', 'dsg.json'):
                if os.path.exists(f'{backend}/{f}'):
                    shutil.copy2(f'{backend}/{f}', f'{backend}/{BEFORE}/{f}')
            src = f'{backend}/{BEFORE}/dsg_with_mesh.json'
    out = out or backend
    os.makedirs(out, exist_ok=True)
    fr = load_frames(frames); c = json.load(open(f"{next(iter(fr.values()))['_dir']}/camera_calib.json"))['sensors']['camera']
    s = 1280.0 / c['width']  # Fusion takes the intrinsics of a 1280 px wide image
    F = Fusion((c['fx'] * s, c['fy'] * s, c['cx'] * s, c['cy'] * s), join=False)  # joined in build()
    files = sorted((int(os.path.basename(f)[3:-4]), f) for d in kf_dirs for f in glob.glob(f'{d}/kf_*.npz'))
    used = 0; T = []
    for st, f in files:
        r = fr.get(st)
        if r is None:
            continue
        d = np.load(f); used += 1
        Rt = (quat_to_R(r['world_T_sensor']['q_wxyz']), np.array(r['world_T_sensor']['t']))
        depth = cv2.imread(f"{r['_dir']}/{r['depth_path']}", cv2.IMREAD_UNCHANGED)[::2, ::2].astype(np.float32) / 1000
        masks = np.unpackbits(d['masks'], axis=-1)[:, :, :depth.shape[1]].astype(bool)
        t = time.perf_counter()
        F.add_keyframe(st, Rt, depth, masks, d['emb'].astype(np.float32), d['top1'], d['boxes'], d['stuff_frac'])
        T.append(time.perf_counter() - t)
    F.merge()
    v = np.load(f'{OV_BENCH}/siglip2_vocab.npz'); vocab = [str(w) for w in v['labels']]
    res = F.results(v['text'], vocab)
    for r in res:
        r['emb'] = base64.b64encode(F.objs[r['id']].unit().astype(np.float16).tobytes()).decode()
    n_core = core_cells(res, fr, kf_dirs)
    build(backend, res, src, out, dict(frames=frames, kf_dirs=list(kf_dirs), keyframes=used, keyframes_found=len(files),
                                       fusion_ms_p50=float(np.median(T) * 1e3) if T else 0, objects_all=len(F.objs),
                                       small_objects_core=n_core))


def remove(backend, ids):
    """Mark the objects removed after the manual check and build the graph again from before_ov_objects (Hydra's
    objects they had replaced come back; long furniture is joined among the remaining ones)."""
    backend = backend_dir(backend)
    if os.path.exists(ids):
        t = open(ids).read().strip()
        ids = json.loads(t) if t.startswith('[') else t.split()
    else:
        ids = ids.split(',')
    ids = {int(i) for i in ids}
    O = json.load(open(f'{backend}/ov_objects.json'))
    found = {o['id'] for o in O} & ids
    for o in O:
        if o['id'] in ids:
            o['removed'] = True
    S = json.load(open(f'{backend}/ov_summary.json')) if os.path.exists(f'{backend}/ov_summary.json') else {}
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    src = f'{backend}/{BEFORE}/dsg_with_mesh.json'
    if not os.path.exists(src):
        sys.exit(f'{src} missing: the graph without open-vocabulary objects is needed (run fuse into this folder)')
    build(backend, O, src, backend, keep)
    print(json.dumps(dict(removed_now=len(found), not_found=sorted(ids - found))))


def label_closed(backend, pairs):
    """Relabel Hydra's own objects by symbol after a check ("O162=partition,..."; "O162=" clears it). Kept in
    closed_labels.json; builds the graph again."""
    backend = backend_dir(backend)
    f = f'{backend}/closed_labels.json'
    relab = json.load(open(f)) if os.path.exists(f) else {}
    for p in pairs.split(','):
        s, w = p.split('=', 1)
        if w.strip():
            relab[s.strip()] = w.strip()
        else:
            relab.pop(s.strip(), None)
    json.dump(relab, open(f, 'w'), indent=0)
    O = json.load(open(f'{backend}/ov_objects.json'))
    S = json.load(open(f'{backend}/ov_summary.json'))
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend, keep)


def box_closed(backend, entries):
    """Set Hydra objects' boxes by hand: "O162=x0,y0,z0,x1,y1,z1;..." ("O162=" clears). closed_boxes.json; rebuilds."""
    backend = backend_dir(backend)
    f = f'{backend}/closed_boxes.json'
    cb = json.load(open(f)) if os.path.exists(f) else {}
    for e in entries.split(';'):
        s, v = e.split('=', 1)
        if v.strip():
            cb[s.strip()] = [float(x) for x in v.split(',')]
        else:
            cb.pop(s.strip(), None)
    json.dump(cb, open(f, 'w'), indent=0)
    O = json.load(open(f'{backend}/ov_objects.json'))
    S = json.load(open(f'{backend}/ov_summary.json'))
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend, keep)


def remove_closed(backend, symbols):
    """Remove Hydra's own objects by symbol after a check ("O364,O354"; duplicates of open-vocabulary objects);
    "-O364" puts one back. Kept in closed_removed.json; builds the graph again."""
    backend = backend_dir(backend)
    f = f'{backend}/closed_removed.json'
    gone = json.load(open(f)) if os.path.exists(f) else []
    for s in symbols.split(','):
        s = s.strip()
        if s.startswith('-'):
            gone = [g for g in gone if g != s[1:]]
        elif s and s not in gone:
            gone.append(s)
    json.dump(gone, open(f, 'w'))
    O = json.load(open(f'{backend}/ov_objects.json'))
    S = json.load(open(f'{backend}/ov_summary.json'))
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend, keep)


def label(backend, pairs):
    """Set labels by hand after a check: <fusion id>=<word>[,...] or a file of such lines (the word need not be in
    the vocabulary: it joins the labelspace); <id>= clears it. Builds the graph again."""
    backend = backend_dir(backend)
    if os.path.exists(pairs):
        pairs = ','.join(l.strip() for l in open(pairs) if l.strip() and not l.startswith('#'))
    O = json.load(open(f'{backend}/ov_objects.json')); byid = {o['id']: o for o in O}
    for p in pairs.split(','):
        i, w = p.split('=', 1); o = byid[int(i)]
        if w.strip():
            o['label_manual'] = w.strip()
        else:
            o.pop('label_manual', None)
    S = json.load(open(f'{backend}/ov_summary.json'))
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend, keep)


def merge(backend, groups):
    """Join objects by hand: groups "<id>+<id>[+...]" separated by commas (one thing seen as several objects);
    "<id>+" clears that id's group. Stored as `merge_manual` on the group's first id; builds the graph again."""
    backend = backend_dir(backend)
    O = json.load(open(f'{backend}/ov_objects.json')); byid = {o['id']: o for o in O}
    for g in groups.split(','):
        ids = [int(i) for i in g.split('+') if i]
        byid[ids[0]].pop('merge_manual', None)
        if len(ids) > 1:
            byid[ids[0]]['merge_manual'] = ids[1:]
    S = json.load(open(f'{backend}/ov_summary.json'))
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend, keep)


def box(backend, entries):
    """Set boxes by hand: "<id>=x0,y0,z0,x1,y1,z1" (graph frame; z0 <= 0.05: down to the floor) separated by ';',
    or "<id>=" to clear; with a group joined by hand, give the group's first id. Builds the graph again."""
    backend = backend_dir(backend)
    O = json.load(open(f'{backend}/ov_objects.json')); byid = {o['id']: o for o in O}
    for e in entries.split(';'):
        i, v = e.split('=', 1); o = byid[int(i)]
        if v.strip():
            x = [float(q) for q in v.split(',')]; o['box_manual'] = [x[:3], x[3:]]
        else:
            o.pop('box_manual', None)
    S = json.load(open(f'{backend}/ov_summary.json'))
    keep = {k: S[k] for k in ('frames', 'kf_dirs', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all',
                              'small_objects_core') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend, keep)


def core(backend, kf_dirs):
    """Add the small objects' `core` voxels to an existing hydra/backend (fused before core_cells existed) and build
    the graph again (the removals in ov_objects.json stay)."""
    backend = backend_dir(backend)
    O = json.load(open(f'{backend}/ov_objects.json')); S = json.load(open(f'{backend}/ov_summary.json'))
    n = core_cells(O, S['frames'], kf_dirs)
    keep = {k: S[k] for k in ('frames', 'keyframes', 'keyframes_found', 'fusion_ms_p50', 'objects_all') if k in S}
    build(backend, O, f'{backend}/{BEFORE}/dsg_with_mesh.json', backend,
          dict(keep, kf_dirs=list(kf_dirs), small_objects_core=n))


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'extract':
        extract(a[0], a[1], a[2], float(a[3]) if len(a) > 3 else 0.5)
    elif cmd == 'fuse':
        fuse(a[0], a[1].split(','), a[2], a[3] if len(a) > 3 else None)
    elif cmd == 'remove':
        remove(a[0], a[1])
    elif cmd == 'label_closed':
        label_closed(*a)
    elif cmd == 'box_closed':
        box_closed(*a)
    elif cmd == 'remove_closed':
        remove_closed(*a)
    elif cmd == 'label':
        label(a[0], a[1])
    elif cmd == 'merge':
        merge(a[0], a[1])
    elif cmd == 'box':
        box(a[0], a[1])
    elif cmd == 'core':
        core(a[0], a[1].split(','))
    else:
        sys.exit(__doc__)
