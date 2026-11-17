"""Put open-vocabulary objects (realtime/ov_fusion.py results) into a Hydra scene graph.

Standard layout (default): the objects become nodes of Hydra's objects layer (layer 2, partition 0, symbols 'O<n>'
numbered after Hydra's own), with one labelspace '_l2p0' (the SigLIP2 vocabulary, then the closed-set class names
it lacks; Hydra's remaining objects are relabelled into it and get their class name), mesh_connections = mesh
vertices in the object's cells, and every object (also Hydra's) has one parent place. --separate: own partition
(PARTITION 1, symbol 'V<id>', labelspace '_l2p1'), next to Hydra's closed-set objects:
  ObjectNodeAttributes  position / bounding_box (axis aligned), name = SigLIP2 label, semantic_label = index
                        into the vocabulary (labelspace '_l2p1'), semantic_feature = mean SigLIP2 crop embedding
                        (when given), metadata 'ov' = n_obs, top3, text_sim, structural share, wall / confirmed flags
  edge                  nearest place (layer 3, partition 0) -> object, like Hydra's own objects

Duplicates: a closed-set object (partition 0) is removed from the graph when >= `dedup` of its mesh vertices lie in
cells occupied by the added objects (their `cell` m voxels, corrected like the objects): the open-vocabulary
objects take over (one per thing instead of one per run of same-class things, e.g. a 5 m 'cabinet' made of a row
of lockers). Removed ids go to the graph metadata 'ov_replaced'. dedup=0 keeps everything.

Pose correction: the objects are fused in the odometry frame, the backend graph is corrected by loop closures.
Each agent (keyframe) node k gives C_k = corrected pose x odometry pose^-1 at its stamp; an object is moved
with the C of the agent nearest in time to its best view. odom_pose(stamp_ns) -> 4x4 (or None) supplies the
odometry pose of the agent body (Hydra's robot frame) at an agent stamp: TF in the live node, the recorded
frames offline.

Offline:  ov_dsg.py <dsg.json> <objects.json> <out dsg.json> [--frames frames_raw.jsonl] [--all]
"""
import base64
import json
import sys

import numpy as np
import spark_dsg as dsg

LAYER, PARTITION, CATEGORY = 2, 1, 'V'
AGENT_LAYER, AGENT_PARTITION = 2, ord('a')
PLACE_LAYER = 3
MAX_PLACE_DIST = 3.0


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def stamp_ns(ts):
    """agent timestamp (datetime.timedelta from the bindings, or int ns) -> int ns"""
    if hasattr(ts, 'days'):
        return (ts.days * 86400 + ts.seconds) * 10**9 + ts.microseconds * 1000
    return int(ts)


def corrections(G, odom_pose):
    """sorted agent stamps and their 4x4 world corrections (corrected x odometry^-1)"""
    if not G.has_layer(AGENT_LAYER, AGENT_PARTITION):
        return np.zeros(0, np.int64), []
    rows = []
    for n in G.get_layer(AGENT_LAYER, AGENT_PARTITION).nodes:
        a = n.attributes; st = stamp_ns(a.timestamp); T_odom = odom_pose(st)
        if T_odom is None:
            continue
        q = a.world_R_body; T = np.eye(4); T[:3, :3] = quat_R(q.w, q.x, q.y, q.z); T[:3, 3] = a.position
        rows.append((st, T @ np.linalg.inv(T_odom)))
    rows.sort(key=lambda r: r[0])
    return np.array([r[0] for r in rows], np.int64), [r[1] for r in rows]


def nearest(stamps, st):
    if len(stamps) == 0:
        return None
    i = int(np.searchsorted(stamps, st))
    return min((j for j in (i - 1, i) if 0 <= j < len(stamps)), key=lambda j: abs(int(stamps[j]) - st))


def decode_emb(s):
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32) if s else None


def cell_centers(cells, size):
    k = np.asarray(cells, np.int64); m = (1 << 21) - 1
    return (np.stack([(k >> 42) & m, (k >> 21) & m, k & m], 1) - (1 << 20) + 0.5) * size


def cell_keys(p, size):
    c = np.floor(p / size).astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def is_ov(n):
    m = n.attributes.metadata.get()  # a read-only mapping (mappingproxy) from the bindings
    try:
        return 'ov' in m
    except TypeError:
        return False


MIN_MESH = 10


def mesh_edges(G, weld=1e-3):
    """vertex index pairs of the mesh faces' sides (2 x 3 n_faces), plus the pairs of vertices at the same position
    (within `weld` m): Hydra's mesh repeats the vertices on the borders of its blocks, so faces alone are not
    connected from one block to the next"""
    F = np.asarray(G.mesh.get_faces(), np.int64).reshape(3, -1)
    E = [F[[0, 1]], F[[1, 2]], F[[2, 0]]]
    V = G.mesh.get_vertices()[:3].T
    _, inv = np.unique(np.round(V / weld).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel(); o = np.argsort(inv, kind='stable'); same = inv[o[1:]] == inv[o[:-1]]
    E.append(np.stack([o[:-1][same], o[1:][same]]))
    return np.concatenate(E, 1)


def mesh_region(V, E, idx, above=0.12, margin=0.5, frac=0.1):
    """The object's region on the mesh: of its vertices (those in its fused cells) the ones above the local floor
    (2nd percentile height of the mesh within `margin` m around them, + `above` m; the fusion's cells also take the
    floor beside an object: library O879, a bench, 2.7 x 3.1 m of mostly floor), then the pieces connected by mesh
    faces (E, mesh_edges()) that hold >= `frac` of them (stray bits go; a desk ring around a planter, whose sides
    the fused cells reach in two pieces, stays whole). Objects with fewer than MIN_MESH such vertices keep their vertices.
    Returns (vertex indices, box lo, box hi); an object whose region reaches down to the floor cut stands on the floor:
    its box goes down to the floor."""
    idx = np.unique(np.asarray(idx, np.int64))
    if len(idx) < MIN_MESH:
        return idx, None, None
    lo, hi = V[idx, :2].min(0) - margin, V[idx, :2].max(0) + margin
    near = np.all((V[:, :2] >= lo) & (V[:, :2] <= hi), 1)
    floor = np.percentile(V[near, 2], 2)
    k = idx[V[idx, 2] > floor + above]
    if len(k) < MIN_MESH:
        return idx, V[idx].min(0), V[idx].max(0)

    def box(r):
        lo, hi = V[r].min(0), V[r].max(0)
        if lo[2] < floor + above + 0.05:
            lo[2] = floor
        return r, lo, hi
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return box(k)
    pos = np.full(len(V), -1, np.int64); pos[k] = np.arange(len(k))
    e = E[:, (pos[E] >= 0).all(0)]
    n, lab = connected_components(coo_matrix((np.ones(e.shape[1]), (pos[e[0]], pos[e[1]])), shape=(len(k), len(k))),
                                  directed=False)
    cnt = np.bincount(lab); keep = k[cnt[lab] >= frac * len(k)]
    return box(keep if len(keep) >= MIN_MESH else k)


WALL_SNAP = ('door',)
LIGHT_FIT = ('light',)


def light_region(V, bright, lo, hi, reach=0.3, thr=0.78, cell=0.1, min_pts=15, row_tol=0.25, gap=0.7):
    """Ceiling lights: the white panels instead of the ceiling around them. Vertices within `reach` (x / y) and
    0.25 m (z) of the box with brightness >= `thr` (mean RGB, 0-1), joined into pieces over a `cell` grid (x / y),
    pieces of >= `min_pts`; the pieces lie in rows along the box's long axis (panels with gaps between them) and,
    where two passes left two rows (pose offsets between them), the row that stands out most from the ceiling
    around it is kept (mean over its panels of: panel brightness - median brightness within 0.25 m of the panel).
    Returns (idx, lo, hi) or (None, None, None)."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    m = np.all((V >= lo - [reach, reach, 0.25]) & (V <= hi + [reach, reach, 0.25]), 1) & (bright >= thr)
    idx = np.nonzero(m)[0]
    if len(idx) < min_pts:
        return None, None, None
    k = np.floor(V[idx, :2] / cell).astype(np.int64); cells = {}
    for i, kk in enumerate(map(tuple, k)):
        cells.setdefault(kk, []).append(i)
    seen, pieces = set(), []
    for c0 in cells:
        if c0 in seen:
            continue
        comp, todo = [], [c0]; seen.add(c0)
        while todo:
            c = todo.pop(); comp += cells[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (c[0] + dx, c[1] + dy)
                    if q in cells and q not in seen:
                        seen.add(q); todo.append(q)
        if len(comp) >= min_pts:
            pieces.append(idx[comp])
    if not pieces:
        return None, None, None
    across = 0 if (hi - lo)[0] < (hi - lo)[1] else 1  # the short horizontal axis
    along = 1 - across
    pos = np.array([V[p, across].mean() for p in pieces]); n = np.array([len(p) for p in pieces])
    ext = np.array([[V[p, along].min(), V[p, along].max()] for p in pieces])
    row = list(range(len(pieces)))  # rows: pieces end to end along the axis (gap <= `gap`), across within row_tol

    def root(i):
        while row[i] != i:
            i = row[i]
        return i
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            g = max(ext[i, 0], ext[j, 0]) - min(ext[i, 1], ext[j, 1])
            if g <= gap and abs(pos[i] - pos[j]) <= row_tol:
                row[root(j)] = root(i)
    rid = np.array([root(i) for i in range(len(pieces))])
    con = []
    for p in pieces:
        plo, phi = V[p].min(0), V[p].max(0)
        near = np.all((V >= plo - [0.25, 0.25, 0.15]) & (V <= phi + [0.25, 0.25, 0.15]), 1)
        con.append(bright[p].mean() - np.median(bright[near]))
    con = np.array(con)

    def score(r):
        r = rid == r
        return (con[r] * n[r]).sum() / n[r].sum()
    best = max(set(rid.tolist()), key=score)
    sel = np.concatenate([pieces[i] for i in np.nonzero(rid == best)[0]])
    return sel, V[sel].min(0), V[sel].max(0)


def snap_to_wall(V, lo, hi, reach=0.25, thick=0.16):
    """A door's box on its wall: of the mesh vertices inside the box or within `reach` m of it across the wall (0.3..2.2 m
    high), the densest plane across the box's thin horizontal side (5 cm bins, smoothed over 3) becomes the
    box's centre on that axis, `thick` m thick (the fused or hand-set box floats off the wall when the passes'
    poses disagree, lab_03 room 667: 0.2 m). Returns lo, hi (unchanged when there is no wall)."""
    ax = int(np.argmin(hi[:2] - lo[:2])); ot = 1 - ax; c = (lo[ax] + hi[ax]) / 2
    m = (V[:, ot] > lo[ot]) & (V[:, ot] < hi[ot]) & (V[:, 2] > 0.3) & (V[:, 2] < 2.2) & (np.abs(V[:, ax] - c) < reach + (hi[ax] - lo[ax]) / 2)
    if m.sum() < 30:
        return lo, hi
    e = np.arange(c - reach - (hi[ax] - lo[ax]) / 2, c + reach + (hi[ax] - lo[ax]) / 2 + 0.05, 0.05)
    h = np.convolve(np.histogram(V[m, ax], bins=e)[0], [1, 1, 1], 'same')
    w = e[int(np.argmax(h))] + 0.025
    lo, hi = lo.copy(), hi.copy(); lo[ax], hi[ax] = w - thick / 2, w + thick / 2
    return lo, hi


def add_objects(G, objects, vocab, odom_pose=None, only_confirmed=True, dedup=0.5, standard=True):
    """Add the objects to G (in place; open-vocabulary objects of an earlier call are replaced). Returns counts.
    standard: the objects go into Hydra's own objects layer (layer 2 partition 0, symbols 'O<n>' after Hydra's), one
    labelspace for all objects, mesh_connections and a parent place for every object; else partition 1 ('V<id>')."""
    for part in (PARTITION, 0):
        if G.has_layer(LAYER, part):
            for n in list(G.get_layer(LAYER, part).nodes):
                if part == PARTITION or is_ov(n):
                    G.remove_node(n.id)
    stamps, C = corrections(G, odom_pose) if odom_pose is not None else (np.zeros(0, np.int64), [])
    # parent places: Hydra's 3D places, or the 2D (floor) places when the graph has only those
    # (realtime/places_2d_only.py exports; nearest in x / y like there)
    pp = 0 if G.has_layer(PLACE_LAYER, 0) and G.get_layer(PLACE_LAYER, 0).num_nodes() else 1
    places = [(n.id, np.asarray(n.attributes.position)) for n in G.get_layer(PLACE_LAYER, pp).nodes] \
        if G.has_layer(PLACE_LAYER, pp) else []
    P = np.array([p for _, p in places]).reshape(-1, 3)
    dims = 3 if pp == 0 else 2
    has_mesh = G.has_mesh() and not G.mesh.empty()
    V = G.mesh.get_vertices()[:3].T if has_mesh else np.zeros((0, 3))
    VA = G.mesh.get_vertices() if has_mesh else None
    bright = VA[3:6].mean(0) if (VA is not None and VA.shape[0] >= 6) else None
    widx = {w: i for i, w in enumerate(vocab)}
    # objects in the graph frame
    sel = []; shift = []
    for o in objects:
        if only_confirmed and not o.get('confirmed'):
            continue
        c = np.asarray(o['center'], float)
        k = nearest(stamps, int(o['best_stamp']))
        R, t = (C[k][:3, :3], C[k][:3, 3]) if k is not None else (np.eye(3), np.zeros(3))
        if k is not None:
            c2 = R @ c + t; shift.append(float(np.linalg.norm(c2 - c))); c = c2
        occ = cell_keys(cell_centers(o['cells'], o['cell']) @ R.T + t, o['cell']) if o.get('cells') else None
        sel.append((o, c, occ))
    cell = next((o['cell'] for o, _, occ in sel if occ is not None), None)
    # numbers after all of Hydra's objects, also the ones replaced below (no symbol reused by another object)
    next_id = 1 + max([n.id.category_id for n in G.get_layer(LAYER).nodes], default=-1) if G.has_layer(LAYER) else 0
    # closed-set objects the open-vocabulary ones cover
    replaced = []
    occ_all = [occ for _, _, occ in sel if occ is not None]
    if dedup > 0 and occ_all and G.has_layer(LAYER) and len(V):
        occ_all = np.unique(np.concatenate(occ_all))
        for n in list(G.get_layer(LAYER).nodes):
            idx = np.asarray(n.attributes.mesh_connections, np.int64); idx = idx[idx < len(V)]
            if len(idx) and np.isin(cell_keys(V[idx], cell), occ_all).mean() >= dedup:
                replaced.append(n.id.str()); G.remove_node(n.id)
    Vkeys = cell_keys(V, cell) if (standard and cell is not None and len(V)) else None
    E = mesh_edges(G) if Vkeys is not None else None
    part = 0 if standard else PARTITION
    if standard:  # one labelspace: the vocabulary, then the closed-set names it lacks
        names = dict(vocab=list(vocab))
        old = G.get_labelspace(LAYER, 0)
        old = dict(old.labels_to_names) if old is not None else {}
        extra = []
        for n in G.get_layer(LAYER).nodes:
            a = n.attributes; w = a.name or old.get(a.semantic_label, f'label_{a.semantic_label}')
            if w not in widx:
                widx[w] = len(vocab) + len(extra); extra.append(w)
            a.name = w; a.semantic_label = widx[w]
        for o, _, _ in sel:  # labels set by hand outside the vocabulary
            w = o.get('label', '')
            if w and w not in widx:
                widx[w] = len(vocab) + len(extra); extra.append(w)
        G.set_labelspace(dsg.Labelspace({i: w for i, w in enumerate(list(vocab) + extra)}), LAYER, 0)
    else:
        G.set_labelspace(dsg.Labelspace({i: w for i, w in enumerate(vocab)}), LAYER, PARTITION)
    n_add = n_edge = 0; ids = {}
    for o, c, occ in sel:
        size = np.asarray(o['size'], float)
        a = dsg.ObjectNodeAttributes()
        a.position = c; a.name = o.get('label', ''); a.semantic_label = widx.get(o.get('label'), 0)
        a.bounding_box = dsg.BoundingBox(size.astype(np.float32), c.astype(np.float32))
        a.last_update_time_ns = int(o['last'])
        if Vkeys is not None and occ is not None:
            idx, lo, hi = mesh_region(V, E, np.nonzero(np.isin(Vkeys, occ))[0])
            a.mesh_connections = idx.tolist()
            if o.get('core'):  # small object: the voxels its masks agree on (ov_session.core_cells)
                X = cell_centers(o['core'], o['core_voxel']) @ R.T + t; h = o['core_voxel'] / 2
                clo, chi = X.min(0) - h, X.max(0) + h
                if lo is None:
                    lo, hi = clo, chi
                else:  # per axis: one range inside the other -> their overlap (a thin plate stays thin, the mesh
                    # beyond the core goes); ranges shifted against each other -> the core's
                    ilo, ihi = np.maximum(lo, clo), np.minimum(hi, chi); il = ihi - ilo
                    nested = (il >= 0.8 * np.minimum(hi - lo, chi - clo))
                    lo, hi = np.where(nested, ilo, clo), np.where(nested, ihi, chi)
                inb = np.all((V[idx] >= lo - h) & (V[idx] <= hi + h), 1) if len(idx) else np.zeros(0, bool)
                if inb.any():
                    a.mesh_connections = idx[inb].tolist()
            if o.get('box_manual'):  # set by hand (e.g. a door from its round windows: ov_session.py box)
                lo, hi = np.asarray(o['box_manual'][0], float), np.asarray(o['box_manual'][1], float)
                lo, hi = np.minimum(R @ lo + t, R @ hi + t), np.maximum(R @ lo + t, R @ hi + t)
                if lo[2] <= 0.05:  # down to the local floor
                    flo = V[np.all((V[:, :2] >= lo[:2] - 0.5) & (V[:, :2] <= hi[:2] + 0.5), 1), 2]
                    f2 = np.percentile(flo, 2) if len(flo) else lo[2]
                    lo[2] = f2 if f2 < 0.1 else lo[2]  # no floor mesh there: keep z0
                inb = np.all((V[idx] >= lo - 0.05) & (V[idx] <= hi + 0.05), 1) if len(idx) else np.zeros(0, bool)
                a.mesh_connections = idx[inb].tolist() if inb.any() else a.mesh_connections
            if lo is not None and o.get('label') in WALL_SNAP and len(V):  # doors on their wall
                lo, hi = snap_to_wall(V, np.asarray(lo, float), np.asarray(hi, float))
                inb = np.all((V[idx] >= lo - 0.05) & (V[idx] <= hi + 0.05), 1) if len(idx) else np.zeros(0, bool)
                a.mesh_connections = idx[inb].tolist() if inb.any() else a.mesh_connections
            if lo is not None:  # the box of the object's mesh (or core), not of its fused cells
                c = (lo + hi) / 2
                a.position = c; a.bounding_box = dsg.BoundingBox((hi - lo).astype(np.float32), c.astype(np.float32))
        e = decode_emb(o.get('emb'))
        if e is not None:
            a.semantic_feature = e.reshape(-1, 1)
        a.metadata.set({'ov': {k_: o.get(k_) for k_ in ('id', 'n_obs', 'top3', 'text_sim', 'text_margin', 'stuff',
                                                          'wall', 'confirmed', 'first', 'last', 'best_stamp', 'joined')}})
        if standard:
            sym = dsg.NodeSymbol('O', next_id); next_id += 1
        else:
            sym = dsg.NodeSymbol(CATEGORY, int(o['id']))
        G.add_node(LAYER, sym, a, part); n_add += 1; ids[sym.str()] = int(o['id'])
        if len(P):
            j = int(np.argmin(np.linalg.norm(P[:, :dims] - c[:dims], axis=1)))
            if (standard or np.linalg.norm(P[j] - c) <= MAX_PLACE_DIST) and G.insert_edge(places[j][0], sym):
                n_edge += 1
    refit = 0
    if E is not None:  # Hydra's own objects too: their region and box from their connected mesh
        for n in G.get_layer(LAYER).nodes:
            if is_ov(n):
                continue
            idx = np.asarray(n.attributes.mesh_connections, np.int64); idx = idx[idx < len(V)]
            r, lo, hi = mesh_region(V, E, idx)
            if lo is not None and n.attributes.name in LIGHT_FIT and bright is not None:
                r2, lo2, hi2 = light_region(V, bright, lo, hi)
                if lo2 is not None:
                    r, lo, hi = r2, lo2, hi2
            if lo is not None:
                c = (lo + hi) / 2; a = n.attributes
                a.mesh_connections = r.tolist(); a.position = c
                a.bounding_box = dsg.BoundingBox((hi - lo).astype(np.float32), c.astype(np.float32)); refit += 1
    orphans = 0
    if standard and len(P):  # every object with a parent place (Hydra's own ones too)
        for n in G.get_layer(LAYER).nodes:
            if not n.has_parent():
                j = int(np.argmin(np.linalg.norm(P[:, :dims] - np.asarray(n.attributes.position)[:dims], axis=1)))
                orphans += bool(G.insert_edge(places[j][0], n.id))
    G.metadata.add({'ov': dict(replaced=replaced, ids=ids, standard=standard)})
    if replaced:
        G.metadata.add({'ov_replaced': replaced})
    return dict(replaced=len(replaced), replaced_ids=replaced, objects=n_add, place_edges=n_edge,
                closed_set_kept=G.get_layer(LAYER).num_nodes() - (n_add if standard else 0), orphans_attached=orphans, closed_set_refit=refit,
                corrected=len(shift), shift_max=max(shift, default=0.0), shift_mean=float(np.mean(shift)) if shift else 0.0)


def standardize(G, drop_2d_places=False):
    """An object with several parent places keeps the nearest one. drop_2d_places: also drop Hydra's 2D mesh
    places (layer 3 partition 1) when the graph has its 3D places too. Returns (2D places removed, extra parent edges removed)."""
    n = extra = 0
    if drop_2d_places and G.has_layer(PLACE_LAYER, 0) and G.get_layer(PLACE_LAYER, 0).num_nodes() \
            and G.has_layer(PLACE_LAYER, 1):
        for x in list(G.get_layer(PLACE_LAYER, 1).nodes):
            G.remove_node(x.id); n += 1
    if G.has_layer(LAYER):
        for o in G.get_layer(LAYER).nodes:
            ps = list(o.parents())
            if len(ps) > 1:
                p0 = np.asarray(o.attributes.position)
                ps.sort(key=lambda q: np.linalg.norm(np.asarray(G.get_node(q).attributes.position) - p0))
                for q in ps[1:]:
                    G.remove_edge(q, o.id); extra += 1
    return n, extra


def remove_objects(G, symbols=(), before=None, dedup=0.5, ov_ids=()):
    """Remove open-vocabulary objects (node symbols, e.g. after a manual check) from a standard-layout graph made
    by add_objects, by symbol or by fusion id (`ov_ids`, metadata ov.id). With `before` (the graph add_objects started from), Hydra's objects that were replaced only
    because of removed objects come back: those with < dedup of their mesh vertices in the kept objects' cells
    (cells from the kept nodes' metadata via their mesh vertices: the kept nodes' mesh_connections)."""
    gone = []
    ov_ids = set(int(i) for i in ov_ids)
    if ov_ids:  # fusion ids (metadata ov.id, stable across rebuilds) -> node symbols
        symbols = list(symbols) + [n.id.str() for n in G.get_layer(LAYER).nodes
                                   if is_ov(n) and int(n.attributes.metadata.get()['ov']['id']) in ov_ids]
    for sym in symbols:
        ns = dsg.NodeSymbol(sym[0], int(sym[1:]))
        if G.has_node(ns) and is_ov(G.get_node(ns)):
            G.remove_node(ns); gone.append(sym)
    back = []
    if before is not None and G.has_mesh() and not G.mesh.empty():
        V = G.mesh.get_vertices()[:3].T
        kept = [np.asarray(n.attributes.mesh_connections, np.int64) for n in G.get_layer(LAYER).nodes if is_ov(n)]
        cov = np.zeros(len(V), bool)
        for k in kept:
            cov[k[k < len(V)]] = True
        old_ls = before.get_labelspace(LAYER, 0); old = dict(old_ls.labels_to_names) if old_ls is not None else {}
        ls = G.get_labelspace(LAYER, 0); names = dict(ls.labels_to_names) if ls is not None else {}
        widx = {w: i for i, w in names.items()}
        pp = 0 if G.has_layer(PLACE_LAYER, 0) and G.get_layer(PLACE_LAYER, 0).num_nodes() else 1
        places = [(n.id, np.asarray(n.attributes.position)) for n in G.get_layer(PLACE_LAYER, pp).nodes]
        P = np.array([p for _, p in places]).reshape(-1, 3); dims = 3 if pp == 0 else 2
        for n in before.get_layer(LAYER).nodes:
            if G.has_node(n.id):
                continue
            idx = np.asarray(n.attributes.mesh_connections, np.int64); idx = idx[idx < len(V)]
            if len(idx) and cov[idx].mean() >= dedup:
                continue
            a = n.attributes
            w = a.name or old.get(a.semantic_label, f'label_{a.semantic_label}')
            if w not in widx:
                widx[w] = max(names, default=-1) + 1; names[widx[w]] = w
            G.add_node(LAYER, n.id, a, 0)
            b = G.get_node(n.id).attributes; b.name = w; b.semantic_label = widx[w]
            if len(P):
                j = int(np.argmin(np.linalg.norm(P[:, :dims] - np.asarray(a.position)[:dims], axis=1)))
                G.insert_edge(places[j][0], n.id)
            back.append(n.id.str())
        G.set_labelspace(dsg.Labelspace(names), LAYER, 0)
    m = dict(G.metadata.get().get('ov', {})); m['replaced'] = [r for r in m.get('replaced', []) if r not in back]
    m['removed_after_check'] = sorted(set(m.get('removed_after_check', [])) | set(gone), key=lambda x: int(x[1:]))
    G.metadata.add({'ov': m, 'ov_replaced': m['replaced']})
    return dict(removed=len(gone), not_found=len(symbols) - len(gone), restored=back)


def frames_odom(path):
    """odometry pose lookup from a session's frames_raw.jsonl (uncorrected world_T_body per frame)"""
    st, T = [], []
    for line in open(path):
        d = json.loads(line)
        if d.get('status') != 'written' or not d.get('world_T_body'):
            continue
        p = d['world_T_body']; M = np.eye(4); M[:3, :3] = quat_R(*p['q_wxyz']); M[:3, 3] = p['t']
        st.append(int(d['timestamp_ns'])); T.append(M)
    o = np.argsort(st); st = np.array(st, np.int64)[o]; T = [T[i] for i in o]

    def lookup(s):
        j = nearest(st, s)
        return T[j] if j is not None and abs(int(st[j]) - s) < 50_000_000 else None
    return lookup


if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('dsg'); ap.add_argument('objects'); ap.add_argument('out')
    ap.add_argument('--frames', help='frames_raw.jsonl for the loop-closure correction')
    ap.add_argument('--vocab', default='/root/data/ov_bench/siglip2_vocab.npz')
    ap.add_argument('--all', action='store_true', help='also objects that are not confirmed')
    ap.add_argument('--separate', action='store_true', help='partition 1 instead of the standard objects layer')
    ap.add_argument('--drop-2d-places', action='store_true', help='drop layer 3 partition 1 when 3D places exist')
    ap.add_argument('--dedup', type=float, default=0.5, help='remove closed-set objects this much covered (0: keep)')
    a = ap.parse_args()
    G = dsg.DynamicSceneGraph.load(a.dsg)
    vocab = [str(s) for s in np.load(a.vocab)['labels']]
    r = add_objects(G, json.load(open(a.objects)), vocab, frames_odom(a.frames) if a.frames else None, not a.all, a.dedup, not a.separate)
    if not a.separate:
        r['places_2d_removed'], r['extra_parents_removed'] = standardize(G, a.drop_2d_places)
    G.save(a.out, include_mesh=a.dsg.endswith('with_mesh.json'))
    print(json.dumps(r))
