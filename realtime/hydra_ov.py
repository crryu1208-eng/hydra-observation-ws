"""hydra_ov.py <backend> <frames.jsonl> [every]: SigLIP2 embeddings for Hydra's own objects left in the graph, so they
can carry the same open-vocabulary attributes as the fused ones. For each object (no metadata.ov): frames (every
`every`-th, corrected poses) where its box is fully in front, >= 70% inside the image, >= 24 px, and not hidden
(median depth inside its projected box >= its distance - 0.3 m - half its size); up to 30 of them, the largest;
crop = projected box + 10% context, SigLIP2-B/16 (TensorRT), mean of the unit embeddings. Writes
<backend>/closed_ov.json: {symbol: {feature (base64 float16), n_obs, first, last, best_stamp, best_box, top3,
text_sim, text_margin}}."""
import os, sys, json, base64, numpy as np, cv2, torch
sys.path.insert(0, '/root/hydra_ws/src/hydra-realtime/realtime')
from ov_session import load_frames, OV_BENCH
from ov_fusion import quat_to_R
os.environ['EVSAM_NO_FRAMES'] = '1'; sys.path.insert(0, OV_BENCH)
import evsam_bench as eb
backend, FR = sys.argv[1], sys.argv[2]; every = int(sys.argv[3]) if len(sys.argv) > 3 else 6
G = json.load(open(f'{backend}/dsg.json'))
fr = load_frames(FR); st = sorted(fr)[::every]; cals = {}
def cal(d):
    if d not in cals:
        p = f'{d}/camera_calib.json'
        cals[d] = json.load(open(p))['sensors']['camera']
    return cals[d]
Rs = np.stack([quat_to_R(fr[s]['world_T_sensor']['q_wxyz']) for s in st]); ts = np.stack([np.array(fr[s]['world_T_sensor']['t']) for s in st])
emb = eb.Embedder(os.path.join(OV_BENCH, 'ViT-B-16-SigLIP2.engine'), 'siglip')
v = np.load(f'{OV_BENCH}/siglip2_vocab.npz'); vocab = [str(w) for w in v['labels']]; text = v['text']
out = {}
for n in G['nodes']:
    a = n['attributes']
    if n['layer'] != 2 or n['partition'] != 0 or (a.get('metadata') or {}).get('ov'):
        continue
    sym = 'O%d' % (n['id'] & ((1 << 56) - 1))
    b = a['bounding_box']; c = np.array(b['world_P_center']); dm = np.array(b['dimensions']); lo, hi = c - dm / 2, c + dm / 2
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    P = np.einsum('fji,fkj->fki', Rs, C[None] - ts[:, None]); Pc = np.einsum('fji,fj->fi', Rs, c[None] - ts)
    cand = []
    for i, s in enumerate(st):
        if (P[i, :, 2] < 0.2).any() or Pc[i, 2] > 6:
            continue
        k = cal(fr[s]['_dir']); W, H = k['width'], k['height']
        u = k['fx'] * P[i, :, 0] / P[i, :, 2] + k['cx']; vv = k['fy'] * P[i, :, 1] / P[i, :, 2] + k['cy']
        x0, x1, y0, y1 = u.min(), u.max(), vv.min(), vv.max(); A = (x1 - x0) * (y1 - y0)
        ins = max(0, min(x1, W) - max(x0, 0)) * max(0, min(y1, H) - max(y0, 0)) / max(A, 1)
        if ins < 0.7 or min(x1 - x0, y1 - y0) < 24:
            continue
        cand.append((A, s, (max(0, x0), max(0, y0), min(W, x1), min(H, y1)), Pc[i, 2]))
    cand.sort(key=lambda q: -q[0]); got = []
    for A, s, (x0, y0, x1, y1), dist in cand:
        d = fr[s]; z = cv2.imread(f"{d['_dir']}/{d['depth_path']}", -1)
        if z is None:
            continue
        zz = z[int(y0):int(y1), int(x0):int(x1)].astype(np.float32) / 1000; zz = zz[zz > 0.2]
        if len(zz) and np.median(zz) < dist - 0.3 - dm.max() / 2:
            continue  # something in front of it
        img = cv2.cvtColor(cv2.imread(f"{d['_dir']}/{d['rgb_path']}"), cv2.COLOR_BGR2RGB)
        w, h = x1 - x0, y1 - y0
        bx = torch.tensor([[max(0, x0 - .1 * w), max(0, y0 - .1 * h), min(img.shape[1], x1 + .1 * w), min(img.shape[0], y1 + .1 * h)]], device='cuda', dtype=torch.float32)
        e = emb(img, bx).float().cpu().numpy()[0]; e /= np.linalg.norm(e)
        got.append((s, e, [float(x0), float(y0), float(w), float(h)]))
        if len(got) >= 30:
            break
    if not got:
        print(sym, a.get('name'), 'not seen'); continue
    m = np.mean([g[1] for g in got], 0); m /= np.linalg.norm(m)
    sc = text @ m; o = np.argsort(-sc)
    out[sym] = dict(feature=base64.b64encode(m.astype(np.float16).tobytes()).decode(), n_obs=len(got),
                    first=int(min(g[0] for g in got)), last=int(max(g[0] for g in got)), best_stamp=int(got[0][0]),
                    best_box=got[0][2], top3=[vocab[j] for j in o[:3]], text_sim=float(sc[o[0]]),
                    text_margin=float(sc[o[0]] - sc[o[1]]))
    print(sym, a.get('name'), 'kf', len(got), 'top3', out[sym]['top3'])
json.dump(out, open(f'{backend}/closed_ov.json', 'w'))
print(len(out), 'objects ->', f'{backend}/closed_ov.json')
