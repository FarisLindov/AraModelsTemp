# Board-side helper for the LAVA ARA240 RSTPReid-100 benchmark: TinyCLIP-61M vision +
# original text DVM (77 tokens) vs SOT-baked text DVM (76 tokens). Needs numpy + Pillow only.
# Subcommands:
#   prep-vision DATASET OUTDIR SIZE DTYPE QSCALE ZP
#       SIZE = 224x128 or 224x224. Reads the preprocessed PNGs in images_SIZE (already
#       letterboxed on host), applies CLIP mean/std, quantizes, writes CHW input_XXXX.dat.
#   prep-text DATASET OUTDIR MODE DTYPE QSCALE ZP
#       MODE = orig -> rows 0..76 [77,512]; MODE = sot -> rows 1..76 [76,512] (SOT baked in DVM).
#   decode LOG OUTDIR COUNT
#   latency LOG CSV TAG
#   params LOG
#   metrics DATASET VIS_DIR TXT_DIR REPORT SIZE MODE VDTYPE VSCALE VZP TDTYPE TSCALE TZP
#       orig: hidden row eot; sot: hidden row eot-1. Then W @ h + b on host.
import base64
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

D = 512
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32) * 255.0
SCALE = 1.0 / (np.array([0.26862954, 0.26130258, 0.27577711], np.float32) * 255.0)
DT = {'int8': (np.int8, -128, 127), 'uint8': (np.uint8, 0, 255), 'int16': ('<i2', -32768, 32767)}


def quant(x, dtype, qs, zp):
    if dtype == 'float32':
        return x.astype('<f4')
    t, lo, hi = DT[dtype]
    assert np.isfinite(qs) and qs > 0, 'qscale must be positive'
    return np.clip(np.rint(x / qs + zp), lo, hi).astype(t)


def dequant(raw, dtype, qs, zp):
    if dtype == 'float32':
        return np.frombuffer(raw, '<f4').astype(np.float32)
    return (np.frombuffer(raw, DT[dtype][0]).astype(np.float32) - zp) * qs


def prep_vision(ds, out, size, dtype, qs, zp):
    ds, out = Path(ds), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    h, w = map(int, size.split('x'))
    names = (ds / 'image_set.txt').read_text().split()
    for k, n in enumerate(names):
        with Image.open(ds / f'images_{size}' / n) as im:
            x = np.asarray(im.convert('RGB'), np.float32)
        assert x.shape == (h, w, 3), f'{n}: {x.shape}'
        x = (x - MEAN) * SCALE
        q = np.ascontiguousarray(quant(x, dtype, qs, zp).transpose(2, 0, 1))
        q.tofile(out / f'input_{k:04d}.dat')
    print(f'vision inputs ({size}):', len(names))


def prep_text(ds, out, mode, dtype, qs, zp):
    ds, out = Path(ds), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    E = np.fromfile(ds / 'text' / 'emb_subset.f32', np.float32).reshape(-1, D)
    ids = np.load(ds / 'text' / 'token_ids.npy')
    sat = 0
    for k, row in enumerate(ids):
        x = E[row] if mode == 'orig' else E[row][1:]
        q = np.ascontiguousarray(quant(x, dtype, qs, zp))
        if dtype != 'float32':
            sat += int((np.abs(x / qs) > DT[dtype][2]).sum())
        q.tofile(out / f'input_{k:04d}.dat')
    print(f'text inputs ({mode}, {x.shape}): {len(ids)}  saturated values: {sat}')


def decode(log, out, count):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    txt = Path(log).read_text(errors='replace')
    for r in range(int(count)):
        mt = re.search(rf'^=== OUTPUT_{r}_0_START(.*?)\n(.*?)^=== OUTPUT_{r}_0_END ===$', txt, re.M | re.S)
        assert mt, f'missing output block {r}'
        raw = base64.b64decode(''.join(mt.group(2).split()))
        md5 = re.search(r'md5=([0-9a-fA-F]+)', mt.group(1))
        assert not md5 or md5.group(1).lower() == hashlib.md5(raw).hexdigest(), f'md5 mismatch {r}'
        (out / f'embedding_{r:04d}.bin').write_bytes(raw)
    print('decoded', count, 'bytes each', len(raw))


def latency(log, csv, tag):
    rows = []
    for line in Path(log).read_text(errors='replace').splitlines():
        if line.startswith('TIMING'):
            v = dict(re.findall(r'([A-Za-z_]+)=([-+]?[0-9]*\.?[0-9]+)', line))
            if 'run' in v and 'hw_ms' in v:
                rows.append((int(float(v['run'])), float(v['hw_ms']), float(v.get('app_ms', 'nan'))))
    assert rows, 'no TIMING rows'
    rows.sort()
    hw = sorted(r[1] for r in rows)
    print(f'[{tag}] n={len(rows)} HW min {hw[0]:.4f} mean {statistics.mean(hw):.4f} '
          f'median {statistics.median(hw):.4f} max {hw[-1]:.4f} ms  {1000 / statistics.mean(hw):.2f} IPS')
    Path(csv).write_text('run,hw_ms,app_ms\n' + ''.join(f'{a},{b:.6f},{c:.6f}\n' for a, b, c in rows))


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def evaluate(S, ip, qp):
    m = ip[np.argsort(-S, axis=1)] == qp[:, None]
    f = m.argmax(1)
    r = {f'R{k}': 100 * float((f < k).mean()) for k in (1, 5, 10)}
    p = np.cumsum(m, 1) / np.arange(1, m.shape[1] + 1)
    r['mAP'] = 100 * float(((p * m).sum(1) / m.sum(1)).mean())
    return r


def text_diag(ds, traw, tdt, tqs, tzp, L, off, eot, Wp, b, Tr):
    # Diagnostics for the text DVM output. Compares the decoded hidden states with the
    # fp32 ONNX hidden states (ref/text_hidden_fp32.npy, [N,77,512]) to separate
    # quantization error from wrong layout, wrong scale or wrong readout row.
    print('---- TEXT DIAG ----')
    bpe = {'int8': 1, 'uint8': 1, 'int16': 2}.get(tdt, 4)
    print(f'text dtype={tdt} qscale={tqs} zp={tzp} L={L} bytes/output={len(traw[0])} expected={L * D * bpe}')
    if tdt in DT:
        raw0 = np.frombuffer(traw[0], DT[tdt][0]).astype(np.int64)
        lim = float(np.isin(raw0, [DT[tdt][1], DT[tdt][2]]).mean())
        print(f'raw[0] min {raw0.min()} max {raw0.max()} mean {raw0.mean():.2f} frac_at_limits {lim:.4f}')
    f = Path(ds) / 'ref' / 'text_hidden_fp32.npy'
    if not f.exists():
        print('no ref/text_hidden_fp32.npy, skipping hidden-state diag')
        return
    Hr = np.load(f).astype(np.float32)[:, off:]
    n = min(len(traw), len(Hr))
    H = np.stack([dequant(r, tdt, tqs, tzp).reshape(L, D) for r in traw[:n]])
    Hr = Hr[:n]

    def cos(a, c):
        return float((l2n(a) * l2n(c)).sum(-1).mean())

    idx = np.arange(n)
    rows = np.array([eot[k] - off for k in range(n)])
    hre = Hr[idx, rows]
    print(f'hidden cos: all tokens {cos(H, Hr):.4f}  eot row {cos(H[idx, rows], hre):.4f}  row0 {cos(H[:, 0], Hr[:, 0]):.4f}')
    print(f'hidden rms ratio ara/fp32 {float(np.sqrt((H ** 2).mean() / (Hr ** 2).mean())):.4f}')
    Ht = np.stack([dequant(r, tdt, tqs, tzp).reshape(D, L).T for r in traw[:n]])
    print(f'hidden cos if output is [512,L]: {cos(Ht, Hr):.4f}')
    sim = np.einsum('nld,nd->nl', l2n(H), l2n(hre))
    best = sim.argmax(1)
    print(f'best row for fp32 eot: {best[:10].tolist()} expected {rows[:10].tolist()} best cos {float(sim.max(1).mean()):.4f}')
    per_tok = (l2n(H) * l2n(Hr)).sum(-1).mean(0)
    print('per-token cos (first 12):', ' '.join(f'{v:.3f}' for v in per_tok[:12]))
    # Full per-row profile (mean over captions) to see where accuracy starts to fall.
    pt = (l2n(H) * l2n(Hr)).sum(-1)
    for s in range(0, L, 12):
        print(f'per-row cos {s:2d}-{min(s + 12, L) - 1:2d}:', ' '.join(f'{v:.3f}' for v in pt.mean(0)[s:s + 12]))
    # Same, aligned to each caption's EOT row: offset -8..+4 around EOT.
    al = []
    for d in range(-8, 5):
        r = rows + d
        ok = (r >= 0) & (r < L)
        al.append(f'{d:+d}:{float(pt[idx[ok], r[ok]].mean()):.3f}' if ok.any() else f'{d:+d}:n/a')
    print('cos vs offset from eot:', ' '.join(al))
    # Content rows (before EOT) vs padding rows (after EOT), averaged per caption.
    pre = np.mean([pt[k, :rows[k]].mean() for k in range(n)])
    post = np.mean([pt[k, rows[k] + 1:].mean() if rows[k] + 1 < L else np.nan for k in range(n)])
    print(f'cos content rows (<eot) {pre:.4f}  eot {float(pt[idx, rows].mean()):.4f}  padding rows (>eot) {np.nanmean(post):.4f}')
    Tb = l2n(np.stack([Wp @ H[k, best[k]] + b for k in range(n)]))
    print(f'text cos using best row: {float((Tb * Tr[:n]).sum(1).mean()):.4f}')
    print('---- END TEXT DIAG ----')


def metrics(ds, vdir, tdir, report, size, mode, vdt, vqs, vzp, tdt, tqs, tzp):
    ds = Path(ds)
    gt = json.loads((ds / 'gt.json').read_text())
    pids = {p: k for k, p in enumerate(sorted({i['pid'] for i in gt['images']}))}
    ip = np.array([pids[i['pid']] for i in gt['images']])
    qp = np.array([pids[c['pid']] for c in gt['captions']])
    Gr = np.load(ds / 'ref' / f'vision_fp32_{size}.npy')
    Tr = np.load(ds / 'ref' / 'text_fp32.npy')
    vraw = [(Path(vdir) / f'embedding_{k:04d}.bin').read_bytes() for k in range(len(ip))]
    traw = [(Path(tdir) / f'embedding_{k:04d}.bin').read_bytes() for k in range(len(qp))]
    Wp = np.fromfile(ds / 'text' / 'text_projection_weight.f32', np.float32).reshape(D, D)
    b = np.fromfile(ds / 'text' / 'text_projection_bias.f32', np.float32)
    eot = np.load(ds / 'text' / 'eot_index.npy')
    L, off = (77, 0) if mode == 'orig' else (76, 1)

    def vis(z):
        return l2n(np.stack([dequant(r, vdt, vqs, z) for r in vraw]))

    def txt_(z):
        return l2n(np.stack([Wp @ dequant(r, tdt, tqs, z).reshape(L, D)[eot[k] - off] + b
                             for k, r in enumerate(traw)]))

    # Offset sign convention is undocumented; keep the one closest to fp32.
    G, vzp = max(((vis(z), z) for z in {vzp, -vzp}), key=lambda t: float((t[0] * Gr).sum(1).mean()))
    T, tzp = max(((txt_(z), z) for z in {tzp, -tzp}), key=lambda t: float((t[0] * Tr).sum(1).mean()))
    cv, ct = (G * Gr).sum(1), (T * Tr).sum(1)
    text_diag(ds, traw, tdt, tqs, tzp, L, off, eot, Wp, b, Tr)
    out = [f'combo=vision_{size}_text_{mode}']
    for tag, r in (('ara', evaluate(T @ G.T, ip, qp)),
                   ('ara_vision_fp32_text', evaluate(Tr @ G.T, ip, qp)),
                   ('fp32_vision_ara_text', evaluate(T @ Gr.T, ip, qp)),
                   ('fp32_onnx_ref', evaluate(Tr @ Gr.T, ip, qp))):
        out += [f'{tag}_{k}={v:.2f}' for k, v in r.items()]
    out += [f'vision_zp_used={vzp}', f'text_zp_used={tzp}',
            f'images={len(ip)}', f'queries={len(qp)}', f'ids={len(pids)}',
            f'cos_vision_mean={cv.mean():.5f}', f'cos_vision_min={cv.min():.5f}',
            f'cos_text_mean={ct.mean():.5f}', f'cos_text_min={ct.min():.5f}']
    txt = '\n'.join(out) + '\n'
    print(txt)
    Path(report).write_text(txt)


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'prep-vision':
        prep_vision(a[0], a[1], a[2], a[3], float(a[4]), int(a[5]))
    elif cmd == 'prep-text':
        prep_text(a[0], a[1], a[2], a[3], float(a[4]), int(a[5]))
    elif cmd == 'decode':
        decode(*a)
    elif cmd == 'latency':
        latency(*a)
    elif cmd == 'metrics':
        metrics(a[0], a[1], a[2], a[3], a[4], a[5], a[6], float(a[7]), int(a[8]),
                a[9], float(a[10]), int(a[11]))
    else:
        raise SystemExit('unknown command ' + cmd)
