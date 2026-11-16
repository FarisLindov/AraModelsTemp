# Board-side helper for the LAVA ARA240 RSTPReid-500 TinyCLIP-61M benchmark. Needs numpy + Pillow only.
# Subcommands:
#   prep-vision DATASET OUTDIR DTYPE QSCALE ZP
#       PIL RGB -> BICUBIC stretch to 128x224 (H x W) -> (x - mean) * scale (CLIP, 0..255)
#       -> quantize -> CHW -> OUTDIR/input_XXXX.dat (order = image_set.txt)
#   prep-text DATASET OUTDIR DTYPE QSCALE ZP
#       token rows from emb_subset.f32 -> [77,512] -> quantize -> OUTDIR/input_XXXX.dat (order = gt captions)
#   decode LOG OUTDIR COUNT          base64 OUTPUT blocks in the nnapp log -> OUTDIR/embedding_XXXX.bin
#   latency LOG CSV TAG              TIMING rows -> stats + CSV
#   metrics DATASET VIS_DIR TXT_DIR REPORT VDTYPE VSCALE VZP TDTYPE TSCALE TZP
#       text side: dequant [77,512] -> EOT row -> W @ h + b. Text-to-image R1/R5/R10/mAP
#       on Ara, the fp32 ONNX reference, and cosine(Ara, fp32) per side.
# NOTE: for the text DVM the output scale DOES matter (the projection bias is added after
# dequantization), so TEXT_OUTPUT_QSCALE must match the "PARAM output=" line.
import base64
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

H, W, L, D = 128, 224, 77, 512
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
        v = np.frombuffer(raw, '<f4').astype(np.float32)
        assert np.all(np.isfinite(v)), 'NaN/Inf in fp32 output'
        return v
    return (np.frombuffer(raw, DT[dtype][0]).astype(np.float32) - zp) * qs


def prep_vision(ds, out, dtype, qs, zp):
    ds, out = Path(ds), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = (ds / 'image_set.txt').read_text().split()
    for k, n in enumerate(names):
        with Image.open(ds / 'images' / n) as im:
            x = np.asarray(im.convert('RGB').resize((W, H), Image.BICUBIC), np.float32)
        x = (x - MEAN) * SCALE
        q = np.ascontiguousarray(quant(x, dtype, qs, zp).transpose(2, 0, 1))
        assert q.shape == (3, H, W)
        q.tofile(out / f'input_{k:04d}.dat')
        if k < 3 or k == len(names) - 1:
            print(f'run={k:04d} {n} norm[{x.min():.3f},{x.max():.3f}] q[{q.min()},{q.max()}]')
    print('vision inputs:', len(names))


def prep_text(ds, out, dtype, qs, zp):
    ds, out = Path(ds), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    E = np.fromfile(ds / 'text' / 'emb_subset.f32', np.float32).reshape(-1, D)
    ids = np.load(ds / 'text' / 'token_ids.npy')
    sat = 0
    for k, row in enumerate(ids):
        x = E[row]
        q = np.ascontiguousarray(quant(x, dtype, qs, zp))
        if dtype != 'float32':
            sat += int((np.abs(x / qs) > DT[dtype][2]).sum())
        assert q.shape == (L, D)
        q.tofile(out / f'input_{k:04d}.dat')
    print(f'text inputs: {len(ids)}  emb range [{E.min():.4f},{E.max():.4f}]  saturated values: {sat}')


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
    app = [r[2] for r in rows if r[2] == r[2]]
    print(f'[{tag}] n={len(rows)} HW min {hw[0]:.4f} mean {statistics.mean(hw):.4f} '
          f'median {statistics.median(hw):.4f} p95 {hw[round(0.95 * (len(hw) - 1))]:.4f} max {hw[-1]:.4f} ms  '
          f'{1000 / statistics.mean(hw):.2f} IPS' + (f'  app mean {statistics.mean(app):.4f} ms' if app else ''))
    Path(csv).write_text('run,hw_ms,app_ms\n' + ''.join(f'{a},{b:.6f},{c:.6f}\n' for a, b, c in rows))


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def evaluate(S, img_pid, q_pid):
    match = img_pid[np.argsort(-S, axis=1)] == q_pid[:, None]
    first = match.argmax(1)
    r = {f'R{k}': 100 * float((first < k).mean()) for k in (1, 5, 10)}
    prec = np.cumsum(match, 1) / np.arange(1, match.shape[1] + 1)
    r['mAP'] = 100 * float(((prec * match).sum(1) / match.sum(1)).mean())
    return r


def metrics(ds, vdir, tdir, report, vdt, vqs, vzp, tdt, tqs, tzp):
    ds = Path(ds)
    gt = json.loads((ds / 'gt.json').read_text())
    pids = {p: k for k, p in enumerate(sorted({i['pid'] for i in gt['images']}))}
    img_pid = np.array([pids[i['pid']] for i in gt['images']])
    q_pid = np.array([pids[c['pid']] for c in gt['captions']])
    G = l2n(np.stack([dequant((Path(vdir) / f'embedding_{k:04d}.bin').read_bytes(), vdt, vqs, vzp)
                      for k in range(len(img_pid))]))
    Wp = np.fromfile(ds / 'text' / 'text_projection_weight.f32', np.float32).reshape(D, D)
    b = np.fromfile(ds / 'text' / 'text_projection_bias.f32', np.float32)
    eot = np.load(ds / 'text' / 'eot_index.npy')
    T = []
    for k in range(len(q_pid)):
        h = dequant((Path(tdir) / f'embedding_{k:04d}.bin').read_bytes(), tdt, tqs, tzp).reshape(L, D)
        T.append(Wp @ h[eot[k]] + b)
    T = l2n(np.stack(T))
    Gr, Tr = np.load(ds / 'ref' / 'vision_fp32.npy'), np.load(ds / 'ref' / 'text_fp32.npy')
    cv, ct = (G * Gr).sum(1), (T * Tr).sum(1)
    out = []
    for tag, r in (('ara', evaluate(T @ G.T, img_pid, q_pid)),
                   ('ara_vision_fp32_text', evaluate(Tr @ G.T, img_pid, q_pid)),
                   ('fp32_text_ara_vision_swap', evaluate(T @ Gr.T, img_pid, q_pid)),
                   ('fp32_onnx_ref', evaluate(Tr @ Gr.T, img_pid, q_pid))):
        out += [f'{tag}_{k}={v:.4f}' for k, v in r.items()]
    out += [f'images={len(img_pid)}', f'queries={len(q_pid)}', f'ids={len(pids)}',
            f'cos_vision_mean={cv.mean():.5f}', f'cos_vision_min={cv.min():.5f}',
            f'cos_text_mean={ct.mean():.5f}', f'cos_text_min={ct.min():.5f}']
    txt = '\n'.join(out) + '\n'
    print(txt)
    Path(report).write_text(txt)


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd in ('prep-vision', 'prep-text'):
        (prep_vision if cmd == 'prep-vision' else prep_text)(a[0], a[1], a[2], float(a[3]), int(a[4]))
    elif cmd == 'decode':
        decode(*a)
    elif cmd == 'latency':
        latency(*a)
    elif cmd == 'metrics':
        metrics(a[0], a[1], a[2], a[3], a[4], float(a[5]), int(a[6]), a[7], float(a[8]), int(a[9]))
    else:
        raise SystemExit('unknown command ' + cmd)
