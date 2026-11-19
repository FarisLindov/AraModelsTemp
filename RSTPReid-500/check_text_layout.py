# Layout check for the Ara text DVM output. Needs numpy only, runs on the board or a PC.
#
# Usage (after the LAVA run, same arguments as the "metrics" step of lava_rstpreid500.py):
#   python3 check_text_layout.py DATASET_DIR TEXT_EMB_DIR DTYPE QSCALE ZP
#   e.g. python3 check_text_layout.py /root/rstpreid500/repo/RSTPReid-500 \
#            /root/rstpreid500/text_embeddings int8 0.123712 6
#
# It reads every text output (embedding_XXXX.bin), dequantizes it and tries several ways of
# reading the bytes back into a [77, 512] hidden-state tensor. For each way it takes the EOT
# row, applies W @ h + b, and reports the mean cosine vs the fp32 ONNX reference (ref/text_fp32.npy).
#   - If "77x512 (current)" is best and still low (< 0.9): the compiled DVM itself is the problem.
#   - If another layout is much higher (> 0.95): the DVM is fine, only the read-back was wrong.
import sys
from pathlib import Path

import numpy as np

L, D = 77, 512
DT = {'int8': np.int8, 'uint8': np.uint8}


def l2n(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def main():
    ds, tdir, dtype, qs, zp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
    ref = np.load(ds / 'ref' / 'text_fp32.npy')
    W = np.fromfile(ds / 'text' / 'text_projection_weight.f32', np.float32).reshape(D, D)
    b = np.fromfile(ds / 'text' / 'text_projection_bias.f32', np.float32)
    eot = np.load(ds / 'text' / 'eot_index.npy')
    files = sorted(tdir.glob('embedding_*.bin'))[:200]
    sizes = sorted({f.stat().st_size for f in files})
    print(f'{len(files)} outputs checked, byte sizes: {sizes} (expected {L * D} for int8 [77,512])')

    def raw(f, z):
        if dtype == 'float32':
            return np.fromfile(f, '<f4')
        return (np.fromfile(f, DT[dtype]).astype(np.float32) - z) * qs

    # Each layout maps the flat buffer to a [77, 512] array (None if the size does not fit).
    def layouts(v):
        n = v.size
        out = {}
        if n >= L * D:
            out['77x512 (current)'] = v[:L * D].reshape(L, D)
            out['512x77 transposed'] = v[:L * D].reshape(D, L).T
        # Padded rows, e.g. 77 rows padded to 80/96/128 or 512 columns padded.
        for rows in (80, 96, 128):
            if n == rows * D:
                out[f'{rows}x512 padded rows'] = v.reshape(rows, D)[:L]
                out[f'512x{rows} padded, transposed'] = v.reshape(D, rows).T[:L]
        if n % L == 0 and n // L > D:
            out[f'77x{n // L} padded cols'] = v.reshape(L, n // L)[:, :D]
        # Channel-tiled (NHWC-like blocks of 16 or 32 channels).
        for t in (16, 32, 64):
            if n == L * D:
                out[f'tiled {t}ch blocks'] = v.reshape(D // t, L, t).transpose(1, 0, 2).reshape(L, D)
        return out

    results = {}
    for z in sorted({zp, -zp}):
        for k, f in enumerate(files):
            for name, h in layouts(raw(f, z)).items():
                c = float(l2n(W @ h[eot[k]] + b) @ ref[k])
                results.setdefault((name, z), []).append(c)
    print(f'{"layout":32s} {"zp":>4s}  mean cos  min cos')
    for (name, z), cs in sorted(results.items(), key=lambda t: -np.mean(t[1])):
        print(f'{name:32s} {z:4d}  {np.mean(cs):8.4f} {np.min(cs):8.4f}')


if __name__ == '__main__':
    main()
