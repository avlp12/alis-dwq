"""Manual format audit of the Bonsai 1-bit pack.

Stock mlx 0.31.2 cannot even dequantize bits=1 (no kernel), so this unpacks
the codes in numpy directly: MLX affine packing is little-endian within each
uint32 word — 32 one-bit codes per word, groups of 128 along the last axis,
value = code * scale + bias. Per tensor: code balance, projection agreement
vs sign(original), and shipped-scale ratio vs the analytic binary optimum
(mean|w| per group — for 2-level quantization the MSE-optimal levels are
±mean|w|, i.e. exactly the absmean rule).
"""
import glob
import json
import struct
import sys

import numpy as np

PACK = "/Users/gesicht/bonsai_audit/Bonsai-27B-mlx-1bit"
ORIG = "/Users/gesicht/bonsai_audit/Qwen3.6-27B-mlx"
GS = 128


def load_tensor(dirpath, name):
    """Read one tensor from a safetensors dir without mlx/torch."""
    for shard in sorted(glob.glob(f"{dirpath}/model*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            if name not in hdr:
                continue
            meta = hdr[name]
            off = 8 + n + meta["data_offsets"][0]
            size = meta["data_offsets"][1] - meta["data_offsets"][0]
            f.seek(off)
            buf = f.read(size)
            dt = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16,
                  "F32": np.float32}[meta["dtype"]]
            arr = np.frombuffer(buf, dtype=dt).reshape(meta["shape"])
            if meta["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
            return arr
    raise KeyError(name)


def unpack_1bit(wq):
    """(rows, words) uint32 -> (rows, words*32) codes in {0,1}, LSB-first."""
    bits = np.unpackbits(wq.view(np.uint8), bitorder="little", axis=-1)
    return bits.reshape(wq.shape[0], -1)


def audit(base):
    wq = load_tensor(PACK, base + ".weight")
    sc = load_tensor(PACK, base + ".scales").astype(np.float32)
    bi = load_tensor(PACK, base + ".biases").astype(np.float32)
    wo = load_tensor(ORIG, base + ".weight").astype(np.float32)
    codes = unpack_1bit(wq)                    # (rows, in_dim)
    rows, in_dim = codes.shape
    assert in_dim == wo.shape[-1], (in_dim, wo.shape)
    G = in_dim // GS
    cg = codes.reshape(rows, G, GS)
    # levels per group: {bias, bias+scale}
    lev0, lev1 = bi, bi + sc
    ones = cg.mean()
    # dequantized values, and level symmetry: binary {-s,+s} iff bias=-scale...
    sym = np.abs(lev0 + lev1) / np.maximum(np.abs(lev1 - lev0), 1e-12)
    # projection agreement vs sign(original): code 1 should mean the larger level
    wog = wo.reshape(rows, G, GS)
    pred = (wog > (lev0 + lev1)[..., None] / 2).astype(np.uint8)  # nearest-level
    agree = float((pred == cg).mean())
    sign_pred = (wog > 0).astype(np.uint8)
    agree_sign = float((sign_pred == cg).mean())
    # scale ratio vs analytic binary optimum: levels ±mean|w| around mean(w)
    # optimal 2-level (MSE) for group g: m ± mean|w - m|; compare half-spread
    m = wog.mean(-1)
    half_opt = np.abs(wog - m[..., None]).mean(-1)
    half_ship = (lev1 - lev0) / 2
    ratio = half_ship / np.maximum(half_opt, 1e-12)
    # code balance entropy
    p = np.clip(cg.mean(-1), 1e-9, 1 - 1e-9)
    ent = float((-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).mean())
    print(f"{base}")
    print(f"  ones={ones*100:.2f}%  group-entropy={ent:.4f}/1.0  "
          f"level-symmetry |l0+l1|/|l1-l0| med={np.median(sym):.4f}")
    print(f"  agree(nearest-level)={agree*100:.2f}%  agree(sign)={agree_sign*100:.2f}%  "
          f"scale-ratio med={np.median(ratio):.4f} ({np.percentile(ratio,10):.3f}..{np.percentile(ratio,90):.3f})")
    return agree, ent


if __name__ == "__main__":
    bases = [
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.model.layers.20.mlp.up_proj",
        "language_model.model.layers.40.mlp.gate_proj",
        "language_model.model.layers.60.mlp.down_proj",
        "language_model.model.layers.30.self_attn.q_proj",
        "language_model.model.layers.30.linear_attn.in_proj_a",
    ]
    ags, ents = [], []
    for b in bases:
        try:
            a, e = audit(b)
            ags.append(a); ents.append(e)
        except KeyError as e:
            print(f"skip {b}: {e}", file=sys.stderr)
    print(f"\naggregate: agree(nearest)={np.mean(ags)*100:.2f}%  "
          f"code-entropy={np.mean(ents):.4f}/1.0")
    print("AUDIT_1BIT_DONE")
