"""--bits-override spec parsing, matching, and a tiny CPU end-to-end clip run
with a per-projection promotion (the shippable ds4-recipe asymmetry)."""
import json
import sys

import mlx.core as mx
import numpy as np
import pytest

from alis_dwq import clip_quantize as cq


def test_parse_specs():
    assert cq.parse_bits_override("down_proj=3") == ("down_proj", None, None, 3, None)
    assert cq.parse_bits_override("down_proj@L3-L6=3:gs64") == \
        ("down_proj", 3, 6, 3, 64)
    assert cq.parse_bits_override("down_proj@L4=3") == ("down_proj", 4, 4, 3, None)
    assert cq.parse_bits_override("switch_mlp.down_proj@L3-6=4") == \
        ("switch_mlp.down_proj", 3, 6, 4, None)


@pytest.mark.parametrize("bad", [
    "down_proj", "down_proj=1", "down_proj=7", "down_proj=3:gs48",
    "down_proj@L6-L3=3", "=3", "down_proj@=3",
])
def test_parse_rejects(bad):
    with pytest.raises(SystemExit):
        cq.parse_bits_override(bad)


def test_match_override():
    ov = [cq.parse_bits_override("down_proj@L3-L6=3"),
          cq.parse_bits_override("down_proj=4:gs32")]
    # in range: first match wins
    assert cq.match_override(ov, "model.layers.4.mlp.down_proj") == (3, None)
    # out of range: falls through to the unranged spec
    assert cq.match_override(ov, "model.layers.9.mlp.down_proj") == (4, 32)
    # ranged specs never match tensors without a layer index
    assert cq.match_override(ov[:1], "lm_head.down_proj") is None
    assert cq.match_override(ov, "model.layers.4.mlp.gate_proj") is None


def build_model_dirs(tmp_path, rows=16, in_dim=128):
    """Source (bf16 dump) + student (2-bit/gs64) with two FFN projections."""
    rng = np.random.default_rng(3)
    src_dir, st_dir = tmp_path / "src", tmp_path / "student"
    src_dir.mkdir(), st_dir.mkdir()
    names = ["model.layers.0.mlp.down_proj", "model.layers.0.mlp.gate_proj"]
    source, student = {}, {}
    originals = {}
    for name in names:
        w = mx.array(rng.normal(size=(rows, in_dim)).astype(np.float32))
        originals[name] = w
        source[name + ".weight"] = w.astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=2)
        student[name + ".weight"], student[name + ".scales"], \
            student[name + ".biases"] = q, s, b
    mx.save_safetensors(str(src_dir / "model.safetensors"), source,
                        metadata={"format": "mlx"})
    mx.save_safetensors(str(st_dir / "model.safetensors"), student,
                        metadata={"format": "mlx"})
    json.dump({"quantization": {"group_size": 64, "bits": 2},
               "quantization_config": {"group_size": 64, "bits": 2}},
              open(st_dir / "config.json", "w"))
    return src_dir, st_dir, originals


def run_clip(tmp_path, src_dir, st_dir, extra):
    out = tmp_path / "out"
    argv = ["clip_quantize", "--source", str(src_dir), "--model", str(st_dir),
            "--out", str(out)] + extra
    old = sys.argv
    sys.argv = argv
    try:
        cq.main()
    finally:
        sys.argv = old
    return out


def test_promotion_end_to_end(tmp_path):
    src_dir, st_dir, originals = build_model_dirs(tmp_path)
    out = run_clip(tmp_path, src_dir, st_dir,
                   ["--bits-override", "down_proj@L0=3"])
    got = mx.load(str(out / "model.safetensors"))
    in_dim = 128
    # promoted tensor packs at 3 bits, untouched projection stays 2-bit
    assert got["model.layers.0.mlp.down_proj.weight"].shape[-1] * 32 == 3 * in_dim
    assert got["model.layers.0.mlp.gate_proj.weight"].shape[-1] * 32 == 2 * in_dim
    # config records the per-module override in both quantization sections
    cfg = json.load(open(out / "config.json"))
    for key in ("quantization", "quantization_config"):
        assert cfg[key]["model.layers.0.mlp.down_proj"] == \
            {"bits": 3, "group_size": 64}
        assert "model.layers.0.mlp.gate_proj" not in cfg[key]
    # promotion buys reconstruction error, it doesn't just reshape
    def mse(name, bits):
        dq = mx.dequantize(got[name + ".weight"], got[name + ".scales"],
                           got[name + ".biases"], group_size=64, bits=bits)
        return float(((dq - originals[name]) ** 2).mean().item())
    st = mx.load(str(st_dir / "model.safetensors"))
    name = "model.layers.0.mlp.down_proj"
    base = mx.dequantize(st[name + ".weight"], st[name + ".scales"],
                         st[name + ".biases"], group_size=64, bits=2)
    base_mse = float(((base - originals[name]) ** 2).mean().item())
    assert mse(name, 3) < base_mse


def test_group_size_override(tmp_path):
    src_dir, st_dir, _ = build_model_dirs(tmp_path)
    out = run_clip(tmp_path, src_dir, st_dir,
                   ["--bits-override", "down_proj=2:gs32"])
    got = mx.load(str(out / "model.safetensors"))
    # gs64 -> gs32 doubles the scales resolution at unchanged packed width
    assert got["model.layers.0.mlp.down_proj.scales"].shape[-1] == 128 // 32
    cfg = json.load(open(out / "config.json"))
    assert cfg["quantization"]["model.layers.0.mlp.down_proj"] == \
        {"bits": 2, "group_size": 32}


def test_override_refuses_unrequantizable_match(tmp_path):
    src_dir, st_dir, _ = build_model_dirs(tmp_path)
    # drop down_proj from the source: the override target cannot requantize,
    # and a silent pass-through would ship the old bits
    src = mx.load(str(src_dir / "model.safetensors"))
    del src["model.layers.0.mlp.down_proj.weight"]
    mx.save_safetensors(str(src_dir / "model2.safetensors"), src,
                        metadata={"format": "mlx"})
    (src_dir / "model.safetensors").unlink()
    with pytest.raises(SystemExit) as e:
        run_clip(tmp_path, src_dir, st_dir,
                 ["--bits-override", "down_proj=3"])
    assert "cannot be requantized" in str(e.value)


def test_no_override_is_shape_preserving(tmp_path):
    src_dir, st_dir, _ = build_model_dirs(tmp_path)
    out = run_clip(tmp_path, src_dir, st_dir, [])
    got = mx.load(str(out / "model.safetensors"))
    st = mx.load(str(st_dir / "model.safetensors"))
    for k in st:
        assert got[k].shape == st[k].shape, k
    cfg = json.load(open(out / "config.json"))
    assert cfg["quantization"] == {"group_size": 64, "bits": 2}
