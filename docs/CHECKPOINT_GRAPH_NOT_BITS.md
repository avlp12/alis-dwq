# Checkpoint graph is not bit-width

*A model-agnostic rule from the GLM-5.3-Flash campaign (2026-08-29/30).
Receipts: [examples/glm-5.3-flash](../examples/glm-5.3-flash/README.md).*

## The problem

Two affine g64 checkpoints of the same official FP8 source, served on the same
oMLX 0.6.3 process on the same M3 Ultra, decoded at **8.4 tok/s** and
**28 tok/s**. The slow one was our first mlx-lm convert. The fast one was
first seen as oMLX oQe; a later convert that writes the same VLM tree
without the oQ wrapper matched it (29.46 / 28.24 / 27.50). Affine q4, q6,
and q8 all sat on the slow shelf. Raising leftover
KDA to 8-bit and dequanting routers at load bought **1.5×** (5.69 → 8.4) and
stopped. Isolated `gather_qmm` 4-bit vs 6-bit is **~1.2×**.

If you treat that gap as “our bits are worse” you will spend a DWQ campaign
tuning scales on a student that cannot take the compiled serve path.

## What is not the difference

| Hypothesis | Why it looked plausible | What was measured | Verdict |
|---|---|---|---|
| Lower bits are slower | 4-bit file is smaller, so maybe more kernel overhead | Affine q4/q6/q8 same tok/s; oQ4e 27–29 vs oQ6e 26–28 | Discard |
| Expert codec (4 vs 6) is the leftover 3× | MoE is ~97% of bytes | `gather_qmm` E=288 H=4096 I=2048 topk=8: 0.47 ms @4b vs 0.55 ms @6b | Discard |
| DWQ / ALIS / imatrix is why oQ is fast | oQe collects an imatrix | oQ4e is still affine g64. No DWQ. VLM q4 with no oQ / no imatrix matches oQ4e | Discard |
| `fuse_in` mixed-bit fallback is the 3× | `Glm5NextLinearAttention._fused_in_proj` splits when bits differ | oQ6e left many KDA leaves at global 6-bit; stream still 26–28 tok/s | Discard as the 3×; keep as a ~5% / load-contract item |
| Raw `language_model` loop (~0.42 tok/s) is the engine gap | It is 66× slower than oQ stream | Same unfair loop is 0.42 on *both* trees. Fair metric is oMLX HTTP `generation_tokens_per_second` | Discard as a comparison |

## What actually differs

oQXe is not a different weight codec. Inside the file it is still MLX affine
(`mode=affine`, group 64, packed uint32 + scales + biases). The product
difference is **which module tree the convert writes, and therefore which
compiled kernels the server is allowed to attach**. The oQ wrapper is not
required once that tree is on disk.

```
mlx-lm LinearAttention (first build)
  q_proj, k_proj, v_proj          → 3 GEMMs
  q_conv, k_conv, v_conv          → 3 ShortConv1d
  forget / g_a / b / g_b / o      → more unfused GEMMs
  DecoderLayer.mlp                → eager SwitchGLU
  no fuse_in, no compile_ffn

oMLX VLM Glm5NextLinearAttention (oQXe)
  q/k/v/f_a/g_a/b                 → one concat + one quantized_matmul
                                  (only if all six share bits, group, mode)
  one grouped conv1d              (sanitize will also fuse q/k/v conv weights
                                  if the file still has them split)
  DecoderLayer.compile_ffn        → mx.compile(_ffn_block) at B=1, S=1
```

The first convert also wrote a **predicate that the fast graph cannot use**:

- kimi-style `mlp.gate` → 8-bit. Discrete top-k. oQ and the Playbook leave
  the router dense. Load-time dequant recovered this.
- zai-FP8-style “skip every KDA tensor”. Those leaves stayed bf16, so
  `fuse_in` saw a mixed quantized/dense set and fell back every token.
  Packing the leftovers to 8-bit at load recovered 1.5× and then plateaued.

After that bandage, the remaining **3.3×** (8.4 vs 28) is still the same
engine against two checkpoints. The *existence* of that gap is measured.

A/B on unchanged oQ4e (2026-08-30, same stream metric, flags logged at
load): `fuse_in=0` is **−2%**, `compile_ffn=0` is **−6%**, both off is
**−7%** (p512 28.48 → 26.39). Both-off is still **3.1×** aligned affine
8.4. Remapped affine already ran with both flags **on** (class defaults)
and stayed at 8.4. Those two Python switches are not the leftover 3.3×.

oQ6e is the complementary fact on bits: many KDA leaves packed at global
6-bit still streamed 26–28 tok/s. Uniform KDA-8 is a fuse/load contract,
not the 28 tok/s shelf.

## Rules for later quants

1. **Name the serve artifact first.** The student DWQ will train is the
   file the server will load. An mlx-lm affine export that the server only
   accepts after remap is a different model class, even if every tensor
   is “the same bits”.
2. **First affine is `baseline` only when it is that serve artifact.**
   Otherwise it is a diagnostic dump. Do not Hub-label it ALIS/DWQ, and
   do not start DWQ on it to “catch” a 28 tok/s teacher.
3. **DWQ tunes scales/biases. It cannot compile a different graph.**
   Distillation will not turn three ShortConvs into `fuse_in` +
   `compile_ffn`.
4. **Price fused members as a set.** A concat-`quantized_matmul` path
   requires one `(bits, group_size, mode)` for every leaf it concatenates.
   Promoting one of them and leaving the others at the global width is a
   load bug (shape `(N, 1024)` vs `(N, 768)`), not a quality win.
5. **Do not skip the per-token path to “save quality bits”.** On this
   model KDA/indexer at 8-bit is a few GB on a 180–340 GB expert file.
   Skipping them was a quality guess borrowed from the FP8 source. It
   broke the fuse contract and cost the 1.5× bandage. Recurrent state
   (`A_log`, `dt_bias`, conv, mHC, norms) stays dense; the GEMMs do not.
6. **Router stays dense.** `mlp.gate` is top-k, not a matmul you want
   in the quantized lattice. `mlp.gate_proj` is an expert SwiGLU leaf.
   Do not confuse the two.
7. **Measure decode on the serve stream.**
   `usage.generation_tokens_per_second` on oMLX HTTP SSE. Not
   `completion_tokens / wall` (mixes prefill). Not a raw
   `language_model` loop (~0.42 tok/s on this model). Not
   `mlx_vlm.generate_step`.

## Transfer test (closed 2026-08-30)

`stream_convert_vlm.py` writes the VLM tree with `make_predicate("q4")`
and no `quantize_oq_streaming`. Same host, same oMLX 0.6.3 stream
metric: p512 **29.46** / p2048 **28.24** / p4096 **27.50**. oQ4e was
29.26 / 27.94 / 27.30. Receipt:
[vlm_q4_decode_stream.json](../examples/glm-5.3-flash/vlm_q4_decode_stream.json).

We can emit the fast graph ourselves. Remaining open items are quality
(KL / PPL vs oQ4e / oQ8e), not tok/s.

Still true, and not the 3.3×:

- **Falsified (2026-08-30):** `fuse_in` and `compile_ffn` as the 3.3×.
  Receipts in [examples/glm-5.3-flash](../examples/glm-5.3-flash/README.md).
  `qwen35_moe_gate_up` does not attach to `glm5_next` (family token miss
  and `SwitchGLU` class miss) — that path is not in this comparison.

## What the leftover 3.3× actually is (2026-08-30)

After VLM load, remapped mlx-lm affine and a native VLM file instantiate
the **same Python classes**: `Glm5NextLinearAttention`,
`omlx.patches.deepseek_v4.switch_layers.SwitchGLU`,
`QuantizedSwitchLinear` → `mx.gather_qmm`. Affine6 logged
`L3.switch=QuantizedSwitchLinear`. A 288-expert Python loop is not the
gap. Decode `topk=8` has `indices.size=8 < 64`, so `do_sort` is false
and the DeepSeek pair/block kernels do not fire either.

The 3.3× is the **operand the compiled graph is allowed to see**.

Isolated `gather_qmm` on a freshly quantized contiguous 3D bank
`(E=288, I=2048, topk=8)` is 0.47 ms @4b. Budget: 42 MoE layers × 3
projections × 0.47 ms ≈ 59 ms plus attention → the 28 tok/s shelf.
The same primitive on a strided or 2D-quantized-then-stacked bank is
the slow shelf. oMLX's own Inkling path states this directly:
`gather_mm / gather_qmm are much slower when fed strided views`.

What the VLM convert writes that mlx-lm convert does not (or writes in
a form sanitize cannot make contiguous):

- stack **bf16** experts, then `mx.quantize` the 3D tensor once →
  `switch_mlp.*.weight` U32 `[288, 2048, 512]`, scales `[288, 2048, 64]`
  in 126 dedicated shards (VLM-q4 / oQ4e headers match bit-for-bit)
- fused `conv1d` already `[24576, 4, 1]` bf16, not three PyTorch
  `[8192, 1, 4]` leaves
- `language_model.*` keys so `nn.quantize` hits the modules it expects
- drop HF layer 45 (MTP)

Epsilon leftovers (`mlx-4bit-quasar`) already have stacked
`switch_mlp` at the same `[288, 2048, 512]`, but still 102 split convs
and `model.*` keys. Official deleted affine6 was 354 shards / 1152
files vs VLM 174 — that shard count is the 2D-per-expert packing
signature. `compile_ffn` then compiles whichever graph the file built:
eager VLM is already ~26 tok/s (`fuse_in=0 compile_ffn=0` = 26.39);
compiled remapped affine stays 8.4 because it compiled a slow operand.

VLM-q6 (no oQ) closed the bit-width check on this tree: 26.28 / 25.31 /
24.63 vs oQ6e 27.73 / 26.58 / 25.95. Bits move ~5–11% here. The graph
moves 3×.
