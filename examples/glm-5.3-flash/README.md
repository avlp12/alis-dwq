# Case study: GLM-5.3-Flash — the first affine and oQXe are not the same model

A decode-graph case, not a DWQ one. The first mlx-lm affine builds of
`zai-org/GLM-5.3-Flash` were deleted the night they were measured. What
survives is the comparison that made us delete them: **same host, same
oMLX 0.6.3, same stream metric, 8.4 tok/s vs 28 tok/s**. Bits were not
the remaining 3×. DWQ on that affine student would have trained the slow
graph.

Reusable rule: [docs/CHECKPOINT_GRAPH_NOT_BITS.md](../../docs/CHECKPOINT_GRAPH_NOT_BITS.md).

## The model

**zai-org/GLM-5.3-Flash** `@ 84c6a6aa9497188e15a635ba793b0f95a79b1033` —
`glm5_next`, 320B-A18B, 45 body layers + Lightning MTP 45, 288 routed
experts, hybrid KDA/DSA, VLM. Official FP8 on gesicht:
`/Volumes/Crucial X10/glm53flash/GLM-5.3-Flash-fp8` (62 shards,
328,337,455,672 B). Do not recopy; epsilon has the same bytes.

Host for every number below: gesicht, M3 Ultra 512 GiB, oMLX 0.6.3,
mlx 0.32.0, mlx-lm 0.31.3.

## Fair metric

oMLX HTTP SSE `usage.generation_tokens_per_second`. Not
`completion_tokens / wall`. Not a raw `language_model` / `generate_step`
loop (that path is **0.42 tok/s** on both trees and is not the engine
gap).

## Measured shelf

| Artifact | Convert | Serve tree | Stream decode | Receipt |
|---|---|---|---|---|
| Unaligned mlx-lm affine q4 | `stream_convert.py`, kimi `mlp.gate` 8-bit, KDA skipped | remapped into oMLX VLM | **5.69** tok/s | cited in `affine6_align_speed_verdict.json` (`omlx_glm53_q4_stream.json` lives under `~/glm53flash/prep/serving/logs/`) |
| Load-aligned affine q6 | same dump + dequant 42 routers + pack 405 leftover KDA GEMMs to 8-bit | oMLX VLM | **8.46 / 8.40** (two runs, p≈573 / 128) | [affine6_align_speed_verdict.json](affine6_align_speed_verdict.json), [omlx_affine6_align_stream.json](omlx_affine6_align_stream.json), [omlx_affine6_align_stream2.json](omlx_affine6_align_stream2.json) |
| Recipe-correct mlx-lm affine q6 / q4 | dense router, 8-bit KDA from convert | mlx-lm `generate` | **8.4 / 8.7** | campaign note; trees deleted |
| **oQ4e** | oMLX `quantize_oq_streaming` 4 + imatrix | native VLM | p512 **29.26** / p2048 **27.94** / p4096 **27.30** | [oq4e_decode_stream.json](oq4e_decode_stream.json) |
| **oQ6e** | same converter, 6 | native VLM | p512 **27.73** / p2048 **26.58** / p4096 **25.95** | [oq6e_decode_stream.json](oq6e_decode_stream.json) |
| **oQ8e** | same converter, 8 | native VLM | p512 **24.59** / p2048 **23.66** / p4096 **23.09** | [oq8e_decode_stream.json](oq8e_decode_stream.json) |
| **VLM q4 (no oQ)** | `stream_convert_vlm.py` + `make_predicate("q4")` | native VLM | p512 **29.46** / p2048 **28.24** / p4096 **27.50** | [vlm_q4_decode_stream.json](vlm_q4_decode_stream.json) |
| **VLM q6 (no oQ)** | `stream_convert_vlm.py` + `make_predicate("q6")` | native VLM | p512 **26.28** / p2048 **25.31** / p4096 **24.63** | [vlm_q6_decode_stream.json](vlm_q6_decode_stream.json) |

oQ6e is ~5–6% slower than oQ4e and still on the 26–28 shelf. That is the
bit-width effect once the graph is the VLM tree. oQ8e sits ~16% under
oQ4e at p512; p2048/p4096 dip under the old 24 tok/s gate.

**VLM q6 transfer (2026-08-30 08:23):** same convert, `make_predicate("q6")`,
no oQ. p512 **26.28** / p2048 **25.31** / p4096 **24.63**. ~5% under oQ6e
(27.73 / 26.58 / 25.95) and ~11% under VLM-q4. Still the VLM shelf, not
the mlx-lm 8.4 shelf. Warmup 48.61 (8 tokens) is not the fair metric.

**Transfer test (2026-08-30 02:29):** a convert that writes the VLM tree
with our predicate and **no** `quantize_oq_streaming` matches oQ4e
(29.46 / 28.24 / 27.50). The 28 tok/s shelf is the file layout, not the
oQ wrapper.

## Isolated expert codec (not the 3×)

From `affine6_align_speed_verdict.json`:

```
shape: E=288 H=4096 I=2048 topk=8 layers=42
eager_ms: 2→0.37  3→0.43  4→0.47  6→0.55  8→0.67
native_block_kind: null
```

6-bit vs 4-bit is ~1.2×. Compiled similar. Expert width does not explain
8.4 vs 28.

## Two mechanisms, stacked

### 1. Predicate mistakes (measured 1.5×, closed)

The first recipe mixed two borrowed rules that are each wrong for this
serve path:

- kimi/mlx-lm: quantize `mlp.gate` to 8-bit. Discrete top-k. oQ and
  `~/glm5.2/QUANT_PLAYBOOK.md` §0 leave it dense.
- zai FP8: skip every KDA tensor. oQ packs those GEMMs at 8-bit so
  `fuse_in` sees a uniform quantized set.

Load bandage (dequant 42 routers, write 405 leftover `self_attn.*.scales`
at 8-bit) moved 5.69 → 8.4. It did **not** close the rest.

### 2. Checkpoint graph (measured 3.3×, not closed by bits)

mlx-lm `LinearAttention` is three `ShortConv1d` + six unfused input
GEMMs and has no `fuse_in` / fused `conv1d` / `compile_ffn`.

oMLX VLM `Glm5NextLinearAttention` concatenates q/k/v/f_a/g_a/b into
one `quantized_matmul` when those six share `(bits, group_size, mode)`,
runs one grouped `conv1d`, and `Glm5NextDecoderLayer.compile_ffn`
wraps the 288-expert FFN in `mx.compile` at `B=1, S=1`.

oQXe writes that VLM tree. Our first convert wrote the mlx-lm tree and
then remapped keys at load. Same server, different file. The A/B below
shows `fuse_in` / `compile_ffn` are only a few percent on an oQ
checkpoint — they do not explain 8.4 vs 28.

`fuse_in` mixed-bit fallback is **not** the 3.3×: oQ6e's config only
listed 254 explicit 8-bit keys; many KDA leaves were packed at global
6-bit; stream stayed 26–28 tok/s. Forcing those leaves to 8-bit in the
load remap expected shape `(8192, 1024)` and died on `(8192, 768)`.
Infer bits from packed width; do not assume leftover KDA is 8-bit.

## A/B: `fuse_in` / `compile_ffn` on unchanged oQ4e (2026-08-30)

Same host, same oMLX 0.6.3 stream metric, same oQ4e file. Env flags
read at module init; logged at load in `oq4e_graph_ablate_server.log`.

| arm | fuse_in | compile_ffn | p512 | p2048 | p4096 | vs baseline p512 |
|---|---|---|---|---|---|---|
| baseline | on | on | **28.48** | 27.25 | 26.43 | 1.00 |
| no_fuse_in | off | on | 27.90 | 26.75 | 26.03 | 0.98 |
| no_compile_ffn | on | off | 26.84 | 25.85 | 25.21 | 0.94 |
| neither | off | off | **26.39** | 25.37 | 24.74 | 0.93 |
| aligned affine q6 (earlier) | on (class default) | on (class default) | **8.4** | — | — | 0.29 |

`qwen35` gate+up fusion does not attach to `glm5_next`. Both flags off
is still **3.1×** the aligned affine. Remapped affine already had both
flags on and stayed at 8.4. **These two switches are not the 3.3×.**

Receipts: [oq4e_graph_ablate.json](oq4e_graph_ablate.json),
[oq4e_ablate_baseline.json](oq4e_ablate_baseline.json),
[oq4e_ablate_no_fuse_in.json](oq4e_ablate_no_fuse_in.json),
[oq4e_ablate_no_compile_ffn.json](oq4e_ablate_no_compile_ffn.json),
[oq4e_ablate_neither.json](oq4e_ablate_neither.json).

## What this means for ALIS / DWQ

- Ship format for this model is a **native VLM tree** (fused `conv1d`,
  `language_model.*` keys, stacked `switch_mlp`), not mlx-lm affine.
  `stream_convert.py` (mlx-lm keys) is not a serve artifact.
  `stream_convert_vlm.py` is: VLM q4 hit the oQ4e shelf without oQ.
  Hub stubs `avlp12/GLM-5.3-Flash-Alis-MLX-{4,6,8}bit` stay withdrawn
  until a VLM-tree artifact is labeled `runtime-verified` for quality,
  not just tok/s.
- Do not DWQ the deleted affine student. Student is VLM-q4 (RTN
  baseline). Teacher, if any, is oQ8e (stream done 2026-08-30 02:05).
- Changing only the predicate on an mlx-lm export will replay the 8.4
  shelf.
- Bit allocation (this repo's usual work) is a quality/size lever on
  top of that tree. It is not how you buy the 3×.

## Adopt / reject

| Move | Verdict |
|---|---|
| Serve mlx-lm affine GLM-5.3 | **Reject.** 8.4 tok/s after the load bandage. Trees deleted. |
| DWQ that affine student | **Reject.** Distillation does not compile a different graph. |
| oQ4e as the 4-bit serve baseline | **Adopt.** Stream 27–29 tok/s. Keep intact. |
| VLM q4 (no oQ) as *our* 4-bit serve baseline | **Adopt.** Stream 27–29 tok/s. Same shelf as oQ4e. Label `baseline`. |
| VLM q6 (no oQ) as *our* 6-bit serve artifact | **Adopt for stream.** 26.28 / 25.31 / 24.63. ~5% under oQ6e, VLM shelf. Label `baseline-converted`. |
| oQ6e as the 6-bit serve / teacher-gate | **Adopt.** Stream 26–28 tok/s. Gate passed. |
| oQ8e as DWQ teacher | **Adopt for stream.** 24.59 / 23.66 / 23.09. Not a quality number. |
| Skip KDA to “protect quality” | **Reject for this serve path.** Cost the 1.5× bandage. Recurrent state stays dense; KDA GEMMs do not. |
| Quantize `mlp.gate` | **Reject.** Router. |

## Raw receipts (this directory)

| File | What |
|---|---|
| [affine6_align_speed_verdict.json](affine6_align_speed_verdict.json) | 5.69 → 8.4, gather_qmm microbench, 0.42 unfair loop, leftover 3.3× |
| [omlx_affine6_align_stream.json](omlx_affine6_align_stream.json) | aligned affine6 run 1: 8.46 tok/s |
| [omlx_affine6_align_stream2.json](omlx_affine6_align_stream2.json) | aligned affine6 run 2: 8.40 tok/s |
| [oq4e_decode_stream.json](oq4e_decode_stream.json) | oQ4e p512/p2048/p4096 |
| [oq6e_decode_stream.json](oq6e_decode_stream.json) | oQ6e p512/p2048/p4096 |
| [oq8e_decode_stream.json](oq8e_decode_stream.json) | oQ8e p512/p2048/p4096 |
| [vlm_q4_decode_stream.json](vlm_q4_decode_stream.json) | VLM q4 no-oQ p512/p2048/p4096 |
| [vlm_q6_decode_stream.json](vlm_q6_decode_stream.json) | VLM q6 no-oQ p512/p2048/p4096 |
| [oq4e_graph_ablate.json](oq4e_graph_ablate.json) | fuse_in / compile_ffn A/B summary |
| [oq4e_ablate_*.json](oq4e_ablate_baseline.json) | per-arm stream logs |

Working campaign notes (not copied; local to the port repo):
`local-llm-serving/ports/glm53flash-mlx/quant/{RECIPES,CAMPAIGN_2026-08-29}.md`.
