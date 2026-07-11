# Case study: the MLX floor — 2-bit/gs128 experts on GLM-5.2 (2.32 bpw), with a teacher A/B

**Question.** How low can the MLX affine container usefully go on a 745B MoE, and at that floor, does a sharper teacher still lose to a closer one? (Third point for the teacher-precision / student-capacity curve: 3.5 bpw gained from an 8-bit teacher, 2.56 bpw lost.)

**Build ("E1").** Take the shipped 2.56-bpw recipe and change exactly one thing: routed-expert modules go 2-bit/gs64 → **2-bit/gs128** (2.5 → 2.25 bpw effective on the 724.8B expert params). Routers/norms/attention/shared expert unchanged. Source was the public Q8 checkpoint (`pipenetwork/GLM-5.2-MLX-8bit@531a2ab`, dequantize-then-requantize, streamed shardwise) because no bf16 copy was on disk — **from-Q8 is a disclosed second variable** vs the from-bf16 2.56/3.5 baselines; Q8's ~38 dB SNR headroom over 2-bit noise makes it a reasonable stand-in, and gs64 affine preserves each group's min/max so the outliers that set 2-bit scales survive. Result: **46 shards, 215.81 GB dec / 200.99 GiB, 2.3225 bpw measured** over 743,377,019,904 params, tokenizer byte-identical to golden (sha `19e77364…`).

Two gates ran before any big spend, and both earned their keep:

- **Pre-download probe** (no 790 GB pulled if it fails): dequantize sampled routed-expert tensors from a local build, requantize at gs64 vs gs128, compare relative reconstruction error. Measured ratio ≈ **1.07–1.08** median (early/mid/late layers) against a ≥2.0 kill rule — gs128 is nearly free at the tensor level.
- **G1 raw-quality gate caught a real artifact bug**: the first E1 emitted Q8-packed MLA `QuantizedMultiLinear` weights under a 4-bit config (the requant path missed that module class) and died at the first forward. Cost: minutes, because the gate runs before anything downstream. Rebuild passed.

**DWQ A/B.** Both arms started from the same E1-raw, layerwise K=6 (13 rounds), 145/32 seq-512 samples, batch 1, lr 1e-6, seed 7 — the exact recipe of the shipped runs. Only the target dump differs:

```bash
# arm A — 4.5-bpw teacher targets
PYTHONPATH=~/alis-dwq ALIS_DWQ_LAYERS_PER_ROUND=6 ALIS_DWQ_DATA_DIR=/Users/Shared/glm5.2/dwq_data \
python -m alis_dwq.run --model builds/GLM-5.2-E1-2.3bpw --quantized-model builds/GLM-5.2-E1-2.3bpw \
  --target-dir dwq_targets_glm512 --mlx-path builds/GLM-5.2-E1-2.3bpw-DWQ-A45 \
  --num-samples 145 --max-seq-length 512 --batch-size 1 --grad-checkpoint --learning-rate 1e-6 --seed 7

# arm B — 8-bit teacher targets (reused dump; the teacher never loaded again)
… --target-dir /Users/Shared/glm5.2/dwq_targets_glm8bit512 --mlx-path builds/GLM-5.2-E1-2.3bpw-DWQ-B8 …
```

Pre-registered before either arm finished: prediction = A wins; primary metric = teacher-independent strided PPL (wikitext/code) + tulu; KL-vs-4.5-ref is structurally **pro-A** (A's teacher *is* the reference), per-teacher valid loss is structurally **pro-B-looking** and never cross-comparable.

**Training behavior.** Initial valid vs the same 8-bit targets ordered exactly by student capacity — E1 0.5840 > 2.56 0.5125 > 3.5 0.1776 — a free alignment tripwire. Arm A: 13/13 rounds accepted, 0.5235 → 0.3355 (−35.9%). Arm B: rounds 1–11 accepted, 12–13 REVERTED (consecutive-revert natural stop), 0.5840 → 0.3682 (−36.9%). ~21 min/round, peak 263 GB, ~4.6 h/arm on one 512 GiB M3 Ultra.

## Results (ctx 2048 / stride 1024; tulu 50×2048 seed 123; KL 3-slice T=3072)

| | E1-raw 2.32 bpw | **E1-DWQ-A45** (4.5-t) | E1-DWQ-B8 (8-bit-t) | 2.56 bpw main (from-bf16) |
|---|---|---|---|---|
| wikitext PPL | 4.7109 | **4.0364** (−14.3%) | 4.0870 (−13.2%) | 3.774 |
| code PPL | 2.2715 | **2.1221** (−6.6%) | 2.1395 (−5.8%) | 2.069 |
| tulu PPL | 3.963±0.031 | **3.660±0.028** (−7.6%) | 3.687±0.028 (−7.0%) | – |
| KL vs 4.5-ref *(pro-A aux)* | 0.7411 | **0.5099** | 0.5537 | 0.379 |
| ZH-slice KL | 1.1541 | **0.7927** | 0.8887 | 0.562 |
| per-teacher valid *(not cross-comparable)* | – | 0.5235→0.3355 | 0.5840→0.3682 | – |

## Findings

1. **The floor is usable, not free.** 2-bit/gs128 experts + DWQ lands at wikitext 4.036 / code 2.122 — +7.0% / +2.6% over the 2.56-bpw build for −26.6 GB (215.8 vs 242.4 GB dec). The tensor-level ~7% error premium of gs128 showed up as roughly that much end-to-end raw-PPL premium; DWQ recovered −14.3% wikitext on top. Decode format is unchanged by DWQ (scales/biases only), so raw's measured 23.26 tok/s carries.
2. **Sweet-spot, third point: the closer teacher won everything.** A45 beat B8 on wikitext (−1.24%), code (−0.81%), tulu (−0.73%), and both aux KL views — same-token paired comparisons, consistent direction everywhere. Below 2.56 bpw the sharper teacher stays inferior; but unlike the 2.56 *re-tune* incident, B8 was not destructive here (−13.2% vs raw) — trained from the same raw student, it's just consistently ~1% behind. A sharper teacher at low capacity wastes capacity, it doesn't necessarily poison.
3. **Valid loss stayed a process metric, not a verdict metric.** B's valid dropped *more* (−36.9% vs −35.9%) while it lost every held-out metric — the same overfit-toward-teacher signature that produced the 2.56 reversal, now reproduced with a clean paired design.
4. **The gates were the story, operationally.** Pre-probe spared a possible dead 790 GB download; G1 caught a malformed artifact for the cost of one forward; the initial-valid ordering check would have caught target corruption (as it did in the 2026-07-09 incident); and one arm launch died to the documented jetsam trap (relaunched after the reclaim-stability gate — see the memory-gate note in the main README, now validated the hard way *twice*).
5. **Instruction-following at this bpw needs its own eval.** Raw-E1 generation ran clean but leaked English analysis scaffolding instead of following the requested language/format; PPL doesn't measure that. The staged card carries the caveat; a generation-quality pass is future work.

**Ship decision.** A45 staged to HF as an experimental build (private until flipped); B8 kept local; E1-raw preserved for recipe iteration. Marginal value over the 2.56 build is a 256 GB-box KV-headroom trade, not a new hardware tier — see the Floor-notes section in the main README for why nothing below ~220 GB ships for this model anywhere today.
