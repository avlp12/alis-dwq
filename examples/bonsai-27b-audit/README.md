# Bonsai 27B: audit + method reverse-engineering protocol

PrismML ships binary/ternary Qwen3.6-27B ([whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf))
claiming 94.6% FP16 retention at 1.71 bpw — post-training, end-to-end
(embeddings through LM head), method proprietary. Both the **output**
([prism-ml HF collections](https://huggingface.co/collections/prism-ml/bonsai-27b),
Apache-2.0) and the **input** (Qwen3.6-27B, public) are downloadable, so the
transformation can be characterized from its endpoints. Analyzing lawfully
obtained, permissively licensed artifacts is standard model forensics; what
follows recovers the method's *class*, not its code.

## Provenance (already narrows the hypothesis space)

PrismML's CEO/founder is **Babak Hassibi (Caltech)** — co-author of
**Optimal Brain Surgeon** (1993), the Hessian-compensation framework that
GPTQ/OBC descend from; co-founders include a second-order-optimization
researcher. "Years of mathematical theory" + this lineage makes the prior
hypothesis: **curvature-compensated ternarization** — quantize a weight to
{−s, 0, +s}, update remaining weights to absorb the error under a
(layer-wise) Hessian metric — possibly followed by short distillation.
Public analogs to diff against: PT²-LLM, PTQTP, TWLA, BitNet-Distillation.

## Tier 1 — claims audit (any Mac, the MLX ternary pack is 8.49 GB)

- `code_entropy --model <bonsai-mlx>`: the pack rides MLX 2-bit affine with
  only 3 levels used → utilization ceiling 1.585/2 = 79%. Reported per-layer
  entropy + level histogram gives the **zero fraction** (ternary sparsity)
  per tensor — a core fingerprint of the objective (fixed threshold vs
  per-group optimized support show different sparsity spreads).
- `eval_kld` vs FP16 Qwen3.6-27B (`--save-ref` on FP16, `--ref` on Bonsai):
  the first token-level KL/flip numbers for these models — the whitepaper
  publishes only sampled, judge-scored benchmarks (with temp 1.0 vs 0.7
  asymmetry). `--loop-probe 256` for degeneration.
- `--kv-probe 4`: reproduce their KV-tolerance table on our harness; then
  run the same probe on our own DWQ'd students (the transferable hypothesis:
  discretization-shaped models absorb cache noise — if true for E1, that's
  a ~4× long-context memory cut we can put on the model card).

## Tier 2 — method forensics (needs Bonsai + original weights side by side)

The instrument ships here and its discrimination is validated on synthetic
ground truth (analytic TWN projection / toy GPTQ-style column compensation /
rotated-basis ternarization → three distinct verdicts):

```bash
python -m alis_dwq.weight_forensics \
  --original <Qwen3.6-27B-mlx> --transformed <Ternary-Bonsai-27B-mlx> \
  --pattern "mlp" --max-tensors 40
```

Per tensor it reports: best-k projection agreement, head-vs-tail column
drift (compensation fingerprint), singular-spectra vs elementwise weight
correlation (rotation test), shipped-vs-closed-form scale ratio, and
4th-level usage (nonzero usage falsifies a plain "ternary" label). The
tests, spelled out:

1. **Projection test.** Compute the closed-form ternary projection of the
   original (TWN-style: support `|w| > Δ`, `s = mean|w|` over support; sweep
   Δ) and measure code agreement with Bonsai's codes. **High agreement
   (>95%) ⇒ direct rounding** (the theory is in scale/threshold choice);
   **low ⇒ the weights themselves were changed** (compensation or training).
2. **Compensation signature.** OBS/GPTQ-family methods process columns in
   order, later columns absorbing earlier errors: plot per-column agreement
   with the naive projection vs column index. A drift with position is the
   compensation fingerprint; flat disagreement points to gradient training.
3. **Rotation test.** If codes decorrelate from `W_o` but quality holds,
   test an orthogonal-transform hypothesis: singular-value spectra are
   rotation-invariant, so `svdvals(W_b) ≈ svdvals(W_o)` with low elementwise
   correlation ⇒ rotated basis; spectra that moved ⇒ retraining.
4. **Scale objective.** Compare shipped group scales against the
   MSE-optimal closed form for the shipped codes (`s* = ⟨|w|⟩ over support`)
   — exact match ⇒ analytic scales; systematic deviation ⇒ scales were
   trained (distillation), and the deviation size says how much.
5. **Residual bookkeeping.** RMSNorm gains and any FP16 tail vs original —
   scale folding shows up here; so would hidden auxiliary capacity.
6. **KV-tolerance mechanism.** Their models' unusual cache tolerance
   suggests flattened activation distributions; capture per-layer activation
   kurtosis on both models (one forward, `_capture_hiddens`) — flattening
   without weight-basis rotation would indicate noise-injection training.

## What this can and cannot recover

Recoverable: rounding vs compensation vs training; rotated vs original
basis; analytic vs learned scales; sparsity objective; whether distillation
touched the weights. Not recoverable: the exact Hessian approximation,
schedules, calibration data, or any patented formulation — for those, watch
for the Caltech patent publication (18-month window from filing) and the
founders' arXiv trail.

## If the fingerprints confirm the prior

The open reproduction is within this repo's existing machinery: ternary
projection of the source (test 1's code), layerwise distillation with
straight-through re-projection between rounds (the layerwise trainer with a
projection step where clip_quantize's requantize sits), gated by eval_kld +
the degeneration probes. That is a research arc, not a weekend — but every
piece (projection, layerwise rounds, rollback, gates) already exists here.

## Tier 1 results (measured 2026-07-15, M3 Ultra 512 GB, stock mlx-lm 0.31.3 / mlx 0.31.2 Metal)

Pack: `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (8.49 GB, 2-bit/gs128 affine,
26.9 B language params scanned; vision tower is HQQ and not affine —
excluded). Reference: `Qwen/Qwen3.6-27B` bf16 (53.8 GB), loaded by stock
`mlx_lm`. The pack **runs on stock mlx-lm** — the custom kernels are a speed
path, not a format requirement.

**Code histograms — the "ternary" label is true.**

- 4th-level usage: **exactly 0 across all 498 tensors** (no `code 3`
  anywhere). No high-precision escape hatches in the language weights.
- Global level fractions: **35.2% −1 / 29.7% 0 / 35.1% +1** — near the
  max-entropy point for 3 levels. Effective **1.58 b/w of the nominal 2.00
  (79% — the ternary ceiling log₂3/2 = 79.2%)**, uniform across all 64
  layers; per-tensor entropy min 1.551 / median 1.581 / max 1.585.
- Zero fraction is *tight*: median 0.299, std 0.013 across tensors —
  a fixed-threshold-like criterion, not per-group-optimized support
  (which would spread wider). `linear_attn.in_proj_a/b` are the outliers
  (0.234–0.341) and also the lowest-utilization tensors (77.5–78.5%).
- Low-entropy groups: **0.0% everywhere** — no stretched-grid pathology at
  all, i.e. nothing for `clip_quantize` to fix; the grids are already
  min-MSE-ish over 3 levels. (Tool note: this scan exposed and fixed a
  float32 histogram-saturation bug in `code_entropy` — counts froze at 2²⁴
  per bin and briefly cosplayed as an exact-⅓ top-k fingerprint. int64
  accumulation now; re-measured numbers above.)

**First token-level KL/flip vs FP16** (3,072-token EN/code/ZH slice,
KL(ref‖cand), T=3072):

| slice | KL | top-1 flip |
|---|---|---|
| EN | 0.580 | 21.0% |
| code | 1.324 | 22.8% |
| ZH | 1.325 | **45.6%** |
| overall | **1.077 ± 0.040** | 29.8% |

The non-English damage concentration we measured on two MoE families holds
on this dense hybrid too: ZH flips nearly half its tokens while the
benchmark-facing EN slice looks mildest. The whitepaper's benchmark-only
evidence (temp-asymmetric, judge-scored) would not show this.

**Degeneration probe (greedy 256 from 64-token raw-text prompts): Bonsai
loops on every slice; FP16 is clean on the identical probe.**

| | EN | code | ZH |
|---|---|---|---|
| FP16 distinct-4gram / cycle | 0.632 / none | 0.941 / none | 0.996 / none |
| Bonsai distinct-4gram / cycle | **0.134 / len-13** | **0.411 / len-2** | **0.079 / len-18** |

That is the REAP signature (eval parity, loop rate up) in an extreme form —
consistent with their own IQ2_XXS AIME-collapse observation that long-form
behavior dies before short-form metrics notice. Caveat: raw-text greedy
continuation is out-of-distribution for an instruct-tuned base; but the
FP16 control on the same probe is what isolates the quantization as the
cause. Their published 80.49/15-benchmark average was sampled at their
serving temperature through chat templates — both can be true.

**Temperature-matched control (run 2026-07-15, after the above): the hard
cycles are greedy-specific.** At the model's own serving distribution
(generation_config: T=1.0, top-k 20, top-p 0.95; 5 seeded samples per
slice, `--loop-temp`): **Bonsai 15/15 cycle-free** — the anticipated
"it doesn't loop at serving temperature" rebuttal is *confirmed*, and the
greedy result must be quoted with that qualifier. What persists at
temperature is a heavier low-diversity tail: Bonsai's worst samples fall to
distinct-4gram 0.265 (EN) and 0.589 (code) while the FP16 control's worst
is 0.660 (means 0.839 vs 0.875, n=15 each) — consistent with sharpened
output distributions, short of degeneration. Practical reading: greedy or
near-greedy decoding regimes (the default for math/code evaluation
harnesses, constrained decoding, some agent stacks) are where this build
degenerates; sampled chat is substantially safer.

**KV-tolerance: reproduced, 56×.** Self-KL of a 4-bit/gs64 KV cache vs each
model's own FP16-KV run (16 of 64 layer caches quantizable on this hybrid —
linear-attention states stay FP16, so both rows are lower bounds):

| | EN | code | ZH | overall | flip |
|---|---|---|---|---|---|
| FP16 weights | 0.0458 | 0.4148 | 0.0261 | 0.1622 | 5.1% |
| Bonsai ternary | 0.0030 | 0.0029 | 0.0028 | **0.0029** | 2.2% |

**56× lower self-KL on the ternary build in aggregate — but the per-slice
ratios are 15× (EN) / 145× (code) / 9.4× (ZH)**, i.e. the aggregate is
dominated by the FP16 model's anomalous code slice (0.415 vs 0.03–0.05
elsewhere), and two of three slices fall *outside* PrismML's 12–95× band
(3-lens review catch). The honest statement: the tolerance phenomenon
reproduces strongly and in the claimed direction, with slice-dependent
magnitude 9–145×. Running the same probe on our own DWQ'd students is §3b
of the validation backlog. Two caveats before quoting numbers: (a) a model
this loop-prone has low-entropy output distributions, and peaked outputs
are cheap to reproduce under cache noise — tolerance and degeneration may
share a cause (the Tier 2 activation-kurtosis capture, test 6, is the
discriminator); (b) the loop probe is one greedy continuation per slice
(n=3 total) with no temperature-matched control — whether the loops
persist at the model's served sampling temperature is untested, which is
also PrismML's most legitimate rebuttal line.

## Tier 2 results (measured 2026-07-15, wording hardened by 3-lens review): direct rounding excluded; training vs order-permuted compensation NOT separable from endpoints

`weight_forensics` against the bf16 original (`mlx_lm convert` of
Qwen/Qwen3.6-27B — sanitize aligns naming and drops the vision tower),
40 MLP tensors + 36 attention tensors:

| fingerprint | MLP (40) | attention (36) | reading |
|---|---|---|---|
| projection agree | 85.1% @k=0.5 | 88.3% | not direct rounding (>98%), not rotation (<60%) |
| column drift | −0.001 (flat) | −0.003 agg., ±0.04 per-tensor | no *storage-ordered* compensation signature |
| spectra corr / w-corr | 0.993 / 0.853 | 0.985 / 0.787 | basis not rotated |
| scale ratio vs conditional centroid | ~0.831 (per-tensor means 0.822–0.843) | 0.78–1.01 | absmean-family scales; NOT a training arrow (below) |
| 4th-level usage | 0.00% | 0.00% | ternary, again |

**What the fingerprints exclude:** direct analytic rounding of the original
(agreement is 15 points short of a projection), a rotated basis, and
compensation applied in storage column order.

**What they cannot exclude — the 3-lens review broke our first reading.**
The initial verdict here was "OBS/GPTQ refuted ⇒ QAT/distillation". Two
independent reviewers demolished the inference:

- **The drift test is blind to act-order GPTQ.** desc_act (standard in
  AutoGPTQ/GPTQModel) processes columns in Hessian-diagonal order and
  inverse-permutes afterwards, scrambling the head-vs-tail signature:
  a reviewer's simulated ternary GPTQ shows drift −0.078 (storage-ordered,
  detected) collapsing to −0.011 (act-order — reads as "flat"). The
  observed −0.001/−0.003 is consistent with *both* training and act-order
  compensation.
- **scale-ratio ≈ 0.83 is not a training signature.** BitNet-style absmean
  scales (s = mean|w| over the whole group — analytic, no training) land at
  0.75 ± 0.03 of this tool's conditional-centroid reference on Gaussian
  weights, with ~31% zeros — matching the observed zero fraction and the
  k=0.5 best threshold. The shipped 0.83 sits between absmean and the
  centroid; only ≈1.0 would have been a sharp (analytic-centroid) reading.
- The projection sweep thresholds per row while the pack scales per group
  (gs128), which depresses agreement for *any* group-wise analytic method.

**Bounded verdict:** the transformation is **not direct rounding**; it is
either **QAT/distillation-style conversion** (BitNet-Distillation family)
or **order-permuted / Hessian-weighted compensated PTQ** (act-order GPTQ
class, which the Hassibi/OBS lineage makes the natural in-house method) —
the endpoints' *weights* alone cannot separate the two. Provenance mildly
favors the compensation family; the moved-code volume (~15%) is reachable
by either.

**Test 6 ran (2026-07-15, later the same day) and the discriminator moved
— toward training.** Per-layer excess kurtosis of each decoder block's
output on a fixed 768-token EN/code/ZH batch (`kurtosis_probe.py`, logs
published):

| | median | p90 | verdict input |
|---|---|---|---|
| FP16 Qwen3.6-27B | 2372 | 4685 | heavy-tailed, as expected |
| zero-training PTQ control (clip arm, FFN 2-bit) | **2353** | 4355 | **low-bit weights alone do NOT flatten** |
| Bonsai ternary | **94.6** | 2151 | **~25× flatter, all 64/64 layers** |

The control is the load-bearing row: naive/clip PTQ at comparable FFN
bit-width leaves the activation geometry essentially at FP16 levels, so the
flattening is a property of Bonsai's *conversion*, not of ternary weights.
Compensated PTQ's objective is to *preserve* layer outputs — a systematic
25× kurtosis collapse across every layer is the opposite of that objective,
and is exactly the predicted signature of noise-injection/QAT-style
training. It also supplies the mechanism for the KV-tolerance result
(flat activations → cache-quantization noise proportionally benign) and is
consistent with the sharpened-output/greedy-loop behavior. Weights-only
verdict stays bounded as stated; **with the activation evidence, the
balance of evidence now favors the QAT/distillation branch.**

The second discriminator also ships: `weight_forensics --selftest` now
includes an act-order GPTQ synthetic — storage-order compensation is
detected (drift −0.163) while act-order reads flat (−0.008), pinning the
blind spot in code. Nuance the synthetic exposed: the blindness requires
the Hessian processing order to be *decorrelated from storage order*; a
model whose high-energy channels are contiguous would leave act-order
partially detectable.

Corollary for this repo either way: the open-reproduction path (ternary
projection + layerwise distillation with straight-through re-projection)
targets a method class consistent with the endpoints, but it is a research
campaign, and the loop-prone-but-KV-tolerant profile shows the conversion
cost long-horizon behavior that short-form metrics — including the ones
Tier 1 ran — do not price in.

## The 1-bit companion pack (format audit, 2026-07-15)

`prism-ml/Bonsai-27B-mlx-1bit` (4.1 GB, config `bits: 1, group_size: 128`)
**does not run on stock mlx** — 0.31.2 has no 1-bit kernels (`quantize`
raises; `dequantize` fails kernel lookup), so unlike the ternary pack this
one is usable only on PrismML's MLX fork. The *format* still audits without
their runtime: manual LSB-first unpack (bit order cross-validated 100%
against `mx.dequantize` codes on the 2-bit pack), 6 tensors sampled across
depth and module types:

- **True symmetric binary.** Per-group levels are exactly {−s, +s}
  (|l0+l1|/|l1−l0| median 0.0000); code balance 49.9% ones; per-group code
  entropy 0.994/1.0 — the same at-the-information-ceiling signature as the
  ternary pack. Effective storage (128·1+32)/128 = **1.25 bpw** (4.1 GB /
  26.9 B checks out).
- **Same "moved weights" class as the ternary pack.** Codes agree with
  sign(original) at only **88.3%** — ~11% of signs moved (ternary: ~15% of
  codes). Nearest-level and sign agreement coincide (symmetric levels), and
  the projection shortfall is again largest on `linear_attn.in_proj_a`
  (83.6%), the same tensor family the ternary pack trained hardest.
- **Scales sit at the analytic optimum: ratio 1.01** vs the per-group
  MSE-optimal binary half-spread (mean|w − mean(w)|). For 2 levels the
  absmean rule *is* the MSE optimum — so the 1-bit pack's scales are
  consistent with pure analytic absmean, while its codes are not pure sign
  rounding. One more point for the "absmean-family projection + weight-
  moving optimization" picture, and one more reason the training-vs-
  compensated-PTQ split can't be settled from endpoints.

Runtime behavior (KL, loops, KV) of the 1-bit pack is **not measured** —
it needs PrismML's fork built locally, which is queued behind an explicit
approval gate (third-party fork build/run).

## Same-harness quality triangle (2026-07-15): trained ternary vs zero-training PTQ vs FP16

512-window non-overlapping PPL (llama.cpp-matched scorer, identical
tokenizer and corpus bytes) on the same base model:

| build | size | wikitext | code | ZH |
|---|---|---|---|---|
| FP16 Qwen3.6-27B | 54 GB | 6.34 | 2.22 | 8.78 |
| **Bonsai ternary 1.71 bpw** (trained, end-to-end incl. embeddings) | **8.5 GB** | 11.31 | **2.82** | **26.85** |
| ours, naive mixed PTQ (FFN 2-bit/gs128, rest 8-bit) | 14 GB | 11.19 | 3.23 | 17.12 |
| ours, + anchor-guarded clip (zero training, ~minutes) | 14 GB | **9.91** | 3.04 | 15.80 |

Not size-matched (8.5 vs 14 GB — theirs quantizes everything including
embeddings/attention to ternary; ours leaves non-FFN at 8-bit), so read it
as recipe-vs-recipe, not a scoreboard. Three honest observations:

1. **Bonsai's training shows exactly where its calibration lived**: code is
   its best slice (1.27× FP16 — beats our clip arm), while **ZH collapses
   to 3.06× FP16** — worse than *naive* min-max PTQ (1.95×), which trains
   on nothing at all. The KL/flip slice asymmetry (ZH flip 45.6%) and the
   PPL triangle agree: the conversion bought EN/code retention at the
   direct expense of the non-English mass. Our 45%-ZH-mix lesson, third
   family in a row.
2. A zero-training clip pass on a mid-bit mixed recipe beats the trained
   1.71 bpw build on wikitext and ZH at 1.6× the bytes — the practical
   floor trade-off (quality-per-byte vs training budget) in one table.
3. Their loop-prone slices (EN/ZH greedy) are also their high-PPL slices;
   the clean slice (code) is their PPL-strongest. Consistent with the
   degeneration being distribution-sharpening damage where training data
   was thin.

## Status

- [x] Toolchain validated on **real mlx** (0.32 Linux CPU backend,
  2026-07-14): `code_entropy` (incl. `--per-expert`), `clip_quantize`
  (incl. `--permute-ffn` and the anchor guard) ran end-to-end on synthetic
  quantized models; `weight_forensics` separates its three method classes
  on synthetic ground truth (projection agree 1.000 / compensation drift
  −0.044 / rotation spectra-corr 0.999 with w-corr −0.02).
- [x] Tier 1 on the actual Bonsai pack (2026-07-15, above): ternary label
  verified; KL/flip vs FP16 measured; loop-probe LOOPED ×3 vs clean FP16
  control (greedy-only, no temperature-matched control yet); KV-tolerance
  reproduced, per-slice 9–145×.
- [x] Discriminators run (2026-07-15 pm): activation-kurtosis capture — Bonsai ~25× flatter than FP16 while a zero-training PTQ control matches FP16 ⇒ evidence favors the QAT/distillation branch; act-order synthetic added to `weight_forensics --selftest`.
- [x] Tier 2 run (2026-07-15, above; wording hardened by a 3-lens review
  that broke the first "training, period" reading): direct rounding and
  rotation excluded; **training vs act-order-compensated PTQ not separable
  from endpoints**. Discriminators (kurtosis capture; act-order synthetic
  for the tool) unrun. Reproduction decision escalated to the owner.
- Compiled 2026-07-14 from the whitepaper, Bonsai-demo repo, and public
  reporting.
