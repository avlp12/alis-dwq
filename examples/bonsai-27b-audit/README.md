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
  layers; per-tensor entropy min 1.551 / median 1.585 (max 1.585).
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

**KV-tolerance: reproduced, 56×.** Self-KL of a 4-bit/gs64 KV cache vs each
model's own FP16-KV run (16 of 64 layer caches quantizable on this hybrid —
linear-attention states stay FP16, so both rows are lower bounds):

| | EN | code | ZH | overall | flip |
|---|---|---|---|---|---|
| FP16 weights | 0.0458 | 0.4148 | 0.0261 | 0.1622 | 5.1% |
| Bonsai ternary | 0.0030 | 0.0029 | 0.0028 | **0.0029** | 2.2% |

**56× lower self-KL on the ternary build** — inside PrismML's claimed
12–95× band, reproduced on an independent harness. The transferable
hypothesis (discretization-shaped models absorb cache noise) survives its
first outside test; running the same probe on our own DWQ'd students is
§3b of the validation backlog. One mechanism caveat before quoting the
number: a model this loop-prone has low-entropy output distributions, and
peaked outputs are cheap to reproduce under cache noise — tolerance and
degeneration may share a cause. The Tier 2 activation-kurtosis capture
(test 6) is the discriminator.

## Tier 2 results (measured 2026-07-15): the OBS prior is REFUTED — it's training

`weight_forensics` against the bf16 original (`mlx_lm convert` of
Qwen/Qwen3.6-27B — sanitize aligns naming and drops the vision tower),
40 MLP tensors + 36 attention tensors:

| fingerprint | MLP (40) | attention (36) | reading |
|---|---|---|---|
| projection agree | 85.1% @k=0.5 | 88.3% | not rounding (>98%), not rotation (<60%) |
| column drift | **−0.001** | **−0.003** | **flat — no OBS/GPTQ column-compensation signature** |
| spectra corr / w-corr | 0.993 / 0.853 | 0.985 / 0.787 | basis not rotated |
| scale ratio vs closed form | **0.835 ± 0.035** | 0.78–1.01 | scales systematically ~17% below MSE-optimal ⇒ **trained** |
| 4th-level usage | 0.00% | 0.00% | ternary, again |

**VERDICT (both sweeps): "weights moved position-independently
(training/distillation)".** The Hassibi/OBS lineage suggested curvature
compensation; the fingerprints say otherwise — no positional error-absorption
structure anywhere, weights moved ~15% of codes away from any threshold
projection of the original, and the shipped group scales sit consistently
*below* the analytic optimum (a distillation-pressure signature, not a
closed form). Supporting detail: `linear_attn.in_proj_a` — the tensor family
with the anomalous zero-fraction spread in Tier 1 — has the *lowest*
projection agreement (76.3%), i.e. it was trained hardest. The method class
is **QAT/distillation-style conversion** (BitNet-Distillation family), not
OBS/GPTQ-style PTQ. What stays unrecoverable from endpoints: schedule, data,
and the exact objective.

Two practical corollaries for this repo:

1. The open-reproduction path (ternary projection + layerwise distillation
   with straight-through re-projection, §"If the fingerprints confirm")
   targets exactly the *measured* method class — it just needs more than the
   "short distillation" the OBS prior assumed (they moved ~15% of codes).
2. Their loop-prone-but-KV-tolerant profile now reads as a *training*
   outcome, not a rounding artifact — distillation on finite data at 1.71
   bpw bought benchmark retention and cache robustness at the price of
   long-horizon behavior our loop probe caught in one minute.

## Status

- [x] Toolchain validated on **real mlx** (0.32 Linux CPU backend,
  2026-07-14): `code_entropy` (incl. `--per-expert`), `clip_quantize`
  (incl. `--permute-ffn` and the anchor guard) ran end-to-end on synthetic
  quantized models; `weight_forensics` separates its three method classes
  on synthetic ground truth (projection agree 1.000 / compensation drift
  −0.044 / rotation spectra-corr 0.999 with w-corr −0.02).
- [x] Tier 1 on the actual Bonsai pack (2026-07-15, above): ternary label
  verified; KL/flip vs FP16 measured; loop-probe LOOPED ×3 vs clean FP16
  control; KV-tolerance reproduced at 56×.
- [x] Tier 2 run (2026-07-15, above): **OBS/GPTQ prior refuted** — flat
  column drift, unrotated basis, trained scales ⇒ QAT/distillation-class
  conversion. Reproduction decision escalated to the owner (it targets the
  measured class, but is a training campaign, not a weekend).
- Compiled 2026-07-14 from the whitepaper, Bonsai-demo repo, and public
  reporting; the OBS-lineage prior was tested and rejected on 2026-07-15.
