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

## Status

- [x] Toolchain validated on **real mlx** (0.32 Linux CPU backend,
  2026-07-14): `code_entropy` (incl. `--per-expert`), `clip_quantize`
  (incl. `--permute-ffn` and the anchor guard) ran end-to-end on synthetic
  quantized models; `weight_forensics` separates its three method classes
  on synthetic ground truth (projection agree 1.000 / compensation drift
  −0.044 / rotation spectra-corr 0.999 with w-corr −0.02).
- [ ] Tier 1 on the actual Bonsai pack pending — the remote container's
  proxy blocks Hugging Face, so weight downloads need a normal box (any
  Apple Silicon, ~an hour; the MLX CPU backend also works on Linux for
  `code_entropy`/`weight_forensics`, just not for fast forwards).
- [ ] Tier 2 pending (both weight sets on disk, ~70 GB total).
- Compiled 2026-07-14 from the whitepaper, Bonsai-demo repo, and public
  reporting; the OBS-lineage prior is a hypothesis to test, not a finding.
