# Auditing Bonsai 27B: what "ternary" and "1-bit" actually measure like

*Unless explicitly attributed to PrismML's materials, every number below was
measured by me on a single M3 Ultra (512 GB) using open tools
([alis-dwq](https://github.com/avlp12/alis-dwq)) — on stock mlx-lm 0.31.3,
except the 1-bit pack's runtime rows, which required PrismML's MLX fork
(noted in that section). The raw
measurement logs and the audit scripts are published in the repo
([examples/bonsai-27b-audit](https://github.com/avlp12/alis-dwq/tree/main/examples/bonsai-27b-audit),
`logs/` + `audit_1bit.py`), and both weight sets are public (Bonsai:
Apache-2.0; base: Qwen/Qwen3.6-27B) — commands at the end.*

PrismML's [Ternary Bonsai 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit)
advertises a 1.71-bits/weight ternary representation of Qwen3.6-27B — the
audited MLX artifact stores those codes in 2-bit affine slots and weighs
8.49 GB — claiming ~95% of FP16 quality retained across 15 thinking-mode
benchmarks. The method is proprietary; the endpoints are not. I downloaded
the pack and the FP16 original and measured: token-level divergence,
per-language slices, degeneration probes in two decoding regimes, KV-cache
tolerance, and weight-space forensics on the transformation itself.

The one-line result: **the format claims are honest, the headline quality
claim is true only along the axis the benchmarks measure, and off that axis
the model degrades in ways per-benchmark averages structurally cannot see.**

Framing notes, stated up front. The retention figures (94.6% / 89.5%), the
15-benchmark table, tok/s and battery numbers circulating in press coverage
are PrismML's own, measured with EvalScope + vLLM on H100s — their CUDA
path, sampled, judge-scored. Nothing below disputes those numbers on their
own terms; this audit supplies what that methodology structurally can't.
Scope: the **ternary pack** runs on stock mlx-lm; the **1-bit companion**
required building PrismML's MLX fork (isolated venv) — both packs get the
full runtime battery below.

## What checks out

Credit first — three of their claims survived adversarial measurement:

- **The ternary label is true where it can be checked.** Recovering the code
  histogram of all 498 affine language-model tensors (through
  `mx.dequantize`, no assumptions about packing): every one uses exactly 3
  levels — the 2-bit container's 4th level is used by *zero* weights. Level
  usage is nearly uniform (35.2% / 29.7% / 35.1%), putting the effective
  payload at 1.58 b/w of the nominal 2.00 — 99.7% of the log₂3 ceiling
  (per-tensor entropy: min 1.551 / median 1.581 / max 1.585). The 4-bit-HQQ
  vision tower is outside this scan, as their card discloses. Whatever
  produced this pack squeezes its alphabet about as hard as information
  theory allows.
- **It runs on stock mlx-lm and generates coherent text.** The custom
  kernels are a speed path, not a format requirement (for the ternary pack —
  not the 1-bit one).
- **The 4-bit-KV-cache tolerance claim reproduces.** Self-KL induced by
  quantizing the KV cache to 4 bits, against each model's own FP16-KV run:
  the ternary build absorbs cache noise **9–145× better than FP16**
  depending on slice (EN 15×, code 145×, ZH 9.4×; the whitepaper claims
  12–95×). A real, independently reproduced phenomenon — with caveats
  covered below.

## What the benchmarks don't show

**Token-level divergence from FP16 is large, and it is not evenly
distributed.** On a fixed 3,072-token slice (⅓ English wikitext, ⅓ code,
⅓ Chinese):

| slice | KL(FP16‖Bonsai) | top-1 flip |
|---|---|---|
| EN | 0.58 | 21.0% |
| code | 1.32 | 22.8% |
| ZH | 1.32 | **45.6%** |
| overall | 1.08 | 29.8% |

Nearly half of Chinese tokens change their argmax. An aggregate benchmark
suite that doesn't report language-separated results will not surface this.

**Same-harness perplexity triangle.** 512-token non-overlapping windows,
identical scorer (verified mirror of `llama-perplexity`'s chunk scoring),
identical corpus bytes, same base model:

| build | size | wikitext | code | ZH |
|---|---|---|---|---|
| FP16 Qwen3.6-27B | 54 GB | 6.34 | 2.22 | 8.78 |
| Bonsai ternary (trained) | **8.5 GB** | 11.31 | **2.82** | **26.85** |
| Bonsai 1-bit (trained; on their fork) | **4.1 GB** | 12.19 | 3.14 | **41.81** |
| naive mixed PTQ, zero training (FFN 2-bit) | 14 GB | 11.19 | 3.23 | 17.12 |
| + anchor-guarded clip search, zero training | 14 GB | **9.91** | 3.04 | 15.80 |

This is not size-matched (8.5 vs 14 GB) and not method-controlled, so it
does not isolate a training effect — read it as recipe-vs-recipe. What it
does establish: on **code**, the Bonsai endpoint reaches 1.27× FP16 and
beats both larger zero-training baselines. On **Chinese**, it reaches
**3.06× FP16 — worse than naive min-max PTQ (1.95×)**, which trains on
nothing at all. Whatever the conversion optimized, the non-English mass
paid for it. (Third model family in a row where I've measured low-bit
damage concentrating in non-English slices; it's why my own recipes
calibrate with a 45% target-language mix.)

**Decode-regime damage that eval parity hides.** Greedy-decoding 256 tokens
from 64-token raw-text prompts, per slice:

| | EN | code | ZH |
|---|---|---|---|
| FP16 distinct-4gram / cycle | 0.632 / none | 0.941 / none | 0.996 / none |
| Bonsai distinct-4gram / cycle | **0.134 / len-13** | **0.411 / len-2** | **0.079 / len-18** |

The ternary build settles into short repeating cycles on every slice; the
FP16 control is clean on the identical probe. One prompt per slice, so
read it as an endpoint difference under this probe — the conversion bundle
(quantization + whatever retuning produced the pack) is what differs, and
its components are not separable here. It matches an observation PrismML
themselves publish (a conventional IQ2_XXS build holding MMLU-Redux at
88.93 while falling to 57.5 on AIME26): sub-4-bit damage concentrates in
long-horizon behavior, exactly where short-form benchmarks don't look.

**I then ran the control PrismML would rightly ask for** — the same probe
with their documented sampling parameters (generation_config: T=1.0,
top-k 20, top-p 0.95; 5 seeded samples per slice, same raw-text prompts).
Result: **the hard cycles vanish — 15/15 samples cycle-free on both
models.** The greedy result stands, but it must be quoted with this
qualifier: the degeneration is specific to greedy decoding on this probe;
chat-template behavior and near-greedy settings remain unmeasured. The full
distinct-4gram picture at temperature (mean, and worst of 5, per slice):

| | EN | code | ZH |
|---|---|---|---|
| Bonsai mean / min | 0.723 / **0.265** | **0.841** / 0.589 | 0.952 / 0.913 |
| FP16 mean / min | 0.840 / 0.660 | 0.808 / 0.672 | 0.978 / 0.964 |

Honest reading: Bonsai's worst samples fall much lower on EN, but its code
*mean* actually beats FP16 — at serving temperature this is a heavier
low-diversity tail on two slices, not a collapse, and n=15 per model makes
it directional. Practical takeaway: workloads that decode greedily (many
math/code harnesses, constrained decoding, some agent stacks) should run
this probe themselves; sampled chat looks substantially safer.

## What the transformation actually is (weight forensics)

Both endpoints are public, so the method's *class* can be constrained by
comparing them — 40 MLP + 36 attention tensors, four fingerprints each
(spectra on 8 sampled tensors per run):

- **It is not direct analytic rounding.** Codes disagree with the best
  threshold projection of the original weights by 14.9% (MLP) / 11.7%
  (attention) — the weights moved.
- **The basis was not rotated** (singular-spectrum correlation 0.993 / 0.985
  on the sampled tensors, with high elementwise correlation).
- **The scales are absmean-family** (MLP scale-ratio ≈0.83 of the
  conditional-centroid optimum, tight across tensors; attention spans
  0.78–1.01). Consistent with BitNet-style analytic scales — *not* evidence
  of training by itself.
- **No aggregate storage-order compensation signature** (drift −0.001 /
  −0.003; individual attention tensors reach ±0.04) — but act-order
  variants process columns in permuted order and are invisible to this
  test. So the honest bound is: **QAT/distillation-style conversion OR
  order-permuted compensated PTQ (act-order GPTQ class). The endpoints
  alone cannot separate these.** Given the founders' lineage (Optimal Brain
  Surgeon), the compensation family is a live hypothesis, not a dismissed
  one.

(An earlier draft concluded "it's training, GPTQ refuted" — an adversarial
review broke that inference by demonstrating act-order GPTQ produces
exactly the flat aggregate drift measured here. The bounded verdict above
is what the *weights* support.)

**So I ran the discriminating experiment.** If the conversion is
noise-injection/QAT-style training, activation distributions should be
*flattened*; if it's output-preserving compensated PTQ, they should match
FP16. Per-layer excess kurtosis of every decoder block's output, one fixed
mixed batch, with a zero-training PTQ build as the control:

| | median excess kurtosis |
|---|---|
| FP16 Qwen3.6-27B | 2372 |
| zero-training PTQ control (clip, FFN 2-bit) | **2353 — at FP16 level** |
| Bonsai ternary | **94.6 — ~25× flatter, all 64/64 layers** |

The control row is load-bearing: low-bit weights by themselves leave the
activation geometry intact, so the flattening is a property of Bonsai's
conversion — and a systematic 25× kurtosis collapse is the opposite of a
compensation objective (which preserves outputs) while being exactly the
predicted signature of noise-robust training. It also supplies the
mechanism for the KV tolerance below. The weights-only verdict stays
bounded as stated; **with the activation evidence, the balance tips to the
QAT/distillation branch.**

Worth noting the convergence: the format PrismML describes publicly —
`w = s_g · t_i`, one shared FP16 scale per 128-weight group — is exactly
what the fingerprints recover. The heart of what remains proprietary is how
`t_i` is selected: the moved codes that no analytic projection explains.

## The 1-bit pack: honest binary, similar fingerprints, fork-only runtime

The ~4 GB "1-bit" companion is the more radical artifact, and its weight
format audits cleanly even though it is **not runnable on stock mlx** —
`bits: 1` has no kernels in mlx 0.31.2 (`quantize` refuses, `dequantize`
fails kernel lookup); PrismML's MLX fork is the documented runtime path.
Unpacking the codes manually (script + full output in the repo; bit order
cross-validated against `mx.dequantize` on the ternary pack), six tensors
sampled across depth and module types:

- **True symmetric binary in the sample**: per-group levels exactly
  {−s, +s}, 49.9% sign balance, per-group code entropy 0.994 of the 1.0
  ceiling — matching the format PrismML describes. Container detail: the
  advertised **1.125 bpw** counts one FP16 scale per group; the MLX affine
  container also ships a per-group *bias*, so the pack stores **1.25 bpw**
  (the 4.1 GB on disk checks out against that, not 1.125).
- **The same "moved weights" signature**: codes agree with sign(original)
  at 88.3% — **11.7% of signs moved** away from the source model, the
  binary analogue of the ternary pack's moved codes. (Within this 6-tensor
  sample the lowest agreement was `linear_attn.in_proj_a` at 83.6%; the
  ternary pack's attention sweep had a *different* worst family,
  `in_proj_b` at 58.9% — so no cross-pack "same tensor moved most" claim.)
- **Scales sit near the analytic optimum** (median ratio 1.01 vs the
  per-group MSE-optimal half-spread — which for two levels *is* the absmean
  rule). Consistent with, though six tensors can't prove, analytic absmean
  scales under a weight-moving optimization.

And with the fork built (isolated venv; the build has teeth — cmake grabs
the newest system Python unless you pin the venv's, and the wheel omits
`libmlx.dylib` under default build isolation), the 1-bit pack got the full
runtime battery too:

- **KL vs FP16: 1.351 overall, flip 33.6% — ZH flips a majority of its
  tokens (50.9%)** and lands at 4.76× FP16 windowed perplexity (41.8 vs
  8.78). Meanwhile wikitext holds 1.92× and code (3.14) *still beats the
  zero-training PTQ baseline at ~4× the bytes* — the conversion's
  EN/code-facing competence and its non-English cost both scale with
  aggressiveness.
- **The temperature defense starts to breach.** The ternary pack was
  cycle-free in 15/15 sampled continuations at serving temperature; the
  1-bit pack looped in 1 of 15 (EN, period 21). The "it only loops under
  greedy" qualifier that protects the ternary pack is thinner one bit down.
- **KV immunity strengthens as bits drop: 82× vs FP16** (self-KL 0.00198;
  ternary: 56×). Two points now trace the capacity trend the kurtosis
  mechanism predicts: more aggressive conversion → flatter activations →
  more cache robustness, paid for in output sharpness and degeneration.

## The KV-tolerance finding deserves its own paragraph

The KV story is load-bearing for their phone pitch: by their launch
materials, an FP16 cache at the full 262K context would cost ~17.2 GB —
more than the model — and 4-bit KV is what brings it to ~4.3 GB on a hybrid
backbone that caches only 16 of 64 layers. So it matters that the
tolerance claim *reproduces* (9–145× by slice, on the same 16 quantizable
caches per model). And with the kurtosis measurement above, the tolerance
now has a mechanism: **activations ~25× flatter than FP16** mean
cache-quantization noise is proportionally benign — the immunity is a
property of the conversion, measured, not conjectured (a zero-training PTQ
control keeps FP16-level kurtosis *and*, on another model family in the
same session, PTQ+DWQ builds showed nothing near this immunity). The flip
side of the same mechanism: flattened/re-shaped internal distributions
coexist with the sharpened-output greedy-loop behavior — robustness and
degeneration appear to be two faces of one training choice. Whether the
robustness can be induced *without* the damage (e.g., KV-noise injection
during distillation-based retuning) is an open experiment, and probably
the most useful research direction this release points at.

## Takeaways

1. **The format claims are honest — audit tools confirm them in minutes.**
   True ternary in every scanned language tensor, true symmetric binary in
   the 1-bit sample, information-ceiling code usage, no hidden precision.
2. **"95% of FP16" is a benchmark-shaped number.** Token-level: 30% of
   argmaxes move; Chinese: 46% flip, and windowed PPL lands below the
   zero-training PTQ baseline measured here (size/method-unmatched — but
   the baseline trained on *nothing*). Slice your evals by language before
   trusting any low-bit headline.
3. **Short-form metrics hide decode-regime damage** — hard loops under
   greedy (FP16 control clean), cycle-free at their documented sampling
   parameters. If your workload decodes greedily — many math/code harnesses
   do — that distinction is the whole ballgame. Ship a degeneration probe,
   and run it in both regimes.
4. **"Labels are not bit-widths" cuts both ways.** PrismML's launch thread
   fairly attacks mixed-precision labels (their accounting: Q4_K_XL ≈ 5.2
   b/w, IQ2_XXS ≈ 2.8) — and their *codes* are exactly as advertised. Apply
   the same lens to their artifacts, though: "1.71 b/w → 5.9 GB" and
   "1.125 b/w → 3.9 GB" describe ideal representations. The shipped MLX
   ternary pack stores 3 levels in 2-bit slots at **8.49 GB**; the shipped
   MLX 1-bit pack stores **1.25 b/w** including the per-group bias — and
   runs only on their fork. The advertised number describes the
   representation, not the deployable artifact. Sound familiar?
5. **The KV tolerance is real, and now has a measured mechanism** (25×
   flattened activations that plain PTQ does not produce). Whether it's
   manufacturable without the greedy-mode damage is the open question worth
   stealing — that, not the ternary itself, is what I'd chase.

## Reproduce it

```bash
git clone https://github.com/avlp12/alis-dwq && cd alis-dwq
python3 -m pip install "mlx-lm>=0.31"

# format audit (minutes, student pack only)
python3 -m alis_dwq.code_entropy --model <bonsai-pack> --save entropy.npz

# token-level KL + greedy degeneration + KV probes, both endpoints
python3 -m alis_dwq.eval_kld --model <qwen36-fp16> --save-ref ref.npz \
  --loop-probe 256 --kv-probe 4
python3 -m alis_dwq.eval_kld --model <bonsai-pack> --ref ref.npz \
  --loop-probe 256 --kv-probe 4

# temperature-matched control (both endpoints)
python3 -m alis_dwq.eval_kld --model <qwen36-fp16> --save-ref /tmp/x.npz \
  --loop-probe 256 --loop-temp 1.0 --loop-top-k 20 --loop-top-p 0.95 --loop-samples 5
python3 -m alis_dwq.eval_kld --model <bonsai-pack> --save-ref /tmp/y.npz \
  --loop-probe 256 --loop-temp 1.0 --loop-top-k 20 --loop-top-p 0.95 --loop-samples 5

# 512-window PPL (llama.cpp-matched scorer)
python3 -m alis_dwq.ppl_windows --model <any-build> --window 512 \
  --text data/wikitext.txt data/code.txt data/zh.txt

# method forensics (needs both weight sets on disk)
python3 -m alis_dwq.weight_forensics --original <qwen36-mlx-bf16> \
  --transformed <bonsai-pack> --pattern mlp --max-tensors 40
python3 -m alis_dwq.weight_forensics --original <qwen36-mlx-bf16> \
  --transformed <bonsai-pack> --pattern "linear_attn|self_attn" --max-tensors 36

# 1-bit pack format audit (no fork needed)
python3 examples/bonsai-27b-audit/audit_1bit.py

# activation-kurtosis discriminator (edit the model paths at the top)
python3 examples/bonsai-27b-audit/kurtosis_probe.py
```

Full raw logs, per-tensor fingerprints, and the audit protocol:
[examples/bonsai-27b-audit](https://github.com/avlp12/alis-dwq/tree/main/examples/bonsai-27b-audit).
