# Case study: Kimi-K2.7-Code (~1T MoE) — the INT4-master ceiling, and when DWQ / calibration *don't* apply

This is the counter-case to the rest of this repo. Everywhere else the lever is a
higher-bit teacher (clip-search, then DWQ toward it). Kimi-K2.7-Code has **no such teacher
to reach for** — its routed experts ship as a *native INT4 master*, so there is nothing
above 4-bit to distill from. The useful output here is the negative result, stated with its
mechanism, plus the honest quality metric for a build whose floor is set by physics, not by
recipe.

Shipped: [`avlp12/Kimi-K2.7-Code-Alis-MLX-Dynamic-3.6bpw`](https://huggingface.co/avlp12/Kimi-K2.7-Code-Alis-MLX-Dynamic-3.6bpw)
(text) and [`-VLM`](https://huggingface.co/avlp12/Kimi-K2.7-Code-Alis-MLX-Dynamic-3.6bpw-VLM)
(image+video; byte-identical LLM weights + a bf16 MoonViT tower).

## The model

`moonshotai/Kimi-K2.7-Code` — `model_type: kimi_k25` (text_config `kimi_k2` → the DeepSeek-V3
engine), ~1T params / 32B active, 61 layers (0 dense + 1–60 MoE), MLA, 384 routed experts
top-8 + 1 shared, plus a MoonViT vision tower. The decisive fact for quantization:

> The **routed experts** (≈99% of the weights) ship as a native **INT4** master —
> `compressed-tensors`, group-size 32, symmetric, QAT. Attention (MLA) / shared / dense /
> embed / head / vision are bf16. **INT4 is the master; there is no higher-precision source
> for the experts.**

## The #907 fix — the enabling lesson for any compressed-tensors source

Re-quantizing an already-quantized checkpoint silently *keeps the experts at their source
bit-width*. After load they are `QuantizedSwitchLinear` (no `to_quantized`), so a lower-bit
`quantize_model` predicate skips them — ask for 3-bit and you get **~5 bpw / 640 GB**
([mlx-lm#907](https://github.com/ml-explore/mlx-lm/issues/907)). Fix: detect quantized
modules **on the model**, dequantize first, then re-quantize:

```python
if any(hasattr(m, "bits") and hasattr(m, "scales") for _, m in model.named_modules()):
    config.pop("quantization", None); config.pop("quantization_config", None)
    model = dequantize_model(model)          # already imported in convert.py
model, config = quantize_model(model, config, ..., quant_predicate=pred)
```

MLX lazy eval keeps the dequant→requant streaming per-tensor — **~51 GB peak** converting a
1T model, no TB-scale intermediate. Detect on the *model*, not `config["quantization"]`,
which isn't reliably populated at this point (mlx-vlm in particular carries it under
`text_config`).

**The maintainer closed #907 as intended** ("already-quantized weights aren't dequantized and
requantized on convert; if you want that, ask for it explicitly … quantizing int4 to 3-bit
is quite lossy"). Our KLD below *confirms* that caution. So this is an explicit, opt-in
operation, not a default — and the two-step he suggests (`--dequantize` to disk, then
`--quantize`) needs a ~1.3 TB full-precision intermediate at 1T scale; the in-memory lazy
form above is the only practical way to retarget the bit-width of a 1T INT4 checkpoint.

## Recipe & the KLD

Sensitivity-graded, data-free RTN — experts gate/up **3-bit g64**, `down_proj` **4-bit on
16/60 layers** else 3-bit, attention/shared/dense/embed/head **6-bit**, router **bf16** →
**3.629 bpw / 465 GB**. Fits one clean 512 GB M3 Ultra, or splits ~233 GB/box over TB.

Quality was measured as **KL divergence + top-1 flip vs a 4-bit reference** (experts 4-bit
g64 + rest 6-bit), over a fixed 4096-token slice (wikitext + code heads, 8×512 non-overlapping
chunks; identical chunking on both builds → apples-to-apples):

| metric | value |
|---|---|
| mean KL(4-bit ‖ 3.6bpw) | **0.199 ± 0.009 nats** (median **0.006**) |
| mean KL on the ~90% **non-flipped** positions | 0.076 |
| **top-1 greedy flip** | **10.2 %** (416/4096) |
| — of those flips, "near-ties" (gap < 0.10) | only **15 %** |
| — at a flip: ref's own pick / this build's pick (ref prob) | ~0.59 / ~0.14 |
| slice-PPL (this vs 4-bit, same slice) | 2.256 vs 2.053 (**+9.9 %**) |

Read it two-sidedly: **typical tokens are near-lossless** (median KL 0.006), but on ~10% of
positions the greedy top-1 *flips*, and those flips are mostly **decisive, not coin-flips**.
That tail (mean KL ≈1.29 on flipped positions) lifts the overall mean to 0.199. It is exactly
the "Accuracy is Not All You Need" effect: the tulu PPL (3.735) looked benign; the flip is the
real cost that averaging hid.

## The INT4-master ceiling — why the flip is bit-starvation, and why nothing cheap fixes it

Because the experts' master *is* INT4, **every 4/6/8-bit build sits at ≈ that master** (you
cannot add information above INT4) and differs only in size — an 8-bit build is *larger than
the 595 GB source* for zero expert-quality gain. So a 4-bit build is effectively **the
original**, the KL/flip above is the cost **vs the original itself**, and the only real
fidelity lever is *more bits* (which just grows the file). The flip is **bit-starvation, not
scale-starvation** — and that is precisely why re-fitting scales (what calibration does) can't
close it.

## When DWQ / AWQ / calibration *don't* apply (the negative result, verified)

A 3-lens review (first-principles / red-team / practitioner) converged unanimously on **ship
as-is**. Each calibration path is blocked on this stack, for a reason worth carrying forward:

- **DWQ has no real teacher.** DWQ distills a low-bit student toward a *higher-bit* teacher's
  outputs. Here the best available teacher is 4-bit — itself a lossy RTN sibling of the same
  INT4 weights — so DWQ's ceiling is "look like a 4-bit copy", not "look like truth". The
  famous DWQ gains all assume a bf16/fp16 teacher this model does not have.
- **DWQ is also memory/engineering-infeasible.** Single-node backward OOMs (the student is
  larger than one box's working set; cf. the GLM 238 GB student that peaked ~470 GB). And the
  distributed path dies at the first training step: MLX has **no vjp for pipeline `Send`**
  (`[Primitive::vjp] Not implemented for Send`) — forward works after the CPU-stream watchdog
  fix, but autograd cannot cross the rank boundary.
- **AWQ is blocked and useless here.** `kimi_k25` is absent from mlx-lm's `AWQ_MODEL_CONFIGS`
  (raises `NotImplementedError`); and AWQ-on-MoE is a proven dead-end on this identical
  DeepSeek/MLA+MoE shape (per-block thrash, scale-search-at-uniform-bits, all-or-nothing
  block fallback). Even if it ran, AWQ multiplies a fixed affine grid by learned scales — it
  cannot move the experts off 3-bit.
- **GPTQ / HQQ / imatrix**: GPTQ/HQQ need an fp source we lack (HQQ also probed negative on
  this stack — weight-MSE, not output error); imatrix is a llama.cpp/GGUF path with no route
  from a compressed-tensors MLX build.

**Verdict:** the 10.2% flip is the near-floor cost of 3-bit experts from an INT4 master. The
only data-free lever that respects the physics is **bit re-allocation** (lift the flip-driving
gate/up projections toward 4-bit on the dominant layers, or ship 4-bit experts outright) — not
calibration. We shipped as-is and disclosed the numbers.

## Operational residue (reusable)

- **A `compressed-tensors` master is not directly measurable on a single box.** 595 GB > 512
  GiB → single-box inference OOM-kills *during the forward* even with `lazy=True` (MLX makes
  each layer's weights resident as the graph evaluates; a >RAM model can't complete one
  forward). And `pipeline_load` / `sharded_load` **rejects it**: `ValueError "Pipeline loading
  is only supported for MLX converted models"` — compressed-tensors sources aren't shardable.
  It *does* load single-box lazily as **4-bit g32** (mlx-lm maps compressed-tensors → affine
  `{bits:4, group_size:32}` at load). So the runnable stand-in for "the original" is a 4-bit
  MLX build; the true master's own PPL is not obtainable on this stack without first
  converting it to an MLX build (which, being ≥4-bit, ≈ the 4-bit reference anyway).
- **Distributed KLD capture** used the ring pipeline (`pipeline_load`, each rank ~half) with a
  **chunked 512-token forward** — a single 2048-token forward overruns the macOS GPU watchdog
  on the CPU-stream comm path. Identical chunking across builds keeps the comparison exact.
- **Never name an `mlx.launch` script flag `--n`.** Its argparse abbreviation-matches
  `--nccl-port` / `--no-verify-script` → `ambiguous option` abort before your script runs.
  (unknown `--long-flags` otherwise pass through to the script; `--n` is the one landmine.)

## Bottom line

Kimi-K2.7-Code is the case where the DWQ toolbox is the wrong toolbox. The enabling lesson is
the #907 dequant-first fix (any compressed-tensors source); the shipping lesson is that when
the master is already INT4, 4/6/8-bit are all one quality shelf, the sub-4-bit flip is
bit-starvation, and no calibration method on this stack can buy it back — so measure it
honestly (KL + flip vs the 4-bit ≈ original) and ship.
