# Case study: the gather_qmm autograd memory laws — layer-local DWQ on a 975B multimodal MoE at a 23 GiB peak

**Question.** What does it actually cost, in memory, to differentiate the affine scales/biases of a *production-size* quantized expert bank in MLX — and can a single 512 GB box run layer-local DWQ on a 256-expert layer whose naive backward wants ~150 GiB of workspace, under a hard watchdog envelope of *min 90% system RAM free, zero swap growth*?

**Setting.** Inkling (975B-class multimodal MoE): 66 hybrid decoder layers, 256 routed experts (top-6) + 2 shared per MoE layer, hidden 6144. Layer-local DWQ (one strict-loaded `DecoderLayer`, BF16 teacher boundaries h_l→h_{l+1}, valid-token NMSE, stop-gradient targets) — the same primitive as the main pipeline, but each layer here is ~15B params with a 5.9 GiB *quantized* expert bank (3-bit g128; logical bf16 extent ≈ 86 GiB). MLX 0.32, M3 Ultra 512 GB.

The naive step — `mx.value_and_grad` over the four expert tensors (`w13/w2 × scales/biases`) through the fused `gather_qmm` — peaked at **150.35 GiB** for one 140-token batch. The watchdog kills that long before swap. Everything below is the empirical path to a **24.9 GiB** peak with gradient cosine ≥ 0.9985 against the fused reference.

## The three memory laws (measured, MLX 0.32)

Isolated probes on the real layer-65 bank (`w13`: 256 experts × 6144 × 6144-logical, 3-bit g128; single gather, 4 tokens × top-8, grads w.r.t. scales+biases only):

| differentiated extent | backward peak |
|---|---|
| all 256 experts | 45.86 GiB |
| 128-expert slice | 27.16 GiB |
| 32-expert slice | 12.92 GiB |
| 8-expert slice | 9.36 GiB |
| forward only (any) | 5.92 GiB |

**Law 1 — scale/bias grads dequantize the *whole differentiated tensor*.** The VJP materializes the dequantized fp32 extent of whatever scales/biases you differentiate, independent of how few tokens routed there. Slicing the expert axis of the *live* (differentiated) tensors shrinks it proportionally.

**Law 2 — activation grads dequantize *one expert matrix per routed assignment*.** In a `w13 → silu·gate → w2` chain, the gradient w.r.t. the intermediate activation (needed to reach `w13`'s scales) pays ~one dequantized expert matrix (~226 MB fp32 for this bank) *per (token, k) assignment*. This is why expert-slicing alone doesn't save you: a 32-expert slice with all 1120 assignments still peaked at **103.4 GiB**, and a full-bank pass over 20-token blocks (160 assignments) peaked at **76.3 GiB** — both trace straight to `assignments × per-expert-matrix`, not to the sliced extent.

**Law 3 — fp32 scales on the module poison the inference path too.** If you `layer.update()` fp32 scale tensors into the quantized modules (the classic "cast params up for Adam" move), even *forward* calls fall off the fused kernel into dequantize-fallback: a probe with fp32 module scales doubled the peak (197.9 vs 101.9 GiB). Keep module state in its original dtype; carry fp32 only in the explicit differentiated arguments.

Corollary worth knowing: the in-grad (differentiable) forward and the inference forward are *both deterministic but numerically different kernels* — a loss evaluated inside `value_and_grad` differs from the plain forward at the ~0.2% level on this layer. Pin your reported metric to one of them (we report the inference-kernel loss) or your receipts won't reproduce.

## The streamed backward

Both laws must be beaten at once: **expert-group slicing** (Law 1) × **token-block serialization** (Law 2), with each (group, block) micro-backward evaluated and freed before the next — grouping inside a single lazy graph does *not* cap the peak (measured 198 GiB: the slice-scatter grads of all groups coexist at eval).

The reconstruction that makes each micro-objective exact:

```python
# once, no grad, inference kernels — capture the fused bank output
out = bank(x, idx)                      # stashed, stop_gradient

# per (expert-group g, token-block b): replay the layer forward with
def replay(_x, _idx):
    const = mx.stop_gradient(bank_g(x[b], idx[b], raw_scales_g,  raw_biases_g))
    live  =                  bank_g(x[b], idx[b], live_scales_g, live_biases_g)
    return out + concat([zeros_before_b, live - const, zeros_after_b], axis=1)
```

`bank_g` gathers on the *sliced* packed weights with in-group masking (out-of-group assignments compute expert 0 of the slice and are zeroed — summing groups reproduces the fused bank exactly, verified bitwise on loss). At the current parameters `live == const` bitwise, so `out + (live − const)` *is* the captured output — write it in that association order; `(out − const) + live` reintroduces rounding. Gradients flow only through `live`, so each micro-backward's workspace is `slice-extent + block-assignments × expert-matrix`, both chosen: group 16 × block 8 →

| variant | peak | grad cosine vs fused |
|---|---|---|
| fused reference | 150.35 GiB | 1.0 |
| grouped, single graph | 198.36 GiB | 0.9999 |
| expert-serial only (g=32) | 103.44 GiB | 0.9999 |
| token-block only (b=20) | 76.25 GiB | 0.9999 |
| **group 16 × block 8, serialized** | **24.9 GiB** | **0.9985+** |

Loss stays bitwise-equal to the plain inference forward; per-tensor gradient cosine ≥ 0.9985 (the residual gap is the in-grad-vs-inference kernel difference above, not the decomposition). Cost: compute scales with `groups × blocks` re-forwards — ~0.4–0.6 s per token of sequence on this layer (61 s at seq 140, 613 s at seq 1022, ~54 min at seq 3234). Memory is flat in sequence length; time is linear. That trade is the whole point.

**Hybrid layers.** Real recipes quantize more than the routed bank. The same skeleton extends:
- **shared experts** (2-expert gather): token-blocked pass with its own capture (`shared_out + (live − const)`), routed bank held at its captured constant — Law 2 applies to *any* gather, and a 2-expert bank on a seq-3234 batch otherwise wants ~245 GB of per-assignment dequant;
- **attention / dense projections** (plain `quantized_matmul`): one fused tail pass with both banks constant — a dense matmul's VJP dequantizes its matrix once (~150 MB), so it needs no blocking. This pass deliberately treats the banks as constants w.r.t. their *inputs* (the exact bank input-gradient is precisely the per-assignment cost the decomposition exists to avoid) — an approximation for the tail modules only, disclosed and acceptable at DWQ learning rates;
- **shared-only layers** (no routed bank at all — Inkling's layer 2): same streamed pass with the routed part skipped. Easy to miss: an eligibility check keyed on "routed experts present" silently falls back to the fused path and dies at 82 GiB.

## Allocator drift — the failure you only meet at seq 3000+

The serialized loop allocates and frees thousands of short-lived buffers. MLX's cache happily retains them: on a seq-3234 round the process footprint drifted from ~25 GiB active to **>96 GiB held**, tripping the watchdog's footprint guard mid-round with hard/soft reasons empty and swap at zero — the model was healthy; the *allocator* wasn't. `mx.set_cache_limit(8 GiB)` around the streamed step (restore in `finally`) pins the footprint at `active + 8 GiB` for the whole schedule. If your harness kills on RSS/footprint rather than swap, this is load-bearing.

## Operational residue

- **Sequence-length variance dominates wall-clock.** Our text calibration batches ranged 140→3234 tokens; identical layers differ 50× in round time. Budget schedules on Σ(seq), not on layer count, and set per-layer timeouts from the *longest* batch, not the mean.
- **Sample the process, don't theorize.** Every wrong hypothesis in this campaign (QoS throttling, cache-key misses, hashing) died to `/usr/bin/sample <pid>` + `lsof` snapshots in minutes. The laws above came out of five focused probes, not from reading kernel code.
- The reference implementation (capture/replay, hybrid passes, cache ceiling, eligibility incl. shared-only) lives in the Inkling pipeline's `layerlocal_trainer._streamed_experts_value_and_grad`; the probes are ~80-line standalone scripts and reproduce on any quantized SwitchLinear bank.

## What it bought — two full schedules, and where the bits actually recovered

The streamed backward was not a probe; it certified the scales for two shipped quantizations of this model, each run as a full 66-layer sealed schedule (do-no-harm acceptance: a layer's tuned scales are kept only if held-out boundary NMSE does not regress *at all* — allowed regression 0.0 — else it rolls back byte-exact).

| tier | routed-expert width | layers tuned | **accepted improvements** | which layers |
|---|---|---|---|---|
| quality (3.71 bpw) | 3-bit g128 | 63 | **1** | 40 |
| capacity (2.72 bpw) | 2-bit g128 floor | 66 | **7** | 0, 4, 6, 9, 10, 11, 12 |

The asymmetry is the interesting part: **the 2-bit build accepted 7× more layers than the 3-bit build, and every accepted layer sits early (0–12).** Quantization error is larger at a 2-bit floor, so there is simply more for the do-no-harm pass to recover — and it concentrates where early-layer error compounds through the most downstream depth. At 3-bit the conversion is already close enough that only one layer had a boundary-NMSE gap worth committing. Same optimizer, same contract, same streamed gradients; the bit width decides how much headroom exists to claw back. If you run this pass at 3-bit and see mostly neutral certifications, that is the expected outcome, not a dead optimizer — drop the floor and the accepts appear.

**Does per-layer do-no-harm hold end-to-end?** The contract is enforced on *boundary* NMSE, one layer at a time. To check it survives composition, we ran teacher-forced NLL on 32 held-out text batches (5,926 scored tokens) through the fully merged 2.7 bpw model versus the raw baseline conversion it was built from:

| | aggregate NLL | better on |
|---|---|---|
| baseline conversion | 3.20269 | 13/32 batches |
| **DWQ-certified (merged)** | **3.20349** | **19/32 batches** |

A **+0.0008-nat** difference (≈0.03%) — parity inside the batch-to-batch noise, candidate ahead on more batches than not. Exactly what a do-no-harm pass should produce end-to-end: the seven committed layer deltas neither visibly help nor hurt the composed logits, while the streamed backward's per-layer certification is what *guarantees* nothing regressed. The point of the exact decomposition was never a benchmark bump; it was the right to say "provably not worse, with improvements where they existed" and have the end-to-end numbers agree.

**Takeaway.** MLX's quantized-gather autograd has a simple cost model — *full differentiated extent + one expert matrix per assignment, and fp32-in-module breaks the fused kernels* — and once you shape the backward to that model, a 975B MoE's layer-local DWQ fits in ~25 GiB with gradients you can trust. The 150 GiB "requirement" was never about the model; it was about the op.
