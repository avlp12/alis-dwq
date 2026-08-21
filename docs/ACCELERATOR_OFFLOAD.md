# Offloading part of a quantized model to a second accelerator

A pattern that keeps reappearing: a machine has a matrix unit sitting idle next
to the GPU — Apple's Neural Engine, an NPU, a DSP — and someone splits each
linear layer so both work at once. The wiring is usually the hard part and is
often already written by someone else. What is rarely written down is how to
decide whether the split is worth taking.

This is the method we arrived at after measuring one such path end to end. It is
model- and vendor-agnostic; the numbers are from a 27B hybrid-attention model on
an M3 Ultra, but nothing here depends on that.

## 1. Decide by arithmetic before you benchmark

Split the question by what the phase is bound on.

**Memory-bound phases cannot be helped by a second compute unit.** Compute the
roofline first: weights-in-bytes divided by achievable bandwidth. If the existing
path is already at 70-80% of it, there is no headroom, and a second unit that
shares the same memory — usually through a *narrower* port — adds nothing. In our
case single-stream decode sat at 77% of roofline and speculative decoding was
already past it (110%) by amortising one weight read across several accepted
tokens. No benchmark was needed, and the vendor's own notes agreed.

**Compute-bound phases are where the split can pay.** Prefill, batch scoring,
long-context ingestion. Measure those.

## 2. Measure quality before you tune speed

We tuned the split ratio, the layer count, the tile size, the thread count, and
the sequence length — four hours — and then measured quality and threw all of it
away. The correct order is the reverse: a path that changes numerics has to clear
the quality bar first, because until it does, every speed number is about a model
you would not ship.

## 3. Do not evaluate an approximate path with cosine similarity

Vendors report cosine because it looks reassuring. For a per-layer weight
substitution, `cos ~= 1 - err^2/2`, so a relative error of 4.5e-03 — enormous by
quantization standards — presents as **0.99999**. Sixty-four layers of that
compounded into a model whose top-1 predictions agreed with the original on
**1.6%** of positions.

Use, in this order:

- **Mean KL over every position**, against the unmodified path on the same input.
  Single-position KL is chaotic: if that position sits near a decision boundary a
  tiny perturbation amplifies without bound. Our first sweep produced KL that
  *fell* as we gave the accelerator more work, which is physically impossible and
  was purely an artifact of reading one position.
- **Top-1 agreement over every position**, reported next to KL. They fail
  differently. We measured a configuration at mean KL 0.025 — comfortably inside
  the gap between our own quantization tiers — that still flipped the argmax on
  **6.7%** of positions. KL alone would have passed it.

## 4. Know the accelerator's precision floor before hunting for tricks

Before reaching for clip search, AWQ scaling, or Hadamard rotation, compute what
the accelerator's arithmetic can deliver on *this* weight. For symmetric
per-channel INT8 the relative RMS error is about

```
(max/rms) / (127 * sqrt(12))
```

Our weights had row max/rms = 3.99 — Gaussian, no outliers — giving 9.1e-03, and
we measured 8.75e-03. The floor was already reached.

**That single number tells you which tools can help.** Clip search, AWQ, and
rotation all work by suppressing outliers. On a weight with no outliers they have
nothing to grip: our clip search over alpha in [0.6, 1.0], scored against real
captured activations, bought **1.9%**, and per-row rescaling before handing the
weight over moved the error by **0.04%** (the runtime already quantized per
channel). Run the arithmetic first and you skip a day.

The corollary is worth stating: **these techniques are for outlier-heavy tensors.**
When they pay — and on many models they pay a great deal — it is because
`max/rms` is 10 or 30, not 4.

## 5. Build a simulator of the accelerator's quantizer, then search in it

Every trial that touches the real accelerator costs a compile. Ours was 0.2 s per
program, which makes a grid search impractical and a per-row optimization
impossible.

Model the quantizer in your own framework and *validate the model against the
hardware* before trusting it. Ours — symmetric per-channel INT8 with fp32
accumulation — reproduced the measured error to three significant figures
(8.7405e-03 simulated vs 8.7486e-03 measured; the simulator/hardware discrepancy
was 3.67e-04, twenty-four times smaller than the effect). A per-tensor model gave
7.7e-02 and would have sent us hunting a bug that did not exist.

With a validated simulator, search costs seconds and only the winner needs
hardware confirmation.

## 6. Find the throughput cliff, not the throughput peak

A split has an optimum, and past it the accelerator becomes the critical path and
the whole step waits on it. Ours in the higher-precision mode:

| accelerator share | speedup | error |
|---:|---:|---:|
| 0.10 | 1.09x | 3.36e-04 |
| 0.15 | **1.15x** | 3.88e-04 |
| 0.20 | 1.12x | 4.34e-04 |
| 0.30 | 0.75x | 5.08e-04 |

Sweep past the peak. A sweep that stops at the peak cannot tell you how sharp the
cliff is, and the cliff is what decides whether the operating point is safe.

## 7. Suspect silent no-ops, and read a counter to settle it

Five times on this path a change *succeeded*, logged nothing, and simply did not
run: a package installed without its native extension; a source tree ahead of the
installed package on `sys.path` shadowing the compiled module; a fraction below an
undocumented threshold disabling a whole subsystem; an input shape that did not
match the accelerator's fixed geometry; a module called outside the model's own
forward. Twice the silence read as *good news* — no numerical difference — when
it meant nothing had happened.

**When a change measures as "no difference", confirm it ran before concluding it
was harmless.** Find the counter the runtime already keeps (ours was
`profile_snapshot()["mlp"]["operations"]`), or add one. A no-difference result
without a positive engagement signal is not evidence.

## 8. Watch for first-execution and asynchrony bugs

The first execution of each accelerator program returned garbage; every later
execution of the *same* program was correct, and a warm-up call did not clear it.
The signature — wrong on first read, right on re-read — points at the kernel
returning before the accelerator's write has landed, with the following operation
acting as an accidental barrier.

Test for it explicitly: run the same program twice and compare both results
against a reference. If only the first differs, you have an ordering bug, not a
precision one, and no amount of quantization tuning will fix it.

## 8b. Know what "enough" means before you optimize toward it

A partial gain is often worth nothing — if the split only pays at full exposure,
a configuration that preserves quality at low exposure has not solved anything.
So before spending effort on error reduction, compute the target.

Sweep the exposure (layers, or share) against quality and find where the model
stops tolerating the substitution. Ours absorbed six to eight layers out of
sixty-four. That ratio, run back through how error compounds, said we needed
roughly **an order of magnitude** less error per layer — not a factor of two.

With that number in hand the candidate techniques sort themselves immediately:
clip search offered 1.02x, channel selection 2.2x, and only a precision change
offered 10x. Two of those three were never worth attempting, and we attempted
them because we had not computed the target first.

**One caution about partial improvements.** Selecting which channels to offload —
by per-channel quantization sensitivity, which spanned 296x in our weights, and
exact because output channels permute freely through an elementwise gate — cut
per-layer error 2.2x and moved end-to-end quality **not at all** (KL 9.94 to
9.93). Past a threshold the model is in a saturated regime where the output is
already unrelated, and local improvements do not register. Always confirm a
per-layer win end to end before building on it.

## 8c. The speed usually *is* the precision

The uncomfortable pattern this ends in: the accelerator was fast because it ran
INT8, and INT8 was what the model could not absorb. Its accurate mode existed and
was ten times better, but sustained only a fraction of the work — past a 15%
share it became the critical path and the split turned negative. There was no
setting with both.

Before starting, ask what the accelerator's fast path actually does differently.
Ours applied **one scale per output channel** where the model's own container
carries **one per group of 64 inputs**. That single structural difference — the
loss of local scale adaptation — is the whole story, and it was visible in the
data sheet before any measurement. A model quantized group-wise has already spent
its error budget on the assumption that scales adapt locally; an accelerator that
flattens them is not offering a cheaper version of the same thing.

## 9. Isolation, so the experiment cannot damage a working install

Vendor runtimes usually read a configuration tree and hold a port. Find the
environment override (ours was `OMLX_BASE_PATH`), point it at a scratch
directory, and verify the redirect *before* the first run. A separate virtualenv
is not enough on its own — the configuration tree is the shared state.

## What this is worth knowing for

We refused the path: the quality-preserving operating point returned about 1.15x
on the affected matmul, roughly +5% end to end, against a machine-level
alternative already delivering +72%. The measurements are in
[qwen38_alis_mlx/docs/ane-hybrid.md](https://github.com/avlp12/qwen38_alis_mlx/blob/main/docs/ane-hybrid.md).

The method survives the verdict. The next accelerator split will be decided by
the same six numbers: the roofline share, the precision floor from `max/rms`, the
simulator's fidelity, the cliff location, mean KL, and top-1 agreement.
