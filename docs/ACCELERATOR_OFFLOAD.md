# Offloading part of a quantized model to a second accelerator

> **CORRECTED (2026-08-21).** This document was written to explain why an
> accelerator offload was not worth taking. It was worth taking: driven through
> the vendor's own entry sequence — pre-load patches, then a load-time warm-up of
> every compiled program — the path delivers **+19% prefill at mean KL 0.000264
> and 100% top-1 agreement**. My harness called the enable function directly and
> skipped the warm-up, so every program's first execution returned garbage, and I
> read that as precision loss compounding over sixty-four layers.
>
> The method below stands and has been extended with the check that would have
> caught it (§7b). The worked example it was drawn from reached the wrong verdict,
> and that is exactly why §7 and §8 are in it.


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

## 7b. Reproduce through the vendor's entry sequence, never around it

The single most expensive mistake in the campaign behind this document: I called
the vendor's `enable_*` function directly on a model I had loaded myself, rather
than driving their engine's own initialisation path. The function worked — the
accelerator ran, and the runtime's profiler confirmed it with a positive
operation count. What it skipped was a load-time warm-up of every compiled
program, without which each program's *first* execution returns garbage.

Every engagement check I had came back positive, so I spent days explaining a
numerical catastrophe that was an initialisation bug I had introduced.

**Two rules follow.**

*Drive the vendor's path, not the vendor's function.* Find where their engine or
server initialises the feature and reproduce that whole sequence, in order.
A public function reachable from outside that sequence is not a supported entry
point just because it is importable.

*A positive engagement signal is not a positive correctness signal.* §7 says to
confirm a change ran before concluding it was harmless, and I applied it — the
accelerator was running. The question I did not ask was whether it had been
brought up correctly. Find the initialisation the vendor logs (ours printed
`Warmed N procedures at load`) and confirm **that** line appears, not merely that
work happened.

The corollary for reading logs: absence of a log line proves nothing until you
have checked that the module's logs reach your sink at all. Ours did not — the
patch package's log lines never reached the server's log file, so their absence
was uninformative in both directions.

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

## 8d. An automatic tuner without a quality gate optimizes toward destroying quality

The vendor path we measured ships with a built-in tuner that benchmarks the split
ratio on the user's own machine and recommends a configuration. We read it: it
computes `speedup_percent` and selects by minimum time. There is no perplexity,
no KL, no logit comparison, no accuracy check anywhere in it.

That is not an oversight with a small consequence. The search space is *ordered*
by how much work moves to the lower-precision unit, so time-only optimization
walks straight to the configuration that damages the model most. A user who runs
the tuner and accepts its answer gets exactly that.

**If you build one of these, the gate is the feature.** Score candidates on
`speedup` *and* a quality measure, and refuse any candidate that fails the quality
bar regardless of its speed. If quality is expensive to evaluate, evaluate it on
the finalists rather than dropping it — a tuner that cannot reject is not a tuner,
it is a stopwatch.

The same applies to reading someone else's numbers. Before trusting a published
speedup, find out what their tuning loop optimized. If quality never entered the
objective, the number is real and the configuration is still unusable.

## 9. Isolation, so the experiment cannot damage a working install

Vendor runtimes usually read a configuration tree and hold a port. Find the
environment override (ours was `OMLX_BASE_PATH`), point it at a scratch
directory, and verify the redirect *before* the first run. A separate virtualenv
is not enough on its own — the configuration tree is the shared state.

## 10. A second lever does not multiply with the first — it competes with it

Once the offload paid, we composed it with a lever we already had: a
layer-pipelined prefill across two machines over Thunderbolt, worth about 1.9x on
its own. The expectation was 1.9 x 1.26.

Measured against an offload-off control **inside the same run**, the composition
was **+10.6% at a 32K prompt and -3.9% at 8K**. The offload improved each single
machine everywhere (+17.1% / +11.5%), and yet the two-machine ratio *fell*: 1.90
to 1.80 at 32K, 1.71 to 1.47 at 8K.

The reason generalizes past this hardware. The pipeline ratio is set by the
balance between compute and link transfer. The offload removes compute only, so
transfer becomes a larger share of what remains, and the pipeline's *relative*
gain shrinks. Two levers aimed at different bottlenecks do not stack: relieving
one promotes the other, and the second lever earns less than it did alone.

Three consequences worth carrying:

- **Measure the composition, never multiply the parts.** Multiplying our two
  separately measured gains predicted about +17% at 32K. The truth was +10.6%,
  and at another prompt length it was negative. Both parts were correctly
  measured; the product was still fiction.
- **The control must be in the same run.** Ours alternated arms inside one
  process, so drift, thermal state and page cache could not masquerade as the
  effect.
- **Expect a crossover, and find what parameter moves it.** Ours was prompt
  length, through chunk count: the offload only pays at chunk 2048, the pipeline
  prefers chunk 1024, and a long prompt has enough chunks to amortise the larger
  one while a short prompt does not. A composition that is positive somewhere and
  negative elsewhere is the normal case, not a measurement failure — so ship the
  branch, not a single number.

## 11. Some of the gain is in the loader, not in the accelerator

Nine of our twenty-six points came from a call that must run *before the weights
load* — a layout patch on the quantized MLP. Same accelerator, same split, same
tuning: +15.5% without it, +24.9% with it. It is not an accelerator setting and
it does not appear in any accelerator counter.

When a vendor's own numbers exceed yours with an identical configuration, look
upstream of the feature before you re-tune it. The gap is more often in what the
model looked like when it was handed over than in how it was split. This is the
same failure as §7b in a smaller key: reproduce their *sequence*, not just their
settings.

## What this is worth knowing for

We took the path, after first refusing it for the wrong reason. Tuned and
composed, our prefill went from 733 tok/s to **863.5 tok/s** on two machines at
long prompts, and a single machine gained 11-17% at every length, at a KL two
orders of magnitude below the gap between our own quantization tiers. The
measurements are in
[qwen38_alis_mlx/docs/ane-hybrid.md](https://github.com/avlp12/qwen38_alis_mlx/blob/main/docs/ane-hybrid.md).

The method survives its own worked example reaching the wrong verdict twice — once
by skipping the vendor's entry sequence, once by assuming two levers multiply. The
next accelerator split will be decided by the same numbers: the roofline share,
the precision floor from `max/rms`, the simulator's fidelity, the cliff location,
mean KL, top-1 agreement — and, if a second lever is already in place, a
composition measured against a control in the same run.
