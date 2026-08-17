# Case study: Qwen3.8-27B — the converter that silently dropped the vision tower

> **Dedicated campaign repo: [avlp12/qwen38_alis_mlx](https://github.com/avlp12/qwen38_alis_mlx)** — the full verdict ledger, topical docs (conversion integrity, kernels, speculative, two-box prefill, methodology, external dossiers), code snapshots, harness, and raw results for this model. This case study is the conversion-integrity cut; that repo is the complete record.

A conversion-integrity and decode case rather than a DWQ one. Nothing here is about
bits: the three builds are plain uniform quants. What made the campaign worth writing
down is that the **standard conversion path turned a multimodal model into a text-only
one without a single warning**, that a *second* silent failure lived one function away in
the same file, and that the decode-speed work which followed found the real reason
speculative decoding was underperforming on this stack — a small-`M` gap in MLX's
quantized GEMM, not the algorithm.

**Read Finding 6 before quoting Findings 4 or 5.** The verdict this case study first
reached ("MTP wins, DSpark's overhead is structural") was overturned by re-tracing an
overhead we had never itemized: three defects, all of them ours, one of which was that the
kernel we had just built was never actually called outside its own benchmark. The
superseded verdict is kept in place rather than edited away — the reversal is the more
useful half.

- **Reusable rules + runnable oracles**, generalized off this model:
  [docs/PORTING_INTEGRITY.md](../../docs/PORTING_INTEGRITY.md).
- **Raw receipts** (measurement records, KV sweep, harness scripts, verdict ledger): this
  directory — see [Raw receipts](#raw-receipts-this-directory) below.

## The model

**Qwen/Qwen3.8-27B** — 27B **hybrid**: 48 GatedDeltaNet linear-attention layers +
16 full-attention layers, vocab **248,320**, context 262K, and a **vision tower**
(`model.visual.*`, **333 tensors / 0.461B params / 0.92 GB bf16**). The checkpoint also
carries a 31-tensor MTP head.

Three builds (uniform, g64), all preserving the 333 vision tensors **and** the 31 MTP
tensors:

| build | size | decode | prefill | top-1 vs bf16 (ko) |
|---|---|---|---|---|
| 8-bit | 27.9 GB | 21.8 tok/s | 429 tok/s | 99.1% |
| 6-bit | 21.5 GB | 27.3 tok/s | 424 tok/s | 97.3% |
| 4-bit | 15.2 GB | 37.5 tok/s | 436 tok/s | 85.7% |

(M3 Ultra 512 GB.)

## Finding 1 — the converter drops multimodality, and the artifact can't tell you

mlx-lm's `qwen3_5` `sanitize` discards every `model.visual.*` tensor, and
`utils.save_config` does `config.pop("vision_config")`. Together those two lines mean a
27B **VL** model converts to a text-only model **with no warning**, and — because the
config key is popped as well — **the output is indistinguishable from a conversion of a
text-only model**. There is nothing in the artifact to notice.

The consequence is not hypothetical. We queried the remote weight index of **all 12
public MLX builds of this model**: **every one carries 0 vision tensors**, including the
ones with `-vision` in the repo name. Seven of the twelve ship a
`preprocessor_config.json` — the image pipeline's config — next to weights that cannot
process an image.

### The fix is a general gate, not a model patch

The tempting fix is to special-case `model.visual.*` in `qwen3_5`. We put the mechanism
in **`save()`** instead, so `convert` and `awq` pass through the same gate and any future
architecture gets it for free: a model may declare **`passthrough_patterns`**, and
tensors matching them that the model itself does not consume are copied through **as
original bytes**.

Four design points, each of which was load-bearing:

1. **Skip keys that were already written, by suffix-aware comparison.** `sanitize`
   reparents the MTP head (`mtp.x` → `language_model.mtp.x`), so a naive exact-name
   check re-emits those tensors under their old names and the index ends up with two
   entries for one weight.
2. **Read the bytes back and verify before advertising them in the index.** An index
   entry pointing at an unwritten or short tensor is a load-time failure that looks like
   a corrupt download.
3. **Keep `vision_config` only when the weights were actually preserved.** Both
   directions are bugs: advertising a tower a stripped checkpoint doesn't have, and
   hiding a tower that is present. The config must follow the bytes.
4. **Carry the preprocessor config along with the weights** — the seven repos above are
   what the alternative looks like.

Result: the vision tower survives **byte-exact across all 333 tensors**, text inference
is unchanged (peak 15.7 GB; decode **37.39 vs 37.61 tok/s**, inside measurement noise).

### The implementation was never the missing piece

**mlx-vlm 0.6.13 already supports `qwen3_5`**, so images worked with **zero lines of
porting** the moment the weights were present. We verified on a hand-drawn shapes image
(red circle top-left, blue square top-right, green triangle at the bottom): the model
described colour, shape and position correctly for all three.

The ecosystem was not missing a vision implementation. It was missing the **weights**,
because every build had been made by a converter that threw them away quietly.

## Finding 2 — using MTP presence as a format discriminator breaks silently

Stock `qwen3_5` `sanitize` decides whether to shift the norm weights with:

```python
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
```

The intent is "this is a raw HF checkpoint". But an **already-converted** checkpoint that
*kept* its MTP head still satisfies the first disjunct, so the norms get shifted a
**second** time (γ 0.94 → 1.94). Nothing crashes; nothing warns; generation collapses.
Measured on our Korean slice: **NLL 1.679 → 17.460** — worse than a uniform distribution
over the 248,320-token vocabulary.

The trap has a second stage. Builds *without* the MTP head measure perfectly fine, so the
natural reading of the evidence is "the MTP-preserving build is broken" — i.e. the bug
frames the feature that exposes it.

Upstream already had [issue #1197](https://github.com/ml-explore/mlx-lm/issues/1197) and
[PR #1623](https://github.com/ml-explore/mlx-lm/pull/1623) for this, from a 35B MoE. We
reproduced it independently on a **dense 27B**, filed
[**ml-explore/mlx-lm#1735**](https://github.com/ml-explore/mlx-lm/pull/1735), and left the
validation data on both threads.

**A side lesson that cost more than the bug.** Our harness had been silently importing the
**installed stock mlx-lm** instead of the fork under test. `measure.py` now puts the fork
explicitly at the front of the path and **fails loudly** if stock is what gets imported.
Every measurement in this campaign predating that change had to be re-run.

## Finding 3 — small-`M` non-amortization in quantized GEMM (upstream mlx#4265)

The 27B forward is not flat in token count, and the shape of the curve is not physical.
Against the roofline, **S=8 is 3.65× off**, S=32 is 1.44×, S=128 is 1.19× — the *small*
batches are the inefficient ones, which is backwards.

Queue-batched microbenchmarks located it. **bf16 is flat**: `M=2..16` all sit at
**2.00×** of `M=1`. **4-bit `quantized_matmul` is nearly linear**: **5.28× at M=7**,
**6.32× at M=16**. `M=1` sits on the roofline; `M=7` is **2.75× off** it. Identical on
MLX 0.31.2 and 0.32.0. Filed as
[**ml-explore/mlx#4265**](https://github.com/ml-explore/mlx/issues/4265).

That single fact explains the model's S-curve:

| tokens/forward | 1 | 7 | 8 | 32 | 128 |
|---|---|---|---|---|---|
| forward (4-bit) | 29.7 ms | 62.5 | 77.1 | 105.4 | 347.9 |
| vs roofline | — | — | **3.65×** | 1.44× | 1.19× |

**Speculative decoding lives exactly in the worst region.** Verify widths of 4–8 tokens
are the whole point of drafting, and they are precisely where the quantized GEMM fails to
amortize.

**Acquitted: the linear-attention kernels.** Measured standalone they are **flat in T**
(0.28–0.37 ms for T=1..64) — the recurrent state is held in registers, so the traffic is
already amortized. The hybrid architecture was the obvious suspect and it was innocent.

## Finding 4 — a split-K MMA kernel, and the latency trap in the middle of it

Three versions.

**v1 — scalar + threadgroup staging.** Amortizes correctly, but runs at **0.30×** of MLX.

**v2 — `simdgroup_matrix` 8×8 MMA.** Now **flat in M** (+12% from M=1 to M=8), and moving
the barrier from every k-tile (8) to every **quantization group (64)** took it
0.29 → **0.15 ms**.

**Wired into the model, v2 was 0.56–0.82× — slower.** Every per-shape comparison
said it should win (layer-count-weighted **1.35×**, `lm_head` **3.45×**) and the Python
wrapper cost was measured at zero.

The **dependent-chain benchmark** found it (N=K=5120, M=7):

| | independent | chained |
|---|---|---|
| MLX | 0.0765 ms | 0.0789 ms (**+3%**) |
| ours (v2) | 0.0680 ms | 0.1530 ms (**+125%**) |

**We won throughput and lost latency.** A queue-batched benchmark overlaps independent
calls, so it measures throughput and hides exactly the quantity a decode loop is made of.

**v3 — split-K.** Eight simdgroups split K eight ways, and `x` is loaded from device
memory straight into the MMA registers, removing the staging step entirely. Chained
latency **0.153 → 0.0558 ms**, i.e. **1.41× vs MLX** on the metric that matters.

The model's S-curve flattens accordingly, with output equivalence (top-1 match) held:

| verify width | before | after |
|---|---|---|
| S=6 | 62.5 ms | 44.5 |
| S=7 | 70.5 ms | 44.6 (**1.58×**) |
| S=8 | 77.1 ms | 43.3 (**1.78×**) |

**Crossover is M=6 on a dependent-chain basis** — not on the queue-batched one. At M=4 the
kernel is 0.98–1.10× (neutral), and dispatching it there actually made MTP k=3 *slower*.
The gating window is narrow and it has to be measured with the right benchmark.

**Postscript — none of this reached production.** `fast_qmm.enable()` was called only from
the measurement scripts that produced the numbers above, never from any loading path, so
the kernel was **dead code in every other path** for the rest of the campaign. And the
window it wins in (M ≤ 8) turned out not to be the window the speculative loop actually ran
in. Both are Finding 6 — read that before quoting anything downstream of this section.

## Finding 5 — a reference implementation may not be the canonical one (DSpark)

`RadixArk/Qwen3.8-27B-DSpark` is a 1.359B **block-diffusion** drafter: seven slots are
generated **in parallel in one forward** (not autoregressively), and its attention is
dual-source — it takes the target's intermediate-layer hidden states (layers
[4, 16, 28, 40, 52]) as context.

Our MLX port is **numerically equivalent** to the PyTorch reference: cosine
**1.00000000**, max abs error **1.1e-4**. And it lost end-to-end at **0.73×**.

**The reason: the `spec_generate` bundled in the repo is DFlash's loop.** It never calls
DSpark's two distinguishing heads — `markov_head` (**127.1M**, 9.3% of the drafter) and
`confidence_head`. Both loaded; neither was ever invoked. DSpark's actual inference path
lives inside SGLang and is not in the repo. **We had taken the reference as canonical and
were running DFlash.**

Wiring the Markov head alone:

- acceptance **2.23 → 3.31** on a code prompt;
- under the card's stated conditions (block 7, temp 0.6, English/code), acceptance
  **exceeds the published card**:

| workload | ours | card |
|---|---|---|
| HumanEval-style | 4.32 | 3.47 |
| MBPP-style | 4.17 | 3.67 |
| GSM8K-style | 4.96 | 4.57 |
| MATH-style | 5.19 | 4.08 |
| MT-Bench-style | 3.75 | 3.10 |
| **average** | **4.48** | **3.39** |

**The confidence head is best switched off.** At block 6: **off 45.0 tok/s** vs
tau 0.10 / 0.25 / 0.50 = **37.1 / 38.5 / 42.8**. The head exists to avoid paying for a
wide block when the drafter is unsure — and once split-K flattened verify cost across
M ≤ 8, there is nothing left to save by narrowing the block, only tokens to lose. **Our
kernel work made one of the drafter's two heads obsolete.**

Two alternatives were tried and **rejected**:

- **bf16 drafter** (the card's stated configuration): no gain — acceptance **4.19** vs
  the 4-bit drafter's **4.23**.
- **The reference's draft-slice convention** (`[:, -B+1:]`): even after the Markov head is
  wired, acceptance is **2.35**, far below our shifted read's **4.23**.

### The verdict we reached here — and it was wrong

> **Superseded. Kept deliberately; see Finding 6 for how it broke.**
>
> Final end-to-end, averaged over three English/code prompts:
>
> | path | tok/s | vs plain |
> |---|---|---|
> | plain | 37.4 | — |
> | **MTP k=2** | **49.3** | **1.32×** |
> | DSpark + Markov, block 6 | 45.0 | 1.20× |
>
> *"DSpark wins acceptance decisively (4.23) and still loses end-to-end. The entire gap is
> overhead: a separate 1.36B forward pass plus a 248k-vocabulary `lm_head`, every step. MTP
> pays neither — its extra layer reuses the target's own hidden state."*

The reasoning was clean, the numbers were real, and the conclusion was still wrong. What we
had actually measured was **our own integration**, and we attributed its cost to the
algorithm. In particular, "the entire gap is overhead" was a claim about a **32 ms
per-step residual we had never itemized** — and an unaccounted residual is not evidence
about anything.

## Finding 6 — the reversal: three defects, all of them ours

Re-tracing the residual instead of explaining it turned up three faults on our side. None
of them are about DSpark.

### 1. The kernel was dead code

`fast_qmm.enable()` **was never called anywhere in the repo** — `grep` returns zero hits
outside the file that defines it. It had only ever been switched on inside the ad-hoc
measurement scripts that produced Finding 4's numbers. Every production path —
`mlx_lm.generate`, the server, the MTP loop, and the DSpark loop being judged — ran
**without the kernel**.

The size of that mistake: with the kernel off, S=8 verification costs **70.1 ms**, not the
43.1 ms Finding 4 reports. We built the thing that fixes the campaign's central bottleneck,
measured it correctly, wrote it up — and then judged everything downstream on a stack that
did not have it.

**Building a feature and not wiring it was the single largest loss of this campaign.** It
cost more than any bug in this document, and it left no trace: nothing fails, nothing warns,
the numbers are simply the numbers of the old stack. Fixed by having `utils.load()` call
`fast_qmm.enable()`, with `MLXLM_NO_FAST_QMM=1` as the killswitch.

### 2. The verify width kept leaving the kernel's window

The split-K kernel wins only at **M ≤ 8** (Finding 4). Nobody had checked whether the loop
actually *stays* there. A width histogram of the original block-6 loop:

| verify width | step cost | share |
|---|---|---|
| ≤ 8 | 54.6 ms | — |
| 9–12 | 106.8 ms | — |
| 13+ | 142.8 ms | — |

**mean width 9.50, max 16** — the loop lived mostly *outside* the window it was optimized
for, in the regime that costs 2–2.6× more. The `max_pending` design let width grow to
`L + n_spec`. A one-line clamp, `min(n_spec, 8 - L)`:

- **41.2 → 59.1 tok/s (+43%)**, step 90.5 → 52.0 ms;
- width mean 9.50 → **7.10**, max 16 → **8**.

An optimization with a shape window is not a property of the kernel alone. It is a claim
about the **distribution the model actually runs at**, and that distribution has to be
measured.

### 3. The optimal block is 8, not 7 — the card's number is a reference, not a ceiling

Block 7 is the drafter's trained block width, so we had treated it as the ceiling. Measured
over three prompts:

| block | 6 | 7 | **8** | 9 | 10 | 12 |
|---|---|---|---|---|---|---|
| tok/s | 62.1 | 63.6 | **71.0** | 70.4 | 68.0 | 66.0 |

Slots **beyond** the trained width pay. And block 9 is the cleanest statement of the
acceptance-is-not-throughput rule in this whole document: it has **higher acceptance than
block 8 (3.55 vs 3.46) and still loses**, because the drafter forward costs more than the
extra accepted token returns.

### Two side gains, and one of them is welded to the kernel

- **`pad_lm`** — round the `lm_head` batch up to the kernel window: that stage goes
  **4.02 → 1.45 ms/step**, worth **+5%** end-to-end. With the kernel **off** the same change
  is **−2%**: it is not an independent optimization, it is part of the kernel.
- **`defer_sync`** — keep the Markov chain's `prev` as a 0-d array and collapse draft +
  posterior into a single `.tolist()` after verification: **+1.9%**, output **bit-identical**.

### The step accounting now closes to zero error

With the kernel on, width clamped to 8, block 8, `pad_lm`:

| stage | ms |
|---|---|
| verify | 43.06 |
| draft | 2.75 |
| `lm_head` | 1.45 |
| markov | 1.79 |
| posterior | 0.54 |
| commit | 0.29 |
| **sum** | **49.9** |
| **measured step** | **49.0** |

The earlier "unexplained 32 ms at block 6" had exactly two constituents: the disabled kernel
and the width excursion. **An unaccounted overhead is usually an unmeasured item, not a
mysterious one** — and treating it as mysterious is what let us publish a verdict on top of
it.

### Final numbers — the ranking inverts

Promoted configuration, measured from stock defaults after promotion
(`max_width=8, pad_lm=True, use_conf=False, defer_sync=True, block_size=8`):

| configuration | 4-prompt avg | 3-prompt (en/code/math) |
|---|---|---|
| plain | 37.63 | 37.64 |
| MTP k=2 | 50.36 (1.34×) | 55.71 (1.48×) |
| **DSpark** | ~~62.21 (1.65×)~~ **retracted** | ~~71.86 (1.91×)~~ **retracted** |

> **These two figures were withdrawn on 2026-08-16.** The harness did not stop at
> end-of-sequence, so the math prompt's post-EOS self-copy inflated acceptance to 4.53 and
> carried the headline. Under the corrected protocol (four prompts, EOS-cut, long-form,
> medians of three) the ordering reverses: gated MTP k=4 leads at **52.8 tok/s (1.40×)** and
> DSpark trails at **48.3 (1.28×)**. The struck rows stay so the retraction is legible.


**Lossless**: all four prompts (chat / code / math / ko) are token-identical to plain
greedy. **DSpark is now the production recommendation for the 4-bit build**, with MTP k=2
second — the opposite of the verdict above.

**One honest exception, unresolved.** On Korean alone, **both** speculative paths are
*slower* than plain decoding: plain 37.6 · MTP 34.3 · **DSpark 33.3**. We have not
decomposed the cause. The operational rule for now is to turn speculation off for Korean
workloads. Also left unmeasured: reproducing this on the 8-bit target, and drafter context
growth past 32k (verified harmless to 2.4k — TTFT 5.58 s vs plain 5.74 s).

### External corroboration — circumstantial only

`ARahim3/mlx-dspark` is the only public MLX implementation of DSpark for Apple Silicon, and
its **v0.10.0 added Qwen3.8-27B support on 2026-08-15**. Its registry advertises
`speedup ~1.7x`, consistent with the third-party report we could not reconcile earlier
(58 / 34.9 = 1.66×), and our 3-prompt 71.86 is **+24%** over that 58. Notably, their README
states that on Apple Silicon verify cost grows with token count and caps the achievable
speedup at 2–3× — **that is precisely the constraint the split-K kernel removes**. We have
no statement from the third party themselves, so treat all of this as **circumstantial**: it
is consistent with our numbers, it does not confirm them.

## Measurement-protocol lessons

- **Queue-batched benchmarks hide latency.** They do avoid the ~250 µs per-call
  synchronization floor, which is why we used them — but a decode-path kernel must
  *also* be measured on a **dependent chain**. In this campaign "1.52× in the microbench"
  and "0.74× in the model" were **both true at once**, and only the chained bench could
  say why.
- **A single-prompt benchmark badly overestimates speculative decoding.** The same MTP
  k=2 measured **1.41×** on a code prompt alone, **1.32×** averaged over three
  English/code prompts, and **1.10×** over three prompts including Korean. Report the
  multi-workload number — and report which prompts, because the spread across workloads
  is larger than most of the levers being compared. (Korean is not a mild case here: after
  the Finding 6 fixes, on Korean alone *both* speculative paths are **slower than plain**.)
- **Force the harness to fail loudly about which library it imported.** The silent
  fallback to a stock install was the substance of this campaign's worst incident, not a
  footnote to it.
- **Verify that an optimization is *called*, not merely written.** The split-K kernel was
  enabled only inside the measurement scripts that benchmarked it; every other path ran
  without it for the rest of the campaign, and no test noticed. `grep` for the call sites of
  anything you have just built: zero hits outside its own definition means it does not
  exist as far as production is concerned.
- **A kernel with a shape window needs a histogram, not an assumption.** Ours wins at
  M ≤ 8; the speculative loop was running at **mean width 9.50, max 16**, i.e. mostly in the
  2–2.6× *worse* regime. Instrument the width the loop actually runs at before crediting —
  or debiting — the kernel.
- **An unaccounted overhead is usually an unmeasured item.** A 32 ms per-step residual was
  explained away as "drafter overhead" for most of a campaign; itemized, it was two of our
  own defects and the accounting closed to **49.9 ms predicted vs 49.0 ms measured**. Do not
  reason about a residual you have not decomposed.
- **KV-cache quantization is a long-context lever only** (measured at 16K). The hybrid
  architecture means only **16 of 64 layers hold a KV cache at all**. On the 4-bit build:
  peak **20.80 GB** (bf16 KV) → **20.30** (8-bit) → **20.03** (4-bit), **top-1 100%
  preserved in all three**, decode −3%. The cache itself goes 16.8 → 4.2 GB at the full
  262K context — which is where the trade becomes worth making, and nowhere shorter.

## Raw receipts (this directory)

Small files only — no weights, images or raw logs.

| file | what it is |
|---|---|
| [`m_bf16.json`](m_bf16.json) | the bf16 source measured by the same harness — the reference every top-1 agreement is computed against |
| [`m_q8v.json`](m_q8v.json) / [`m_q6v.json`](m_q6v.json) / [`m_q4v.json`](m_q4v.json) | per-build measurement records (`v` = vision-preserving): size, bits/group_size, peak memory, per-prompt decode tok/s, 2048-token prefill tok/s, and the per-slice probe — `nll` plus the **full top-1 id sequence**, which is what makes agreement recomputable without re-running the model |
| [`kv_q4v.json`](kv_q4v.json) / [`kv_q8v.json`](kv_q8v.json) | the KV-quantization sweep (`kv_bits` ∈ {none, 8, 4}) at 16K context: prefill, decode, peak and top-1 ids per setting |
| [`measure.py`](measure.py) | produces the `m_*.json` records |
| [`table2.py`](table2.py) | renders the comparison table from the `m_*.json` files |
| [`kv_measure.py`](kv_measure.py) | produces the `kv_*.json` sweep |
| [`DSPARK_FINDINGS.md`](DSPARK_FINDINGS.md) | the campaign's **verdict ledger** (Korean, AIF-structured: information / inference / conflict / decision nodes with IDs). Every number above resolves to a node there — including the overturned ones, which are preserved rather than deleted: `[CA5]`/`[CA6]` are the rejections the split-K work forced, `[I25]` is where the missing DSpark head wiring was found, `[CA7]` is the dead-kernel discovery ("building a feature and not wiring it was the campaign's largest single loss"), and `[PA12]` is the reversal of the production recommendation |

| [`HANDOFF.md`](HANDOFF.md) | the **zero-prior-knowledge handoff** for the whole campaign — read this first if you are picking it up |
| [`results/kl_*.json`](results/) | exact full-vocab KL to bf16 for ten builds (mine and the community's), 512-token block SEs |
| [`results/allsum.json`](results/) / [`results/mtp_tp2.json`](results/) / [`results/decode_tp2.json`](results/) | the two-box tensor-parallel spike: RDMA `all_sum` latency, plain TP2, and the TP2 x speculation composite |
| [`results/out_tp2_*.jsonl`](results/) | the served two-box stack, four prompts x three alternated boots |
| [`tier_chart.png`](tier_chart.png) | size-vs-fidelity across the ten builds |

## Where the campaign ended up

| axis | first build | now | protocol |
|---|---|---|---|
| decode, 4-bit | 37.6 tok/s | **52.8** gated MTP · **74.2** on two boxes | four-prompt EOS-cut, dependency-chained |
| decode, served | — | **53.1** one box · **62.9** two boxes | HTTP streaming, same protocol |
| prefill 8K | ~430 tok/s | **733** (layer pipeline) · ~650 (TP2 stack) | bitwise-identical output |
| quality | — | exact full-vocab KL, ten builds | paired ctx-2048 windows, block SEs |

The full account, including six rejected levers with the diagnosis each one bought, is in
[`HANDOFF.md`](HANDOFF.md) and the [ledger](DSPARK_FINDINGS.md).

Reproduce the table — this is the exact command, run from this directory:

```bash
python3 table2.py
```

It reads only the `m_*.json` files and never the build directories, so it works on any
box. Two AWQ rows are listed in the script but not shipped here; they print `(미측정)`.

`table2.py` also carries the campaign's tripwire: if a build's `ko` NLL exceeds the bf16
reference's by more than 3×, it warns you to check whether the harness picked up stock
mlx-lm instead of the fork. That is Finding 2 encoded as a guard.

### The fork pin, verbatim

`measure.py`'s header is the campaign's central lesson expressed as code, so it is worth
reading rather than summarizing:

```python
# 포크를 명시적으로 앞세운다. 이걸 안 하면 설치된 스톡 mlx-lm 이 잡히는데,
# 스톡 qwen3_5.sanitize 는 `has_mtp_weights` 를 raw-HF 판별자로 써서 **이미 시프트된**
# norm 가중치에 +1.0 을 한 번 더 얹는다(γ 0.944→1.944). 증상은 크래시가 아니라
# nll 1.7→17 의 조용한 붕괴이고, 하필 MTP 를 보존한 빌드만 골라서 망가져 보인다.
for _fork in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_fork, "mlx_lm")):
        sys.path.insert(0, _fork)
        break

import mlx.core as mx
import mlx_lm
...
# 조용히 스톡으로 되돌아가면 수치가 전부 무의미해지므로 여기서 크게 실패한다.
_used = os.path.dirname(mlx_lm.__file__)
if "site-packages" in _used:
    raise SystemExit(f"스톡 mlx-lm 이 잡혔다({_used}) — 포크 경로를 확인하라")
print(f"[measure] mlx_lm = {_used}", file=sys.stderr)
```

(The comment says: pin the fork, because stock `qwen3_5.sanitize` uses `has_mtp_weights`
as a raw-HF discriminator and adds +1.0 to already-shifted norm weights — γ 0.944→1.944 —
and the symptom is not a crash but a quiet collapse from nll 1.7 to 17, on precisely the
builds that preserved MTP.)

Two halves, and **both are required**. Putting the fork on `sys.path` is the fix; the
`site-packages` check is what makes the *absence* of the fix loud. A path insert alone
fails silently the first time the fork moves, and every number the harness prints is then
measured against a different library than the one you believe you are testing.

`kv_measure.py` carries the same pin, plus the two timing rules in its docstring: batch the
`mx.eval` (a per-iteration eval lays down a ~250 µs synchronization floor) and start the
decode timer **after the first token**, so prefill is not folded into decode.

## Findings

1. **A converter can drop a whole modality silently, and the artifact cannot tell you.**
   `sanitize` dropping `model.visual.*` plus `save_config` popping `vision_config`
   produces a text-only build that is indistinguishable from a text-model conversion. All
   **12** public MLX builds of this model carry **0** vision tensors — including the ones
   named `-vision`.
2. **Fix pass-through generally, in `save()`, not per model.** A declared
   `passthrough_patterns` set copies unconsumed tensors as original bytes through both
   `convert` and `awq`; suffix-aware skip, read-back verification, config-follows-bytes,
   and carrying the preprocessor config are all required for it to be correct.
3. **Never use the presence of an optional head as a format discriminator.** MTP-as-
   "this is a raw checkpoint" double-shifts the norms of a converted checkpoint that kept
   its head: γ 0.94 → 1.94, ko NLL 1.679 → **17.460**, no crash — and it frames the very
   feature that exposes it.
4. **Throughput wins and latency wins are different wins.** A kernel that beat MLX in a
   queue-batched microbench ran the model 0.56–0.82×; the dependent chain showed +125%
   latency against MLX's +3%. Fixing that (split-K) gave 1.41× chained and flattened the
   model's verify curve to 1.58–1.78× at widths 7–8.
5. **When speculative decoding underperforms, price the verify curve before blaming the
   algorithm.** The weakness here was MLX's quantized GEMM failing to amortize at small
   `M` (5.28× at M=7 where bf16 is flat at 2.00×), not the drafting method.
6. **A published reference implementation is not necessarily the canonical inference
   path.** Numerical parity to the reference (cosine 1.00000000) proved only that we had
   faithfully ported the wrong loop — the repo shipped DFlash's `spec_generate`, and
   DSpark's own path lives in SGLang. Wiring the missing head took acceptance 2.23 → 3.31
   and end-to-end 0.73× → 1.20×.
7. **Building a feature and not wiring it was the single largest loss of this campaign.**
   `fast_qmm.enable()` had zero call sites outside its own definition: the kernel that fixes
   this stack's central bottleneck was live only inside the scripts that benchmarked it,
   while `generate`, the server, and both speculative loops ran without it (S=8 verify
   70.1 ms instead of 43.1). Nothing failed and nothing warned — the numbers were simply the
   old stack's. `grep` for call sites of anything you just built.
8. **An optimization with a shape window is a claim about the run-time distribution.**
   The kernel wins at M ≤ 8; the loop was running at mean width **9.50**, max 16 — mostly in
   the regime that costs 2–2.6× more. A one-line clamp was worth **+43%**. Histogram the
   width before crediting or debiting the kernel.
9. **An unaccounted overhead is usually an unmeasured item.** The "32 ms of drafter
   overhead" that justified our first verdict was findings 7 and 8 wearing a costume; once
   itemized, the step accounting closed at **49.9 predicted vs 49.0 measured**. A residual
   you have not decomposed is not evidence for anything.
10. **Winning the drafter metric is not winning — and the example we first used was
    ourselves being wrong.** We wrote this rule pointing at DSpark (acceptance 4.23, losing
    end-to-end to MTP) and the verdict inverted the moment findings 7–8 were fixed: DSpark
    appeared to lead — until the EOS-cut retraction reversed that too, leaving the gated
    in-weights path ahead at 1.40× and the block drafter at 1.28×. The rule survives its own
    counter-example — block 9 has **higher** acceptance than block 8 (3.55 vs 3.46) and is
    still slower, because the drafter forward costs more than the extra token returns. Judge
    on tok/s; and when the loser is your own integration, fix the integration before writing
    the verdict.
11. **A drafter's trained block width is a reference value, not a ceiling.** Block 8 beats
    the card's block 7 (71.0 vs 63.6 tok/s over three prompts). Where verify cost is flat in
    width, spend the width.
