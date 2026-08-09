# Porting and quantizing Kimi K3 (2.8T) to MLX — the full lesson set

Everything we learned shipping [`avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw`](https://huggingface.co/avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw):
a 2.8T-parameter multimodal MoE (69 KDA linear-attention layers + 24 MLA layers, 896-expert
LatentMoE top-16, MoonViT-3d vision) quantized to 737 GB / 2.096 bpw and served on two
512 GB M3 Ultras over Thunderbolt 5 at ≈4.6 tok/s. The DWQ-specific optimizer lessons are in
the [README](../README.md) ("From a 2.8T ternary-codebook student"); this file is the rest.

## 1. Architecture porting

- **Check the checkpoint, not the paper or the reference code.** K3's `A_log` is per-channel
  `[head_dim]`; the tech report table implies per-head and the release modeling file would
  *generate* `[num_heads]`. The checkpoint is the ground truth — load it first, shape-audit
  everything (`A_log [128]` settled three conflicting sources).
- **A new attention variant can be a one-function swap.** KDA "v2 gate"
  (`exp(lb·σ(e^{A_log}·(z+dt_bias)))`) needed a new `compute_g` only — zero Metal kernel changes —
  because the recurrence itself was unchanged. Diff the math, not the marketing name.
- **The eps convention will bite you exactly once per port.** q/k L2-normalization is
  `rms_norm(eps=1e-6/head_dim)` — an `eps@mean` vs `eps@sum` mismatch is a silent ×128 error that
  parity tests catch only if you test at production scale amplitudes (our Test D). Upstreamed as
  ml-explore/mlx-lm#1624.
- **Attention-residual (AttnRes) mixes are position-independent** (learned per-query depth
  softmax over block-boundary states). This matters twice: (a) truncation payloads for
  layer-parallel training/eval stay small (~143 KB/token); (b) any layerwise training window must
  align to AttnRes block boundaries (12 layers) or your cache balloons 6×.
- **MLA absorb is a decode-time win with a load-time cost.** Precomputing
  `W_uk/W_uv` (28× smaller KV at 1M ctx, perf-neutral) is safe — but cache them in bf16, not
  fp32: at 24 layers the fp32 copies add 1.2 GB of *per-token* read traffic.
- **Multimodal survives extreme quantization if you never quantize the tower.** bf16 passthrough
  vision (0.9 GB of 737 GB) + feature injection at placeholder tokens reproduced
  image→HTML/CSS generation on a 2.1 bpw text stack. Gate your release on an end-to-end
  vision smoke, not tower parity alone.
- **Packed-vs-unpacked expert audits first.** K3's mxfp4 release packs only expert w1/w2/w3
  (`weight_packed` count = layers×experts×3 exactly); latent projections and gates are bf16.
  compressed-tensors e2m1+e8m0 gs32 is bit-identical to MLX's mxfp4 layout — a zero-loss repack,
  verified Δ=0. Never dequantize-requantize what you can repack.

## 2. Quantization campaign mechanics

- **Sensitivity ladders beat uniform budgets.** Expert-quant error is heavy-tailed: protecting the
  top ~10–17% most sensitive expert-instances at native mxfp4 while dropping the rest to a
  1.5625 bpw ternary codebook beat every uniform config at equal bytes.
- **Score promotions with Hessian×mass, allocate with clamp-aware waterfilling.** Relative-error
  scores saturate at the ternary grid floor (~0.45) and carry no signal — use *unnormalized*
  weighted SSE. And when one layer hogs 68% of the mass, naive waterfilling leaves half the budget
  unspent at the per-layer cap: solve τ with the clamps inside the equation.
- **Per-expert curvature beats per-layer curvature.** Encoding with each expert's own imatrix
  rows (−18…−21% on spike experts) dwarfs layer-average weighting (−2.6%). Keep the
  (expert × call-block) table 2-D; don't average it away.
- **Completeness audits are part of the converter.** Our converter once silently dropped 8 dense
  tensor kinds (defaulted-to-1 norms = ×18 activation blowup). The fix that stays: assert
  model-tree ↔ file-key equality after every conversion. "Did every parameter come from a file?"
- **Fix the budget before you evaluate.** Promotion count (14,000/82,432) was frozen pre-eval;
  no eval-set tuning. It makes the card honest and the ablation clean.

## 3. Custom-format Metal kernels (1.5625 bpw ternary codebook)

- **Decode and prefill want different kernels.** A prefill-shaped kernel (threadgroup-per-row)
  collapses at decode: R≈14 rows = 14 threadgroups on an 80-core GPU. The decode kernel wants
  split-K: (output-tile × row) grid, one output per simdgroup, `simd_sum` reduce — 3.8× total
  (307→80 ms/step over 92 layers).
- **Shared-memory LUTs can be *slower* than device loads.** Moving the 16 KB ternary grid into
  threadgroup memory lost 23% vs. plain device reads: random per-lane gathers bank-conflict, and
  Apple's device cache already serves a hot 16 KB table. Measure A/B/C variants; don't trust the
  "shared memory is fast" prior.
- **Kernel-bench gains ≠ end-to-end gains.** Our second kernel round cut 36 ms of bench time but
  only 5 ms of e2e — the graph had started hiding expert time under attention/ring waits. Confirm
  the critical path before optimizing further.
- **Dispatch overlap is free fusion.** A fused w1+w3 kernel was bit-exact and… 1.02× — two
  back-to-back dispatches already overlapped on the GPU. Fuse only what the profiler says is
  serialized.
- **bf16 mx arrays don't round-trip through numpy** (PEP 3118 item-size error) — cast to fp32
  first in every parity harness.

## 4. Evaluation infrastructure

- **Cache teacher logits once; everything downstream becomes offline.** 48 fixed windows
  (24 wikitext + 24 Korean, 2047 targets, f16 logits) let us compute KL/flip/PPL for every build
  variant without ever reloading the teacher — and let a *deleted* run be re-scored from disk.
- **Oracle-check every regenerated asset.** When /tmp cleanup ate our token file, we regenerated
  it and *proved* identity by re-deriving PPL from cached teacher logits (4-decimal match), and
  proved H2H raw-text dumps by token-for-token equality against the npz. Regeneration without an
  oracle is a silent protocol fork.
- **In-loop and offline scoring must cross-validate.** Offline logits→PPL matched the in-loop
  accumulator to 4 decimals — after that, pruning 78 GB of logits mid-campaign was a safe,
  reversible decision instead of a leap.
- **Cross-harness PPL is not a comparison.** llama-perplexity (BOS per chunk, different
  boundaries) reads ~0.3–0.4 lower than our windowed harness on the same text. Publish both
  numbers with the caveat; the defensible cross-quant metric is KL-vs-teacher, which only the
  teacher-logit owner can compute.
- **Audit the audit.** An offset-integrity scan passed a slice directory that was *missing seven
  files* — it only checked files that existed. The standard now: offsets + key-signature majority
  vote + file count + cross-rank same-key checksums.
- **Truncated teacher = poisoned ground truth.** A teacher sweep once ran with top-k 8 on a
  top-k 16 model; every downstream number was quietly wrong. Pin router hyperparameters into the
  sweep config and assert them at load.

## 5. Two-box expert-parallel serving (MLX ring)

- **Per-layer all_sum is mathematically irreducible** (a nonlinearity sits right after the expert
  sum), but *how* you complete it is not: hard-eval after every collective serializes the whole
  step. MLX's official sharded-decode pattern leaves collectives in the lazy graph with one
  async_eval per token — our forced per-layer evals were a historical watchdog defense that
  outlived its cause (a GPU heartbeat thread now covers it).
- **Warm up every graph shape with the ring untouched** (skip-mode), then never introduce a new
  shape mid-flight — JIT-while-collective deadlocks or trips the ring watchdog. New shapes
  (e.g. image injection) get their own eager section.
- **Session KV for chat is a protocol problem, not a cache problem.** Chat templates re-render
  history with thinking channels *stripped*, so retokenized history never prefix-matches what you
  actually fed the model — template-based prefix caching is structurally dead for thinking models.
  Fix: server-side sessions that append glue tokens (probe-extracted once from the template) +
  the new user turn; never re-render history. Turn-start latency went from O(history) (~45 s) to
  ~3 s flat.
- **Don't store or forward the EOS.** The model's stop token is not part of the canonical
  rendered history; feeding it desyncs your cache from the template's token stream by one.
- **Recurrent-state caches cannot trim.** KDA state is cumulative — prefix reuse is
  all-or-nothing, and speculative verify needs snapshot/replay, not KV rollback.
- **Keep the ring alive on idle** (a tiny nop collective under the recv timeout), and kill server
  processes by saved PID only — `pkill -f` pattern-matching killed innocent processes three times
  before we banned it.
- **apply_chat_template may return *text*** (K3's does) — always `tokenize=True` + an
  `isinstance(str)` guard, or your "token ids" are a string.

## 6. Operational pitfalls (the expensive ones)

- macOS purges /tmp on a ~3-day cadence: every launcher, hostfile, token dump and audit script
  lives in a permanent assets dir and is *copied into* /tmp, never authored there.
- Logits at [2047×163840] f16 are 670 MB per window. A 142-window sweep is 95 GB — budget disk
  for evaluation artifacts like you budget for weights, and prune windows the protocol doesn't
  need *as they stream*.
- BatchMode ssh has a minimal PATH: absolute paths for *every* binary (python, cmake), and ship
  remote code as script files, never inline shell with local-expanded variables.
- `sed` with two expressions can double-apply to one line; regenerate scripts with a full Write
  instead of stacked in-place edits.
- `grep -c` prints `0` *and* exits 1 on no-match — `$(grep -c …) || echo 0` yields `"0\n0"` and
  breaks integer comparisons downstream.
- HF: escape literal tildes in cards (single `~…~` pairs render as strikethrough on HF only);
  `upload_large_folder` resumes cleanly across repo renames if you pause, `move_repo`, and
  relaunch with the new id; clearing `~/.cache/huggingface` deletes your auth token with it.

## 7. Performance method (what actually found the wins)

- **First-principles floors before surgery**: weight-read, ring-latency and dispatch budgets put
  the theoretical step at ~90–100 ms vs. 217 ms measured — proving the gap was overhead, not
  physics, and pointing at the eval-serialization assumption as the biggest suspect.
- **Observer-free profiling**: the 92 forced evals double as free per-layer timestamps — append
  to a list (never print) and you get a full KDA/MLA/MoE cost split with zero graph changes.
- **Isolated single-layer microbenches** (load one real layer file, 300 decode reps) decompose a
  layer into kernel / projections / glue in minutes, and re-validated a months-old launch-overhead
  number (246 µs → 262 µs) before we trusted it.
- Every optimization ships with: unit parity (bit-exact or tolerance-justified), a 4-layer 2-box
  protocol smoke, an env killswitch, and an e2e stats measurement. Three of seven candidates in
  our last batch were *rejected* by these gates (one numerically, two by measurement) — the gates
  are the method.

## 8. Serving speed: what worked, what the literature got wrong for MoE

Second optimization round took decode 4.61 → 5.41 tok/s and roughly halved prefill. Ranked by
what actually paid:

- **Delete the per-collective `mx.eval`.** MLX's own sharded-decode path leaves collectives in the
  lazy graph and syncs once per token; our per-layer hard flush (92/step) was a watchdog defense
  from before we had a GPU heartbeat thread, and it was serializing the entire step. Removing it
  was the single biggest win — and it was found by a first-principles pass, not a profiler:
  weight-read + ring-latency + dispatch floors summed to ~90–100 ms against 217 ms measured, which
  said "the gap is an assumption, not physics."
- **`MLX_METAL_FAST_SYNCH=1`** on every rank (upstream measures ~12× on collective sync).
- **Vectorized loads in the codebook kernel** (8 bf16 as one `uint4`): bit-exact 1.25×. Notably an
  fp16-dot variant and a δ-term-precompute variant landed on the *same* time, and combining all
  three gained nothing further — the kernel is load-latency bound, not ALU bound. When three
  unrelated optimizations converge to one floor, stop optimizing that axis.
- **Speculative decoding is a *loss* on a sparse-MoE-over-EP rig.** n-gram/prompt-lookup drafting
  reached a real 1.37 accepted tokens per round — and still cut throughput to 0.58× (5.36 → 3.13
  tok/s), because every draft token routes to *different* experts: verify cost scales with draft
  length (measured ~2.4× per round at k≤13) instead of amortizing a fixed cost. The published
  "speculation is nearly free at batch 1" results assume dense weights shared across the batch.
  For MoE, break-even acceptance is the verify-cost multiplier, not ~1.3. Keep the code behind a
  killswitch; default it off.
- **Chat-template caching**: see §5 — session-based glue-token appending, not history re-render.
- Two more rejections worth stating: a hand-written T=1 depthwise conv (bf16 rounding drifted 2e-2
  from `nn.Conv1d`'s fp32 accumulation) and a fused w1+w3 kernel (bit-exact, 1.02× — the two
  dispatches already overlapped).

Profiling notes that generalize: the forced collective evals double as **free per-layer
timestamps** (append to a list, never print); a single-real-layer microbench decomposes a layer
into kernel/projection/glue in minutes; and `mx.compile(shapeless=True)` is unsafe where the
traced function reshapes with `-1` (a trace specialized on one source count died on another —
per-shape compile caches are the safe form).

## 9. Writing custom Metal kernels that match MLX eager ops bit-for-bit

Fusing MLX's small glue ops (conv/silu/rms_norm/sigmoid chains) into one custom kernel is the
right lever on dispatch-bound decode — but "mathematically identical" is not "bit-identical",
and on a 93-layer recurrent model 1-ULP glue drift compounds into measurable logit drift.
Three MLX semantics we had to pin down *empirically* (each verified to 0 mismatches on 8k
random samples before adoption):

- **Eager `mx.sigmoid` is NOT `1/(1+exp(-x))` in fp32.** The Metal kernel
  (`unary_ops.h::Sigmoid`) computes `y = 1/(1+exp(|x|)); x<0 ? y : 1-y` **in the tensor's own
  dtype** — for bf16 inputs every intermediate (exp, add, divide, subtract) rounds to bf16.
  Emulation that matches exactly: round each step through bf16, compute the transcendental in
  fp32. Note `mx.compile`d graphs promote the same chain to fp32 — so a compiled reference and
  an eager reference disagree with *each other*; match whichever path you are replacing.
- **Python-scalar × bf16-array pre-rounds the scalar to bf16** (weak promotion), then multiplies.
  Passing the exact fp32 scalar into a custom kernel produced 222/8192 mismatches; pre-rounding
  it to bf16 reproduced MLX exactly. Applies to patterns like `(head_dim**-0.5) * x`.
- **`mx.fast.rms_norm` rounds once without weight, twice with weight**: `bf16(x·rsqrt(mean+eps))`
  then `bf16(norm × w)`. Fusing normalize-and-scale into one round is a 1-ULP mismatch on ~26%
  of elements.

Payoff profile for the fusion itself (Kimi K3, KDA layer, M3 Ultra): packing 6 same-input
projections into one QMM (row-concat of 6-bit weights is bit-exact; slice back on the output
axis) gave a measured e2e win; the glue kernel (24 dispatches → 2, threadgroup-per-head with
simd_sum reductions) then reduced per-layer isolated time by a further ~4%. After matching the
three semantics above: conv states bit-exact, recurrent state at 1e-8 (fp32 noise), greedy
generations identical with fusion on/off.

One more compounding pair from the fusion session: a launcher "memory gate" that refuses to
start when wired memory is high cannot distinguish *a healthy server still running* from *a
post-crash wired leak* — ours saw the live server's 400GB, refused silently, and exited. The
readiness watcher then matched the **previous boot's** "server up" line in the un-truncated log,
so the operator believed the new build was live and benchmarked the old one (an entire A/B round
was invalid). Fixes: the gate first SIGTERMs stragglers and re-checks before declaring a leak;
the launcher truncates its log as its first action; and after any deploy, verify the *running
process's env* (`ps eww <pid>`) rather than trusting log lines. Deployment checks must probe the
process, not the logs.

Two more operational traps from the same session: (1) output-axis packing and tensor-parallel
head-slicing touch the same modules — they compose into shape mismatches at runtime; make them
explicitly mutually exclusive. (2) A smoke launcher that shares the production port *and* uses a
broad `pkill -f` cleanup pattern will take down the production server when the bind fails —
isolate smoke runs by port and narrow the kill pattern to the smoke's own arguments.

## 10. MoE decode restructuring on MLX — what moved the needle and what didn't

Measured on the same rig (92 MoE layers, 896 experts top-16, batch-1, 2-box EP), after the
KDA fusion work brought the step to ~179ms. Stage-decomposed the ~90ms MoE region first
(isolated stage benches inflate by ~190µs of per-eval overhead each — subtract it or you will
chase phantoms; the full-call time is the honest denominator).

**Adopted (all bit-exact, each behind a killswitch):**
- **Fused router kernel** — the sigmoid→bias→top-16-of-896→normalize→group-map→hi-top-4 chain
  (~18 small dispatches) as ONE single-threadgroup kernel doing sequential max-extraction.
  Selection set, weights, and hi choices matched the argpartition reference bit-for-bit on
  32/32 trials; only the emission *order* differs (score-descending vs argpartition's arbitrary
  order), which shifts downstream weighted sums by ~1 ULP. Biggest single e2e win of the phase.
- **Shared-experts GLU packing + fused SiTU** — gate+up row-concat into one QMM (bit-exact for
  uniform quant) plus a one-dispatch SiTU elementwise kernel replacing ~8 glue ops.
- **Activation-at-store fusion** — inlining the SiTU nonlinearity into the up-projection
  kernel's store site (round the accumulator to bf16 *first* to preserve the reference's
  rounding point, then apply the fp32 nonlinearity, round once at output). Zero mismatches over
  860k elements. In-pipeline gain exceeded the isolated estimate — removing glue ops also
  removes command-buffer pressure, so glue elimination compounds.

**Rejected, with numbers:**
- **Tensor-parallel on the dense/shared blocks**: halving a 107MB-per-layer weight stream
  changed decode time by ~0%. At batch 1 these QMMs are latency-bound, not bandwidth-bound —
  *every* bytes-reduction lever we tried on this rig (attention TP, dense TP, absorb-matrix
  bf16) was speed-neutral. Measure one before building more.
- **Raising MLX command-buffer budgets** (`MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER` to
  effectively-infinite): a 5% *regression*. The per-op commits those budgets force are not pure
  overhead — they are what lets the CPU encode ahead while the GPU drains. Also note the
  budget math: one stacked expert tensor binds ~1.2GB, so any budget between 50MB and ~1GB
  yields the *same* commit pattern; there is no useful intermediate point.
- (Earlier, same theme) **speculative decoding** loses 0.58× here — but external data
  (Cohere's published MoE profile) reframes this as kernel accounting, not physics: where
  routed-expert bytes are a small fraction of step time, a second token riding the *same*
  gather kernels should cost ~1.05–1.25×. The loss came from doubling kernel count, not bytes.
  A `gather_qmv`-style kernel that amortizes weight loads across 2 tokens reopens the door.

Net for the day across both phases (KDA + MoE): 5.41 → ~5.95 tok/s, every adopted change
bit-exact or ULP-bounded with a killswitch, greedy outputs stable throughout.

## 11. Router mass tells you whether top-k reduction is on the table (novel data)

Nobody had published the router probability-mass profile for a k=16-of-896 fine-grained MoE,
so we measured it: 45k layer-samples across prose/Korean/code/math on Kimi K3. Result: the
sorted, normalized top-16 weights have a **flat tail** — ranks 13-16 each carry ~3% (about a
fifth of the top expert, not a hundredth), cumulative mass at rank 12 is only 86.6% on average
(75.5% worst layer). This is the opposite of the "confident about 2-4, rest indistinguishable"
profile reported for coarse-grained models, and it corroborates the DeepSeekMoE claim that
shared-expert + fine-grained designs make routed experts *less* redundant per-expert.

Consequence, measured end-to-end: k=16→12 bought +5% decode speed but shifted greedy outputs
within 44-190 characters on all four test domains — a real distribution change, not rounding
noise. We shipped it as an opt-in flag, default off. If you are considering top-k reduction on
a fine-grained MoE: dump the mass profile first (it costs a few hundred tokens of serving with
a 10-line hook), and gate any adoption on a real code benchmark — perplexity and short-form
math stay green far past the point where code generation collapses (see OEA Table 7).

One measurement pitfall from the same night: a single custom-kernel dispatch benched in
isolation carries ~190µs of `mx.eval` fixed cost, which completely masks a 15µs/call kernel
improvement. Chain ~16 dependent calls inside one graph and divide — our row-parallel decode
variant (8 rows/simdgroup, after llama.cpp's low-bit N_R0=8 convention) measured 1.01× naively
but is a real 1.14× at the shapes that matter, bit-identical, and is now the default engine.

## 12. Speculative decoding on a sparse MoE, revisited — the premium was kernel accounting

Section 8 reported speculative decoding as a 0.58× net loss on this rig. That number deserved
an autopsy, and the autopsy changed the conclusion: the verify forward (T=2..4) was expensive
mostly because every decode-path fusion we had built was gated `T==1` — the verify tokens fell
off the fast path entirely. Extending the fused KDA glue kernel to T≤4 (the causal-conv taps
become a sliding window over [state, raw inputs]; the last position's threads write the new
state) and the fused router to T≤8 kept everything bit-exact at T=2/3 and cut the measured
verify-2 premium from 1.38× to ~1.28×.

Live result with an n-gram (prompt-lookup) drafter, greedy, 192-token generations:
draft k=2 is still a net loss on all domains (accept 33-51%); draft k=1 is break-even —
+2.5% on repetitive code (72% accept), neutral on prose, −1.5% on step-by-step math. We ship
it default-off. The honest summary: **the verify-cost half of the speculation problem is now
solved on this stack; the drafter half is not.** Any drafter with >70% acceptance across
domains (a learned per-layer draft expert, MTP-style heads) would flip this positive
immediately. One API trap for reproducers: our `K3_SPEC_K` is a *total feed budget* — the
draft count is `K3_SPEC_K − 2`, so setting it to 2 silently disables drafting entirely.

## 13. An EAGLE-style draft head on a fine-grained-MoE trunk: a measured negative

To reopen speculative decoding (whose verify cost we had already halved via T≤4 bit-exact
fusion), we trained EAGLE-1-style draft heads (~230M params: fuse + 1 transformer block,
frozen trunk embed/lm_head) on a 6M-token pilot of trunk hidden states (wiki-EN/KO/code mix,
dumped at 62 tok/s via the expert-major prefill path). Two recipes, 3000 steps each:

- CE-only: holdout trunk-greedy agreement α plateaued at **0.134**
- + feature regression (smooth-L1 to the next hidden, EAGLE's core loss, weight 1.0 vs CE 0.1):
  slower start, late crossover, final **0.141** — the celebrated ingredient bought +5% here.

Both are a factor ~4 below the acceptance needed (α≈0.55 break-even, α≈0.8 for the 1.6×+
regime), and the curves' diminishing slopes do not extrapolate to the gate even at 8× data.
Our working hypothesis for *why*, consistent with the router-mass finding above: a 896-expert
fine-grained trunk produces hidden dynamics that a small dense head cannot approximate the way
it can for dense trunks — the same flat-tailed, high-entropy routing that defeats top-k
reduction also defeats cheap drafting. Practical guidance: on this model class, budget
drafter-based speculation as a *research project with real failure risk*, not an engineering
line item — or wait for vendor-trained MTP heads. The verify-side engineering (bit-exact T≤4
fused paths, synchronized sampling) is model-agnostic and keeps its value regardless.

Bonus negative from the same window: mirroring the dense (non-expert) weights to bf16 for
prefill GEMMs made long-prompt prefill 25% *slower* — at 16k-token prefill the dense matmuls
are bandwidth-sensitive, and trading 6-bit reads for bf16 (4.6× bytes) loses despite faster
math. The winning prefill lever was expert-major dequant-once for the codebook experts
(long-prompt TTFT 100s→37s, adopted).

## 14. EP sharding must reach the kernel: the dummy-row full-compute trap

Expert-parallel sharding here rewrites the routing maps so that non-local experts point at a
zero-weight dummy row (row 0) — memory is halved per box, outputs stay correct because the
weighted sum multiplies those rows by zero. What we missed for weeks: the *kernel* still ran a
full matvec for every dummy row. With two boxes, ~56% of routed-expert rows per rank were
full-cost zero-contribution work.

The fix is four lines per kernel — read `eidx[r]`, and if it is the dummy, write zeros and
return before touching weights:

```metal
uint e = eidx[r];
if (e == 0u) { if (lane == 0u) { /* write NR zeros */ } return; }
```

Measured: **+0.48 tok/s live (6.04→6.52)**, greedy transcripts unchanged (the elided work was
exactly the zero-weight rows). Two portable lessons. First, *sharding a tensor is not sharding
the compute* — any indirection that remaps "not mine" to "cheap placeholder" needs an explicit
kernel-side elision, or you pay placeholder cost at full rate. Second, this hid for weeks
because our decomposition microbenches normalized per-call rather than per-useful-byte; an
external audit (fresh-eyes agent) spotted it from the code, not the numbers. Adversarial
code-reads of "settled" hot paths are cheap compared to what they find.

## 15. Fusing two matvecs + activation into one dispatch — and the kernel-generation trap

The decode up-projection ran as three dispatches per expert group: w1 matvec → w3 matvec with
the SwiGLU-variant activation fused at the store → w2 matvec. A w1+w3 fused kernel existed in
the tree but was never wired — and wiring it as-is would have been a *regression*: it predated
the current row-parallel kernel generation (1.14× faster) and the dummy-row exit from §14.
Defined-but-unwired kernels rot. Re-derive them against your current best generation instead
of resurrecting them.

The rewrite (one dispatch: both matvecs + activation): each simdgroup owns NR=4 output rows ×
2 weight stacks — same register budget as the single-stack NR=8 kernel — shares the x-block
loads across both accumulations, applies the activation in-register, and keeps the dummy exit.
To stay bit-identical to the two-kernel path, round *at the same three points* the split path
rounds: gate to bf16 (it used to cross device memory as bf16), up to bf16, output once.

Measured (chained, [layer-level]): up-phase 176.7 → 137.3 µs/layer-call (**+22.3%**), bit-equal
including dummy rows; live serving **6.58-6.60 → 6.64-6.68 tok/s**, full greedy transcripts
identical with the fusion on/off. The win decomposed: one dispatch fewer, x/codebook
threadgroup loads amortized once instead of twice, and the gate tensor no longer round-trips
through device memory.

## 16. Framework upgrades move greedy transcripts; re-certify against the teacher, cheaply

Upgrading MLX 0.31.2→0.32.0 (for its new RDMA collective backend) shifted greedy decoding at
one near-tie token in one of three fixed probes — transcripts elsewhere byte-identical. Bisect
discipline: with the new framework fixed, toggle your own change on/off (ours was bit-exact
either way → framework guilty); never attribute a transcript shift without that split.

A transcript shift is not a quality verdict. The cheap definitive test: recompute a *small
window subset* of teacher-forced logits under the new framework (same prefill path as the
original certification) and compare KL(teacher‖build) per window against the certified run:

| window | KL before | KL after |
|---|---|---|
| wikitext w000 | 0.2313 | 0.2314 |
| wikitext w001 | 0.3637 | 0.3633 |
| korean w000 | 0.1130 | 0.1128 |
| korean w001 | 0.1169 | 0.1175 |

Identical to the 4th decimal (4 windows, ~20 min including model load) — certification stands
without rerunning the full 48-window suite. The direct old-vs-new logit KL (2.4-3.6e-3 nats,
1.3-2.0% argmax flips over all teacher-forced positions) is the framework's rounding footprint:
real, harmless, and now on file so the next transcript drift has a reference scale. Retire
transcript baselines recorded under the old framework the moment this lands — stale baselines
turn every future A/B into a false alarm.

## 17. RDMA collectives on a two-Mac TB5 cluster (MLX 0.32 jaccl): +5% e2e, and the topology surgery

MLX 0.32 ships a `jaccl` distributed backend: RDMA over Thunderbolt, no InfiniBand hardware.
On our 2× M3 Ultra ring it replaced TCP collectives outright — measured on the same chained
harness (each all_sum consumes the previous result, `mx.eval` per step):

| payload | ring (TCP) | jaccl (RDMA) |
|---|---|---|
| [1] f32 | 278.8 µs | 212.8 µs (−24%) |
| [1, 8192] bf16 | 285.1 µs | 200.5 µs (−30%) |
| [1, 65536] bf16 | 320.8 µs | 222.7 µs (−31%) |

At 92 per-layer all_sums per decode step that projected to −6~8 ms/step; live serving landed
exactly there: **6.6 → 6.9-7.1 tok/s**, greedy transcripts bit-identical (a two-rank sum has
one addition order — switching transports cannot change the arithmetic; still verify).

What it actually takes, beyond `pip install -U mlx`:
- `rdma_ctl status` must say enabled on every box (ours already were — check before planning
  recovery-mode work).
- The Thunderbolt interface carrying the link must own an IP *directly*. macOS puts TB ports
  in `bridge0` by default, and a bridge member has no address/GID, so jaccl's queue-pair setup
  dies at RTR with errno 22. The surgery, per box: pull the port out of the bridge and move
  the existing IP onto it (`ifconfig bridge0 deletem enX` + `ifconfig enX inet <same-ip>
  netmask ...`) — same addresses, so ssh/hostfiles/launchers survive untouched. It does not
  persist across reboots.
- Manual rendezvous without `mlx.launch` (our launchers are hand-rolled ssh): set
  `MLX_JACCL_COORDINATOR=<rank0-ip:port>` and `MLX_IBV_DEVICES=<file>` holding the device
  matrix `[[null,"rdma_enX"],["rdma_enX",null]]` — device names are `rdma_` + the *local
  interface name*, world size is inferred from the matrix, `MLX_RANK` as usual. No hostfile.

Surgery pitfalls that cost us an hour at 1 a.m.: applying it half on one box (port un-bridged
but IP left on the bridge) severs the link **mid-serving** — the ring dies, the server wedges
at 100% CPU with the GPU near-idle, and SIGTERM won't land because the ring-blocked thread
never reaches the graceful handler (verify wired memory has drained, then SIGKILL is safe).
And on the other box the IP moved but the connected route silently failed to install, so
replies left via the VPN interface — `route get <peer>` from *both* sides plus a resolved ARP
entry distinguishes every one of these half-states in under a minute; ping alone distinguishes
none of them.

## 18. Prefill for codebook experts: token-parallel decode amortization (v5) beats both incumbents in the chat-turn regime

Per-row LUT re-decode is the prefill killer for codebook-quantized experts: at R routed rows
per layer, the same expert weights get decoded R-ish times. Our first fix (§ earlier,
"expert-major") dequantizes each expert once into a dense buffer and runs a GEMM — great at
long prompts, but its fixed dequant cost loses below ~250 tokens, so chat turns stayed on the
naive path.

The v5 kernel closes that gap without materializing anything: sort rows by expert (host-side
argsort, one plan shared across w1/w3/w2), tile them (expert × ≤8 tokens), and give each
simdgroup one output row — each lane decodes its 32-value weight chunk *once* into registers
and reuses it across the tile's 8 token vectors. Accumulation order is kept identical to the
decode kernels (lane-strided fp32 + simd_sum), which bought bit-identity with the decode path
at R=64 and p99 bit-identity with the old prefill path at R=8192.

Measured (M3 Ultra, 3072×3584 stacks, realistic 44%-local EP routing):

| tokens | naive per-row | v5 | expert-major |
|---|---|---|---|
| 32 | 5.6 ms | **1.7 ms (3.3×)** | — |
| 126 | 18.4 ms | **5.6 ms (3.3×)** | — |
| 512 | 69.4 ms | **50.7 ms/3-mat best** | 59 ms/3-mat |
| 2048 | 271.9 ms | 197 ms/3-mat | **65 ms/3-mat** |

Crossover ≈ 650 tokens → gate v5 for 64 < R < 10240, keep expert-major above. Live serving:
**chat-turn prefill 5.6 → 2.7 s (2.1×)**, long-prompt TTFT 37 → 33.6 s (the latter mostly from
also extending §14's dummy-row elision to the prefill paths — we had patched only the decode
kernels; audit *every* path that consumes the sharded maps). Decode throughput unchanged.

Transcript note for adopters: switching the mid-R kernel changes <1% of outputs at the last
ulp, which flipped exactly one of three fixed greedy probes — the same near-tie "canary"
prompt that flipped under the MLX 0.31→0.32 upgrade and under speculative verify. Track which
of your fixed probes are hair-triggers; a canary flipping alone (with kernel-level p99 bit
identity in hand) is evidence of rounding sensitivity, not regression — but bundle a
self-consistency PPL check in the relevant window-length regime with the next deploy anyway.

## 19. Two invalid experiments, one rule: interventions must print evidence of themselves

Two of our most consequential "findings" this week were artifacts of interventions that never
actually happened:

- **The phantom top_k result.** "Reducing routed experts 16→1 changes nothing" drove days of
  prioritization (killed expert-side levers, killed expert-reduction drafting, spawned a wrong
  "MoE is latency-hidden" theory). In reality our EP installer replaces `layer.mlp` with a
  lambda closure; the sweep's `isinstance(...)` scan found zero MoE modules and changed
  nothing. The corrected sweep (recovering modules from `lambda.__defaults__`, printing the
  kernel-observed row count per setting) shows top_k=1 is **−11.2%** — the original experiment
  was a no-op, not a discovery.
- **The phantom TP certification.** Our KL harness accepted `K3_ATTN_TP=1` but never called
  the TP installer — and produced four windows of KL **identical to six significant digits**
  with the non-TP run. That impossible-looking agreement was the tell: the intervention wasn't
  installed. (The perfect reproduction was, at least, a free determinism check of the harness.)

The standing rule we adopted: **an intervention experiment is valid only if it emits evidence
of its own application** — the kernel-side row count actually dispatched, the number of layers
actually wrapped, or a measurable divergence from a known-identical baseline. Silence plus a
plausible number is how no-ops masquerade as physics. Corollary from the same week: eager
distributed measurement harnesses must include a settle pass before timing (skipping it cost
us a 3-hour watchdog deadlock that presented as "slow").

## 20. A read-bound decode model, cross-validated by two byte-removal experiments

With the phantom result corrected, two independent experiments that each remove a known number
of bytes from the per-token read stream agree on one exchange rate:

| experiment | bytes removed / token / rank | measured saving | implied marginal BW |
|---|---|---|---|
| top_k 16→1 (corrected) | −11.3 GB | −20.2 ms | 560 GB/s |
| attention 6→4 bit | −9.4 GB | −14.5 ms | 650 GB/s |

**Decode step ≈ bytes-read ÷ ~550-650 GB/s effective.** The whole step checks out too (62.1 GB
÷ 460 GB/s ≈ the 135 ms step). The actionable currency: **1 GB removed ≈ 1.8 ms**.

Byte census (per token per rank, batch-1, 2-way EP): attention projections 33.2 GB + shared
experts 9.8 GB + MoE latent/router ~5 GB + embeddings/head 1.9 GB — the **always-read stack is
80%** — routed experts only 12.1 GB (20%). Under this model, replicated-weight tensor
parallelism is the only parallelism that reduces bytes; pipeline parallelism does not (a
batch-1 token walks the layers serially either way, so PP's gain is only the non-overlapped
collective time). Flipping the long-rejected `K3_ATTN_TP=1` (head-sharded q/k/v/gate/o,
−16.6 GB/token/rank, one extra partial-sum all_sum per layer) delivered **7.4 → 8.1 tok/s
(+9%)** live. The original TP rejection dated from the TCP-ring, pre-fusion era — the same
vintage as the phantom top_k result, and equally stale.

Operational caveat discovered in the same push: **jaccl (TB5 RDMA) collectives stall on
large payloads** (tens of MB — e.g. TP partial-sums during a 2k-token prefill, ~59 MB/layer).
Small decode-time collectives are fine. Until a chunked-collective fix lands, run TP prefill
over the TCP ring backend or cap chunk sizes.

## 21. The precision axis closes: 4-bit attention fails its KL gate, and sensitivity is not
where requant error says it is

The read-bound model made the 6-bit always-stack the biggest byte target (6.0 bit stored;
experts are 2.10 bpw). Speed delivered as predicted (KDA layers −9.6%; MLA layers unchanged —
their kv_b projection is dequantized **once** into absorb caches, so quantizing it saves
nothing per-step). Quality did not: teacher-anchored KL rose ~10% (0.2313→0.2586 wt) and a
selective variant that kept the highest-requant-error tensors (the tiny gate low-rank
projections) at 6 bit recovered almost none of it (0.2587). Two lessons:

- **Requant rel-error is not a sensitivity proxy.** The tensors with the worst per-tensor
  reconstruction error (2.4 MB of gate projections) contributed almost nothing to the KL hit;
  the well-reconstructed large projections carried it. Grade sensitivity by output/KL
  perturbation, never by weight-space error.
- The axis reopens only with distillation-corrected low-bit (our DWQ pipeline) — the exchange
  rate (−17 ms for attention alone) now prices that campaign precisely.

Also on file: a bf16 DSpark-style drafter against this 2.10 bpw target accepts ~1.55
tokens/round (k=2) versus 3.85-5.4 reported against bf16 targets — the quantized-target
acceptance degradation is real on this stack, resolving a conflict in the community record.

## 22. Attention TP lands as default: graph-size deadlocks, a 4-layer bisection harness, and
byte-identity as the closing receipt

Head-sharded attention TP (q/k/v/gate/o split across ranks, fp32 partial sums, one extra
per-layer all_sum) is the final +12% of the campaign — decode 7.4 → **8.28-8.29 tok/s**
measured on freshly booted boxes, now the default serving configuration. Three things had to
be true before we could ship it, and each produced a reusable lesson.

**1. The long-prompt stall was a graph-size problem, not a payload-size problem.** Under TP,
prefills ≳1.5k tokens wedged the scheduler on *both* backends (RDMA jaccl and TCP ring) —
so it was never the transport. Chunking only the all_sum payloads (splitting the [1,T,7168]
tensor along T into 256-token all_sums) did **not** fix it: the deadlock tracks the size of
the single lazy forward graph containing 93 interleaved collectives, not the size of any one
collective. The fix that works is chunking the forward itself: under TP the server prefills
in 256-token forwards (`K3_PREFILL_CHUNK`). Verified end to end with a 2k-token prompt
(42.6 s prefill, no stall). Cost: ~25% slower long-prompt prefill than non-TP (33.6 s at 2k)
— an acceptable trade against +0.9 tok/s decode; prefill-heavy batch workloads can keep
`K3_ATTN_TP=0`.

**2. Bisect deadlocks with a partial-load mini harness, never the full model.** Each wedged
full-scale attempt cost a 395 GB wired-memory leak and a reboot (§21's kill-discipline rules
exist because of this). The decisive experiment loaded **4 layers (~15 GB)**: chunked
prefill (256×8) completed in 1.9 s where the single T=1993 forward wedged even at 4 layers —
root cause confirmed and fix validated in one run, with nothing at risk. If a distributed
hang reproduces at all, reproduce it at the smallest layer count that still shows it.

**3. Certify TP by construction + identity, not by re-running the quality battery.**
Head-TP is algebraically the same sum in a different accumulation order. So certification is
(a) single-box numerical parity — shard the same weights into both ranks in one process, sum
the partial outputs locally, compare against non-TP: rel_max ≈5e-3 on KDA and MLA layers,
bf16 rounding scale, with partial sums accumulated in fp32 before the all_sum — and
(b) **byte-identical greedy transcripts** vs the non-TP configuration (3 prompts × 96 tokens,
reasoning traces included). Together these are strictly stronger than a PPL smoke, and
cheaper: no 4-window KL run needed for a numerically-equivalent rewiring. (The KL battery
remains the gate for anything that changes *values*, per §16.)

Operational footnote: the TB5 point-to-point link that jaccl needs (bridge0 dismantled,
static IPs on the raw interface) does not survive a reboot on macOS — a LaunchDaemon that
re-applies interface config at boot (with retry, since interfaces come up late) makes the
RDMA topology surgery permanent. Wired-leak recovery without a reboot: six candidate paths
(memory-pressure eviction, GPU-restart trigger, IOKit user clients, purge calls, kext-level
resets, sysctl VM knobs) all fail against IOGPU non-reclaimable pages from a killed process
— prevention (never SIGKILL a loaded rank; chunked graphs so collectives can't hang) is the
only strategy that works.
