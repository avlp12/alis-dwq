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
