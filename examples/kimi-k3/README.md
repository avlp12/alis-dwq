# Kimi-K3 2.8T → 2.10 bpw, and the decode campaign that followed

Case study for the largest build in this repo: **Kimi K3 (2.8T-param multimodal MoE,
93 layers = 69 KDA linear-attention + 24 MLA, 896 experts top-16)** quantized to
**2.096 bpw** (1.5625 bpw ternary codebook for 82k experts + mxfp4 promoted tier +
6-bit dense) and served on **two M3 Ultra 512GB Macs** over expert parallelism.

- Weights + runtime + install guide: [avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw](https://huggingface.co/avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw)
- Full lesson set (porting, campaign mechanics, kernels, DWQ-v3, pitfalls):
  [docs/kimi-k3-mlx-port.md](../../docs/kimi-k3-mlx-port.md) — this directory holds the
  **raw receipts** referenced there.

## Quality (certified)

| Measure | Value |
|---|---|
| KL vs fp teacher (48 windows) | **0.2253 nats** (−35% vs the 211 GB larger v2 build) |
| top-1 flip vs teacher | 13.77% |
| PPL wikitext / korean | 2.078 / 3.311 |
| Decode-path fusion drift (this dir) | **1.1e-3 nats** ≈ 1/200 of the quantization's own KL |

## Serving speed: 2.31 → 6.0 tok/s on unchanged weights

Every adopted lever is bit-exact or ULP-bounded with a killswitch; every rejected lever
is recorded with numbers. Highlights (full narrative in the docs file):

| Lever | Verdict | e2e |
|---|---|---|
| session KV + lazy-graph collectives + Metal fast-synch | adopted | 2.31 → 4.6 |
| ternary-codebook decode kernels (split-K → row-parallel v4) | adopted, bit-identical | → 5.4 |
| KDA projection packing (6 QMM → 1) + fused glue kernels (T≤4) | adopted, bit-exact decode | → 5.75 |
| fused MoE router (top-16-of-896, one dispatch) | adopted, set+weights bit-identical | → 5.9 |
| SiTU-at-store + shared-experts packing | adopted, bit-exact | → **6.0** |
| attention/dense tensor-parallel, buffer-budget raise, f_b in-kernel | **rejected** (measured neutral/regression) | — |
| top-k 16→12 | **rejected for quality** (flat router tail — see below) | (+5% if opted in) |
| speculative decoding (n-gram PLD) | break-even at k=1; verify premium halved to 1.28×, drafter is the binding constraint | ±0 |

## Novel data in this directory

- **`router_mass_45k.npz` + summary** — the first published router probability-mass profile
  for a k≥16 fine-grained MoE (45k layer-samples, 4 domains). Key finding: the tail is
  *flat* (ranks 13-16 carry ~3% each), the opposite of coarse-MoE folklore — fine-grained
  designs are per-expert fragile, so top-k reduction needs a real code eval, not PPL.
- **`kl_decode_bundle_results.txt` + script** — decode-mode logit-drift certification of the
  fused serving stack. Method caveat that cost us almost a wasted run: prefill-based KL is
  *vacuous* for decode-gated fusions; you must dump per-step decode logits.

## Reproduce

The serving runtime (fusion kernels, EP harness, web chat, 2-box launcher) ships in the
HF repo with an install guide (including a note addressed to AI coding agents). The
quantization pipeline is this repo's `alis_dwq` + the campaign notes in the docs file.

## Addendum: expert-placement balancing is a dead lever here (measured)

With per-token expert-ID samples (26.6k routing sets per layer, 92 layers), the best static
2-way partition found by direct E[max] minimization improves the straggler expectation only
from 9.561 to 9.182 active-experts-per-box (floor 8.0) — a ~4% MoE-latency potential. The
telling number: the naive index split (9.561) already sits at the uncorrelated-routing
expectation (9.571), i.e. **K3's fine-grained routing shows no exploitable co-activation or
frequency skew at the box-partition level**, and three quarters of the 19.5% straggler excess
is irreducible per-token variance. Static placement optimization is not worth the reshard on
this class of model; only dynamic (per-token) dispatch could recover it, at protocol
complexity far exceeding the ~3ms prize. Raw optimizer + data hooks in the serving repo.

## Addendum 2: dummy-row elision, single-dispatch GLU fusion, framework re-cert (2026-08-06)

Receipts:
- [`glu_fusion_bench.txt`](glu_fusion_bench.txt) — chained layer-level A/B of the fused
  w1+w3+activation kernel vs the two-kernel path: 176.7 → 137.3 µs/layer-call (+22.3%),
  bit-equal output including EP dummy rows; reproduced identically on MLX 0.31.2 and 0.32.0.
- [`mlx032_recert_kl.txt`](mlx032_recert_kl.txt) — 4-window teacher-anchored KL smoke after
  the MLX 0.32 upgrade: per-window KL unchanged to the 4th decimal; the framework's rounding
  footprint (direct old-vs-new KL 2.4-3.6e-3 nats) documented for future drift triage.

Narrative and portable lessons: docs/kimi-k3-mlx-port.md §14-16. Live serving across the day:
6.04 → 6.52 (dummy-row early-exit) → 6.64-6.68 tok/s (fusion), greedy-verified at each step.

## Addendum 3: jaccl (TB5 RDMA) collectives adopted (2026-08-07)

Same-harness collective latency ring vs jaccl and the live outcome (6.6 → 6.9-7.1 tok/s,
transcripts bit-identical) are in docs §17, with the bridge0 surgery and manual-rendezvous
recipe. Day arc for the serving campaign: 6.04 → 6.52 (dummy-row elision) → 6.66 (GLU fusion)
→ 7.0 (RDMA collectives) — +16% in one day, every step transcript- or teacher-anchored.

## Addendum 4: read-bound model, revived attention-TP (+9%), precision-axis closure (2026-08-09)

Receipts: [`topk_probe_corrected.txt`](topk_probe_corrected.txt) (the corrected sweep with
kernel-side R evidence — top_k=1 is −11.2%, not 0%), [`dspark_alpha_quantized_target.txt`](dspark_alpha_quantized_target.txt)
(bf16 drafter vs 2.10bpw target: accept-len 1.55). Narrative and rules: docs §19-21.
Serving arc this week: 6.04 → 7.4 (certified) → **8.1 tok/s** (attention head-TP, final KL
smoke pending at commit time). 4-bit always-stack rejected on KL (+10%); DWQ-corrected low-bit
is the priced reopening path.

## Addendum 5: TP ships as default — 8.3 tok/s, byte-identity certification (2026-08-09)

Receipts: [`tp_parity.txt`](tp_parity.txt) (single-box head-shard parity, rel_max ≈5e-3 on
KDA and MLA layers = bf16 rounding scale), [`tp_greedy_identity.txt`](tp_greedy_identity.txt)
(**byte-identical greedy transcripts** TP vs non-TP, 3 prompts × 96 tokens — the "KL smoke
pending" note in Addendum 4 is resolved by something strictly stronger: exact output identity).
Final decode **8.28-8.29 tok/s** on fresh boots; the ≳1.5k-token TP prefill stall is fixed by
chunking the forward graph itself (256-token prefill chunks — chunking only the collective
payloads does not help; the deadlock tracks lazy-graph size, not payload size), verified with
a 2k-token prompt (42.6 s, no stall). Root cause isolated with a 4-layer partial-load
bisection harness (~15 GB at risk instead of 395 GB). Full story: docs §22.

## Addendum 6: shared-expert TP composes with fusion — 8.8 tok/s (2026-08-09)

Receipts: [`dense_tp_parity.txt`](dense_tp_parity.txt) (single-box parity incl. the
sliced-then-packed fused path, rel_max ≈1e-3), [`kl_densetp_4win.txt`](kl_densetp_4win.txt)
(4-window teacher-anchored KL — every window within 0.002 nats of the non-TP baseline).
The "conflict" that had shelved shared-expert TP was an install-order bug (fusion packed
full-width weights before TP sliced them); shard-then-fuse composes cleanly with zero extra
collectives. Decode 8.3 → **8.8 tok/s**, 2k-depth decode 7.3 → 8.7. Full story incl. the
eval-harness wedge saga: docs §23.

## Addendum 7: v7 MMA fused-codebook prefill — 2k prefill 24.8s under full TP (2026-08-09)

Receipts: [`v7_parity_bench.txt`](v7_parity_bench.txt) (parity vs certified kernel 4e-3 =
bf16 scale; microbench 1.55×/2.11×/3.59× by regime), [`kl_v7_4win.txt`](kl_v7_4win.txt)
(4-window KL within 0.0006 nats). Ternary-codebook dequant fused into the MMA threadgroup
loader — beats both the token-parallel kernel (3.6×) and dequant-then-GEMM (2.65×); with
the per-call regime ladder and 512-token chunks, TP prefill now beats non-TP outright.
Design + regime map: docs §24.

## Addendum 8: 4-bit attention + DWQ — a precisely-priced rejection (2026-08-10)

Receipts: [`attn4_dwq_verdict.txt`](attn4_dwq_verdict.txt). DWQ recovery worked
(KL +12% → +6.5%), but the speed side collapsed: attention TP had already consumed half the
lever (+0.3 tok/s measured, not +18%). Rejected; artifact preserved. The 13-nats debugging
saga (grid-scales marriage, recipe lineage, control-experiment-beats-audits) : docs §25.

## Addendum 9: speed-neutral Korean re-tiering — ko-KL −20/−30% at unchanged speed (2026-08-11)

Receipt: [`ko_retier_kl.txt`](ko_retier_kl.txt). Mixed-corpus hi-tier experts are ~orthogonal
to Korean routing mass (overlap 17.7%); re-tiering to the Korean oracle (same per-layer count
→ speed invariant) cuts Korean KL 20-30% and PPL 3.9% for free. Method + disk/verify
discipline: docs §26.

## Addendum 10: prefill top-k gating promoted (τ=0.90) — prefill −5%, decode untouched (2026-08-11)

Receipt: [`pf_topk_tau_gate.txt`](pf_topk_tau_gate.txt). Cumulative-router-mass gating on
prefill drops the low-mass expert tail (E[k]≈13 of 16); the tail is ternary-noisy, so quality
is neutral-to-better (Korean KL −5%) while prefill gains ~5%. Chunk-size 1024 rejected: full-scale
wedge despite mini-harness clearance — clearance at small scale is not a promotion ticket.

## Addendum 11: on-policy drafter alignment + adaptive speculative decoding (2026-08-12)

Receipt: [`spec_decode_campaign.txt`](spec_decode_campaign.txt). On-policy tap dumps (target
self-generation) lift offline accept +33% (doc) and close the chat-distribution gap (+59% on
chat); live spec decode reaches **10.0-10.6 tok/s on code (+15-21%)** — first double-digit on
this build — with an adaptive accept-gate falling back to plain when drafting loses. Korean
long-form still nets negative (dump-stack vs serve-stack hidden drift, next campaign). Lessons:
measure accept on the SERVING distribution and stack, not the training one; a once-only gate
misses mid-generation collapse — roll it.

## Addendum 12 (cross-model, Motif-3): fence profiles lie — ablate for wall shares (2026-08-13)

Motif-3 Q8 decode sat at 14.4 tok/s (25% of BW ceiling). A fence-instrumented profile said
mHC=15%; compile-fusing all the glue it blamed moved nothing (+0-2%). **Whole-component
ablation** told the truth: mHC = 43% of wall — its 20-iteration Sinkhorn loop issues ~4,200
16-element GPU kernels per token, invisible to fences (async dispatch hides under GEMMs) and
untouched by mx.compile (which cuts CPU graph cost, not GPU kernel count). One
`mx.fast.metal_kernel` doing the whole loop in registers: **14.4 → 20.2 tok/s (+40%)**,
KL vs eager 2.2e-3 / 0% top-1 flips. Lessons: (1) fence shares are for hypotheses, ablations
are for decisions; (2) tiny-op loops are dispatch bombs — single-kernel them; (3) A/B one
sample lever before building a fusion campaign on a profile. Follow-up (same model): GEMV
concat-fusion and expert gather-fusion both measured NULL — launch-count is not the wall
either; after the dispatch bombs are gone, decode is bound by the serial dependency chain
(~1k dependent tiny kernels/token), which neither fusion nor mx.compile shortens. Structural
plateau ≈ 20.2 tok/s; the next real lever is shortening the chain itself (MTP self-spec).

## Addendum 13 (cross-model, Motif-3): acceptance is a *rule*, not a number — and check your metric's denominator (2026-08-14)

Chasing a "38–41% vs vendor 70–80%" MTP draft-acceptance gap produced three portable lessons.
(1) **Denominator check first**: our "acceptance" was the fraction of *emitted* tokens that came
from the draft — a/(1+a) for k=1 — not per-draft acceptance. 41% draft-fraction *is* 69%
acceptance; the measured 1.21× end-to-end was arithmetically impossible under the literal
reading. Before debugging a cross-stack metric gap, reconcile definitions — vLLM's reported
rate comes from its RejectionSampler, ours from greedy equality. (2) **Acceptance is a rule**:
strict sampler-equality discards argmax-mismatched tokens that Leviathan rejection sampling
(accept draft x w.p. min(1, p(x)/q(x)), resample from (p−q)+ on reject) accepts losslessly.
Implemented in ~40 lines by reshaping only the verify-side token vector (on rejection the
residual has zero mass at the draft token, so the existing equality loop needs no changes).
(3) **On a well-trained native head the two rules converge** (~80% either way at T=0.8 —
the draft IS the target's own MTP block, p≈q), so rejection buys little there; its value is
the **mismatch regime** — on a mismatched pair (Beta backbone driving the final-release MTP
head) it lifted acceptance 52%→85%, +21% end-to-end. Ship it as mismatch insurance with a
kill switch. Bonus findings: a 300-token NLL probe overstated the 4.5bpw-vs-8bit Korean delta
3× (+9.9% → +2.4% at 1.5k tokens — size probes before trusting them), and a ladder artifact
reused across model *generations* must be provenance-checked (shard mtimes + config knobs +
the bundled card caught a Beta-era build masquerading as current). Speed economics of the
4.5bpw tier: batch-1 decode is read-bound, so 8.5→4.55 bpw took 24 → ~30 tok/s (+25%) for
+2.4% KO NLL on a 315B MoE.

## Addendum 14 (cross-model, Motif-3): latency-bound regimes invert vendor speculative-decoding guidance (2026-08-14)

Three results from pushing the same build 24 → 43 tok/s in one day. (1) **Verify width is
nearly free when decode is latency-bound**: a 4-token verify forward costs ~1.5x a 1-token
one (46 → 70 ms measured), so chained MTP drafting at k=3 beats k=1 by ~35% on Apple
silicon — directly inverting the vendor's (compute-bound, vLLM) "1 speculative token is
optimal" guidance. Check which regime you are in before importing serving folklore.
(2) **The chain-norm detail is load-bearing**: feeding chained drafts back through the
backbone's final norm a second time (instead of the MTP block's own output norm) crushed
chained acceptance enough to make k=2 net-negative (19.3 tok/s); fixing one norm turned the
same k=2 into +15% (34.6). When a speculative chain underperforms, audit the anchor/chain
normalization path before blaming the head. (3) **Dispatch-bomb hunting generalizes and
compounds**: after the Sinkhorn mega-win, compiling the remaining elementwise glue
(activation polynomials, router, mHC mixes, attention epilogue — all bit-exact, each
kill-switched) added +5% and then +9-11% on top; plain decode went 20.2 → 24.5 (+21%)
from fusion alone. The same sweep also produced a clean rejection: a fused rmsnorm→qmv
Metal kernel passed accuracy but died on regime analysis — with k=3 the hot path runs
width-4 forwards, and an N=1 GEMV kernel only touches the (tiny) drafting side. Kill the
lever before integration when its applicability window has already moved.

## Addendum 15 (cross-model, Motif-3): the transition mega-kernel — when a mega-kernel finally works (2026-08-14)

Addendum 12 recorded a mega-kernel failure (fusing big GEMMs into one threadgroup serializes
the memory system). The fix is scope: fuse only the *small-tensor bookkeeping* between
blocks — here a 16K rmsnorm, a 24-output gate projection, 20 Sinkhorn iterations, the
residual premix, and the next layernorm, i.e. ~7 serial dispatches per block transition —
into one threadgroup per position, keeping every large GEMM outside. Result: transition
latency 175 → 84 µs at verify width 4, end-to-end **+17% (4.5bpw, 50.2 tok/s) / +21%
(8-bit, 37.1)**. Two corollaries worth keeping: (1) as latency is removed, decode slides
toward bandwidth-bound and the optimal speculative width *shrinks* for fatter builds —
the 8-bit optimum moved from k=3 to k=2 while 4.5bpw stayed at k=3; re-tune k after every
big latency win. (2) A follow-up that folded the (multi-threadgroup-parallel) residual
postmix into the same kernel measured neutral: removing a dispatch only pays when the
removed op's lost parallelism is worth less than the inter-kernel gap. Small serial ops
fold; mid-size parallel ops don't. Also budget for a ±5% acceptance-trajectory noise band
in MTP end-to-end numbers — adjudicate levers only above it.

## Addendum 16 (cross-model, Motif-3): oracle-first evaluation kills two plausible levers for the price of none (2026-08-14)

Two directed levers died on cheap measurement before any build. (1) *Tree drafting*: a
768-step oracle probe — just log the verify target's rank in the existing draft logits —
showed rank-2 coverage of 11.6% (the entire theoretical prize of a second branch) and a
30% depth-2 conditional on the alternate branch vs ~62% on the argmax branch: this head's
misses are hard misses, not near-misses. Upper bound +4.7% at zero cost, net negative
after the tree's real machinery (per-depth rope groups, KV compaction, wider verify).
Speculative trees only pay when the drafter's rank distribution says so — probe it first
(half an hour) before building masks and cache surgery. (2) *Fused norm→GEMV at verify
width*: accuracy-correct, but at N≥2 a hand-rolled per-lane kernel loses 3-5x to the
library's simdgroup-matrix quantized matmul — a ~14 µs stage-removal target buried under
a ~170 µs kernel deficit. Custom kernels earn their keep on N=1 GEMVs and non-GEMM
bookkeeping; past N=1, join the library, don't fight it.
