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
sample lever before building a fusion campaign on a profile.
