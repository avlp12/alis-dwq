# Case study: porting **and** quantizing Motif-3-Beta to MLX (314.84B MoE)

Unusually for this repo, the compression was the *easy* half. Motif-3-Beta is a
from-scratch architecture whose published HF reference could not run; almost all of
the work was making a *correct* MLX forward exist at all, and the transferable lessons
are about diagnosing a broken model, not about bits. **Part 1 is the port; Part 2 is
the (comparatively routine) DWQ ladder.**

## The model

Motif-3-Beta — **314.84B MoE, 97.84% of the mass in routed experts (≈13B active),
text-only**:

- **53 layers**: 2 dense + 51 MoE.
- **MoE**: 384 routed experts, top-8 + 1 shared; Grouped PolyNorm per-expert activation.
- **GDLA attention**: MLA-style low-rank q/kv + differential-v2 + elementwise gate.
  **The KV cache is plain GQA (16-head), NOT latent** — the low-rank projection is a
  weight-side factorization, so at serve time it caches like ordinary GQA. (This is why
  `wkv_a` gets its own bit treatment and why the 4-bit KV probe below behaves.)
- **mHC 4-wide hyper-connections** (Sinkhorn, 20 iterations).
- **Interleaved SWA**: window 129; YaRN on the full-attention layers, plain
  `swa_rope_theta` on the SWA layers.
- vocab 220160; stop tokens `{0, 3, 6}`.

None of the load-bearing behavior lives in a library — all of it had to be re-derived
and checked against a reference that was itself broken.

## Part 1 — the port

### A published reference can be broken; parity proves faithfulness, not correctness

The initial Motif HF reference (SHA `32b0305`) **could not run**. Three independent
defects:

1. **yarn-rope `inv_freq` built from `head_dim=192` instead of `qk_rope_head_dim=64`**
   → first-forward crash.
2. **custom eager attention missing the GQA repeat** (80 vs 16 heads) → work around with
   `attn_implementation="sdpa"`.
3. **`grouped_mm` expert dispatch calls `_apply_gate` with no expert index** →
   expert-0's PolyNorm coefficients get applied to all 384 experts, **silently** → work
   around with `experts_implementation="eager"`.

We reported all three (HF discussion #6); Motif fixed them (SHA `d2c9ac6`). The
load-bearing lesson for anyone porting a fresh architecture: **a parity harness that
agrees with the reference proves your port is *faithful to that reference*, not that
either is correct.** Point parity at the *fixed* reference, and keep "matches reference"
and "is correct" as two separate gates.

### Root cause of incoherence was the activation, not RoPE

The port produced fluent-looking-but-garbage output. On a model stacked with YaRN, SWA,
and differential attention, the instinct is to blame rope. **It was not rope.**

**The position-0 test settled it.** The garbage was already present generating from a
single BOS token — position 0, where the attention softmax over one key is exactly 1, so
the rope phase and the q·k scores are irrelevant to the output. **Garbage at position 0
is a position-independent defect** → the suspect is anything that runs identically at
every position: activations, norms, MoE combine — *not* rope or attention scores. That
single observation redirected the entire investigation.

**The fix was PolyNorm.** Grouped PolyNorm must:

- apply `sigmoid(weight)` to its three coefficients (`polynorm_sigmoid_weight`, default
  true),
- multiply by `polynorm_output_scale = 0.5`, and
- for the routed GroupedPolyNorm, clamp the bias to ±`polynorm_bias_clamp` (0.5).

These were config keys **present but unread** in the buggy code. Applying just this took
held-out NLL **≈23 → ≈2.1** and produced fluent output.

Secondary corrections found alongside it:

- SWA layer assignment is `layer_idx % period != 0` (**not** `(i+1) % period`).
- the mscale² softmax factor applies on **full-attention layers only**.
- mHC `h_post = 1.0·sigmoid` for Motif-3 (config `mhc_h_post_alpha_end = 0`), **not** the
  paper's `2·sigmoid`.

### The diagnostic playbook (reusable)

What actually localized the bug, in rough order of leverage:

- **Depth-sliced logit probes** — decode logits after each layer. Dense layers were
  fine; the **first MoE layer** broke — localizing the defect to the MoE stage before a
  single component was touched.
- **Position-0 test** — isolates position-independent bugs (above); the single
  highest-leverage probe here.
- **Component-ablation NLL matrix** — zero/neutralize each component, measure recovery.
  **Caveat, learned the hard way:** ablating a large-magnitude *broken* component only
  *dampens* the garbage; it does **not** prove that component is the root cause. We were
  briefly misled by a "shared-expert" ablation signal that was really the whole model
  being wrong downstream of it.
- **Weight-statistics forensics** — per-row-norm CV/kurtosis/skew to tell gate from up;
  subspace alignment to tell k from v; period-32 rope-pair correlation.
- **Layout sweeps** — an 8-combo rope sweep + a 64-combo layout search, both of which
  came back "HF layout is correct," *exonerating* rope/layout and further pointing at the
  activation.
- **Independent corroboration** — the community FP8 mirror hit the same wall, confirming
  the defect was in the shared reference, not our port.

Operationally important: **these are forward-math fixes — the weights never change — so
none of them required re-quantization.** Every fix was verified by hot-patching the
existing Q8 build.

### Parity harness

- **4-layer truncation** (dense×2 + MoE×2; SWA×3 + full×1) — exercises every layer type
  cheaply.
- torch **fp32** reference, `sdpa` + eager experts.
- prompts **<129 tokens** (torch sdpa ignores the SWA window, so staying under it keeps
  the two forwards comparable).
- target **KL ≈1e-7/token, top-1 100%**.
- point it at the **fixed** reference and keep `rope_scaling` (the fixed code handles it).

### Two container bugs worth their own note

- **`mx.split` silently corrupts tensors over 2³¹ elements**
  ([ml-explore/mlx#3836](https://github.com/ml-explore/mlx/issues/3836)). `mx.split` on
  the 8 GB fused `gate_up_proj` (a >2³¹-element bf16 tensor) returns **corrupted data
  past the 4 GiB offset** — corruption begins around expert 205/384. Basic strided slices
  are correct, so **slice, don't split** any large fused-expert MoE. Reported-but-unfixed
  on mlx 0.31.2 / 0.32.0.
- **SWA mask built from the wrong cache (our own bug, from the `is_sliding` change).**
  Build the windowed mask from the **first SWA layer's** cache, not `cache[0]`. Once
  layer 0 is full-attention (`i % period == 0`), `cache[0]` is a plain `KVCache` with the
  wrong key length for a windowed mask. This broke eval's chunked forward and any
  >129-token decode.

## Part 2 — the quantization

With a correct forward in hand, the DWQ ladder was routine by this repo's standards.

### Routing census first

`expert_traffic` on Motif came back **flat** — the top-64 of 384 experts carry ≈40% of
the mass, the same GLM-5.2-family shape (§0 of the main README). Verdict: **uniform
expert bits — no salient/tail split, no REAP.**

### Recipe (`motif_quant_predicate` profiles)

| path | treatment |
|---|---|
| routed experts | lowest bits (the profile's expert bits) |
| `wkv_a` (low-rank kv down-projection) | **8-bit** — the precision chokepoint |
| embed / head | 6-bit |
| attn / shared expert / dense | the profile's attn bits |
| router / mHC / `lambda_proj` / norms / `act_fn` | **kept fp** |

Three profiles: **FLOOR** = 2b/g128 · **C6** = 4b/g64 + attn6 (the DWQ teacher) ·
**Q8** = 8b.

### Build ladder (FLOOR, vs the 8-bit Q8 reference)

KL is per-slice EN/code/KO against the Q8 build; degeneration is the greedy
distinct-4gram ratio.

| stage | KL EN | KL code | KL KO | degeneration (distinct-4gram) |
|---|---|---|---|---|
| FLOOR raw | 1.007 | 0.350 | 1.235 | **code LOOPED** (0.011) |
| + clip | ≈1.0 | ≈0.35 | ≈1.2 | code 0.011→0.054; KO 0.13→0.86 |
| + DWQ | **0.277** | **0.100** | **0.522** | EN 0.46 / code 0.96 / KO 0.88 |

Two things this ladder isolates cleanly:

1. **Clip fixed the degeneration while barely moving KL.** Raw FLOOR *looped* on the
   code slice (distinct-4 0.011); clip lifted code to 0.054 and KO from 0.13 to 0.86 —
   the model stopped degenerating — yet KL-vs-Q8 stayed essentially flat. This is the
   **reference-lattice effect**: KL against one specific reference can sit still while
   real generation quality improves (the generation-space twin of the quantized-sibling-
   KL note in the main README). Gating on KL alone would have missed clip's entire
   contribution.
2. **DWQ then cut KL by −58% to −72%** (KO 1.235→0.522, EN 1.007→0.277) and pushed
   distinct-4 to 0.46 / 0.96 / 0.88. The two passes fix **different** failure modes —
   generation-space (clip) and distribution-space (DWQ) — and they stack.

**KV probe:** 4-bit KV self-KL **0.006** → **4-bit KV is viable** for this model (the
plain GQA cache, not a latent one, quantizes cleanly — a ≈4× long-context memory win to
state on the card).

### DWQ

- **Teacher: C6 (4.5 bpw).** The teacher-precision / student-capacity sweet spot from the
  main README, applied as a prior rather than re-litigated: a **4.5 bpw teacher for
  ≤2.56 bpw students**. FLOOR sits well under that line, so the rule picks C6 directly.
- **K=8** layerwise; **lr 1e-5** (the ≈85 GB student tolerates it — consistent with the
  87 GB-student anchor in the main README, and 10× the 1e-6 a 242 GB student needed);
  **45% Korean** calibration mix (the target-language slice for this model); targets
  dumped once, then the teacher freed.
- valid **0.379 → 0.200**; rounds 1–6 accepted, **round-7 revert = the natural stop** (a
  late rollback is the gate working, as on both GLM retunes). Peak **125 GB**.

## Shippable — three tiers

| tier | size | role | recipe |
|---|---|---|---|
| **8-bit (Q8)** | 312 GB | reference / teacher-of-record | 8b |
| **4.5 bpw (C6)** | 167 GB | golden daily driver + DWQ teacher | 4b/g64 + attn6 |
| **2.3 bpw (FLOOR)** | 85 GB | the 128 GB-Mac floor | 2b/g128 + clip + DWQ |

Non-commercial research license (carried from the base model). All three run via the
mlx-lm fork branch:

```bash
pip install git+https://github.com/avlp12/mlx-lm.git@motif
```

Operational asides that carried over from the GLM campaigns and paid off again here:
offload the big intermediate builds to an external SSD via a **non-destructive symlink**;
publish the HF repos **public from the first commit** to sidestep the private-storage cap
entirely (see the HF-publishing notes in the main README).

## The one that got away — a shipped 8-bit build was corrupt

Four days after release a user reported the 8-bit tier producing collapsing text and
infinite repetition. It reproduced on a fresh download, **at greedy decoding**, so it was
not sampling. The 2.3 bpw tier from the *same checkpoint* was fine.

**Root cause: the build predated its own fix.** The fused `moe.experts.gate_up_proj` is
`384 × 4096 × 2560 = 4.03e9` elements — **1.88× the 2³¹ limit** where `mx.split` silently
returns corrupted data past the 4 GiB offset ([ml-explore/mlx#3836](https://github.com/ml-explore/mlx/issues/3836)).
`sanitize()` was later changed to strided slices, and every build after that is clean. The
Q8 was converted *before* it and was never re-made, because of the rule in Finding 5 —
which is true for forward-math fixes and **false for this one**.

**Confirmation.** The rebuilt Q8 fixed generation (same greedy prompt: the old build ran
`대한민국의 수도 and double-checking and double-checking…` forever, the new one answers
`서울특별시` in 74 tokens). Comparing a preserved sample of the old build against the new one,
experts 0–3 of a layer are **byte-identical** — the two builds agree below the corruption
offset and diverge only past it. Exactly the mx.split signature.

### What this costs downstream — the reference poisons its own metrics
The Q8 was the **KL reference** for the whole ladder. Every `KL vs Q8` number, and the
"DWQ cut KL 58–72%" claim, was measured against a corrupted baseline and had to be
withdrawn from the published cards. The models themselves were fine; the *measurements*
were not. Port parity (`KL ≈1e-7` vs the fixed torch reference) was unaffected because it
never went through Q8.

### Rules taken from this
1. **Generation-verify every shipped tier, not just the headline one.** The floor build was
   probed exhaustively; the reference build shipped on a smoke test run before the fixes.
2. **A KL anchor must be generation-verified before anything is measured against it.**
   An anchor is load-bearing for every number in the family.
3. **When a weight-materialization fix lands, list every artifact built before it and
   re-make or re-verify each.** Nothing else catches this — the corrupt build loads,
   passes shape and index checks, has healthy per-tensor statistics, and quantizes to
   byte-identical size.
4. **Sample forensics where the bug is, not where it is convenient.** First-pass statistics
   over experts 0–3 looked perfectly healthy; the corruption started at expert 205.

### Conversion gotcha (transformers ≥5.14)
`AutoTokenizer` resolves through `AutoConfig`, whose rope standardization does
`rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)`
and raises `AttributeError` on Motif's config. Removing `rope_scaling` from the source
config clears it — which is also what the known-good shipped builds already carry
(`rope_scaling: null`), so this restores the established shape rather than deviating from it.

## Findings

1. **On a fresh architecture, correctness is the project; compression is the epilogue.**
   The DWQ ladder here was a day's work; the port was the campaign. Budget accordingly.
2. **Garbage at position 0 is a position-independent bug.** It cost one probe to redirect
   the hunt off rope and onto the activation, where the real defect (unread PolyNorm
   config keys) lived. NLL ≈23 → ≈2.1 from that one fix.
3. **Parity to a reference is faithfulness, not correctness** — and a *published*
   reference can be the broken one. Gate on both.
4. **Clip and DWQ fix different things.** Clip bought back the degeneration
   (generation-space), DWQ bought back the KL (distribution-space); neither alone would
   have shipped this floor build.
5. **Forward-math fixes are free of re-quantization — *weight-materialization* fixes are not.**
   Most correctness fixes here were forward-math and were verified by hot-patching the
   existing Q8; the weights never moved. But a fix that changes how weights are *produced*
   — `sanitize()`, the tensor split, the predicate — invalidates every build made before it.
   Sorting fixes into these two buckets is not bookkeeping: getting it wrong is how this
   campaign shipped a broken 8-bit build (§ *The one that got away*).
