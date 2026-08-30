# Porting integrity checklist (checkpoint → a number you can trust)

Every item below cost a real mistake porting **Qwen3.8-27B** to MLX, and every one of them
is **silent**: the model loads, the shapes check out, the artifact passes every structural
test, and the number at the end is wrong — or the *capability* at the end is missing and
nothing says so. This file is the reusable half; the campaign narrative and the raw
receipts are in [examples/qwen3.8-27b](../examples/qwen3.8-27b/README.md).

Each item has the **why**, the **fix**, and where possible an **oracle** you can run.
There is a battery at the bottom.

---

## 1. A converter silently drops the tensors the model class does not consume

**Why.** A conversion pipeline is written around one model class, and that class's
`sanitize` returns only the tensors it knows how to use. Anything else — a vision tower,
an audio encoder, a draft head, an auxiliary predictor — is dropped, usually without a
warning, because from the converter's point of view nothing is missing. The measured case:
mlx-lm's `qwen3_5` `sanitize` discards `model.visual.*` (**333 tensors / 0.461B params**)
and `utils.save_config` does `config.pop("vision_config")`.

**Why it is worse than a normal bug.** Dropping the weights *and* the config key produces
an artifact that is **internally consistent**: a stripped VL model is byte-for-byte
indistinguishable from a conversion of a text-only model. It cannot be caught by
inspecting the output — only by comparing against the source. Measured consequence: **all
12 public MLX builds of this model carry 0 vision tensors**, including ones with `-vision`
in the repo name. Seven of the twelve are the *sloppier* variant — they ship
`preprocessor_config.json` next to weights that cannot process an image — and those seven
are the only ones a self-consistency check can find.

**Fix — put pass-through in `save()`, not in the model.** A model declares
`passthrough_patterns`, and tensors matching them that the model does not consume are
copied to the output **as original bytes**. Doing it at the save layer means `convert` and
`awq` share one gate and future architectures inherit it; doing it per model means the next
architecture re-learns this the same way.

Four parts, each of which is load-bearing:

1. **Skip already-written keys with a suffix-aware comparison.** `sanitize` *reparents*
   modules — in this checkpoint `mtp.x` → `language_model.mtp.x` — so an exact-name check
   sees the reparented tensor as absent, re-emits it under its old name, and the index ends
   up describing one weight twice.
2. **Read the bytes back and verify before advertising them in the index.** An index entry
   pointing at an unwritten or truncated tensor fails at load time and looks exactly like a
   corrupt download.
3. **Keep the sub-config only when the weights actually survived.** Both directions are
   bugs: advertising a tower a stripped checkpoint does not have, and hiding a tower that is
   present. **The config must follow the bytes**, decided at save time from what was
   written — not from what the source config said.
4. **Carry the preprocessor config with the weights.** The seven repos above are what the
   alternative looks like.

**What it buys.** Verified on this model: vision tower preserved **byte-exact, 333/333**;
text inference unchanged (decode **37.39 vs 37.61 tok/s**, inside noise). And because
**mlx-vlm 0.6.13 already supported the architecture**, images worked with **zero lines of
porting** once the weights were present. The ecosystem was never missing an
implementation. It was missing weights, because every build had been made by a converter
that threw them away quietly.

**The rule behind the rule:** *never let an artifact misdescribe itself.* A build that
knows it is text-only is recoverable; a build that presents as a faithful conversion of a
multimodal checkpoint is not, and neither is any measurement taken on it.

**Oracle:** the source-vs-build subsystem census in the battery below.

---

## 2. Never use an optional module's *presence* as a format discriminator

**Why.** Conversion code often needs to know "is this a raw upstream checkpoint or one I
already processed?", and the tempting witness is some module that raw checkpoints happen to
have. Measured case — stock `qwen3_5`:

```python
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
```

The intent is "raw HF checkpoint". But an **already-converted** checkpoint that *kept* its
MTP head still satisfies the first disjunct, so its norms are shifted a **second** time
(γ 0.94 → 1.94). Nothing crashes. Generation collapses quietly: on our Korean probe
**NLL 1.679 → 17.460**, worse than a uniform distribution over the 248,320-token vocab.

**And the failure frames the wrong suspect.** Builds *without* the MTP head measure
perfectly, so the evidence reads as "the MTP-preserving build is broken" — the bug
incriminates precisely the feature that exposes it. Expect to waste a day on the innocent
party.

**Fix — the discriminator must be something the transformation itself destroys.** The
second disjunct, `has_unsanitized_conv1d`, is a correct witness: the raw Conv1D layout
exists only before conversion and conversion consumes it, so it cannot survive into the
converted artifact. The MTP head is *incidental* — it passes through untouched and is
therefore evidence of nothing.

Stated generally: **`sanitize` must be idempotent, and a discriminator that survives its
own transformation breaks idempotence by construction.** Before shipping any
"is-this-raw?" test, ask what the second application does. If the answer is "shifts it
again", the witness is wrong.

Upstream had this from a 35B MoE ([issue #1197](https://github.com/ml-explore/mlx-lm/issues/1197),
[PR #1623](https://github.com/ml-explore/mlx-lm/pull/1623)); we reproduced it independently
on a dense 27B and filed [PR #1735](https://github.com/ml-explore/mlx-lm/pull/1735).
Corollary worth internalizing: **a silent-collapse bug can sit in a popular converter for
months**, because the models it breaks are the unusual ones nobody re-measures.

---

## 3. Make the harness fail loudly about which library it imported

**Why.** A fork under test and an installed release have the same import name. If the fork
is not first on `sys.path`, `import mlx_lm` quietly resolves to the release, and every
number the harness prints describes a different library than the one you believe you are
testing. In this campaign that silent fallback *was* the incident — item 2's collapse was
being measured with stock code while we reasoned about the fork — and every measurement
predating the fix had to be re-run.

**Fix — pin, then assert.** Both halves are required:

```python
for _fork in ("/path/to/fork",):
    if os.path.isdir(os.path.join(_fork, "mlx_lm")):
        sys.path.insert(0, _fork); break
import mlx_lm
if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit(f"stock mlx-lm resolved: {mlx_lm.__file__}")
```

The path insert is the fix; the assert is what makes the *absence* of the fix loud. A path
insert alone fails silently the first time the fork moves. Print the resolved path to
stderr on every run so the log itself carries the provenance.

**Second-order fix: encode the symptom as a tripwire in the reporting tool.** Our
comparison table warns when any build's probe NLL exceeds the reference's by more than 3×,
which is item 2's signature. Cheap, and it catches the class rather than the instance.

---

## 4. Queue-batched benchmarks hide latency — measure decode-path kernels on a chain

**Why.** Firing N independent calls and dividing wall clock by N avoids the ≈250 µs
per-call synchronization floor, which is exactly why it is the default technique. It also
lets the framework **overlap** those calls — so it measures throughput, and a decode loop
is not made of throughput. Measured on a hand-written quantized-matmul kernel
(N=K=5120, M=7):

| | independent | chained |
|---|---|---|
| MLX | 0.0765 ms | 0.0789 ms (**+3%**) |
| ours (MMA v2) | 0.0680 ms | 0.1530 ms (**+125%**) |

The kernel **won the microbench and ran the model at 0.56–0.82×**. Per-shape comparisons
all said it should win (layer-count-weighted 1.35×, `lm_head` 3.45×) and the Python wrapper
cost measured zero. "1.52× in the microbench" and "0.74× in the model" were **both true**;
only the chained benchmark could say why.

**Fix.** For anything on the decode path, report the **dependent-chain** number — each call
consuming the previous call's output — and treat the queue-batched number as a
throughput-only datapoint. Derive gating thresholds from the chained metric too: our
kernel's crossover is **M=6 chained**, and at M=4 it is 0.98–1.10× (neutral), so a
threshold read off the queue-batched curve dispatched it into a regime where it made MTP
k=3 *slower*.

(Related trap from an earlier campaign: a benchmark loop with no serial dependency
evaluates one iteration out of N — that one manufactures a number, this one inverts a
verdict.)

---

## 5. Small-`M` quantized GEMM does not amortize — price the verify curve before blaming the algorithm

**Why.** Speculative decoding assumes verifying `k` tokens costs much less than `k` single
token forwards. On this stack that assumption failed, and the cause was not the drafting
method. Measured, queue-batched, MLX 0.31.2 **and** 0.32.0:

- **bf16 is flat**: `M=2..16` all sit at **2.00×** of `M=1`.
- **4-bit `quantized_matmul` is nearly linear**: **5.28× at M=7**, **6.32× at M=16**.
  `M=1` sits on the roofline; `M=7` is **2.75×** off it.

Filed as [ml-explore/mlx#4265](https://github.com/ml-explore/mlx/issues/4265). It fully
explains the model's forward curve — S=1 29.7 ms, S=7 62.5, S=8 77.1 (**3.65× off
roofline**), S=32 105.4 (1.44×), S=128 347.9 (1.19×). **Speculative decoding lives exactly
in the worst region**: verify widths of 4–8 tokens are the entire point of drafting.

**Acquit the exotic component before suspecting it.** This is a hybrid model (48
GatedDeltaNet linear-attention + 16 full-attention layers) and the linear-attention kernels
were the obvious suspect. Measured standalone they are **flat in T** (0.28–0.37 ms,
T=1..64) — the recurrent state is register-resident, so their traffic is already amortized.
The unusual part of the architecture was innocent; the ordinary quantized GEMM was not.

**Fix (if you must write the kernel).** The winning shape was **split-K**: eight simdgroups
split K eight ways, with `x` loaded from device memory straight into the MMA registers,
removing threadgroup staging entirely. Chained latency **0.153 → 0.0558 ms** = **1.41× vs
MLX**, flattening the model's verify curve at constant output (top-1 match):

| verify width | before | after |
|---|---|---|
| S=6 | 62.5 ms | 44.5 |
| S=7 | 70.5 ms | 44.6 (**1.58×**) |
| S=8 | 77.1 ms | 43.3 (**1.78×**) |

Two intermediate versions are worth knowing about so you skip them: a scalar +
threadgroup-staging kernel amortizes correctly but runs at **0.30×** of MLX, and the
straightforward `simdgroup_matrix` 8×8 MMA version is flat in M (+12% from M=1 to M=8, and
moving the barrier from every k-tile to every **quantization group** took it 0.29 → 0.15 ms)
but is the one that lost on latency in item 4.

**Second-order effect worth anticipating: flattening the verify curve can retire a
feature.** A drafter head whose job is to *narrow* the block when confidence is low has no
premise once verify cost is flat in width — see item 7.

---

## 6. A speculative-decoding number measured on one prompt is a best case, not a result

**Why.** Acceptance length is strongly workload-dependent, and the end-to-end multiplier
moves with it. The same MTP k=2 configuration measured:

| benchmark | speedup |
|---|---|
| one code prompt | **1.41×** |
| three English/code prompts | **1.32×** |
| three prompts including Korean | **1.10×** |

**Fix.** Minimum three prompts of genuinely different character, and report the average. If
you ship a single-prompt number, say which prompt — a code prompt is the friendliest
workload speculation has, and quoting it as *the* result overstates by ≈30% here.

**And the spread can exceed the levers you are comparing.** After every fix in items 8–10,
the winning configuration on this model reads **1.91×** on English/code/math and **below
1.0× on Korean alone** — plain 37.6 vs MTP 34.3 and DSpark 33.3 tok/s, i.e. *both*
speculative paths lose to plain decoding on that workload (cause not yet decomposed; the
operational rule is to disable speculation for it). A language slice can invert the sign of
a lever, so a single-language benchmark is not a result for a multilingual model — the same
lesson the DWQ side of this repo learned about per-slice quality gates, arriving on the
speed axis.

---

## 7. The reference implementation shipped with a checkpoint may not be its inference path

**Why.** A released drafter/adapter repo often carries a generation loop inherited from the
**parent** method it was derived from. Measured case: our MLX port of the DSpark drafter
matched the PyTorch reference to **cosine 1.00000000 / 1.1e-4 max abs error** — and lost
end-to-end at **0.73×**. The bundled `spec_generate` is **DFlash's** loop; it never calls
DSpark's two distinguishing heads (`markov_head`, 127.1M = 9.3% of the drafter, and
`confidence_head`). Both loaded; neither was ever invoked. DSpark's real inference path
lives inside SGLang and is not in the repo. **Parity proved we had faithfully ported the
wrong loop.**

Wiring the Markov head took acceptance **2.23 → 3.31** on a code prompt and end-to-end
**0.73× → 1.20×**, landing *above* the model card's own acceptance numbers under the card's
conditions (average **4.48** vs the card's 3.39) — and then to **1.91×** once items 8–10
were fixed, which is how the method went from "rejected" to "the production
recommendation".

**Fix — a checkpoint's module list is a contract; check the loop against it.** Enumerate
the top-level modules in the checkpoint and grep the inference loop for each one. Any
module that is loaded but never called is either dead weight or a missing feature, and the
difference matters (see the battery). This is a ten-second check that would have saved the
campaign a day.

**Three corollaries measured here:**

- **Winning the drafter metric is not winning.** Higher acceptance loses whenever the
  drafter forward costs more than the extra accepted token returns — measured on this
  drafter, **block 9 has higher acceptance than block 8 (3.55 vs 3.46) and is slower**
  (70.4 vs 71.0 tok/s). **Judge a speculative scheme on tok/s, never on acceptance length.**
  We first wrote this rule pointing at DSpark losing to MTP end-to-end, and *that* example
  inverted the moment our own integration was fixed (item 8). The rule held; our instance
  of it was an artifact.
- **A drafter's trained block width is a reference value, not a ceiling.** This drafter is
  trained at block 7 and runs fastest at **block 8** (71.0 vs 63.6 tok/s over three
  prompts). Where verify cost is flat in width, spend the width — sweep it rather than
  inheriting the card's number.
- **Reproducing a paper's configuration is not the same as reproducing its result.** Two
  card-specified conditions measured *worse* for us: a bf16 drafter bought nothing over the
  4-bit one (acceptance 4.19 vs 4.23), and the reference's draft-slice convention
  (`[:, -B+1:]`) reached only 2.35 against our shifted read's 4.23. And the
  `confidence_head` measured **best switched off** (block 6: off 45.0 tok/s vs
  tau 0.10/0.25/0.50 = 37.1/38.5/42.8) — its whole purpose is skipping a wide block when the
  drafter is unsure, and item 5's kernel work removed the cost it was avoiding.

---

## 8. A feature you built but never wired does not exist

**Why.** An optimization usually arrives with its own benchmark script, and the script
enables it explicitly. That is the *only* place it gets enabled unless someone
deliberately wires it into a loading path — and nothing detects the omission: no test
fails, no warning fires, the model runs correctly and merely slowly. Measured case: our
split-K kernel's `fast_qmm.enable()` had **zero call sites outside the file that defines
it**. It was live inside the scripts that benchmarked it and nowhere else, so
`mlx_lm.generate`, the server and both speculative loops ran without it for the rest of the
campaign — S=8 verification at **70.1 ms instead of 43.1 ms**.

**This was the single largest loss of the campaign**, larger than any bug in this file, and
it produced something worse than a slow stack: an entire verdict (which speculative method
to ship) reasoned out on top of a stack that silently lacked the fix, and published.

**Fix.** When you finish an optimization, `grep` for its entry point across the whole repo
before you benchmark anything downstream of it. **Zero hits outside its own definition means
it does not exist in production.** Then wire it at the *loading* boundary rather than at
each call site — ours now goes through `utils.load()` — and give it a killswitch env var so
the A/B is still one command.

**Corollary — a side optimization can be welded to the main one.** Rounding the `lm_head`
batch up to the kernel's window is **+5%** with the kernel on and **−2%** with it off. Any
lever tuned against a feature must be re-measured whenever that feature's status changes,
including "was never on in the first place".

---

## 9. An optimization with a shape window is a claim about the run-time distribution

**Why.** Kernels, fused paths and fast branches typically win inside a window — a batch
range, a sequence length, a dtype. Proving the win inside the window says nothing about how
often the model is *in* it. Measured case: our kernel wins at **M ≤ 8**, and the speculative
loop it was meant to accelerate ran at **mean verify width 9.50, max 16**:

| verify width | step cost |
|---|---|
| ≤ 8 | 54.6 ms |
| 9–12 | 106.8 ms |
| 13+ | 142.8 ms |

The loop lived mostly *outside* its own optimization's window, in the regime costing
2–2.6× more. A one-line clamp holding width inside it (`min(n_spec, 8 - L)`, mean width
9.50 → 7.10, max 16 → 8) was worth **41.2 → 59.1 tok/s (+43%)**.

**Fix.** Instrument the shape the loop actually runs at — a histogram, not a spot check —
and either clamp the loop into the window or widen the window. Do this *before* attributing
any end-to-end result to the optimization, in either direction: outside its window a
"fast path" is not neutral, it is a regression.

---

## 10. An unaccounted overhead is usually an unmeasured item

**Why.** When a step is slower than the sum of its known parts, the residual is
overwhelmingly a component nobody has instrumented — not an intrinsic tax of the method.
Measured case: a ≈32 ms per-step residual was rationalized for most of a campaign as
"the drafter's structural overhead", and used to justify shipping the other method. It was
items 8 and 9, in a costume. Once both were fixed the accounting closed exactly:

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

**Fix.** Require a step budget that closes before publishing any verdict that depends on
it. If the parts do not sum to the whole, the missing time is a measurement you have not
taken — name it and instrument it. A residual you have not decomposed is not evidence for
or against anything, and "structural overhead of method X" is the most expensive way to
spell "I did not measure that".

---

## Pre-ship battery

**A. Subsystem census, source vs build** — the only check that catches item 1's clean
variant. Counts are matched by name *substring*, not prefix, because `sanitize` reparents
modules (`model.language_model.*` → `language_model.model.*`), and a prefix diff therefore
reports false drops.

```bash
python3 - <<'PY' <source_dir> <build_dir> visual.,mtp.
import json, os, sys
keys = lambda d: list(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"])
src, build = sys.argv[1], sys.argv[2]
markers = sys.argv[3].split(",") if len(sys.argv) > 3 else ["visual.", "mtp."]
s, b = keys(src), keys(build)
print(f"{'subsystem':16s} {'source':>7s} {'build':>7s}")
print(f"{'(all tensors)':16s} {len(s):7d} {len(b):7d}")
for m in markers:
    ns, nb = sum(m in k for k in s), sum(m in k for k in b)
    print(f"{m:16s} {ns:7d} {nb:7d}" + ("   <-- DROPPED" if ns and not nb else ""))
cfg = json.load(open(os.path.join(build, "config.json")))
print("\nvision_config in config.json:", "vision_config" in cfg,
      "| preprocessor_config.json:", os.path.exists(os.path.join(build, "preprocessor_config.json")))
PY
```

Reading it: **total counts are not comparable** (quantization splits one weight into
`weight`/`scales`/`biases`), so read the per-subsystem rows. A passed-through subsystem
matches its source count **exactly** — the vision tower reads 333 → 333 on our build
because it is copied as original bytes. Our stripped build reads 333 → **0**, with
`vision_config` **absent** and no preprocessor config: **self-consistent and lobotomized**,
which is precisely why the source column is mandatory.

**B. Import provenance** — run inside the harness, not beside it:

```bash
python3 -c "import os,sys; sys.path.insert(0,'<fork>'); import mlx_lm; p=os.path.dirname(mlx_lm.__file__); assert 'site-packages' not in p, f'stock resolved: {p}'; print('mlx_lm =', p)"
```

**C. Loaded-but-never-called modules** — item 7 in two commands. Print the checkpoint's
top-level modules, then grep the inference loop for each:

```bash
python3 - <<'PY' <checkpoint_dir>
import glob, os, sys
from safetensors import safe_open
mods = set()
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.safetensors"))):
    with safe_open(f, framework="np") as h:
        mods |= {k.split(".")[0] for k in h.keys()}
print("top-level modules:", sorted(mods))
PY
grep -c "markov\|confidence" <every .py in the repo>   # substitute the names printed above
```

Run on the DSpark drafter, this prints
`['confidence_head', 'fc', 'hidden_norm', 'layers', 'markov_head', 'norm']`, and the grep
across the repo's two source files returns **`dspark.py:42`, `dflash.py:0`** — while
`grep -ln "def spec_generate"` returns **`dflash.py`**. In one line: the heads are defined
in the modeling file, and the only shipped generation loop is the parent method's and never
touches them. That is item 7, visible before a single token is generated.

**D. Call sites of every optimization you added** — item 8, and the cheapest check in this
file. For each feature's entry point:

```bash
grep -rn "fast_qmm" --include='*.py' . | grep -v "/fast_qmm\.py:"   # substitute your own symbol
```

**Zero lines out means the feature is dead code**, however good its benchmark was. On our
repo this now returns `mlx_lm/utils.py:530` and `:532` — the wiring at the loading boundary.
For most of the campaign it returned nothing, and every downstream verdict was wrong
because of it. (Quote the `--include` glob: unquoted, zsh expands it and the grep silently
matches nothing — a fitting way to fail this particular check.)

**E. Shape-window residency** — item 9. If any adopted path wins only in a window, log the
shape every iteration and histogram it before trusting an end-to-end number:

```python
from collections import Counter
hist = Counter()          # in the loop:  hist[verify_width] += 1
# after the run:
print("mean", sum(k*v for k,v in hist.items())/sum(hist.values()), "max", max(hist))
```

Ours read **mean 9.50 / max 16** against a kernel window of M ≤ 8. Clamping into the window
was +43%.

**F. Benchmark protocol** — before quoting any decode-path number:

- kernel numbers reported from a **dependent chain**, not a queue batch (item 4);
- speculative-decoding numbers averaged over **≥3 dissimilar prompts**, and state which
  (item 6) — on this model the same configuration reads 1.91× on en/code/math and **below
  1.0× on Korean alone**;
- decode timer started **after the first token**, so prefill is not folded into decode;
- a **step budget that closes** (item 10): if the instrumented stages do not sum to the
  measured step, do not publish a verdict that rests on the difference.

A build that clears A–F has cleared the failure modes that load cleanly, pass every
structural check, and still ship the wrong thing. It does not check quality — measure that
separately.

## 11. Audit a port's numerics against *multiple* references before trusting any single deviation story (DeepSeek-V4-Flash case)

**Why.** A port can be "coherent at short context, broken at long context" for several
stacked reasons at once, and fixing the loudest one makes the output *look* healed while
quieter deviations keep degrading quality in ways you stop noticing. Measured case:
mlx-lm PR #1189's DeepSeek-V4-Flash port carried **three independent deviations** —
(a) per-layer rope/YaRN assignment collapsed to one global rope (the layer-conditional
`compress_rope_theta` instance was created but never called — dead code is the tell),
(b) the compressed-pool prefill mask was all-zeros where the reference clamps per query
`i < (p+1)//ratio` (future leakage → optimistic teacher-forced evals + prefill/decode
inconsistency), and (c) pool rows were never rotated (the reference applies
compress-theta rope at each row's block-start position, `i·ratio`). Fixing (a) alone made
4.9K-token generation coherent; the residual "occasional CJK token slips at 19K" — easy
to shrug off as quantization noise — was (c).

**Fix — triangulate with ≥3 references, then adversarially verify.** The procedure that
held up: pull the official reference, the transformers implementation, and one
independent port (FreeToken); derive the disputed rule from each *separately*; only claims
where all three agree go into the bug report. Then run two adversarial passes — one agent
briefed to *refute* each claim, one blank-slate agent that re-derives the rules without
seeing the claims — before publishing. The red team caught a real error in our own
follow-up (a "semantically identical" alternative that silently dropped a √D factor)
that the maintainer would have found in minutes.

**Oracles.**
- *Differential coherence:* same long-document prompt through the port and through an
  independent runtime (llama.cpp) — localizes breakage to the port without needing logits.
- *CJK-slip count:* for an English-output task, `len(re.findall(r'[一-鿿぀-ヿ가-힯]', out))`
  is a cheap long-range-degradation counter. Before the pool-rope fix: slips present at
  19K on every run; after: 0/250 tokens on the same task. A metric this crude still
  cleanly separated fixed from unfixed.
- *Attribution humility:* if a threshold coincides with a config constant
  (`index_topk × ratio = 2048` ≈ the observed ~2K breakage onset), report the
  differential measurement and *decline* to name a single cause. The upstream comment
  survives review; the confident version would not have.

Receipts: [examples/deepseek-v4-flash](../examples/deepseek-v4-flash/README.md) —
verified fix, validation harness, and the published upstream comments.

## 12. Fine-tuning through a quantized MoE forward: four VJP blockers, in the order you hit them, and why a working gradient still isn't a working result

**Why.** A draft/speculative-decoding head sitting on top of a quantized MoE backbone is a
common shape now (DeepSeek-family MTP, most speculative-decoding retrofits). If its acceptance
rate is disappointing, the fix looks like "fine-tune the head" — except every layer in the path
is some mix of frozen quantized weights and custom fused Metal kernels, none of which expose a
gradient. The blockers don't announce themselves as a group; they surface one at a time, each
looking like a different bug, in a fixed order dictated by how far the backward pass gets before
it hits the next wall. Measured case: aligning a DeepSeek-V4-style MTP head's non-expert weights
(attention, projections, norms — 74M params, bf16) via teacher-forced chain SFT, backbone
producing hidden states as a frozen teacher (`stop_gradient` on the hidden, never on the loss
graph downstream of the trainable block).

**The four blockers, in the order they appear:**

1. **Routed-expert gather has no VJP.** `RuntimeError: [Primitive::vjp] Not implemented for
   <QuantizedMxfp4GatherBlocks-style primitive>`. The MoE forward's routed path (`switch_mlp` /
   equivalent) selects a handful of experts per token via a gather over the packed quantized
   weight table — differentiating through that gather isn't implemented, full stop, regardless
   of what's frozen. Fix: `stop_gradient` on exactly the routed-expert output, computed inline
   inside a reimplementation of the MoE forward — **not** the whole MoE output, because the next
   term (below) needs gradient. Reference implementation:
   ```python
   def moe_forward_grad_safe(self, x, input_ids):
       inds, scores = self.gate(x, input_ids)
       y = self.switch_mlp(x, inds, scores=scores)
       if y.ndim == scores.ndim + 1:
           y = (y * scores[..., None].astype(y.dtype)).sum(-2)
       y = mx.stop_gradient(y)              # routed path: gather VJP doesn't exist
       y = y + self.shared_experts(x)       # shared path: ordinary matmul, differentiable
       return y
   ```
   The naive fix — monkey-patching `MoE.__call__` to `stop_gradient` its *entire* return value —
   compiles, trains, and silently produces zero gradient anywhere downstream of any MoE call.
   It takes a second bug (below, item 4 in the failure sequence you'll actually hit) to notice.

2. **Custom fused kernels (hyper-connection mixers, gating collapses, anything written as
   `mx.fast.metal_kernel`) have no VJP either**, but for a different reason than #1 — these
   aren't quantized, they're just forward-only kernels with no registered backward. Many such
   modules already carry a `self.training` branch that falls back to a plain-ops implementation
   for numerical-stability or CPU reasons — that fallback is differentiable even though the fast
   path isn't. Fix: call `.train()` on exactly the sub-module you're fine-tuning before the
   forward pass. Cheap and often already there for a different reason, easy to miss.

3. **Residual quantized weights the class instantiates on the side (output projections, packed
   grouped-projection layers) raise on the actual weight, not on a gather.**
   `RuntimeError: [QuantizedMatmul::vjp] no gradient wrt the quantized weights.` This is a
   distinct failure from #1 — it fires only if something (typically a blanket `.unfreeze()` on a
   parent module) leaves a quantized `Linear` unfrozen. `unfreeze()` on a container recursively
   unfreezes every leaf underneath it, including quantized ones you meant to leave alone, so a
   single `block.unfreeze()` followed by "refreeze what should stay frozen" needs an explicit
   sweep, not a name-based exclude list — new quantized submodules the exclude list didn't
   anticipate will surface this same error one at a time as the model architecture evolves:
   ```python
   def freeze_quantized(module):
       stack = [module]
       while stack:
           cur = stack.pop()
           for child in (cur.children().values() if isinstance(cur.children(), dict)
                        else cur.children()):
               if isinstance(child, nn.Module):
                   (child.freeze() if hasattr(child, "scales") else stack.append(child))
   ```

4. **A model-specific fused attention kernel, if one exists in the serving stack, will also lack
   a VJP** — same category as #2, but often not gated behind `self.training` because it was never
   meant to run under gradient tracking. Fix is blunter: monkey-patch the kernel entry point to
   `None`/no-op for the duration of training, so the call site's own stock-attention fallback
   takes over (most serving stacks have one, since the fused kernel usually has a coverage gate
   already — training just needs to force the "kernel unavailable" branch).

**Whether the resulting gradient is worth anything is a separate question from whether it
compiles.** With all four blockers routed around, the *trainable* surface is small — norms,
attention projections, embedding-adjacent linears — 74M parameters here. That's cheap to extend
further: wrap a quantized shared-expert `Linear` in a parallel low-rank adapter (dequantize
nothing, just add a differentiable side path) —

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=16.0):
        super().__init__()
        self.base = base                                    # frozen quantized layer
        out_dim, in_dim = base.weight.shape[0], base.scales.shape[1] * base.group_size
        self.lora_a = mx.random.normal((in_dim, r)).astype(mx.bfloat16) / (r ** 0.5)
        self.lora_b = mx.zeros((r, out_dim)).astype(mx.bfloat16)  # zero-init: starts == base
        self.alpha_over_r = alpha / r
        self.base.freeze()

    def __call__(self, x):
        y = mx.stop_gradient(self.base(x))
        return y + self.alpha_over_r * ((x.astype(mx.bfloat16) @ self.lora_a) @ self.lora_b)
```

This attaches cleanly, the gradient measurably flows into `lora_a`/`lora_b` (verified: nonzero
per-parameter grad norm), teacher-forced held-out accuracy improves after training. **Live
acceptance regressed across every metric anyway** — depth-1 conditional accept 95.6% → 85.7%,
depth-2 66.7% → 58.9%, depth-3 34.5% → 11.3%, tokens/cycle 2.81 → 2.44, on the *same* fixed-depth
chained-decode harness the pre-LoRA checkpoint was measured on. The self-reported eval set (model's
own greedy continuations, teacher-forced) went the other way — 95.9% → 96.5% — which is precisely
the failure mode item 6 of the on-policy port-fidelity work already flagged: an on-policy
*evaluation* set is still an evaluation the model is scored on by itself, and self-evaluation
optimism is not bounded by "on-policy" the way self-evaluation optimism from an *off*-policy
corpus is. The suspected mechanism here: the LoRA delta had nothing new to fit (same corpus,
continued from the checkpoint already converged on it), so 1500 more steps pushed the shared-expert
output slightly off whatever narrow operating point the frozen expert weights and the head had
already found together — a regression a held-out *live* chained-decode measurement caught
immediately and a same-corpus self-eval could not.

**Oracle.** Never trust a fine-tune of a speculative-decode head on self-graded metrics alone —
teacher-forced accuracy on the model's own generations, even nominally "held out," can move
opposite to the number that actually matters (live chained acceptance rate on fresh prompts,
measured through the real serving harness with the actual sampler). Gate every checkpoint
promotion on the live number, not the training log.

## 13. Train on the measured residual, not on a model of it — and verify in the regime you actually serve

**Why.** §12 ends with a draft head that fine-tunes cleanly and regresses live. This is the
sequel — same head, same stack, seven more rounds — and what decided it was not the optimizer or
the adapter but two things neither the training log nor the eval log can show you: *which
distribution the head is trained on*, and *which regime the offline eval runs in*. Measured case:
the same DeepSeek-V4-style MTP head, chain-aligned on single-box hidden states, serving under
2-box tensor parallelism.

**The gap being closed.** The backbone does not produce the same hidden states on one box and on
two. Measured single-box↔TP2 hidden-state drift: **≈0.4% relative std, kurtosis 414** — decidedly not
Gaussian, concentrated in a few positions and channels. Promoting every `all_sum` to fp32 changed
the hidden states *not at all* (bit-identical output), which is itself the diagnosis: the
divergence lives in the K-split partial sums of the row-parallel GEMMs, not in the reduction's
precision — summing already-diverged partials exactly is still exact, and still different. The
phenomenon is independently established (arXiv 2511.17826 on deterministic inference across TP
sizes; DeepSpeed #7500; vLLM's batch-invariant kernel work; NVIDIA NeMo-RL's TP notes), and none
of those fixes reach a speculative-decoding path — which is why this had to be trained around
rather than engineered away.

**Three rounds that never showed the head a single real TP2 hidden state.** Rounds 3–5 each
continued from the same round-2 checkpoint on the same on-policy corpus, and each tried to close
the gap without leaving the single box: a LoRA side path on the quantized shared experts adding
capacity (§12), then Gaussian noise at 1% and at 0.3% of `h.std()` meant to *imitate* the drift.
Live, on a fixed-depth chained-decode harness under TP2 (single prompt, `d1/d2/d3` = conditional
accept at chain depth 1/2/3):

| round | what it trained on | d1 | d2 | d3 | tok/cycle |
|---|---|---|---|---|---|
| 2 (baseline) | single-box hidden, on-policy chain alignment | 81.9% | 61.6% | 18.9% | 2.44 |
| 3 | + LoRA on quantized shared experts | regressed on every metric (§12) | | | |
| 4 | + Gaussian noise, 1% of `h.std()` | 71.8% | 53.6% | 17.8% | 2.19 |
| 5 | + Gaussian noise, 0.3% | 68.9% | 48.8% | 19.5% | 2.10 |
| 6a | **measured TP2 hidden**, 60 windows | 79.1% | 54.0% | 21.3% | 2.33 |
| 6b | measured TP2 hidden, 297 windows (4.9× corpus) | 78.3% | 59.0% | 34.7% | 2.42 |

Each of those rounds moved the *offline* teacher-forced score the wrong way relative to live:
round 4 read d1 98.8% / d2 97.0% offline while live d1 fell ten points. Three losses on three
mechanistically different attempts is the tell that the idea is wrong, not its hyperparameter —
and four independent literatures name the same replacement. Sim-to-real budget allocation
(arXiv 2606.22062) says buy real samples rather than wider randomization; Draft-OPD
(arXiv 2605.29343) trains draft heads on the deployment distribution; structured-perturbation
work (TeKAP, ICLR 2025) says the perturbation's *structure* is the thing that matters; and
residual bootstrap is the century-old statistical form of all three — resample the residual you
measured, don't assume its shape.

**The fix is boring: capture the real hidden states and train on those.** A forward-only TP2 pass
(ring backend, no gradients — the safe kind of distributed run) over exactly the training windows,
hidden states cached to disk, and a `--real-hidden` flag that substitutes the cache where the
trainer would otherwise have run — and noised — a single-box forward. That is not even a
bootstrap; it is paired data. Round 6a (60 windows) pulled d1 back from 68.9% to 79.1% and still
lost overall. **Corpus size was the second lever, and a bigger one than expected**: 4.9× the
corpus (497 KB, 297 windows) took depth-3 acceptance from the baseline's 18.9% to 34.7%, and 11×
(1.16 MB, 697 windows of 384 tokens; 663 train / 34 eval) held the win with clearly diminishing
returns. Both ends of that are cheap: the corpus is the model's own greedy self-continuations from
seed topics, generated on one box (400 more of them in 79 minutes), and capture runs at 0.8 s per
window — the 297-window corpus in 4 minutes.

**A single-prompt live measurement will lie to you about which checkpoint won.** On one topic,
round 6b read 2.42 tok/cycle against the baseline's 2.44 — a loss, and nearly the end of the
line. Pooled over 8 topics with both arms run on the same prompts, 6b **won at +1.1%**. Nothing
changed but the sample size.

**And the regime you verify in has to be the regime you ship.** An audit of that promotion caught
the next fault: the 8-topic comparison had to set `OMLX_MTP_ROWWISE_BATCH=1` to force MTP on at
batch size 8 (the stack auto-disables it there, because plain batched decode is faster) — and
that override exists nowhere in the launcher, so production runs MTP at **batch size 1 only**.
The promotion evidence had been collected in a regime production never enters. Re-run at bs1 over
24 topics (16 deliberately outside the training corpus's CS-heavy seed topics), paired per topic:

| bs1 × 24 topics, pooled | d1 | d2 | d3 | tok/cycle | sign test |
|---|---|---|---|---|---|
| round 2 (baseline) | 78.1% | 59.5% | 24.7% | 2.371 | — |
| **round 6c (promoted)** | **79.3%** | **61.3%** | **34.3%** | **2.459 (+3.68%)** | 19W / 4L / 1T, **p = 0.0026** |

The depth-1 deficit seen at bs8 (74.1% vs 76.5%) *inverted* at bs1, and non-CS topics gained
**more** than CS topics (mean Δtok/cycle +0.106 vs +0.064) — which answers "did the training
corpus just contaminate the eval topics" with the opposite of the feared sign. **Paired
per-prompt, sign test, domain split** is the minimum bar; a single pooled ratio would have thrown
away round 6b and shipped round 6c on a regime nobody runs.

### 13a. The offline eval was scoring a distribution the server never produces

Through every round above, the offline teacher-forced depth-1 score sat between **0.985 and
0.990** while live depth-1 acceptance sat at **76–79%**. §12 attributes that kind of gap to
self-evaluation optimism and prescribes gating on the live number — correct, and not the whole
story. Underneath it were two stacked measurement faults, and only the first is fixable.

**Layer 1 — capture-regime mismatch.** The training data was captured as one 384-token prefill
with `cache=None`. Serving never does that: it bulk-prefills the prompt *through a cache object*
and then decodes new tokens one at a time on top of it. Three measurements, all in one process on
identical token windows, decompose the difference:

- **The cache object itself dominates.** Same model, same 320 tokens, cache-mediated vs
  `cache=None`: raw-hidden relative distance **0.742**. Same tokens, same code, different kernel
  dispatch — the rotating-KV and pooling-cache machinery changes the prefill hidden states, so the
  training data was unfaithful *in the prompt region too*, not only in the decode region.
- **Sequence length changes the prefill.** Comparing the first 320 positions of a 384-token
  prefill against a 320-token prefill (both `cache=None`): rel **0.232**, where pure causal
  attention would give 0 — the sparse indexer and pooling paths are whole-length dependent.
- **Self-comparison 0.000**, as the sanity control that keeps the two numbers above honest.

A serving-faithful probe (320-token bulk prefill through the cache, then 64 teacher-forced
single-token decode steps on that cache) put the prefill-vs-decode hidden distance at median rel
**0.819** — about **205×** the 0.4% TP2 drift the whole campaign was chasing. That number alone
over-alarms: the backbone's own `lm_head` still agreed on **91.7%** of argmaxes across the two
regimes and produced fluent text, because the head's norm absorbs most of the raw distance. The
number that mattered came from feeding both regimes into the *actual promoted draft head*:
depth-1 cross-agreement **78.6%**, teacher-forced accuracy **97.4% on prefill hidden vs 77.6% on
decode hidden (−19.8 pp)**, depth-2 −31.8 pp. And **77.6% ≈ the live depth-1 acceptance of
79.3%** — the five-round mystery explained by one measurement. The backbone is robust to the
regime change; the draft head, trained only on cache-free prefill states, is not.

Recapturing the corpus serving-faithfully restored the offline eval's usefulness immediately: the
*unchanged* baseline head, scored on cache-mediated hidden states, reads d1 **0.807** / d2
**0.570** — the first offline number in the campaign of the same order as live acceptance, against
**0.989 / 0.906** for the same head on the old capture.

**Layer 2 — fixing the capture still does not recover the gap.** Round 6e recaptured all 697
windows serving-faithfully and retrained with the loss restricted to the decode positions
(`--loss-start-pos 320`, attention context still full-sequence). Offline it did exactly what the
probe predicted: **d1 0.807 → 0.846, d2 0.570 → 0.744**. Live, bs1 × 24 paired: **tok/cycle 2.436
vs 6c's 2.459 — −0.91%, 8W / 13L / 3T, p = 0.38**, indistinguishable. The "up to 18 pp of
headroom" the probe appeared to promise did not exist.

**The transferable lesson: offline-eval sensitivity is not recoverable headroom.** The probe
measured how much the head's predictions move when the hidden-state regime changes. That is a
real, useful number — it correctly proved the training data unfaithful and correctly predicted
the offline gain. It says nothing about how much *live* acceptance is winnable, because the
remaining gap is not in the hidden states at all: it is in the live loop's conditional structure.
The positions the drafter is asked about are acceptance-dependent; the chain's depth-2/3 inputs
are its own depth-1 outputs, not teacher-forced tokens; the prompt-priming history differs.
Nothing captured from a dump reproduces that at any fidelity — closing it needs training *through
the live loop* (gradients co-resident with collective communication, the high-risk combination
this campaign crashed on repeatedly), not a better dump. Park it honestly instead of buying a
fourth round of the same idea.

**And the adapter still loses when it rides on correct data.** Round 6d is round 6c's exact recipe
plus the shared-expert LoRA (r=16, α=16), so the adapter is the only variable. Offline it landed
slightly *below* 6c (d2 0.961 vs 0.968); live (8-topic pooled) it fell back to baseline shape —
d1 76.6% / d2 56.4% / d3 24.9% / **tok/cycle 2.317**, against 6c's 2.346 and the baseline's 2.312.
The adapter erases the gain the measured-residual data bought. Two mechanics are worth keeping
even though that round closed the line for good:

- **A path-based sharder will slice your adapter.** The serving stack's in-place sharder walks
  parameter paths and splits everything under the module it is handed — `lora_a` (4096→2048) and
  `lora_b` (16→8) both got cut, a double corruption whose crash was first (wrongly) blamed on the
  model sharder. Fold the adapter into the base weight instead of attaching it, and it shards like
  any ordinary layer.
- **Merge in fp32, cast once at the end.** `W' = W + (α/r)·(A@B)ᵀ`, computed against the fp32
  dequantized base and cast to bf16 only after the addition — accumulating a small delta directly
  in bf16 rounds part of it away. Persist `α` and `r` in a sidecar written at checkpoint time
  rather than inferring them at merge time, and check the merged path against the attached path
  before trusting it (measured: rel 4.8e-3, ≈ 1 bf16 ULP). Wire the merge into *both* the bench
  harness and the server: a `load_weights(..., strict=False)` accepts a checkpoint full of
  `lora_*` keys and silently serves the base weights.

### 13b. Two distributed-capture pitfalls, cheap to avoid and expensive to debug

- **When a distributed script crashes, first check that both ranks see the same input.** A corpus
  manifest holding hard-coded `/Users/<box-a-user>/…` paths meant the second box's corpus builder
  hit a swallowed exception and returned **zero windows**; that rank died indexing `windows[0]`,
  its partner hung waiting for a forward that would never come, and the whole thing surfaced as a
  ring EPIPE cascade with no traceback. Three hours went into GPU-timeout, long-context,
  double-patching and resource-accumulation hypotheses first. Every single-shot isolation test had
  passed because they all called `os.path.expanduser("~/…")` directly and never went through the
  manifest. Make manifests `~`-relative and expand on read.
- **A resumable capture must broadcast its skip decision.** Skipping a window because its file
  already exists is a *collective* decision — the forward does send/recv/all-gather, so a rank
  that skips alone hangs its partner forever. Decide on rank 0, then `all_sum` a 0/1 flag to keep
  the ranks in lockstep. (Adjacent trap: `mx.save_safetensors` appends `.safetensors` when the
  path does not already end in it, so the obvious `x.safetensors.tmp` atomic-write temp name is
  written as `x.safetensors.tmp.safetensors` and the following `os.replace` raises
  `FileNotFoundError` — name the temp file so it ends in `.safetensors`.) With both in place, 697
  windows captured in 82.4 minutes, zero crashes, correct resume across an interruption.

**Oracles.**
- *Regime parity before trusting any offline eval.* Run the eval's capture path and the serving
  path over the same tokens, and compare what the *consumer* does with the result, not the tensors
  themselves. Here the raw-tensor distance (rel 0.82) over-alarmed by two orders of magnitude, the
  backbone's argmax agreement (91.7%) under-alarmed, and only the draft head's own cross-agreement
  (78.6%) predicted live behavior.
- *Promotion gate.* Paired on identical prompts, ≥ 20 prompts, sign test, a domain split that
  deliberately includes prompts unlike the training corpus — and an explicit check that the batch
  size and feature flags you measured under are ones the launcher actually sets.
- *Stop rule.* When the third mechanistically different attempt at the same idea fails, the idea
  is what is wrong. Three rounds of closing the gap without leaving the single box all lost; the
  first round trained on captured hidden states recovered most of it, and the second won.
- *Sensitivity ≠ headroom.* A probe showing that an offline metric is sensitive to some regime
  bounds nothing about live gain. Treat it as a reason to fix the measurement, and forecast the
  live win only from a live paired run.

Receipts: [examples/deepseek-v4-flash](../examples/deepseek-v4-flash/README.md) — capture scripts,
the trainer with `--real-hidden` / `--loss-start-pos` / LoRA merge, the paired analyzer, and the
verified command for every round above.

---

## 14. A convert that writes the wrong module tree cannot be rescued by bits or DWQ

**Why.** Serving stacks attach compiled / fused kernels to a *checkpoint graph*,
not to a bit-width. Two affine g64 files of the same GLM-5.3-Flash FP8 source,
same oMLX 0.6.3, same HTTP stream metric, decoded at **8.4 tok/s** (mlx-lm
tree remapped at load) and **28 tok/s** (native oQ/VLM file). Affine q4/q6/q8
sat on the slow shelf. Isolated `gather_qmm` 4 vs 6 bit is ~1.2×. Load-time
“fix the predicate” (dense routers, leftover KDA → 8-bit) bought 5.69 → 8.4
and stopped. Turning `fuse_in` / `compile_ffn` off on the *fast* file costs
only 2% / 6% / 7% together — both-off is still 26.4 tok/s. Remapped affine
already had those flags on. DWQ tunes scales; it cannot rewrite the file
layout the kernels attach to.

**Why it is worse than a slow baseline.** The slow file is internally consistent.
It loads, generates, and looks like a finished quant. The temptation is to DWQ
it, or to spend bits on KDA/experts hoping tok/s will follow. Both spend days
on the 8 tok/s shelf.

**Fix — name the serve artifact before the first convert.** The student is the
file the server will load. If the server’s fast path is a VLM tree with fused
members, emit that tree (or use the converter that already does). Treat an
mlx-lm affine export as a diagnostic dump, not a `baseline`. Measure decode
on the serve stream (`generation_tokens_per_second`), never a raw
`language_model` loop (0.42 tok/s on this model — both trees).

**The rule behind the rule:** *bits allocate quality and size; the checkpoint
graph decides which kernels exist.* Fused concat members must share
`(bits, group_size, mode)` or load/fuse fails. That is a contract, not a
quality recipe — mixed 6/8 KDA on a native oQ6e tree still decoded at 26–28
tok/s.

**Oracle:** same host, same server, two converts, one stream metric. If tok/s
matches across 4/6/8 on convert A and jumps 3× on convert B, you are looking
at the graph. A `gather_qmm` microbench on the expert shape tells you whether
bit-width can possibly be the leftover. Transfer test: writing the VLM tree
without the oQ wrapper recovered the 28 tok/s shelf (GLM-5.3-Flash VLM q4
29.46 / 28.24 / 27.50 vs oQ4e 29.26 / 27.94 / 27.30).

Full write-up + receipts: [docs/CHECKPOINT_GRAPH_NOT_BITS.md](CHECKPOINT_GRAPH_NOT_BITS.md),
[examples/glm-5.3-flash](../examples/glm-5.3-flash/README.md).
