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

**Why.** Firing N independent calls and dividing wall clock by N avoids the ~250 µs
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
workload speculation has, and quoting it as *the* result overstates by ~30% here.

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
conditions (average **4.48** vs the card's 3.39).

**Fix — a checkpoint's module list is a contract; check the loop against it.** Enumerate
the top-level modules in the checkpoint and grep the inference loop for each one. Any
module that is loaded but never called is either dead weight or a missing feature, and the
difference matters (see the battery). This is a ten-second check that would have saved the
campaign a day.

**Two corollaries measured here:**

- **Winning the drafter metric is not winning.** DSpark's acceptance (4.23) beats MTP's
  decisively and still loses end-to-end — 1.20× vs **1.32×** — because a separate 1.36B
  forward pass plus a 248k-vocabulary `lm_head` are charged every step, neither of which an
  MTP layer reusing the target's hidden state pays. **Judge a speculative scheme on tok/s,
  never on acceptance length.**
- **Reproducing a paper's configuration is not the same as reproducing its result.** Two
  card-specified conditions measured *worse* for us: a bf16 drafter bought nothing over the
  4-bit one (acceptance 4.19 vs 4.23), and the reference's draft-slice convention
  (`[:, -B+1:]`) reached only 2.35 against our shifted read's 4.23. And the
  `confidence_head` measured **best switched off** (block 6: off 45.0 tok/s vs
  tau 0.10/0.25/0.50 = 37.1/38.5/42.8) — its whole purpose is skipping a wide block when the
  drafter is unsure, and item 5's kernel work removed the cost it was avoiding.

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

**D. Benchmark protocol** — before quoting any decode-path number:

- kernel numbers reported from a **dependent chain**, not a queue batch (item 4);
- speculative-decoding numbers averaged over **≥3 dissimilar prompts** (item 6);
- decode timer started **after the first token**, so prefill is not folded into decode.

A build that clears A–D has cleared the failure modes that load cleanly, pass every
structural check, and still ship the wrong thing. It does not check quality — measure that
separately.
