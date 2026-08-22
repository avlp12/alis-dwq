# Graded bit allocation without gradients

*A model-agnostic recipe for deciding which tensors get more bits, when the usual
gradient-based sensitivity signal is unavailable. Written from the Qwen3.8-27B campaign
(2026-08-22), where it improved three published builds at 0.04–1.6% of file size.*

## The problem

Uniform quantization spends the same bits on a tensor that is 0.33% of the model and on one
that is 18% of it. If some small tensors are disproportionately damaging to quantize, promoting
them is nearly free. The question is which ones — and the honest answer is that the popular
proxies do not tell you.

## What does not work: activation energy (imatrix)

llama.cpp's `imatrix` — and the allocation tables that Unsloth and others derive from it — stores
per-tensor `in_sum2`: the sum of squared activations per input channel, over a calibration corpus.
It is cheap, it is public for many models, and it is the obvious thing to reach for.

We built a knapsack over it and shipped the result into an MLX affine container. **It improved its
own objective by 60.7% and made the model measurably worse** — full-vocab KL to bf16 rose 16–37%
against plain uniform 4-bit. A relative-error variant (error divided by signal) did better, +2–3.6%,
still worse than uniform.

The reason is structural, not a tuning failure:

- Sensitivity of tensor *t* is `S_t = E‖G_t · Δ_t x‖²` — the input factor times the **downstream
  gain** `G_t`, i.e. how much the rest of the network amplifies an error introduced there.
- imatrix gives you only the input factor.
- **RMSNorm erases absolute scale.** A tensor whose activations are ten times larger feeds a norm
  that divides the ten back out. Activation energy therefore carries almost no information about
  downstream influence in a normalized transformer.

The tell was concrete: the relative-error criterion wanted to **demote `lm_head` to 3-bit**, while
direct measurement showed promoting `lm_head` was worth two-thirds of the whole gain. `lm_head` is
the one matrix with no downstream normalization — its error lands on the logits unabsorbed. Any
criterion that ranks it low is measuring the wrong thing.

**What does transfer from a competitor's imatrix table is the *list*: which tensors they chose to
protect.** That list is mostly arithmetic — `k_proj` and `v_proj` are tiny under GQA, gate
projections are tiny by construction — and you can derive it yourself from shapes. The algorithm
does not transfer; the shopping list does.

## What works: measured damage

When gradients are available, use them. Ours were not: the model's GatedDeltaNet layers have no
VJP in this framework, so every gradient and DWQ path died in `nan` or OOM.

The forward-only substitute is direct and embarrassingly simple:

> Quantize **one tensor group at a time**, leave everything else in bf16, and measure the KL to the
> unquantized model. That number *is* the group's damage — no proxy, no assumption about what
> amplifies it.

Group tensors by (type × depth bucket) so the sweep stays affordable — we used 106 groups for 506
quantizable tensors — and divide each group's damage by its byte cost to rank by value per byte.

```bash
# one group per step, rest bf16, exact full-vocab KL on NWIN fixed windows per slice
NWIN=3 FORK=~/your/mlx-lm python probe_damage.py damage.json
```

(`probe_damage.py` in this repo's `examples/qwen3.8-27b/` reads its model from `SRC` and its
windows from `CORPUS` near the top of the file — set those two for your model. `BASE=4` switches
from isolated damage to **marginal** value against a 4-bit floor; `PROMO_GS=32` measures
group-size promotion instead of width promotion, which is the only lever left at 8-bit.)

It costs one forward pass per group over a small window set. For a 27B model at 106 groups this
was under an hour, which is cheaper than one bad build.

## Three traps in the measurement

**1. Isolated damage is strongly sub-additive — you cannot rank and sum.** Our per-group damages
summed to 0.18974; the actual all-4-bit build measured 0.09052. The sum overstates by **2.1×**.
Use the ranking to *choose*, then measure the chosen build. Do not predict the build's KL by adding.

**2. A byte-matched control, or the finding is unfalsifiable.** Promoting good tensors also spends
bytes, and spending bytes helps by itself. We built an arm that spent the *identical* byte budget
on arbitrary mid-depth FFN tensors: it bought **−1.2%**, against **−17.5%** for the chosen set, at
budgets matched to within 0.02%. Without that arm the claim is "we made it bigger and it got
better."

**3. Verify the predicate actually fired.** Our first `--bit-map` run matched 9 of 506 tensors
because the map's keys were full paths and the predicate saw module suffixes. It completed, it
produced a build, and the build was silently uniform — we read an hour of AWQ as evidence that
promotion "does not compose with AWQ." Read the shipped weights back and assert the shapes:

```bash
python - <<'PY'
import json, mlx.core as mx
d = "<build>"
idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
for pat in ("self_attn.k_proj.weight", "self_attn.q_proj.weight"):
    k = next(x for x in idx if pat in x and "mtp" not in x)
    print(k, mx.load(d + "/" + idx[k])[k].shape)
PY
```

MLX packs affine weights into `uint32`, so the **last dimension is the receipt**: at
`in_features = 5120`, 8-bit g64 gives 1280, 6-bit gives 960, 4-bit gives 640. Our shipped
6-bit build prints `k_proj (1024, 1280)` — promoted — next to `q_proj (12288, 960)` —
baseline. If the promoted tensor prints the baseline width, the predicate never fired.

Index the map by every path suffix so predicate and map cannot disagree:

```python
for k, v in list(raw.items()):
    parts = k.split(".")
    for i in range(len(parts)):
        bit_map.setdefault(".".join(parts[i:]), v)
```

## The allocation this produced

Ranked by damage per byte, the same five groups came out on top at every precision:

| tensor | share of parameters | why it is cheap to protect |
|---|---|---|
| `self_attn.k_proj`, `self_attn.v_proj` | 0.33% each | GQA shrinks K/V by the head-group ratio |
| `linear_attn.in_proj_a`, `in_proj_b` | 0.04% each | gate projections are rank-narrow by construction |
| `lm_head` | 4.7% | no downstream normalization to absorb its error |

At 4- and 6-bit these move to a wider integer width. At 8-bit there is no wider width in MLX's
affine set, and the remaining lever is **group size** — halving 64 → 32 doubles scale density for
+0.5 bits on the tensors it touches.

## Results, and where the method stops paying

Paired full-vocab KL to bf16, same windows, against each build's own previous published revision:

| build | size cost | English | Korean | code | pooled |
|---|---|---|---|---|---|
| 4-bit (AWQ + graded) | +1.6% | −16.6% | −16.4% | −3.9% | −8.7% |
| 6-bit (graded) | +1.6% | −23.9% | −27.3% | −14.3% | **−18.9%** |
| 8-bit (group size) | +0.04% | −6.0% | −2.7% | +0.1% | −2.3% *(t = −1.6, n.s.)* |

Decode cost −2.3% / −1.2% / −0.2% respectively.

**The gain is largest in the middle and vanishes at the top.** At 8-bit the pooled effect does not
clear significance — the same conclusion we reached when that tier tied every other reasonable
8-bit recipe from three different authors. Bit allocation stops being the dominant error term
somewhere between 6 and 8 bits, and past that point a graded build is worth shipping only because
it costs nothing, not because it measurably helps.

## Baseline hygiene

We first reported the 6-bit gain as "English not significant (t = −1.4)". It was **t = −10.1**. The
comparison had run against a local build whose name matched a variant the model card mentioned but
which was never the published artifact. **Diff against the bytes you actually published**, by
hash, not by filename — and prefer downloading your own release over trusting a local directory
that shares its name.

One more of the same species: our paired-KL harness treats its **first argument as the reference
for all the others**, so passing four builds on one line silently produced comparisons against the
wrong base. Check the output labels before reading the numbers.
