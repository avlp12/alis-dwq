# Model card publishing checklist (Hugging Face)

Every item below cost a real mistake shipping the Motif-3-Beta and GLM-5.2 MLX
quants. HF's card renderer and badge logic have several **silent** failure modes:
the card previews clean and still ships wrong. Each item has the *why*, the *fix*,
and there is a grep/awk battery at the bottom to run before any card goes public.

## 1. The quantization badge shows the wrong bit-width on mixed-precision builds

**Why.** HF derives the "N-bit precision" sidebar badge from `config.json`
**top-level `quantization.bits`** (and `quantization_config.bits` if present) —
**not** from the effective bpw, and **not** from the per-path bit distribution.
`mlx_lm.convert` leaves that top-level `bits` at the `q_bits` default (often 4).
So a 2.3 bpw build whose *experts* (the dominant mass) are 2-bit badges as
**"4-bit"** — wrong, and wrong in the flattering direction on your headline number.

**Fix.** Set the top-level `bits` (in both `quantization` and `quantization_config`,
wherever each is present) to the **dominant-by-parameter-mass** value — `2` for a
2-bit-expert floor build. A C6 build (dominant 4-bit) and a Q8 build (8-bit) already
badge correctly; leave them alone.

**Safe only if every quantized tensor has its own per-path entry.** If any tensor
relies on the top-level default *at load time*, editing it changes the weights that
load, not just the badge. The oracle: the count of dict-valued entries in
`config["quantization"]` must equal the number of `.scales` tensors in the weight
index. If they are equal, the top-level default is never consulted → editing it is
**metadata-only** (badge only).

```python
import json
cfg = json.load(open("config.json"))
q = cfg.get("quantization", {})
per_path = sum(1 for v in q.values() if isinstance(v, dict))   # nested dicts = per-module specs
weight_map = json.load(open("model.safetensors.index.json"))["weight_map"]
n_scales = sum(1 for k in weight_map if k.endswith(".scales"))
print(f"per-path entries={per_path}  scales tensors={n_scales}  "
      f"{'SAFE: top-level bits is metadata-only' if per_path == n_scales else 'NOT SAFE: some tensor uses the default'}")
```

**Do not** touch the per-path entries — only the top-level scalar. Re-check the badge
on the live page after upload (the value is read at page-render time).

## 2. A bare `~` strikes through everything to the next `~`

**Why.** HF's GFM renderer pairs two single `~` into `~~strikethrough~~`. A card that
writes "≈96 GB decompressed … ≈11 GB" using `~` for "approximately" renders the whole
span between the two tildes struck through — and can break a `**bold**` run that falls
inside the pair, leaking literal asterisks. GitHub only strikes on double `~~`, so the
card previews fine on GitHub and corrupts on HF.

**Fix.** Use `≈` (U+2248) for "approximately", never a bare `~`. Escape any literal
tilde as `\~`. Beware `~/` home paths in prose — write the full path or escape it.
**Grep the card for `~` before publishing.**

## 3. Unbalanced or misplaced `**` leaks bold

**Why.** An odd number of `**` leaks bold across the rest of the document. Bold also
needs word boundaries to render: `**단어**` works, but `** 단어 **` (inner spaces) and
some `**word**직후CJK조사` (bold close immediately followed by a CJK particle) do not.
And `**` inside a table cell can collide with the `|` delimiters.

**Fix.** Keep `**` counts even (awk below). No inner spaces; put a space or punctuation
between a bold close and a following CJK particle. Use bold **sparingly** — a card
bolded everywhere reads as bolded nowhere, so reserve it for the numbers that matter.

## 4. Angle-bracket tokens get eaten as HTML

**Why.** Bare special tokens in prose — `<think>`, `<|...|>`, `<eos>` — are parsed as
HTML tags and **vanish** from the rendered card.

**Fix.** Wrap every special token in backticks (`` `<think>` ``) or keep it inside a
fenced code block. Same applies to any `<...>` you actually want the reader to see.

## 5. General render hygiene

- **No leftover placeholders** — `[pending]`, `[FORK_LINK]`, `[EVAL]`, `[TODO]`. Grep
  for `[` followed by a capital letter and eyeball the hits.
- **Balanced code fences** — an even count of ``` per file; an odd count swallows the
  rest of the card into a code block from the stray fence down.
- **Images are valid PNGs referenced by relative path** — `assets/…`; a broken or
  absolute path renders as a broken-image icon on the Hub. Open each PNG to confirm it
  is a real PNG, not a truncated download.
- **Frontmatter modality correct** — `pipeline_tag: text-generation` and
  `library_name: mlx` for an MLX repo. `library_name: mlx` is also what enables HF's
  **download tracking**, so getting it wrong silently zeroes the download counter.
- **No foreign-model artifacts** — a card started from another release's template can
  carry `GLM`/`Hy3`/`nvfp4` strings, the wrong param count, or another model's badge.
  Grep for the other family's name before publishing.

## Pre-publish grep/awk battery

Cheap insurance — run all of these on the card file before it goes public:

```bash
CARD=README.md
grep -n '~'            "$CARD"   # (2) bare tildes → use ≈  (ignore any ~~ you meant)
grep -nE '<[a-z|/]'    "$CARD"   # (4) angle-bracket tokens (eyeball; code-fence hits are fine)
grep -nE '\[(pending|TODO|EVAL|FORK_LINK)' "$CARD"   # (5) placeholders
awk '{n+=gsub(/\*\*/,"")} END{print "** count:", n, (n%2? "ODD — bold leaks":"ok")}' "$CARD"   # (3)
awk '/^```/{f++} END{print "fence count:", f, (f%2? "ODD — unclosed fence":"ok")}' "$CARD"      # (5)
grep -niE 'glm|hy3|nvfp4|deepseek' "$CARD"   # (5) template leftovers — expect only intended mentions
```

And the one metadata check the greps can't do — the badge (item 1):

```bash
python - <<'PY'
import json
cfg = json.load(open("config.json")); q = cfg.get("quantization", {})
pp = sum(1 for v in q.values() if isinstance(v, dict))
wm = json.load(open("model.safetensors.index.json"))["weight_map"]
ns = sum(1 for k in wm if k.endswith(".scales"))
print("top-level bits:", q.get("bits"), "| per-path:", pp, "| scales:", ns,
      "|", "editable (metadata-only)" if pp == ns else "NOT editable — the default is load-bearing")
PY
```

A card that passes this battery has cleared the failure modes that previewed clean and
still shipped broken. It does not check prose — read the thing.
