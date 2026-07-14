# Head-to-head: ds4-recipe GGUF vs Alis-MLX — GLM-5.2 at the ~200 GB floor

Two independent stacks now ship GLM-5.2 in the same size class with opposite
philosophies. This doc pins down what is comparable on paper today and the
exact protocol for the on-device quality head-to-head (not yet run).

## The contenders

| | [antirez/glm-5.2-gguf](https://huggingface.co/antirez/glm-5.2-gguf) (q2) | Alis E1 (this repo, [case study](../glm-5.2-e1-floor-spike/README.md)) |
|---|---|---|
| Size / eff. bpw | **197 GB ≈ 2.1 bpw** (from the [pulsar](https://github.com/giannisanni/pulsar) post) | 215.8 GB / **2.3225 bpw measured** |
| Container | GGUF (llama.cpp / ds4 / pulsar) | MLX safetensors (stock mlx-lm) |
| Routed experts | ~2-bit **non-uniform**: IQ-class codebook up/gate + Q2_K down | 2-bit/gs128 **uniform affine**, clip-searched grids |
| Everything else | Q8 (attention, router, shared experts, embeddings), MTP Q2_K | sensitivity-graded mixed precision (dynamic recipe) |
| Error compensation | **imatrix** (activation-weighted rounding, calibration-time only) | **DWQ** (distillation-trained scales/biases, 45% ZH mix, 4.5 bpw teacher) |
| Published quality | none comparable — ds4 gates on a private 100-case behavioral fixture | strided wikitext PPL 4.711 raw → **4.036** post-DWQ; EN/code/ZH KLD harness |
| Verification | teacher-forced argmax agreement vs ds4 reference engine | eval_kld / flips vs 4.5 bpw reference + held-out PPL gates |

Structural read: the ds4 recipe spends its byte budget exactly where the
4-bitter Lesson and our measurements say (decision-making tensors high, tail
cheap), and its IQ-class expert format is non-uniform — the codebook-family
advantage MLX's affine container lacks. Our side answers with clip-searched
grids + distillation recovery. Whether 2.1 bpw codebook-no-training beats
2.32 bpw affine-with-DWQ is precisely the open question; neither camp's
published numbers answer it.

## Protocol (on-device, ~1 evening)

Both models share the GLM-5.2 tokenizer, so token-level metrics are directly
comparable **if the windowing matches**. The trap is methodology skew:
llama.cpp's `llama-perplexity` uses full non-overlapping context windows by
default, our strided PPL does not.

1. **Corpus**: one fixed file per slice — the same `data/wikitext.txt`,
   `code.txt`, `zh.txt` this repo evaluates with. Hash them; both harnesses
   read identical bytes. (Calibration disjointness note in §1 applies: the
   ds4 imatrix corpus is unknown, so report EN *and* non-EN slices — an
   imatrix built on English-heavy data should show the gap on ZH.)
2. **PPL, matched windows**: run llama.cpp `llama-perplexity -c 512` (window
   512, no overlap) on each slice file for the GGUF; run the same 512-window
   non-overlapping PPL on the MLX side (a ~10-line variant of `eval_kld`'s
   chunked forward — do NOT compare against our *strided* numbers, recompute
   both sides under one windowing).
3. **Token agreement vs a common reference**: teacher-force both quants along
   the same reference continuation (the Q8/8-bit MLX teacher's greedy path on
   the three slice prompts) and report per-position argmax agreement — the
   cross-engine analogue of our flip metric, and the method ds4 itself
   certifies with (15/16, 10/12 on its own builds).
4. **Degeneration probe**: `--loop-probe` equivalent on both (greedy 256 from
   the same three prompts; distinct-4gram + cycle detection). Cheap, and the
   REAP lesson says aggregate parity can hide exactly this.
5. **Report per-slice, never aggregate-only** (house rule), plus GB and
   tok/s on the serving box for the quality-per-byte-per-speed picture.

Fairness notes: E1's DWQ used a 4.5 bpw teacher and ZH-heavy calibration —
if the ds4 imatrix corpus turns out to be EN-only, slice-level results should
be read as recipe-vs-recipe, not format-vs-format. And size is not equal
(197 vs 215.8 GB): if ds4-q2 wins EN PPL at −19 GB, the affine container's
practical floor claim needs revising; if E1 wins ZH by the usual margin, the
DWQ-recovery story extends to cross-stack comparisons.

## Status

- [ ] Protocol run pending (needs the llama.cpp/pulsar box + an M3 Ultra).
- Paper comparison above compiled 2026-07-14; ds4 q2 composition per the
  DwarfStar docs (IQ-class up/gate + Q2_K down; dense/control Q8) — verify
  against the actual GGUF header before publishing numbers.
