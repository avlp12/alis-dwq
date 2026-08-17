# Case study: MiniMax-M3 (427B VL MoE) — multimodality-preserving mixed-precision, and the DWQ that hasn't run yet

The other case in this repo where **DWQ was not applied** (cf. [Kimi-K2.7](../kimi-k2.7/README.md),
which *can't* be DWQ'd). MiniMax-M3 *can* — it ships bf16, so a higher-precision teacher
exists — but it was shipped as a straight sensitivity-graded mixed-precision quant first.
This write-up is here for three things that belong in a DWQ playbook regardless: the
**VL structural lever** (unpacking the always-on shared expert, a pre-DWQ fix that DWQ can't
substitute for), the **8-bit-reference evaluation pattern** used when the bf16 teacher won't
co-load, and the **quantified DWQ opportunity** the mixed-precision ship left on the table.

Shipped: [`avlp12/MiniMax-M3-Alis-MLX-Dynamic`](https://huggingface.co/avlp12/MiniMax-M3-Alis-MLX-Dynamic)
— the **first MLX quant of M3 that keeps the full vision-language model** (existing MLX quants
are text-only extractions). `main` = T256, branches `t512` / `t512ref`.

## The model

`MiniMaxAI/MiniMax-M3` — `model_type: minimax_m3_vl`, **427.04B** total / ~23B active
(exact count summed from the 59 bf16 source shards: LM 426.18B + vision tower 0.63B +
patch-merge 0.19B + projector 0.05B). 60 layers (0–2 dense, 3–59 MoE), 128 routed + 1 shared
expert, MiniMax Sparse Attention (MSA, top-16 of 128-token blocks) at 1M context, and a
32-layer ViT vision tower. No MTP head (checked against the weight index).

Param mass is **97.6% routed experts** — so, as with every big MoE here, expert bit-width
sets the size, and everything else is nearly free to protect.

## Builds

| branch | routed experts | shared | attn/dense | embed / head | vision | size | bpw |
|---|---|---|---|---|---|---|---|
| `main` (T256) | 3-bit g64 | 8-bit | 6-bit | 6b / 8b | bf16 | 194.6 GB | 3.646 |
| `t512` | 6-bit g64 | 8-bit | 8-bit | 8b / 8b | bf16 | 351.6 GB | 6.587 |
| `t512ref` | 8-bit g64 | 8-bit | 8-bit | 8b / 8b | bf16 | 454.9 GB | 8.522 |

Never quantized, every build: **router gate + `e_score_correction_bias` = fp32** (verified
in the source, not assumed); **MSA `index_q_proj`/`index_k_proj` = bf16** (they select the
top-16 attention blocks — a flipped pick reads different history, so the selector stays
exact); **vision tower + projector + patch-merge = bf16** (~1.4 GB; multimodality is the
whole point).

## Lever 1 — unpack the always-on shared expert (a pre-DWQ structural fix)

mlx-vlm packs M3's shared expert as expert #128 inside the same `SwitchLinear` as the 128
routed experts whenever `shared_intermediate_size == intermediate_size` (both 3072 here).
That welds the one tensor that sees **100% of tokens** (a routed expert sees ~3% at
top-4/128) to the routed experts' bit-width — so in the T256 build it would ride at 3-bit,
the single worst quality decision available.

The unpacked layout (separate `shared_experts` module) already exists in the code; it just
wasn't reachable by config. A three-file, backward-compatible switch
([Blaizzy/mlx-vlm#1544](https://github.com/Blaizzy/mlx-vlm/pull/1544)) adds
`text_config.pack_shared_expert`; setting it `false` frees the shared expert to carry 8-bit
while routed experts drop to 3-bit, at **+0.4% size**. Packed vs unpacked forward outputs
match to **7.75e-07** (max |logit diff|) on a shrunken-config model routing identical
HF-style weights through both `sanitize` paths.

**Why this matters for DWQ, not just RTN:** a packed shared expert can't be given its own
learning treatment either — DWQ would tune one welded tensor's scales for a 100%-coverage
path and a 3%-coverage path together. Unpack first; *then* DWQ (or just ship, as here).

## Lever 2 — evaluate against an 8-bit reference when bf16 won't co-load

The bf16 teacher is 869 GB — it doesn't fit beside a student on a 512 GB box, and (unlike
the GLM/Inkling cases) we didn't run a distributed teacher dump for a model we weren't
DWQ-ing. Same fix as every ship here: build an **8-bit anchor** (`t512ref`, 454.9 GB, loads
alone) and KL every other build against it. MSA only engages past ~2.2K tokens, so the eval
has to include a long-context arm or it never exercises the sparse path.

Measured vs `t512ref` (deterministic EN/KO/code/econ contexts; KL on the ref's top-256
support; 16K arm on a document that fully engages MSA; needle-in-a-haystack at 3 depths):

| metric | T512 | T256 (`main`) |
|---|---|---|
| KL vs REF, short (512 tok) | **0.0184** | 0.1243 |
| top-1 agreement, short | 97.4% | 90.4% |
| KL vs REF @16K | 0.0087 | **0.0011** |
| top-1 @16K | 99.2% | 100.0% |
| NIAH @16K (3 depths) | 3/3 | 3/3 |
| sparse block-selection overlap vs REF @16K | 54% | 48% |
| decode tok/s (short / 2.4K ctx) | 22.5 / 17.7 | 28.1 / 20.6 |
| PPL (eval slice) | 2.900 | 2.994 |

REF itself: PPL 2.851, NIAH 3/3, decode 20.6 tok/s.

Two reading notes that generalize:
- **Block-selection overlap reads low (48–54%) and nearly build-independent** despite T512
  being 2.9 bpw richer. With 16K KL at 0.001–0.009 and NIAH 3/3, this says the flips are
  among near-tied blocks on repetitive text — *selection* differs, *retrieval* doesn't
  degrade. Don't gate on the discrete-selection overlap; gate on KL + retrieval.
- **A thinking model needs `max_tokens` budgeted for the thinking.** NIAH first scored 0/3
  at `max_tokens=30` because the model spent the whole budget inside `<mm:think>`; at 400
  tokens it was 3/3 at every depth. Looks exactly like model damage; it's a harness artifact.

## The DWQ opportunity left on the table

T256's **short-context KL of 0.1243 misses the 0.10 ship gate** — and it's not diffuse: the
code context alone is 0.217 while prose contexts run 0.05–0.12. That is precisely the shape
DWQ is for (the GLM 2.56 case recovered a comparable overall-KL gap by −42%). The pieces are
already in place: bf16 teacher exists → dump targets (distributed, §2b, the tower is
forward-only), unpack is done so the shared expert can be tuned independently, and the
eval already isolates the offending slice. **Not yet run** — the mixed-precision build
shipped first; this is the documented next step, not a claimed result.

## Operational scars (full text in the top-level notes)

- **Xet writer-death stall:** after a USB enclosure dropped and re-enumerated mid-download,
  `hf download` sat alive-but-dead (0% CPU, zero writes) — the background writer thread had
  died and the main thread waits forever. Reproduced twice in one day, once at 57/59 shards.
  Watchdog: process alive **and** no write in the target tree for 10 min → kill + relaunch
  (resume keeps completed files).
- **HF branch shard inheritance:** creating `t512`/`t512ref` and uploading each build left
  *both* builds' shards on every branch (39 stale + 87 real). `delete_files` the inherited
  names per branch after upload; verify shard counts per-revision via the API.
- **Sidebar param count:** `mlx_vlm.convert` doesn't write `total_parameters` into
  `model.safetensors.index.json` metadata (`mlx_lm.convert` does), so the Hub counted packed
  U32 elements and showed **427B as "56B"**. Inject the exact count before upload.
- **Don't write builds to the same ExFAT USB drive you're reading the source from** — the
  drive dropped twice under sustained write load (survived sustained *read* fine). Build on
  internal SSD, upload, delete.
