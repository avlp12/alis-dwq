# Laguna S 2.1: MLX + ALIS/DWQ reproducible recipe

This example records the measured procedure used for `poolside/Laguna-S-2.1`.
It deliberately separates a stock Q4 baseline, a compact mixed-precision arm,
and a highest-quality quantized arm. A checkpoint is called **ALIS/DWQ** only
after layerwise training has completed and the held-out gates pass.

## Pinned provenance

| Role | Repository | Revision |
|---|---|---|
| BF16 source | `poolside/Laguna-S-2.1` | `88796b991a17fc691abf1c1ad0d9f459dae73834` |
| MLX runtime | `ml-explore/mlx-lm` | `cf10f962b7a20e63a6df43dbf0faf06070153d40` |
| ALIS/DWQ base | `avlp12/alis-dwq` | `e68c8f708032bfc751d4393b3544c600572e0c16` |

Keep the source OpenMDW-1.1 license, origin notice, and Poolside acceptable-use
notice with every derivative. Use separate, no-clobber directories for source,
baseline, raw, clipped, DWQ work, accepted, and remote-reproduction artifacts.

## Safety boundaries

- Laguna's router, correction bias, norms, and other biases must remain floating.
  Do not use mlx-lm's generic `mixed_3_4` named policy: a named policy replaces
  the model predicate and can quantize the router. Use the checked converter in
  this directory or another predicate with the same exclusions.
- `clip_quantize --source` must point to an **unquantized MLX-layout** checkpoint.
  The original Hugging Face checkpoint has individual expert keys, while the
  MLX runtime uses stacked `switch_mlp` keys.
- A lazy load only defers evaluation. It is not evidence that the model fits in
  memory. Run one large MLX process at a time and retain the memory JSONL.
- Default memory gates use 90% of MLX's recommended wired working set and 16 GiB
  of additional swap. A tripped training-round gate restores that round's
  snapshot before raising.
- `RotatingKVCache` does not support the optional quantized-KV probe. This does
  not affect the ordinary Laguna KLD, generation, or layerwise DWQ path.

## 1. Verify and convert the source

The pinned source contains 46 BF16 shards and 36,769 keys. The converter first
streams and verifies every shard against the pinned LFS SHA-256 manifest,
checks the exact index key/shard set and revision metadata, and refuses drift
before importing MLX. It preserves `LICENSE.md` and the byte-identical upstream
model card as `SOURCE_README.md`, while the derivative card retains the origin,
OpenMDW-1.1, Poolside AUP, and safety-guardrail notice. Laguna sanitization
stacks 36,096 individual expert tensors into 141 projection tensors, producing
814 MLX tensors. Verify every source shard and key before conversion.

```bash
python examples/laguna-s-2.1/convert.py \
  --source /path/to/pinned-hf-source \
  --out /build/bf16-mlx-layout \
  --recipe bf16-mlx-layout

python examples/laguna-s-2.1/convert.py \
  --source /path/to/pinned-hf-source \
  --out /build/baseline-q4-g64 \
  --recipe baseline-q4-g64

python examples/laguna-s-2.1/convert.py \
  --source /path/to/pinned-hf-source \
  --out /build/compact-raw \
  --recipe quality-3p7

python examples/laguna-s-2.1/convert.py \
  --source /path/to/pinned-hf-source \
  --out /build/highest-quality-raw \
  --recipe highest-quality-q4
```

These are the release converter commands, not illustrative aliases. The
converter writes `mlx_lm_base_revision`, the full recipe configuration,
promoted routed-module list, pinned source shard-manifest root, and source-file
verification evidence into `conversion_plan.json`. Every `--out` is exclusive;
rerunning any arm at the same path fails instead of replacing it.

After the unpromoted compact arm has a complete 48-round DWQ stderr log, rank
the two bounded routed-module promotion attempts, then build them into separate
no-clobber directories:

```bash
python ../../scripts/rank_laguna_promotions.py \
  --source /build/bf16-mlx-layout \
  --student /build/compact-dwq-work-1 \
  --dwq-log /logs/compact-dwq.stderr.log \
  --first-count 8 --second-count 24 \
  --output /logs/compact-promotion-plan.json

python examples/laguna-s-2.1/convert.py \
  --source /path/to/pinned-hf-source \
  --out /build/compact-promotion-1-raw \
  --recipe quality-3p7 \
  $(jq -r '.attempts[0].convert_arguments | join(" ")' \
    /logs/compact-promotion-plan.json)

python examples/laguna-s-2.1/convert.py \
  --source /path/to/pinned-hf-source \
  --out /build/compact-promotion-2-raw \
  --recipe quality-3p7 \
  $(jq -r '.attempts[1].convert_arguments | join(" ")' \
    /logs/compact-promotion-plan.json)
```

`rank_laguna_promotions.py` constrains every generated value to
`LAYER:(gate_proj|up_proj|down_proj|*)`, so the whitespace expansion above
contains no free-form paths or text. Promotion is valid only for
`quality-3p7`; the converter rejects promotion flags for all other recipes.
The ranking student is the exact, immutable unpromoted checkpoint produced by
that 48-round DWQ run, not `/build/compact-raw`: the reconstruction scores and
the per-layer DWQ improvements must describe the same weights. Keep the
student checkpoint and stderr log in place for the receipt builder to verify
their recorded paths and hashes.

The `quality-3p7` policy uses routed experts at 3-bit/group-128, attention, dense and
shared experts at 4-bit/group-64, and embedding/LM head at 6-bit/group-64. The
`highest-quality-q4` policy keeps embedding and LM head BF16 and quantizes the other
eligible matrices at 4-bit/group-64. Both keep control paths floating.

Do not infer bpw from a recipe name. Compute it from the finished indexed tensor
bytes and the BF16 source parameter count.

## 2. Structural preflight

The default preflight is report-only. Explicitly enable the gates required for
this model and retain its JSON stdout:

```bash
set -o noclobber

PYTHONPATH=. python -m alis_dwq.preflight \
  --model /build/compact-raw \
  --expect-layers 48 \
  --expect-moe-layers 47 \
  --expect-experts 256 \
  --expect-full-caches 12 \
  --expect-rotating-caches 36 \
  --require-float-routers \
  --require-quantized-layer-coverage \
  > /logs/compact-preflight.json
```

The preflight checks structure and method/attribute contracts. It does not prove
forward correctness, Metal memory safety, generation quality, or long-context
cache behavior.

## 3. Calibration and teacher targets

Build byte-disjoint `train.jsonl`, `valid.jsonl`, and held-out JSONL files using
Laguna's chat template with thinking enabled. The measured run used seed 7 and:

| Split | Total | Code | English reasoning/tool | Korean | Chinese |
|---|---:|---:|---:|---:|---:|
| train | 80 | 36 | 28 | 8 | 8 |
| validation | 40 | 18 | 14 | 4 | 4 |
| held-out | 100 | 45 | 35 | 10 | 10 |

Store dataset revisions, source rows, raw-text SHA-256, and final JSONL hashes.
No held-out byte may be used for clipping, training, rollback, early stopping,
or promotion selection.

Dump teacher targets in a separate process from the pinned BF16 MLX-layout
checkpoint. Setting `ALIS_DWQ_NUM_VALID_SAMPLES=0` consumes all validation rows.

```bash
ALIS_DWQ_DATA_DIR=/build/laguna-data-v4 \
ALIS_DWQ_NUM_VALID_SAMPLES=0 \
ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat \
ALIS_DWQ_TEACHER_IDENTITY=poolside/Laguna-S-2.1 \
ALIS_DWQ_TEACHER_REVISION=88796b991a17fc691abf1c1ad0d9f459dae73834 \
ALIS_DWQ_MAX_PEAK_FRACTION=0.90 \
ALIS_DWQ_MAX_SWAP_INCREASE_GIB=16 \
ALIS_DWQ_MEMORY_EVIDENCE_PATH=/logs/targets-v3-memory.jsonl \
ALIS_DWQ_RUN_EVIDENCE_PATH=/logs/targets-v3-run.jsonl \
python -m alis_dwq.run \
  --model /build/bf16-mlx-layout \
  --targets-only \
  --target-dir /build/teacher-targets-v3 \
  --num-samples 80 \
  --max-seq-length 512 \
  --batch-size 1 \
  --seed 7
```

Before unloading the teacher, validate the numeric tensors and retain the
no-clobber manifest:

```bash
python ../../scripts/verify_dwq_targets.py /build/teacher-targets-v3 \
  --expected-train 80 --expected-valid 40 \
  --max-sequence-length 512 --vocab-size 100352 --sha256 \
  --output /logs/teacher-targets-v3-manifest.json
```

That validator checks the expected shapes, finite and nonzero values,
in-vocabulary per-token unique top-k IDs, and all numeric target hashes. Separately,
`/build/teacher-targets-v3/target-contract.json` binds the tokenizer, sample order,
seed, batch size, maximum length, exact pad-to-32 target sequence shape, raw
data, teacher checkpoint, and target checksums. The launcher verifies that
identity/order contract before loading any model; it does not replace the
numeric-tensor validator.

For the already-complete historical `teacher-targets-v2`, create the same
contract deterministically without replacing any numeric file:

```bash
python -m alis_dwq.target_contract \
  --data-dir /build/laguna-data-v4 \
  --tokenizer /build/bf16-mlx-layout \
  --teacher-checkpoint /build/bf16-mlx-layout \
  --target-dir /build/teacher-targets-v2 \
  --teacher-identity poolside/Laguna-S-2.1 \
  --teacher-revision 88796b991a17fc691abf1c1ad0d9f459dae73834 \
  --num-samples 80 --num-valid-samples 40 \
  --max-seq-length 512 --batch-size 1 --top-k 1024 --seed 7 \
  --tokenization preformatted_chat
```

This helper intentionally rejects the old Laguna data build that lacks persisted
token-ID hashes. Rebuild the calibration directory with the current data builder
first; the unchanged selected text/order can then be bound to the existing
numeric targets without recomputing the teacher forward pass.

## 4. Anchor-guarded clip search

Run clipping before DWQ. The unclipped lattice remains a candidate for every
group, and `--max-err-slack 1.1` bounds worst-case anchor damage. Do not enable
FFN permutation in this release arm.

```bash
python -m alis_dwq.clip_quantize \
  --source /build/bf16-mlx-layout \
  --model /build/compact-raw \
  --out /build/compact-clip-s11 \
  --max-err-slack 1.1 \
  --require-no-skips
```

Repeat with the highest-quality raw checkpoint as the student. Never run clip
search on an already-DWQ checkpoint: it would recompute the packed codes and
discard the learned scales and biases. The clip transaction verifies that the
BF16 and student conversion plans share the same pinned source lineage, records
both canonical input-directory digests and plan hashes, and recomputes them
before publication so an input cannot change during the long lazy-mmap pass.

Use the same clip and DWQ blocks for every release candidate, substituting only
the exact paths in this table. Each path is new and no-clobber:

| Arm | Raw student | Clipped student | DWQ output | Memory JSONL | Run JSONL | DWQ stderr |
|---|---|---|---|---|---|---|
| Compact | `/build/compact-raw` | `/build/compact-clip-s11` | `/build/compact-dwq-work-1` | `/logs/compact-dwq-memory.jsonl` | `/logs/compact-dwq-run.jsonl` | `/logs/compact-dwq.stderr.log` |
| Highest quality | `/build/highest-quality-raw` | `/build/highest-quality-clip-s11` | `/build/highest-quality-dwq-work-1` | `/logs/highest-quality-dwq-memory.jsonl` | `/logs/highest-quality-dwq-run.jsonl` | `/logs/highest-quality-dwq.stderr.log` |
| Promotion 1 | `/build/compact-promotion-1-raw` | `/build/compact-promotion-1-clip-s11` | `/build/compact-promotion-1-dwq-work-1` | `/logs/compact-promotion-1-dwq-memory.jsonl` | `/logs/compact-promotion-1-dwq-run.jsonl` | `/logs/compact-promotion-1-dwq.stderr.log` |
| Promotion 2 | `/build/compact-promotion-2-raw` | `/build/compact-promotion-2-clip-s11` | `/build/compact-promotion-2-dwq-work-1` | `/logs/compact-promotion-2-dwq-memory.jsonl` | `/logs/compact-promotion-2-dwq-run.jsonl` | `/logs/compact-promotion-2-dwq.stderr.log` |

For each row, `--model` in the clip command is **Raw student** and `--out` is
**Clipped student**. In the DWQ block, `--quantized-model` is **Clipped
student**, `--mlx-path` is **DWQ output**, the two evidence variables select
the row's JSONL paths, and stderr is redirected to the row's final column.

## 5. Deepest-first layerwise DWQ

Use one layer per round, full validation after every round, rollback on any KL
regression, and precomputed targets. `ALIS_DWQ_EXTRAS_MODE=skip` keeps the
quantized embedding/head fixed. Router training, LoRA, alternate losses, FFN
permutation, and diagnostic round/step limits stay disabled for release runs.

```bash
set -o noclobber

ALIS_DWQ_DATA_DIR=/build/laguna-data-v4 \
ALIS_DWQ_NUM_VALID_SAMPLES=0 \
ALIS_DWQ_LAYERS_PER_ROUND=1 \
ALIS_DWQ_EXTRAS_MODE=skip \
ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat \
ALIS_DWQ_MAX_PEAK_FRACTION=0.90 \
ALIS_DWQ_MAX_SWAP_INCREASE_GIB=16 \
ALIS_DWQ_MAX_ROUNDS=0 \
ALIS_DWQ_MAX_STEPS_PER_ROUND=0 \
ALIS_DWQ_TRAIN_ROUTERS=0 \
ALIS_DWQ_LORA_RANK=0 \
ALIS_DWQ_ADAPTER_DIR= \
ALIS_DWQ_CKA_MONITOR=0 \
ALIS_DWQ_LOSS=kl \
ALIS_DWQ_MEMORY_EVIDENCE_PATH=/logs/compact-dwq-memory.jsonl \
ALIS_DWQ_RUN_EVIDENCE_PATH=/logs/compact-dwq-run.jsonl \
python -m alis_dwq.run \
  --model /build/bf16-mlx-layout \
  --quantized-model /build/compact-clip-s11 \
  --target-dir /build/teacher-targets-v3 \
  --mlx-path /build/compact-dwq-work-1 \
  --num-samples 80 \
  --max-seq-length 512 \
  --batch-size 1 \
  --grad-checkpoint \
  --learning-rate 1e-6 \
  --seed 7 \
  2> /logs/compact-dwq.stderr.log
```

The one-layer Laguna probe measured 55.79 GB peak working set with zero swap
increase. That was probe evidence, not a guarantee for later rounds; the full
run therefore retained per-step and per-round monitoring. The BF16 teacher was
not resident during DWQ training.

The final run JSONL contains exactly `run_started` and `run_completed` for a
non-diagnostic success, sharing one run ID and binding the pre-DWQ checkpoint,
target contract, and final artifact digests. Immediately before completion the
launcher revalidates the data/tokenizer files, every target, and fresh
teacher/pre-DWQ directory digests. The final student is written under a
run-owned sibling staging path and reaches `--mlx-path` only through an atomic
no-replace move. Round/step limits are diagnostic:
their output path must contain `diagnostic`, the artifact is marked incomplete,
and no release-complete evidence is emitted.

An incomplete target dump, clipped directory, or final shard save is not
resumable. Restart from the immutable input into a fresh sibling directory.

## 6. Acceptance gates

All gates are mandatory for each publishable build:

1. Missing and unexpected strict-load parameters are both zero. Logits are
   finite, thinking on/off generations complete, and caches are exactly 12 full
   plus 36 rotating/window-512 entries.
2. Q4 baseline, pre-DWQ arm, and final arm use identical held-out bytes, prompts,
   tokenizer options, and seed. Record load time, peak memory, prefill/decode
   speed, PPL, full-vocabulary KL, top-1 flip, distinct-4gram, and tail cycles.
3. Every DWQ round is accepted only when validation KL does not regress. The
   final result must not be worse than its pre-DWQ arm outside the paired 95%
   confidence interval on any major held-out slice.
4. Relative to stock Q4, final PPL degradation is at most 3% for code and 5%
   each for English, reasoning, and Korean slices.
5. A new repetition loop or tool-call formatting break is an immediate failure.
6. Cross the 511/512/513 boundary and run actual 8K and 32K needle smokes. Report
   1M context only as calculated support unless it is actually executed.
7. If the compact arm fails, promote only measured high-damage routed
   projections/layers to Q4 and rerun clip, DWQ, and every gate. Allow at most
   two cumulative promotion attempts. If none passes, do not publish it as the
   compact minimum-quality build.

Measured final bpw, quality metrics, throughput, full-run memory peak, promotion
decisions, PR revisions, and immutable Hub commits belong in the completed model
cards and run receipts; never fill them with estimates.
