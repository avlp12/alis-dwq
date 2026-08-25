# DeepSeek-V4-Flash port-fidelity audit (2026-08-23)

Raw receipts for [PORTING_INTEGRITY.md §11](../../docs/PORTING_INTEGRITY.md): three
independent numerical deviations found in mlx-lm PR #1189's DeepSeek-V4-Flash port,
triangulated against three references (official `model.py`, transformers, FreeToken),
adversarially verified, fixed, and validated.

- `deepseek_v4_patched.py` — the port with all three fixes applied: per-layer
  rope/YaRN assignment, per-query causal pool mask (`_pool_causal_mask`), and pool-row
  rotation at block-start positions (`_rope_pool_rows`).
- `rope_pool_test.py` — validation harness: long-document generation with prefill/decode
  throughput and the CJK-slip counter. Verified commands:

```bash
python rope_pool_test.py <model_path> 4900 200    # 4.9K: coherent, CJK slips 0
python rope_pool_test.py <model_path> 19300 250   # 19K: coherent, CJK slips 0 (was: slips present)
```

Published upstream:
- Main report (3 deviations + fixes): https://github.com/ml-explore/mlx-lm/pull/1189#issuecomment-5386017983
- Follow-up (pool-rope fix validated): https://github.com/ml-explore/mlx-lm/pull/1189#issuecomment-5386141413

Note: a pack with restored MTP shards is rejected by the stock loader ("unexpected
weights"); validate through a premtp-index symlink view of the pack.

## On-policy chain-alignment of the MTP head (2026-08-24)

`train_align.py` — teacher-forced chain SFT of the head's non-expert weights (74M params,
bf16), routing around four VJP blockers along the way (routed-expert gather, custom fused
kernels, residual quantized weights, fused attention kernel). Live result: depth-1 conditional
accept 81.5% → 95.6%, depth-2 43.2% → 66.7%, depth-3 9.8% → 34.5%. A follow-up LoRA extension
onto the quantized shared-expert layers trained cleanly (gradient verified nonzero) but
regressed every live metric despite an *improved* self-reported eval score — see
[PORTING_INTEGRITY.md §12](../../docs/PORTING_INTEGRITY.md#12-fine-tuning-through-a-quantized-moe-forward-four-vjp-blockers-in-the-order-you-hit-them-and-why-a-working-gradient-still-isnt-a-working-result)
for the full failure-mode writeup and the reusable VJP-blocker patterns.

## Closing the train/serve numerical gap: measured residuals (2026-08-25)

The sequel, written up in
[PORTING_INTEGRITY.md §13](../../docs/PORTING_INTEGRITY.md#13-train-on-the-measured-residual-not-on-a-model-of-it--and-verify-in-the-regime-you-actually-serve).
The head above was aligned on single-box hidden states while serving runs 2-box tensor
parallel, and the two differ (≈0.4% relative std, kurtosis 414 — K-split partial-sum
nonassociativity, unfixable by promoting `all_sum` to fp32). Three rounds of closing that
gap without leaving the single box all lost (a capacity-adding LoRA, then Gaussian noise at
1% and at 0.3% of `h.std()` meant to imitate the drift); capturing the real TP2 hidden
states and training directly on them won: **+3.68% tok/cycle over the round-2 baseline,
19W/4L/1T over 24 paired topics, sign test p = 0.0026**, with non-CS topics gaining more
than CS ones. A serving-faithful recapture (round 6e) then lifted the offline eval by
+3.9 pp (d1) / +17.4 pp (d2) and moved live acceptance not at all (p = 0.38) — the
section's central lesson.

- `train_align.py` — updated from the §12 copy. New: `--real-hidden` (load a cached TP2
  hidden dump instead of running — and noising — a local forward; accepts either the
  single-file `<base>.safetensors` + `<base>_ids.json` pair or a per-window directory),
  `--loss-start-pos N` (compute CE only at positions ≥ N while keeping full-sequence
  attention context, so training and eval can be restricted to the decode region), and
  `merge_lora_into_shared_experts()` (fold an adapter into the base weight in fp32 —
  attached LoRA parameters do not survive a path-based sharder).
- `dump_hidden_tp2_corpus.py` — forward-only TP2 capture of the whole corpus (bulk prefill,
  `cache=None`), the round-6a/6b/6c training data. Inlines the trainer's `build_corpus`
  verbatim, seed and all, so window *i* here is window *i* there.
- `dump_hidden_6e.py` — the serving-faithful capture: 320-token bulk prefill *through a
  prompt cache* + 64 teacher-forced single-token decode steps on that cache, per window,
  written per-window for resume. Carries the two distributed-capture pitfalls in comments
  (all-rank broadcast of the skip decision; `mx.save_safetensors` renaming a temp file).
- `p1_paired_analysis.py` — the promotion gate: per-topic paired Δtok/cycle, pooled depth
  rates, two-sided sign test, CS/non-CS domain split, parsed straight from harness logs.

Verified commands (capture runs on both boxes under `mlx.launch`; training and analysis are
single-box):

```bash
# 1) capture real TP2 hidden states — forward-only, ring backend, no gradients
cd /Users/Shared/tp2
~/venv_omlx063/bin/mlx.launch --hostfile hostfile_ring2.json \
  /Users/Shared/tp2/exp_chain/r6_dump_worker.sh    # worker: python -u dump_hidden_tp2_corpus.py
~/venv_omlx063/bin/mlx.launch --hostfile hostfile_ring2.json \
  /Users/Shared/tp2/exp_chain/r6e_worker.sh        # worker: python -u dump_hidden_6e.py

# 2) train on the captured residual (round 6c — the promoted checkpoint)
cd ~/dsv4flash/align
~/venv_omlx063/bin/python -u train_align.py --steps 5000 --lr 5e-6 \
  --real-hidden r6c_real_hidden \
  --init-ckpt ckpt_r2/step1000.safetensors --out ckpt_r6c_real

# 2b) round 6e — serving-faithful capture, loss restricted to the decode region
~/venv_omlx063/bin/python -u train_align.py --steps 5000 --lr 5e-6 \
  --real-hidden ~/dsv4flash/align/r6e_h --loss-start-pos 320 \
  --init-ckpt ckpt_r2/step1000.safetensors --out ckpt_r6e_real

# 2c) round 6d — same recipe plus the shared-expert adapter (the isolated LoRA variable)
~/venv_omlx063/bin/python -u train_align.py --steps 5000 --lr 5e-6 \
  --real-hidden r6c_real_hidden --lora-shared --lora-r 16 --lora-alpha 16.0 \
  --init-ckpt ckpt_r2/step1000.safetensors --out ckpt_r6d_real_lora

# 3) paired verdict over two live-harness logs (baseline first, candidate second)
python p1_paired_analysis.py logs/rt_p1_bs1_baseline.log logs/rt_p1_bs1_6c.log
```

The live harness that produces those logs is serving-stack-specific and stays in the serving
repo; the analyzer only needs its telemetry contract — a `[p1-topic i/N] <topic>` line per
prompt followed by the generator's `MTP[0] … tok/cycle=… depth[d1=n/d,d2=n/d,d3=n/d]` line
(duplicated per rank; the first is taken). The regime it runs in must be the regime the
launcher runs — here batch size 1, since the `OMLX_MTP_ROWWISE_BATCH=1` override used for
the earlier batch-8 comparisons is not set in production.

The promoted weights are published at
[avlp12/dsv4flash-mtp-aligned](https://huggingface.co/avlp12/dsv4flash-mtp-aligned)
(`mtp_aligned_r6c_step5000.safetensors`; round 2 kept for rollback).
