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
