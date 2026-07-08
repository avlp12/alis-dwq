# Case study: GLM-5.2 2.56 bpw (745B MoE) on one 512 GB M3 Ultra

The run that motivated layerwise DWQ. Full commands, in order:

```bash
# 0. measure per-language damage first (sizes the opportunity)
python -m alis_dwq.eval_kld --model <4.5bpw-ref> --save-ref ref.npz
python -m alis_dwq.eval_kld --model <2.56bpw> --ref ref.npz
#   -> EN KL 0.727 / code 0.252 / ZH 0.987  (ZH +36% vs EN: the target)

# 1. teacher targets (4.5 bpw nvfp4 teacher, 424 GB — its only load)
python -m alis_dwq.run --model <4.5bpw-ref> --targets-only \
  --target-dir targets --num-samples 145 --max-seq-length 512 \
  --batch-size 1 --seed 7

# 2. layerwise training (student 193 GB; full-layer training OOMs)
ALIS_DWQ_LAYERS_PER_ROUND=8 ALIS_DWQ_DATA_DIR=dwq_data \
python -m alis_dwq.run --model <2.56bpw> --quantized-model <2.56bpw> \
  --target-dir targets --mlx-path <out> --num-samples 145 \
  --max-seq-length 512 --batch-size 1 --grad-checkpoint \
  --learning-rate 1e-6 --seed 7
# peak ~334 GB, ~12 s/it, 10 rounds

# 3. re-evaluate
python -m alis_dwq.eval_kld --model <out> --ref ref.npz
```

Results (T=3072 vs the 4.5-bpw reference):

| slice | before | after |
|---|---|---|
| EN    | 0.727 / 24.9% | 0.383 / 15.6% |
| code  | 0.252 / 12.7% | 0.193 / 10.2% |
| ZH    | 0.987 / 35.7% | 0.562 / 21.9% |
| all   | 0.655 / 24.4% | **0.379 / 15.9%** |

Validation KL during training: 0.458 -> 0.322, all 10 rounds accepted.
Upstream PR for the native flag: https://github.com/ml-explore/mlx-lm/pull/1499
Shipped result: https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-2.56bpw (main = DWQ weights; `pre-dwq` branch = original)
