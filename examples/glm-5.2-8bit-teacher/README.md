# Case study: an 8-bit teacher — one student won, one lost

Same model family, same targets, opposite outcomes. This is the run behind the
*teacher-precision / student-capacity sweet spot* note in the main README.

**Setup.** GLM-5.2 (745B MoE). Teacher: a public **8-bit** quant (790 GB — does
not fit one 512 GB box). Students: the 3.5 bpw (310 GB) and 2.56 bpw (227 GB)
builds, both previously DWQ-retuned against a 4.5 bpw teacher.

## 1. Distributed target dump (teacher's only appearance)

The dump is forward-only, so it pipelines across two 512 GB boxes (§2b of the
main README has the gotchas: embed on every rank, `mx.eval` after the final
`all_gather`, capture the batch size *before* the gather):

```bash
mlx.launch --hosts 10.0.0.1,10.0.0.2 --backend ring \
  --env ALIS_DWQ_DATA_DIR=/Users/Shared/dwq_data \
  --python <venv>/bin/python \
  examples/distributed_dump_entry.py \
  --model <local 8bit teacher dir> --targets-only --pipeline \
  --target-dir targets8bit --num-samples 145 --max-seq-length 512 \
  --batch-size 1 --seed 7
# 177 target files (~430 MB), ~19 s/sample, ~45 min on 2× M3 Ultra
```

Each box holds only its pipeline half of the shards (~404 / ~370 GB). The
dump is tiny and **student-independent** — it was reused as-is for both
students below (same tokenizer, same calibration jsonl, same seed).

## 2. Student A: 3.5 bpw — improved (shipped)

```bash
ALIS_DWQ_LAYERS_PER_ROUND=6 ALIS_DWQ_DATA_DIR=dwq_data \
python -m alis_dwq.run --model <3.5bpw> --quantized-model <3.5bpw> \
  --target-dir targets8bit --mlx-path <out> --num-samples 145 \
  --max-seq-length 512 --batch-size 1 --grad-checkpoint \
  --learning-rate 1e-6 --seed 7
# 13 rounds of K=6, peak ~389 GB, loss 0.1776 -> 0.1381 (12 accepted, 1 late revert)
```

| strided PPL | pre-DWQ | 4.5 bpw teacher | **8-bit teacher** |
|---|---|---|---|
| wikitext | 2.946 | 2.851 | **2.814** (−1.3%, outside the 95% CI) |
| code | 1.893 | 1.841 | **1.832** (within noise) |

Shipped as `main` of the 3.5 bpw repo; the 4.5-teacher retune is preserved on
the `dwq-4.5teacher` branch. Downstream figures (tulu PPL, MC tasks, tok/s)
stayed within noise — the strided PPL is the sensitive metric here.

## 3. Student B: 2.56 bpw — got *worse* (not shipped)

Same targets, K=8 (10 rounds, peak ~307 GB). Training looked *better* than
student A's: loss 0.5125 → 0.3348 (**−35%**, vs A's −22%). Held-out PPL
disagreed:

| strided PPL | 4.5 bpw teacher (shipped) | 8-bit teacher |
|---|---|---|
| wikitext | **3.774** | 3.825 (+1.3%) |
| code | **2.069** | 2.081 (+0.6%) |

At ~2-bit expert precision the student cannot represent the sharper teacher's
distribution; chasing it overfits the 145-sample calibration set instead of
transferring. The 2.56 bpw repo keeps the 4.5 bpw-teacher weights.

## The lesson

- **Teacher precision has a student-capacity-dependent sweet spot.** 3.5 bpw
  sits on the side where a sharper teacher helps; 2.56 bpw sits on the side
  where it hurts. If you upgrade a teacher, re-judge every student.
- **Never judge on the DWQ loss.** Student B had the *bigger* loss drop and
  the *worse* outcome. Held-out PPL (ideally per-language slices) is the only
  verdict that counts.
- **Dumps are reusable across students** — the marginal cost of testing a
  second student is one training run, so test rather than assume (we assumed
  the lower-bpw student would benefit more, and measured the opposite).
