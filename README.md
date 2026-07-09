# alis-dwq

**DWQ for very large MLX models on Apple Silicon** — layerwise rounds + two-phase targets, so quantized models that are *hundreds of GB* can be distillation-tuned on a single Mac (or a pair) without OOM.

Built and battle-tested on a 512 GB M3 Ultra pair while shipping public quants of GLM-5.2 (745B MoE) and Hy3 (295B MoE).

## Why

`mlx_lm dwq` fine-tunes the quantization `scales`/`biases` of a quantized model ("student") against a higher-precision teacher's logits. It works great — until the student is large:

1. **The teacher may not fit next to the student.** Two-phase fixes this: dump the teacher's top-k logits to disk once (teacher alone in memory), then train the student against the dump (student alone in memory — with targets on disk, mlx-lm never loads the teacher for training).
2. **The student's own backward may not fit at all.** Training *every* layer's scales in one step materializes large per-layer intermediates for the quantized-matmul VJP simultaneously. A 242 GB / 745B-parameter MoE student dies in the first step on a 512 GB machine — at sequence length 1024, 512 *and* 256, with `--grad-checkpoint`, at batch size 1. Distributing doesn't rescue it (`Send` has no VJP in the pipeline path).

**Layerwise rounds** fix (2): train at most K layers per round. Since `value_and_grad` only differentiates unfrozen parameters, backward memory is bounded by K instead of by the model. Rounds run deepest-first (best-conditioned gradients — shallow-first diverged in our runs), and every round is validated and rolled back if it didn't improve, so the procedure is monotone by construction.

Measured on the 745B case: **OOM at step 0 → stable training at ~334 GB peak** with K=8.

## Install / use

Works on **stock mlx-lm ≥ 0.31** (the layerwise trainer ships here as a patch module; upstream PR [ml-explore/mlx-lm#1499](https://github.com/ml-explore/mlx-lm/pull/1499) adds `--layers-per-round` natively — once merged you can use either path).

```bash
git clone https://github.com/avlp12/alis-dwq && cd alis-dwq
pip install mlx-lm  # >= 0.31
```

### 1. Build a calibration mix (optional but recommended)

`dwq_data/train.jsonl` + `valid.jsonl`, one `{"text": ...}` per line. Language mix matters: low-bit damage concentrates in the model's non-English mass (we measured ZH slices 1.4–3.9× worse than EN on two different MoE families); a ~45% target-language mix is what recovered it. Without `ALIS_DWQ_DATA_DIR`, mlx-lm's default `--data-path` loader is used.

### 2. Dump teacher targets (teacher's only appearance)

```bash
python -m alis_dwq.run \
  --model <teacher> --targets-only --target-dir ./targets \
  --num-samples 145 --max-seq-length 512 --batch-size 1 --seed 7
```

### 2b. Teacher too big for one box → distributed dump

The dump is forward-only, so it pipelines across N Macs — each box loads only its layer range's shards:

```bash
mlx.launch --hosts <box1>,<box2> --backend ring \
  --python <venv>/bin/python -m alis_dwq.run \
  --model <local teacher dir> --targets-only --pipeline --target-dir ./targets \
  --num-samples 145 --max-seq-length 512 --batch-size 1 --seed 7
```

Point `--model` at a **local** teacher dir on each box (an HF repo id makes every box re-download). Split by the pipeline's own layer assignment — it snaps to DSA "full"-layer boundaries so IndexShare never crosses a rank — and give each box its layers' shards **plus every globally-shared weight** (see below).

Two gotchas cost us a full session on the first real distributed dump (790 GB 8-bit GLM-5.2 teacher across two 512 GB boxes; single-node teachers never hit this path). Both surface **identically** as `[METAL] Command buffer execution failed: GPU Timeout` on *one* rank even though the compute is fine — `eval_every`, wired-limit, and warmup are all red herrings. Diagnose by tracing per-layer (`mx.eval(h); print` after each layer, **both** ranks): you'll see one rank finish every layer while the other never enters its loop.

- **Every box needs the embedding shard — not just the first-stage rank.** The pipeline forward runs `embed_tokens(x)` on *all* ranks (later ranks discard it and overwrite with the `recv`). A rank missing the embed weights hangs at the embedding; its peer then trips the watchdog waiting at the collective. Replicate `embed_tokens` to every box; `lm_head`/final-norm only need the last rank.
- **`mx.eval` the pipeline's final `all_gather` before any GPU op consumes it.** The collective is on the CPU stream (no watchdog) — but the final norm/slice that reads it runs on the GPU, so the GPU command buffer *waits* on the collective while the slowest rank finishes its whole forward. On forward #1, which includes cold Metal kernel compilation on the deep rank, that wait blows past the macOS ~5 s GPU watchdog and kills the rank that arrived first. Fix in the `deepseek_v3`/`v32` pipeline path: `n = h.shape[0]; h = all_gather(h, stream=cpu); mx.eval(h); h = h[:n]` — capture the local batch **before** `all_gather` grows axis 0. (Miss that and the dump silently saves `(ranks, seq, k)` logits — half of them the wrong rank's partial forward — instead of `(1, seq, k)`; slice back to row 0, the last-stage rank that owns `lm_head`.)

**Keep `--batch-size 1` identical between the dump and the training run.** Targets are keyed by batch index, so a dump/train batch-size mismatch silently aligns teacher logits to the wrong samples. Batch 1 also keeps each command buffer small — batch 8 at seq 512 is 4096 tokens/layer and flirts with the same watchdog.

### 3. Train the student, layerwise

```bash
ALIS_DWQ_LAYERS_PER_ROUND=8 ALIS_DWQ_DATA_DIR=./dwq_data \
python -m alis_dwq.run \
  --model <student> --quantized-model <student> \
  --target-dir ./targets --mlx-path <out> \
  --num-samples 145 --max-seq-length 512 --batch-size 1 \
  --grad-checkpoint --learning-rate 1e-6 --seed 7
```

(`--model` only supplies the tokenizer when targets exist — point it at the student; the teacher's 400 GB never move again.)

### 0b. Measure expert traffic first (sizes the recipe)

Expert-hybrid quants (e.g. [NVFP4 top-64 + NF3 tail on GLM-5.2](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)) work because routing traffic is heavily skewed. Measure that skew for your model before picking per-layer bits:

```bash
python -m alis_dwq.expert_traffic --model <model> --save traffic.npz --top 64
python -m alis_dwq.expert_traffic --load traffic.npz --top 32   # re-analyze, no model load
```

Hooks `SwitchGLU`/`SwitchMLP` (architecture-agnostic), runs the same fixed EN/code/ZH slices as `eval_kld`, and reports per layer: top-N traffic share, experts needed for 90%/99% of mass, dead experts, JSD(EN, ZH), and the worst per-slice overlap of the top-N set. Uses for the numbers:

- **top-N share high everywhere** → a salient/tail bit split is on the table; **low or uneven** → spend bits per-layer instead.
- **low min-overlap / high JSD layers** → the salient set is language-dependent there; a static salient set chosen on EN traffic under-serves ZH (the per-expert version of the ZH damage the 45% mix corrects). Give those layers more bits, and make sure the calibration mix covers the language whose experts differ.
- **dead experts** on the calibration slice get no DWQ gradient — if a layer has many, widen the mix before trusting a low-bit build of it.

### 4. Evaluate on language slices, not just overall

```bash
python -m alis_dwq.eval_kld --model <teacher> --save-ref ref.npz   # once
python -m alis_dwq.eval_kld --model <out> --ref ref.npz            # per build
```

Reports KL and top-1 flip per EN/code/ZH third (drop your own corpora in `data/`). Overall averages hide exactly the damage you're trying to fix.

## Results

| Case | Student | Setup | Outcome |
|---|---|---|---|
| [GLM-5.2 2.56 bpw](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-2.56bpw) (shipped: `main` carries the DWQ weights, pre-DWQ at `pre-dwq`) | 745B MoE student (78 quant layers) | K=8, seq 512, lr 1e-6, 45% ZH mix | vs 4.5-bpw ref: overall KL **0.655→0.379 (−42%)**, ZH **0.987→0.562 (−43%)**, flips 24.4%→15.9%; peak 334 GB (full-layer training OOMs); 10/10 rounds accepted, ~5 h |
| Hy3 T128 2.375 bpw | 87.6 GB / 295B MoE | full-layer DWQ (fits), 45% ZH mix | ZH-concentrated damage recovered enough to ship; see model card |

## Hard-won operational notes

- **Gate every big load**: wait until `free+inactive > model + 60 GB` *and stable* — a "done" log line doesn't mean the previous process released its memory, and loads launched into a reclaim window get jetsam-killed silently.
- **Never overlap a 100 GB-class load with heavy disk I/O** (uploads, mass writes): the load wedges with rss/avail frozen. Kill (-9, then verify with pgrep — zombies squat memory), let it reclaim, relaunch on a quiet box.
- **Getting the teacher shards onto each box (distributed dump):** Xet-backed HF repos don't parallelize with `aria2c` — the `resolve/main` redirect is a byte-range-signed CAS URL, so multi-connection splits `403`. Use `snapshot_download` with `HF_XET_HIGH_PERFORMANCE=1` and per-shard `allow_patterns` to hand each box only its half. Xet can also silently wedge mid-repo (shard count frozen, zero net-in) — wrap it in a watchdog that kills + resumes when progress stalls. If one box's uplink is slower, have the faster box pull its peer's remaining shards and ship them over the local (TB/LAN) link.
- Learning rate transfers poorly across student sizes: 1e-5 was fine for an 87 GB student and diverged on a 242 GB one. Start at mlx-lm's default 1e-6; the per-round rollback makes over-stepping cheap.
- If your checkpoint carries extra layers your runtime remaps (e.g. an MTP head), strip them for training with a hardlinked variant and re-attach after. **Never edit a hardlinked file in place** — replace it (`os.replace`), or you rewrite the original through the shared inode.
- When re-attaching a shard, **name it `model-*.safetensors`**: mlx-lm's loader collects shards by that glob, not by the index — a shard named anything else silently never loads and you get "missing parameters" for exactly its keys.
- **`mx.load` is lazy (mmap) — never `save_safetensors` back to the path you loaded from.** The save truncates the file *before* the lazy view is read, so you write zeros over your own data. `mx.eval` the new arrays first, or write a temp file and `os.replace`. (We zeroed a whole target dump post-processing it this way — and the real fix was to not post-process at all: get the shape right in the forward, per §2b.)
- **Verify a dump is non-zero and the right shape before you reclaim the teacher.** A distributed dump is your only local copy of the teacher's logits; the teacher weights are the cheap thing to reconstruct, the dump is not. After the corruption above we'd already deleted one box's teacher half — re-dumping meant re-pulling 400 GB. Keep the *peer* box's half until the targets pass a `min/max != 0` + shape check; and to free a box for the next stage, move the student to the peer over the local link (minutes) instead of deleting and re-pulling from HF (an hour).

## License

MIT
