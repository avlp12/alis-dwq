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

### 0. Measure expert traffic first (optional — sizes the recipe)

Expert-hybrid quants (e.g. [NVFP4 top-64 + NF3 tail on GLM-5.2](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)) work because routing traffic is heavily skewed. Measure that skew for your model before picking per-layer bits:

```bash
python -m alis_dwq.expert_traffic --model <model> --save traffic.npz --top 64
python -m alis_dwq.expert_traffic --load traffic.npz --top 32   # re-analyze, no model load
```

Hooks `SwitchGLU`/`SwitchMLP` (architecture-agnostic), runs the same fixed EN/code/ZH slices as `eval_kld`, and reports per layer: top-N traffic share, experts needed for 90%/99% of mass, dead experts, JSD(EN, ZH), and the worst per-slice overlap of the top-N set. Uses for the numbers:

- **top-N share high everywhere** → a salient/tail bit split is on the table; **low or uneven** → spend bits per-layer instead.
- **low min-overlap / high JSD layers** → the salient set is language-dependent there; a static salient set chosen on EN traffic under-serves ZH (the per-expert version of the ZH damage the 45% mix corrects). Give those layers more bits, and make sure the calibration mix covers the language whose experts differ.
- **dead experts** on the calibration slice get no DWQ gradient — if a layer has many, widen the mix before trusting a low-bit build of it.

### 1. Build a calibration mix (optional but recommended)

`dwq_data/train.jsonl` + `valid.jsonl`, one `{"text": ...}` per line. Language mix matters: low-bit damage concentrates in the model's non-English mass (we measured ZH slices 1.4–3.9× worse than EN on two different MoE families); a ~45% target-language mix is what recovered it. Without `ALIS_DWQ_DATA_DIR`, mlx-lm's default `--data-path` loader is used.

### 1b. Clip-search requantize the student (free KL, before any training)

MLX's affine mode maps each group's exact min/max onto the grid ends, so one outlier stretches the whole group's grid. Borrowing the [four-over-six](https://humansand.ai/blog/nvfp4-rl.html) idea (narrow the range per block only when *measured* reconstruction error drops):

```bash
python -m alis_dwq.clip_quantize \
  --source <bf16/fp16 dump, or a Q8 dump with --dequantize-source> \
  --model <student> --out <student-clip> --max-err-slack 1.1
```

Tries a few clipped ranges per group and accepts one only when it lowers the group MSE **without raising the group's max abs error beyond `--max-err-slack`×** the unclipped grid's (unclipped is always a candidate). Per-tensor bits/group_size are inferred from shapes, so dynamic recipes pass through untouched. **Run it before DWQ, never after**: DWQ trains scales/biases with the packed codes frozen, and re-deciding the codes is exactly what clipping adds (it would also discard an existing DWQ, since everything is recomputed from `--source`).

Two hard-won constraints (mechanisms + numbers in the [E1 case study, part 2](examples/glm-5.2-e1-floor-spike/README.md)):

- **The source must be quasi-continuous** — bf16/fp16, or a ≥8-bit dump via `--dequantize-source`. A dequantized low-bit lattice (nvfp4, ≤4-bit affine) as source produces correlated rounding that kills the model while every per-tensor metric improves (measured: identical rules, nvfp4 source → wikitext 51–12,769; Q8 source → 4.42–4.68). The tool refuses lattice sources unless `--allow-lattice-source`.
- **Keep the anchor guard on** (`--max-err-slack`, default strict 1.0; 1.1 measured best on a 2-bit student). MSE-only selection saturates the group-extreme "super weights" that min-max grids protect — mean −24% with anchors ×4 worse decoded to PPL 51.

### Where to spend bits (measured, from the 4-bitter Lesson Fig. 18)

1. **Shared experts first** — near-free (+0.2 GB on a DeepSeek-style model) and the steepest single quality gain; they run for every token.
2. **Then the last ~15% of layers** (returns taper: 4% → 8% → 15%).
3. **Don't bump the first layer** — ~1 GB for no measured gain; reclaim those bits for 1–2.

Cross-check the split against your own `expert_traffic` numbers before committing a recipe.

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
  --env ALIS_DWQ_DATA_DIR=<identical-path>/dwq_data \
  --python <venv>/bin/python \
  examples/distributed_dump_entry.py \
  --model <local teacher dir> --targets-only --pipeline --target-dir ./targets \
  --num-samples 145 --max-seq-length 512 --batch-size 1 --seed 7
```

`mlx.launch` takes a **script file** it can verify on every host — `-m alis_dwq.run` does not work — hence the 3-line [`examples/distributed_dump_entry.py`](examples/distributed_dump_entry.py). The calibration jsonl must be **byte-identical on every host** (hash it): each rank loads its own copy, and the shared seed only aligns the permutation if the rows match.

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
| [GLM-5.2 3.5 bpw](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw), **8-bit teacher** (shipped: `main`; prior retune at `dwq-4.5teacher`) | 310 GB / 745B MoE | 790 GB teacher → **distributed 2-box dump** (§2b), then single-node K=6 | strided PPL wikitext **2.851→2.814 (−1.3%, significant)** over the 4.5 bpw-teacher retune; code within noise; loss 0.178→0.138 — see [examples/glm-5.2-8bit-teacher](examples/glm-5.2-8bit-teacher/README.md) |
| GLM-5.2 2.56 bpw, 8-bit teacher (**not shipped** — negative result) | 227 GB / 745B MoE | same targets reused, K=8 | DWQ loss −35% but held-out PPL slightly *worse* (+1.3% wikitext) — the sweet-spot lesson above; `main` keeps the 4.5 bpw-teacher weights |
| GLM-5.2 **E1 floor spike**: 2-bit/gs128 experts, 2.32 bpw, teacher **A/B** ([case study](examples/glm-5.2-e1-floor-spike/README.md)) | 215.8 GB / 745B MoE, from-Q8 | both arms from the same raw student, K=6 | 4.5-t arm won *every* held-out metric (wikitext 4.036 vs 4.087; −14.3% vs raw); 8-bit arm's valid dropped more yet lost — sweet-spot 3rd point. +7.0% wikitext vs the 2.56 build for −26.6 GB |

## Hard-won operational notes

- **Gate every big load**: wait until `free+inactive > model + 60 GB` *and stable* — a "done" log line doesn't mean the previous process released its memory, and loads launched into a reclaim window get jetsam-killed silently.
- **Never overlap a 100 GB-class load with heavy disk I/O** (uploads, mass writes): the load wedges with rss/avail frozen. Kill (-9, then verify with pgrep — zombies squat memory), let it reclaim, relaunch on a quiet box.
- **Getting the teacher shards onto each box (distributed dump):** Xet-backed HF repos don't parallelize with `aria2c` — the `resolve/main` redirect is a byte-range-signed CAS URL, so multi-connection splits `403`. Use `snapshot_download` with `HF_XET_HIGH_PERFORMANCE=1` and per-shard `allow_patterns` to hand each box only its half. Xet can also silently wedge mid-repo (shard count frozen, zero net-in) — wrap it in a watchdog that kills + resumes when progress stalls. If one box's uplink is slower, have the faster box pull its peer's remaining shards and ship them over the local (TB/LAN) link.
- Learning rate transfers poorly across student sizes: 1e-5 was fine for an 87 GB student and diverged on a 242 GB one. Start at mlx-lm's default 1e-6; the per-round rollback makes over-stepping cheap.
- If your checkpoint carries extra layers your runtime remaps (e.g. an MTP head), you have two working paths. (a) Strip them for training with a hardlinked variant and re-attach after — **never edit a hardlinked file in place**; replace it (`os.replace`), or you rewrite the original through the shared inode. (b) Train with them attached (the model's `sanitize` remaps them at load) — but then the *saved* checkpoint carries **module-named keys** (e.g. `mtp.layer.*`) instead of the checkpoint convention (`model.layers.<N>.*`), and strict loading breaks with "N parameters not in model". Fix by inverse-renaming those keys in the affected shards + index; **verify the inverse map by asserting the renamed keyset equals the original checkpoint's** before shipping, and write via temp-file + `os.replace` (see the lazy-mmap note above).
- **Stock `mlx_lm.perplexity` / `mlx_lm.evaluate` never set the wired limit** — only generate/server do. On a 100 GB-class model that is a silent ~10× slowdown (throughput thrashing, not an error). Wrap them: `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` after load, then call the module's `main()`. (This repo's `eval_kld` already does.) Custom eval scripts loading an MTP-bearing checkpoint also need `load_model(..., strict=False)` unless the runtime remaps the head.
- **Choosing K (layers per round)**: backward memory scales with K, so pick the largest K that fits. Anchors from the 745B family on a 512 GB box: a 227 GB student at K=8 peaked ~307 GB; a 310 GB student at K=6 peaked ~389 GB. Start high, drop K on OOM — rounds get proportionally cheaper, total wall-clock barely moves.
- **A REVERTED round near the end is the gate working, not a failure.** Both shipped GLM retunes ended with one late-round rollback (best already reached); treat consecutive reverts as the natural stopping signal.
- When re-attaching a shard, **name it `model-*.safetensors`**: mlx-lm's loader collects shards by that glob, not by the index — a shard named anything else silently never loads and you get "missing parameters" for exactly its keys.
- **`mx.load` is lazy (mmap) — never `save_safetensors` back to the path you loaded from.** The save truncates the file *before* the lazy view is read, so you write zeros over your own data. `mx.eval` the new arrays first, or write a temp file and `os.replace`. (We zeroed a whole target dump post-processing it this way — and the real fix was to not post-process at all: get the shape right in the forward, per §2b.)
- **Verify a dump is non-zero and the right shape before you reclaim the teacher.** A distributed dump is your only local copy of the teacher's logits; the teacher weights are the cheap thing to reconstruct, the dump is not. After the corruption above we'd already deleted one box's teacher half — re-dumping meant re-pulling 400 GB. Keep the *peer* box's half until the targets pass a `min/max != 0` + shape check; and to free a box for the next stage, move the student to the peer over the local link (minutes) instead of deleting and re-pulling from HF (an hour).
- **Xet can also wedge after a device-level I/O interruption, not just mid-repo.** After a USB enclosure dropped and re-enumerated mid-download, `hf download` sat alive-but-dead — ~0% CPU, trickle network, zero file writes — because the background writer thread had died (`Internal Writer Error: Background writer channel closed`) and the main thread waits forever. It reproduced twice in one day, once at 57/59 shards. The stall watchdog that catches it: downloader process alive **and** no file in the target tree modified for 10 minutes → kill and relaunch (resume is safe; completed files are kept).
- **VL MoE recipe note — check whether the shared expert is *packed* into the routed bank.** mlx-vlm's MiniMax-M3 packs the shared expert as expert #128 inside the same SwitchLinear when `shared_intermediate_size == intermediate_size`, welding the one tensor that sees **100% of tokens** (a routed expert sees ~3% at top-4/128) to the routed experts' bit-width. Unpacking it and holding it at 8-bit while routed experts drop to 3-bit cost +0.4% size and needs only a config-gated layout switch ([mlx-vlm#1544](https://github.com/Blaizzy/mlx-vlm/pull/1544); packed vs unpacked forward matches to 7.75e-07). The same audit applies before DWQ: a packed shared expert can't be given its own learning treatment either.
- **Evaluating a thinking model: budget `max_tokens` for the thinking, and grade the tail.** A needle-retrieval probe with `max_tokens=30` "failed" 0/3 on an 8-bit reference — the model spent the whole budget inside `<mm:think>` and never emitted the answer; at 400 tokens it was 3/3 at every depth. Harness artifact, looks exactly like model damage.
- **Publishing branch-per-quant on HF: branches inherit `main`'s files at creation.** Create branch → upload build = the branch now carries *both* builds' shards (ours had 39 stale + 87 real; snapshot downloads grow ~190 GB). `delete_files` the inherited shard names on every non-main branch after upload, and verify per-revision shard counts via the API.
- **HF sidebar param count for quantized repos comes from `model.safetensors.index.json` metadata `total_parameters` — and `mlx_vlm.convert` doesn't write it** (`mlx_lm.convert` does). Without it the Hub counts packed-U32 elements as parameters and shows ~1/8 of the truth (427B displayed as "56B"). Inject the exact count (summed from the bf16 source shard headers) before upload; verify with `api/models/<repo>?expand[]=safetensors`.
- **Never re-quantize a dequantized low-bit lattice with a coarser grid.** nvfp4/≤4-bit-affine dequants sit on a coarse lattice; a coarser min-max affine grid over lattice values rounds with correlated bias (continuous sources round pseudo-randomly), and the bias accumulates through activations — the model dies while per-tensor MSE, kernel outputs, and every artifact check look *better*. Paradox signature: the more groups you clip (off-lattice rescale), the more alive the model. Sources for requantization must be bf16/fp16/Q8-class.
- **A mean weight metric cannot see super-weights.** The few largest-magnitude weights per tensor do disproportionate work, and min-max affine grids protect them for free (they *are* the grid ends). Any transform that trades their fidelity for interior precision (MSE-optimal clipping at 2-bit: top-0.01% weights ×4 error, mean −24%) kills the model. Gate weight-space transforms on per-group max-error and held-out PPL, never on mean reconstruction metrics alone.
- **Hash the *resolved* file, not the HF `raw/` endpoint, when verifying tokenizer identity across repos.** `raw/` serves the ~133-byte LFS pointer for LFS-tracked files, so the hash "mismatches" even when the tokenizers are byte-identical; download the real file and `shasum` that. `tokenizer_config.json` may also differ by benign client keys (e.g. `local_files_only`) — identity lives in `tokenizer.json`.
- **Start the decode timer after prefill, and keep Time Machine away from scratch trees.** Timing a generator from its first iteration folds prefill into "decode" and manufactures a fake long-context decode cliff (we chased one for a day). Separately, an unexcluded few-hundred-GB working tree gets picked up by Time Machine mid-run and strangles disk I/O — `tmutil addexclusion <dir>` every big build/download directory at creation.
- **A sandboxed autonomous agent cannot see the GPU.** macOS Seatbelt profiles (e.g. Codex CLI `workspace-write`) block Metal device access: MLX fails with `[metal::load_device] No Metal device available`, and detached/nohup children inherit the sandbox. Probe Metal *before* any download/build step in agent-run pipelines, and give GPU-touching agent runs full (unsandboxed) access — probe-first discipline turned this from a wasted 790 GB download into a zero-cost stop.
- **A sharper teacher is not always better — it depends on the student's bit-width, and only held-out PPL tells you.** Re-tuning a 3.5 bpw GLM-5.2 student against an 8-bit teacher (vs a 4.5 bpw one) improved held-out strided PPL a real, significant **−1.3%** (wikitext). The *same* 8-bit teacher on the more-quantized **2.56 bpw** student made it slightly **worse** (+1.3%, within noise) — even though its DWQ-vs-teacher loss dropped **35%**. At ~2 bpw the student can't represent the finer target distribution, so chasing it overfits the 145-sample calibration set without transferring to the eval corpora. There's a **teacher-precision / student-capacity sweet spot**: a mid-bit student benefits from a sharper teacher, an aggressively-quantized one does not. Always judge on held-out PPL, never on the training/validation loss alone (which will happily drop toward any teacher).

## Absorbed from upstream, not yet validated here

Ideas worth stealing from [omlx's oQ pipeline](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) (reviewed 2026-07-10). None of these carry our own measurements yet — treat as roadmap, not method:

- **Batched-expert GPTQ.** All routed experts in a layer receive the same input hidden states, so they share one Hessian — oQ runs GPTQ over the whole expert bank at once (they report ~15× vs per-expert). Complementary to DWQ: GPTQ is a cheap local error-compensation pass (minutes, forward-only calibration) where DWQ is end-to-end distillation (hours). Likely first target: a mid-2-bpw-class student *before* DWQ, to see whether the passes stack. **Code-audit caveat (omlx `d42528a`): the shipped `oq.py` contains no GPTQ/Hessian implementation — its `enhanced=True` path is imatrix-weighted affine clipping ("oQe").** Verify upstream code, not the docs, before planning around "oQ+ GPTQ".
- **Activation-weighted affine scale fitting.** Fit per-group `scales`/`biases` by activation-weighted least squares instead of min-max — same checkpoint format, better values, stock loading. Same compatibility trick DWQ itself relies on.
- **Normalized layer sensitivity for bit allocation.** `MSE(float, quant) / mean(float²)` per layer — the normalization keeps late layers from looking artificially sensitive under residual accumulation. Our measured-not-heuristic promotion lesson (Hy3: the sensitive band was mid-stack, not early/late) says automate this rather than trust priors.
- **Expert-coverage tracking of the calibration set.** oQ tracks which experts the calibration data actually activates. We have never measured what fraction of a 128–256-expert bank our 145-sample mix reaches per layer — undersampled experts get weak distillation signal in DWQ too. The diagnostic now ships here as `alis_dwq/expert_traffic.py` (§0); measurements on our own mixes still pending.

Caveat on their headline numbers (2-bit MMLU 64% vs mlx-lm's 14%): the baseline is *uniform* mlx-lm quantization. A sensitivity-graded mixed-precision recipe already captures much of that gap; the marginal value here is the error-compensation and scale-fitting passes on top of a good recipe, and it needs to be measured as such.

## Floor notes: how low the MLX container goes (2026-07-10)

Research and probe notes from scoping a sub-2.56-bpw GLM-5.2 build. The build ran on 2026-07-11: experts 2-bit/gs128 → 215.8 GB dec / **2.3225 bpw measured**, raw wikitext 4.711, after DWQ (the 4.5-bpw teacher won the A/B) **4.036** — full numbers in the [E1 floor-spike case study](examples/glm-5.2-e1-floor-spike/README.md).

- **The MLX affine floor is 2-bit** (mlx 0.31.2 probe: supported bits `{2,3,4,5,6,8}`; `bits=1` raises). 2-bit/gs128 is the lowest effective rate: (128·2+32)/128 = **2.25 bpw**. Probed end-to-end at gs128 — `quantize`/`dequantize`/`quantized_matmul` *and* gradients w.r.t. `scales`/`biases` — so layerwise DWQ supports a mixed-group-size student out of the box.
- **Ternary ("BitNet-style") inside MLX is strictly dominated.** Three levels are a subset of the 2-bit affine grid at identical storage, so ternary only pays in a genuinely sub-2-bit container (GGUF `IQ1_*`/`TQ*`), which MLX doesn't have. Conversion-by-training ([BitNet Distillation](https://arxiv.org/abs/2510.13998): SubLN + ~10 B-token continued pretraining + attention distillation) is demonstrated at 0.6–4 B and out of reach for 100 B+ models on Apple-Silicon boxes; pure-PTQ ternary literature so far tops out at 70 B dense with heavy cost ([TWLA](https://arxiv.org/abs/2606.13054): +44% wikitext at W1.58, needs llama.cpp ternary kernels; [PTQTP](https://arxiv.org/abs/2509.16989)'s "trit-planes" are ~4 bpw effective despite the name).
- **The practical floor converges across stacks.** [Unsloth's GLM-5.2 dynamic "1-bit"](https://unsloth.ai/docs/models/glm-5.2) (UD-IQ1_S GGUF) ships at **223 GB ≈ 2.4 bpw effective** for the same 745 B model — the size class a sensitivity-graded MLX 2.56-bpw build already occupies — while the same team's R1-671B went to 131 GB (1.56 bpw). The floor is model-dependent, and no one ships GLM-5.2 under ~220 GB today; below that line the fight is quality-per-byte (recipe + error-compensation passes + DWQ), not container size.

## License

MIT
