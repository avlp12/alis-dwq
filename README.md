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

Measured reference point (GLM-5.2 745B, EN/code/ZH mix, 2026-07-12): routing is **flat** — top-64/256 experts carry only 39–57% of frequency (45–72% of `--norms` mass), n90≈170–207 of 256, dead≈0, busy-but-weak = 0. Salient/tail bit-splits and REAP-style pruning are contraindicated on this family; and JSD(EN,ZH) reaches 0.4 mid-stack with 44–61% min-overlap — the language-dependent-experts effect the 45%-ZH calibration mix exists to cover.

Independently confirmed by the [NF3-hybrid's v3.6 update](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid) (2026-07-09): swapping its top-64 selection from routing frequency to **measured per-expert quantization damage** cut KLD 16% and gained +0.5 GPQA on this same model. The *frequency criterion* was wrong, not expert-level differentiation itself — our student-only damage proxy is `code_entropy --per-expert` (§1a).

Add `--norms` (experimental) to also accumulate each selected expert's output L2 norm — a [REAP](https://arxiv.org/abs/2510.13999)-saliency proxy (their criterion is `gate × ‖out‖`; the gate factor is applied outside `SwitchGLU` and isn't captured). The report then flags **busy-but-weak** experts (selected often, contributing little): REAP's prune candidates, or bit-cut candidates for a hybrid recipe.

### 1. Build a calibration mix (optional but recommended)

`dwq_data/train.jsonl` + `valid.jsonl`, one `{"text": ...}` per line. Language mix matters: low-bit damage concentrates in the model's non-English mass (we measured ZH slices 1.4–3.9× worse than EN on two different MoE families); a ~45% target-language mix is what recovered it. Without `ALIS_DWQ_DATA_DIR`, mlx-lm's default `--data-path` loader is used.

Two hygiene rules from [Unsloth Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)'s methodology notes: **(a) keep calibration and evaluation corpora disjoint** — calibrating on wikitext-like data while gating ships on wikitext PPL overfits the quant to the gate (our held-out gates are wikitext-heavy; check the mix before trusting a small win); **(b) instruct/thinking models should be calibrated through their chat template** — raw text misses the serving token distribution (`gen_calib --chat-template` does this).

**Experimental — synthetic mix with coverage stopping:** [Recover-LoRA](https://arxiv.org/abs/2606.04238) showed 10k *synthetic* samples match curated data for distillation recovery, and [NVIDIA's QAD](https://arxiv.org/abs/2601.20088) showed teacher logits transfer across domains — while our flat-routing measurement (§0) says 145 samples can't reach every expert. `python -m alis_dwq.gen_calib --model <teacher> --out dwq_data --mix EN=0.3,code=0.25,ZH=0.45` self-generates the jsonl, watches expert coverage through the routing hooks, stops when coverage plateaus, and drops degenerate generations via the loop detector. Scale anchor: the NF3-hybrid's per-expert GPTQ used **~23k routed tokens per expert** on this 753B family — our 145-sample mix is ~10× short of that; the tool reports median/p10 routed tokens per expert against this reference (reachability plateau ≠ sufficiency).

### 1a. Scan code entropy (student-only, minutes — decides whether 1b is worth the disk)

`python -m alis_dwq.code_entropy --model <student> --save entropy.npz` recovers every tensor's code histogram (through `mx.dequantize`, no source needed) and reports **effective vs nominal bits**: a group whose min-max grid got stretched by one outlier keeps most weights in 1–2 interior codes — nominal 2 bits, effective 1.x. Low-utilization tensors are exactly where `clip_quantize` pays off, so run this before hauling a multi-hundred-GB source onto the box; per-layer effective-bpw also feeds bit reallocation, and anchor-mass tracks the super-weight sites. (Lens borrowed from [lossless BF16 compression](https://github.com/brianbell-x/weight-compression): measure the information the bits actually carry.)

Add `--per-expert` (experimental) to break expert stacks out along axis 0 (bits ≤ 4): per-expert code entropy is the student-only proxy for **per-expert quantization damage** — the selection criterion the NF3-hybrid v3.6 validated after frequency saliency failed on this family (see §0). Outputs the most-damaged (tensor, expert) list: bit-promotion targets, or routing targets for `gen_calib`.

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

**Experimental — outlier-scattering FFN permutation:** `--permute-ffn` attacks the stretched-grid pathology at the cause instead of the symptom. The FFN hidden axis is residual-free, so permuting gate/up **output** channels together with down_proj **input** channels (per-expert in stacks — the permutation closes inside each expert) is *mathematically identity* at zero runtime cost, while re-dealing which channels share a quantization group: channels are sorted by magnitude and dealt round-robin so every group gets an even outlier share ([PeRQ](https://arxiv.org/abs/2601.22347): calibrated permutations recover most of the full-rotation benefit at 2-bit). Unlike rotations, values are untouched, so the super-weight/anchor lesson above is not violated. Float sources only (blocks with a quantized source, missing member, or ffn-axis bias are skipped whole; a planned block that later fails aborts the run rather than de-align). Perms are saved to `<out>/ffn_perms.npz` for audit. Gate on held-out PPL + `code_entropy` before/after.

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

**Experimental — LoRA error compensators:** `ALIS_DWQ_LORA_RANK=8` wraps every quantized module in a LoRA adapter ([Recover-LoRA](https://arxiv.org/abs/2606.04238) recovered 80–95% of 2-bit damage; [MiLo](https://arxiv.org/abs/2504.02658) is the MoE variant) and trains adapters alongside scales/biases under the same rounds/rollback. This adds the degree of freedom scales/biases (a per-group linear remap) fundamentally lack — the main lever left for the ~2.3 bpw floor builds, and it may reopen sharper teachers for low-bit students (the sweet-spot failure was a student-capacity limit). Adapters are saved to `ALIS_DWQ_ADAPTER_DIR` (default `alis_adapters/`) in mlx-lm's `--adapter-path` format; **the saved checkpoint stays stock** — wrappers are removed before mlx-lm writes it. Never fuse adapters into a quantized base (fusing requantizes = re-rounds the codes).

**Experimental — CKA drift monitor:** `ALIS_DWQ_CKA_MONITOR=1` reports per-round layerwise CKA between accepted states on a fixed valid batch (diagnostic only, ~2 extra forwards/round). Rationale: [CKA-QAD](https://arxiv.org/abs/2606.05682) showed KL-only distillation can degrade internal geometry *while outputs match* — the same valid-vs-held-out inversion we measured twice (8-bit-teacher 2.56 bpw arm; router-KD arm). A round that valid-KL accepts but that craters one layer's CKA is the signature to investigate before shipping. Reading tip: weight **early-layer** drift heaviest — routing is largely decided by the upstream residual stream ([pulsar](https://github.com/giannisanni/pulsar) prefetches by running layer N+1's router on layer N's input, and it works), so early-layer CKA drift is a routing-path damage signal for *every* layer downstream; it also explains why retraining router weights alone (router-KD) bought nothing.

**Experimental — router-KD:** add `ALIS_DWQ_TRAIN_ROUTERS=1` to also train each round's router gate weights (matched by `(gate|router)$`, e.g. `MoEGate.weight`) alongside scales/biases. Rationale: quantized experts change outputs, so the original routing is no longer optimal — [0xSero's REAP 504B](https://huggingface.co/0xSero/GLM-5.2-504B) recovered a 34%-expert prune to eval parity by distilling *only* the gates (0.016% of params); quantization damage has the same routing-shift component. Routers are tiny, so memory cost is nil, and the per-round rollback still gates every change. Judge on held-out PPL/KL as usual — routing changes which experts fire, so training loss alone can mislead (see the sweet-spot note below).

### 4. Evaluate on language slices, not just overall

```bash
python -m alis_dwq.eval_kld --model <teacher> --save-ref ref.npz   # once
python -m alis_dwq.eval_kld --model <out> --ref ref.npz            # per build
```

Reports KL and top-1 flip per EN/code/ZH third (drop your own corpora in `data/`). Overall averages hide exactly the damage you're trying to fix.

**Experimental — degeneration probe:** add `--loop-probe 256` to greedy-generate 256 tokens per slice and report distinct-4gram ratio + tail-cycle detection. Motivation: REAP's 504B held *eval parity* while its loop rate doubled (3.6%→7.2%, z≈5) — aggregate scores hide behavioral degeneration the same way they hide slice damage. Cheap to run on every build; treat a new `LOOPED` on a previously clean slice as a ship blocker until investigated.

**Experimental — KV-tolerance probe:** add `--kv-probe 4` to also report the self-KL a 4-bit KV cache induces vs this model's own FP16-KV run (per slice; DSA/recurrent layer caches without a quantized form stay FP16 and are counted). Motivation: Bonsai 27B measured low-bit-weight models absorbing 4-bit KV **12–95× better** than FP16/Q4-weight builds — if DWQ'd students inherit that tolerance, long-context memory drops ~4× as a free deployment win worth stating on the model card.

**Gate caveat — selective collapse hides in short-form metrics.** Sub-4-bit damage concentrates in *long* reasoning chains: Bonsai's cross-family data shows IQ2_XXS holding MMLU (88.9) while AIME collapses 93→57. Our KL/flip slices and the loop probe are short-form; a build can pass both and still have lost long-CoT. Until a long-form reasoning probe ships here, treat KL/flip parity on an aggressive build as necessary, not sufficient — spot-check a few long thinking-mode problems before shipping a new low.

## Results

| Case | Student | Setup | Outcome |
|---|---|---|---|
| [GLM-5.2 2.56 bpw](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-2.56bpw) (shipped: `main` carries the DWQ weights, pre-DWQ at `pre-dwq`) | 745B MoE student (78 quant layers) | K=8, seq 512, lr 1e-6, 45% ZH mix | vs 4.5-bpw ref: overall KL **0.655→0.379 (−42%)**, ZH **0.987→0.562 (−43%)**, flips 24.4%→15.9%; peak 334 GB (full-layer training OOMs); 10/10 rounds accepted, ~5 h |
| Hy3 T128 2.375 bpw | 87.6 GB / 295B MoE | full-layer DWQ (fits), 45% ZH mix | ZH-concentrated damage recovered enough to ship; see model card |
| [GLM-5.2 3.5 bpw](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw), **8-bit teacher** (shipped: `main`; prior retune at `dwq-4.5teacher`) | 310 GB / 745B MoE | 790 GB teacher → **distributed 2-box dump** (§2b), then single-node K=6 | strided PPL wikitext **2.851→2.814 (−1.3%, significant)** over the 4.5 bpw-teacher retune; code within noise; loss 0.178→0.138 — see [examples/glm-5.2-8bit-teacher](examples/glm-5.2-8bit-teacher/README.md) |
| GLM-5.2 2.56 bpw, 8-bit teacher (**not shipped** — negative result) | 227 GB / 745B MoE | same targets reused, K=8 | DWQ loss −35% but held-out PPL slightly *worse* (+1.3% wikitext) — the sweet-spot lesson above; `main` keeps the 4.5 bpw-teacher weights |
| GLM-5.2 **E1 floor spike**: 2-bit/gs128 experts, 2.32 bpw, teacher **A/B** ([case study](examples/glm-5.2-e1-floor-spike/README.md)) | 215.8 GB / 745B MoE, from-Q8 | both arms from the same raw student, K=6 | 4.5-t arm won *every* held-out metric (wikitext 4.036 vs 4.087; −14.3% vs raw); 8-bit arm's valid dropped more yet lost — sweet-spot 3rd point |
| GLM-5.2 **E1r** (shipped 2.3 bpw `main`): anchor-guarded **clip-search + DWQ** stacked ([part 2](examples/glm-5.2-e1-floor-spike/README.md)) | 215.8 GB, from-Q8 | clip slack 1.1 → same DWQ recipe | raw 4.711 → clip 4.424 → DWQ **3.899** (−17.2%); clip's gain survives DWQ (orthogonal passes). +3.3% wikitext vs the 2.56 build at −26.6 GB; loop-probe clean |
| GLM-5.2 **2.56 bpw rework** (shipped `main`; prior DWQ at `dwq-noclip`): the E1r clip+DWQ stack applied to the mid-low sibling | 242.4 GB, from-Q8 | clip slack 1.1 → same DWQ recipe (4.5-t targets reused) | raw 4.34 → clip 4.174 (−3.8%) → DWQ **3.698** (−14.8% cumulative; **−2.0%** vs its no-clip DWQ 3.774); code 2.054, tulu flat, loop-probe clean. **Clip's raw gain only ~half-survived DWQ here** (vs near-full on E1r): orthogonality is partial and shrinks as the no-clip baseline improves. KL vs the 4.5 bpw sibling *rose* +10% while every real-corpus PPL improved — see the reference-lattice note below |

## Hard-won operational notes

- **Gate disk as well as RAM between sequential arms.** Two ~330 GB DWQ outputs back-to-back filled the volume to 100% and the second arm died at shard-write time ("Unable to write N bytes") — after 6 h of training layerwise cannot resume, so the whole arm re-ran. Check free space >= output size + margin before every arm, not just at campaign start.
- **Gate every big load**: wait until `free+inactive > model + 60 GB` *and stable* — a "done" log line doesn't mean the previous process released its memory, and loads launched into a reclaim window get jetsam-killed silently.
- **Never overlap a 100 GB-class load with heavy disk I/O** (uploads, mass writes): the load wedges with rss/avail frozen. Kill (-9, then verify with pgrep — zombies squat memory), let it reclaim, relaunch on a quiet box.
- **Getting the teacher shards onto each box (distributed dump):** Xet-backed HF repos don't parallelize with `aria2c` — the `resolve/main` redirect is a byte-range-signed CAS URL, so multi-connection splits `403`. Use `snapshot_download` with `HF_XET_HIGH_PERFORMANCE=1` and per-shard `allow_patterns` to hand each box only its half. Xet can also silently wedge mid-repo (shard count frozen, zero net-in) — wrap it in a watchdog that kills + resumes when progress stalls. If one box's uplink is slower, have the faster box pull its peer's remaining shards and ship them over the local (TB/LAN) link.
- Learning rate transfers poorly across student sizes: 1e-5 was fine for an 87 GB student and diverged on a 242 GB one. Start at mlx-lm's default 1e-6; the per-round rollback makes over-stepping cheap.
- If your checkpoint carries extra layers your runtime remaps (e.g. an MTP head), you have two working paths. (a) Strip them for training with a hardlinked variant and re-attach after — **never edit a hardlinked file in place**; replace it (`os.replace`), or you rewrite the original through the shared inode. (b) Train with them attached (the model's `sanitize` remaps them at load) — but then the *saved* checkpoint carries **module-named keys** (e.g. `mtp.layer.*`) instead of the checkpoint convention (`model.layers.<N>.*`), and strict loading breaks with "N parameters not in model". Fix by inverse-renaming those keys in the affected shards + index; **verify the inverse map by asserting the renamed keyset equals the original checkpoint's** before shipping, and write via temp-file + `os.replace` (see the lazy-mmap note above). If a conventional-name copy of the module already exists (e.g. a previously shipped sidecar shard), a third path is cleanest: **strip** the module-named tensors from the base shards and re-attach the sidecar — then assert the stripped tensors **byte-match** the sidecar under the rename map. That equality check is a free integrity oracle: it simultaneously validates the map and proves the module passed through the whole quant/retune pipeline untouched.
- **Stock `mlx_lm.perplexity` / `mlx_lm.evaluate` never set the wired limit** — only generate/server do. On a 100 GB-class model that is a silent ~10× slowdown (throughput thrashing, not an error). Wrap them: `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` after load, then call the module's `main()`. (This repo's `eval_kld` already does.) Custom eval scripts loading an MTP-bearing checkpoint also need `load_model(..., strict=False)` unless the runtime remaps the head.
- **Choosing K (layers per round)**: backward memory scales with K, so pick the largest K that fits. Anchors from the 745B family on a 512 GB box: a 227 GB student at K=8 peaked ~307 GB; a 310 GB student at K=6 peaked ~389 GB. Start high, drop K on OOM — rounds get proportionally cheaper, total wall-clock barely moves.
- **A REVERTED round near the end is the gate working, not a failure.** Both shipped GLM retunes ended with one late-round rollback (best already reached); treat consecutive reverts as the natural stopping signal.
- When re-attaching a shard, **name it `model-*.safetensors`**: mlx-lm's loader collects shards by that glob, not by the index — a shard named anything else silently never loads and you get "missing parameters" for exactly its keys.
- **`mx.load` is lazy (mmap) — never `save_safetensors` back to the path you loaded from.** The save truncates the file *before* the lazy view is read, so you write zeros over your own data. `mx.eval` the new arrays first, or write a temp file and `os.replace`. (We zeroed a whole target dump post-processing it this way — and the real fix was to not post-process at all: get the shape right in the forward, per §2b.)
- **Verify a dump is non-zero and the right shape before you reclaim the teacher.** A distributed dump is your only local copy of the teacher's logits; the teacher weights are the cheap thing to reconstruct, the dump is not. After the corruption above we'd already deleted one box's teacher half — re-dumping meant re-pulling 400 GB. Keep the *peer* box's half until the targets pass a `min/max != 0` + shape check; and to free a box for the next stage, move the student to the peer over the local link (minutes) instead of deleting and re-pulling from HF (an hour).
- **Xet can also wedge after a device-level I/O interruption, not just mid-repo.** After a USB enclosure dropped and re-enumerated mid-download, `hf download` sat alive-but-dead — ~0% CPU, trickle network, zero file writes — because the background writer thread had died (`Internal Writer Error: Background writer channel closed`) and the main thread waits forever. It reproduced twice in one day, once at 57/59 shards. The stall watchdog that catches it: downloader process alive **and** no file in the target tree modified for 10 minutes → kill and relaunch (resume is safe; completed files are kept).
- **VL MoE recipe note — check whether the shared expert is *packed* into the routed bank.** mlx-vlm's MiniMax-M3 packs the shared expert as expert #128 inside the same SwitchLinear when `shared_intermediate_size == intermediate_size`, welding the one tensor that sees **100% of tokens** (a routed expert sees ~3% at top-4/128) to the routed experts' bit-width. Unpacking it and holding it at 8-bit while routed experts drop to 3-bit cost +0.4% size and needs only a config-gated layout switch ([mlx-vlm#1544](https://github.com/Blaizzy/mlx-vlm/pull/1544); packed vs unpacked forward matches to 7.75e-07). The same audit applies before DWQ: a packed shared expert can't be given its own learning treatment either.
- **Evaluating a thinking model: budget `max_tokens` for the thinking, and grade the tail.** A needle-retrieval probe with `max_tokens=30` "failed" 0/3 on an 8-bit reference — the model spent the whole budget inside `<mm:think>` and never emitted the answer; at 400 tokens it was 3/3 at every depth. Harness artifact, looks exactly like model damage.
- **Shipping a 200 GB-class build to HF: mind the private cap and the commit budget.** Private repos have a storage limit a big staging blows through mid-upload ("Private repository storage limit reached") — if the plan is stage-private-then-flip, either budget for it or flip public *first* and push the model card as a small commit before the weights, so the public page is never incoherent while shards land. Separately, `upload_large_folder` spends the 128-commits/hour budget on shard batches and grinds to a crawl with rate-limit retries; a single `upload_folder` commit avoids the ceiling entirely.
- **HF's card renderer pairs *single* tildes into strikethrough.** Two "\~90 GB … \~40 K" approximations in one paragraph render as a struck-through span (and can break `**bold**` parsing mid-pair, leaking literal asterisks). GitHub only strikes on double `~~`, so a card that previews fine on GitHub still corrupts on HF — escape every literal tilde (`\~`) in card prose.
- **Clearing `~/.cache/huggingface` deletes your auth token with it.** Public downloads keep working afterwards, so the loss surfaces only at the next upload as "Invalid username or password", hours later. Preserve the `token` file when reclaiming cache space (or expect to re-run `hf auth login`).
- **Publishing branch-per-quant on HF: branches inherit `main`'s files at creation.** Create branch → upload build = the branch now carries *both* builds' shards (ours had 39 stale + 87 real; snapshot downloads grow ~190 GB). `delete_files` the inherited shard names on every non-main branch after upload, and verify per-revision shard counts via the API.
- **Swapping a build in-place with `upload_folder` leaves the *outgoing* shard set behind whenever shard naming changes.** `upload_folder` adds/overwrites, never deletes: a 3.5 bpw `main` silently carried 153 shards / 661 GB for four days (an older build's `model-*-of-00076` set next to the new `-of-00077` set). The repo still *works* — the index resolves to the new shards — but `snapshot_download` pulls both builds. Same-name uploads mask the issue (a later same-sharding build overwrites fully in place), so it surfaces only when shard count/naming shifts between builds. Detect with an index-vs-tree audit on **every revision**: `set(tree *.safetensors) - set(index weight_map.values())` must be empty per branch (branch snapshots inherit `main`'s orphans at creation — cf. the branch-inheritance note above). Prevent by making swaps **atomic**: one `create_commit` carrying the new files *plus* `CommitOperationDelete` ops for the outgoing set.
- **HF sidebar param count for quantized repos comes from `model.safetensors.index.json` metadata `total_parameters` — and `mlx_vlm.convert` doesn't write it** (`mlx_lm.convert` does). Without it the Hub counts packed-U32 elements as parameters and shows ~1/8 of the truth (427B displayed as "56B"). Inject the exact count (summed from the bf16 source shard headers) before upload; verify with `api/models/<repo>?expand[]=safetensors`.
- **Never re-quantize a dequantized low-bit lattice with a coarser grid.** nvfp4/≤4-bit-affine dequants sit on a coarse lattice; a coarser min-max affine grid over lattice values rounds with correlated bias (continuous sources round pseudo-randomly), and the bias accumulates through activations — the model dies while per-tensor MSE, kernel outputs, and every artifact check look *better*. Paradox signature: the more groups you clip (off-lattice rescale), the more alive the model. Sources for requantization must be bf16/fp16/Q8-class.
- **A mean weight metric cannot see super-weights.** The few largest-magnitude weights per tensor do disproportionate work, and min-max affine grids protect them for free (they *are* the grid ends). Any transform that trades their fidelity for interior precision (MSE-optimal clipping at 2-bit: top-0.01% weights ×4 error, mean −24%) kills the model. Gate weight-space transforms on per-group max-error and held-out PPL, never on mean reconstruction metrics alone.
- **KL against a *quantized sibling* reference penalizes de-resonance — anchor quality judgments to real-text PPL.** A 2.56 bpw build re-derived from a Q8 source (clip) improved every held-out corpus (wikitext −2.0%, code −0.7%, tulu flat) yet its KL vs the 4.5 bpw sibling reference *rose* +10% (flips +1.5 pt): the previous build shared the reference's 4-bit-lattice ancestry, so it sat artificially close to it. Quantized-sibling KL is only comparable *within* one source family; across source classes it mistakes de-resonance for damage.
- **Hash the *resolved* file, not the HF `raw/` endpoint, when verifying tokenizer identity across repos.** `raw/` serves the ~133-byte LFS pointer for LFS-tracked files, so the hash "mismatches" even when the tokenizers are byte-identical; download the real file and `shasum` that. `tokenizer_config.json` may also differ by benign client keys (e.g. `local_files_only`) — identity lives in `tokenizer.json`.
- **Start the decode timer after prefill, and keep Time Machine away from scratch trees.** Timing a generator from its first iteration folds prefill into "decode" and manufactures a fake long-context decode cliff (we chased one for a day). Separately, an unexcluded few-hundred-GB working tree gets picked up by Time Machine mid-run and strangles disk I/O — `tmutil addexclusion <dir>` every big build/download directory at creation.
- **A sandboxed autonomous agent cannot see the GPU.** macOS Seatbelt profiles (e.g. Codex CLI `workspace-write`) block Metal device access: MLX fails with `[metal::load_device] No Metal device available`, and detached/nohup children inherit the sandbox. Probe Metal *before* any download/build step in agent-run pipelines, and give GPU-touching agent runs full (unsandboxed) access — probe-first discipline turned this from a wasted 790 GB download into a zero-cost stop.
- **A sharper teacher is not always better — it depends on the student's bit-width, and only held-out PPL tells you.** Re-tuning a 3.5 bpw GLM-5.2 student against an 8-bit teacher (vs a 4.5 bpw one) improved held-out strided PPL a real, significant **−1.3%** (wikitext). The *same* 8-bit teacher on the more-quantized **2.56 bpw** student made it slightly **worse** (+1.3%, within noise) — even though its DWQ-vs-teacher loss dropped **35%**. At ~2 bpw the student can't represent the finer target distribution, so chasing it overfits the 145-sample calibration set without transferring to the eval corpora. There's a **teacher-precision / student-capacity sweet spot**: a mid-bit student benefits from a sharper teacher, an aggressively-quantized one does not. Always judge on held-out PPL, never on the training/validation loss alone (which will happily drop toward any teacher).

## Absorbed from upstream, not yet validated here

Ideas worth stealing from [omlx's oQ pipeline](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) (reviewed 2026-07-10). None of these carry our own measurements yet — treat as roadmap, not method:

- **Batched-expert GPTQ.** All routed experts in a layer receive the same input hidden states, so they share one Hessian — oQ runs GPTQ over the whole expert bank at once (they report ~15× vs per-expert). Complementary to DWQ: GPTQ is a cheap local error-compensation pass (minutes, forward-only calibration) where DWQ is end-to-end distillation (hours). Likely first target: a mid-2-bpw-class student *before* DWQ, to see whether the passes stack. **Code-audit caveat (omlx `d42528a`): the shipped `oq.py` contains no GPTQ/Hessian implementation — its `enhanced=True` path is imatrix-weighted affine clipping ("oQe").** Verify upstream code, not the docs, before planning around "oQ+ GPTQ". Existence proof at our scale: the [NF3-hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid) ran per-expert GPTQ (Hessians from ~23k routed tokens/expert) across all 753B expert weights — the pass is tractable; only the MLX implementation is missing.
- **Activation-weighted affine scale fitting.** Fit per-group `scales`/`biases` by activation-weighted least squares instead of min-max — same checkpoint format, better values, stock loading. Same compatibility trick DWQ itself relies on.
- **Normalized layer sensitivity for bit allocation.** `MSE(float, quant) / mean(float²)` per layer — the normalization keeps late layers from looking artificially sensitive under residual accumulation. Our measured-not-heuristic promotion lesson (Hy3: the sensitive band was mid-stack, not early/late) says automate this rather than trust priors.
- **Expert-coverage tracking of the calibration set.** oQ tracks which experts the calibration data actually activates. We have never measured what fraction of a 128–256-expert bank our 145-sample mix reaches per layer — undersampled experts get weak distillation signal in DWQ too. The diagnostic now ships here as `alis_dwq/expert_traffic.py` (§0); measurements on our own mixes still pending.
- **Stock mlx-lm AWQ, cascaded.** [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) ships an AWQ pass (activation-aware per-channel scaling + clipping, salient-channel-protecting — consistent with the anchor lesson) and documents cascading it with dynamic quant and DWQ. We have never measured AWQ → clip_quantize → DWQ; channel-level scaling and group-level clip search are plausibly complementary, and the experiment costs zero code.
- **Single-box oversized-teacher dump via mmap streaming.** [pulsar](https://github.com/giannisanni/pulsar) proves 97%-on-disk MoE forwards are workable (743B on dual 16 GB GPUs). Our teacher dump is forward-only at batch 1, and prefill reuses experts across the whole sequence, so a 790 GB Q8 teacher on one 512 GB box ≈ one full-model scan per sample through MLX's lazy mmap + page cache — roughly half a day for 145 samples *if* macOS eviction behaves under the wired limit. Would retire the entire 2-box distributed-dump trap list (§2b). Unvalidated; probe with a small layer-range forward before committing a run.
- **Lossless bf16 archival, if you must keep a bf16 master.** [brianbell-x/weight-compression](https://github.com/brianbell-x/weight-compression) packs BF16 sign+exponent into a 4-bit table code — bit-exact, ~30% smaller, verified on GLM-5.2 (1,403→980 GB). Irrelevant to our shipped quants (packed low-bit ints have no BF16 mass), and for a `clip_quantize --source` the validated Q8 dump is smaller (~790 GB) and loads natively — this only wins when losslessness itself is the requirement.

Caveat on their headline numbers (2-bit MMLU 64% vs mlx-lm's 14%): the baseline is *uniform* mlx-lm quantization. A sensitivity-graded mixed-precision recipe already captures much of that gap; the marginal value here is the error-compensation and scale-fitting passes on top of a good recipe, and it needs to be measured as such.

## Floor notes: how low the MLX container goes (2026-07-10)

Research and probe notes from scoping a sub-2.56-bpw GLM-5.2 build. The build ran on 2026-07-11: experts 2-bit/gs128 → 215.8 GB dec / **2.3225 bpw measured**, raw wikitext 4.711, after DWQ (the 4.5-bpw teacher won the A/B) **4.036** — full numbers in the [E1 floor-spike case study](examples/glm-5.2-e1-floor-spike/README.md).

- **The MLX affine floor is 2-bit** (mlx 0.31.2 probe: supported bits `{2,3,4,5,6,8}`; `bits=1` raises). 2-bit/gs128 is the lowest effective rate: (128·2+32)/128 = **2.25 bpw**. Probed end-to-end at gs128 — `quantize`/`dequantize`/`quantized_matmul` *and* gradients w.r.t. `scales`/`biases` — so layerwise DWQ supports a mixed-group-size student out of the box.
- **Ternary ("BitNet-style") inside MLX is dominated in bytes — but no longer in method.** Three levels are a subset of the 2-bit affine grid at identical storage, so ternary only pays in a genuinely sub-2-bit container (GGUF `IQ1_*`/`TQ*`; mainline llama.cpp now also carries Bonsai's `Q2_0` g64), which MLX doesn't have — and [PrismML's own MLX packs](https://huggingface.co/collections/prism-ml/bonsai-27b) pay exactly this tax (8.49 GB vs 7.17 GB GGUF). Conversion-by-training ([BitNet Distillation](https://arxiv.org/abs/2510.13998)) is demonstrated at 0.6–4 B; pure-PTQ ternary literature topped out at 70 B dense with heavy cost ([TWLA](https://arxiv.org/abs/2606.13054): +44% wikitext at W1.58; [PTQTP](https://arxiv.org/abs/2509.16989) is ~4 bpw effective despite the name). ***Counterexample (2026-07)***: [Bonsai 27B](https://github.com/PrismML-Eng/Bonsai-demo) claims post-training ternary at 1.71 bpw holding 94.6% of FP16 (and 1.125 bpw at 89.5%) on a dense 27B hybrid — method proprietary (Caltech/Hassibi — the Optimal Brain Surgeon author, so a curvature-compensation lineage is the prior), evidence benchmark-only with sampling-temp asymmetry, MoE-745B transfer unknown. Weights are Apache-2.0 and the source model is public, so the claim and the method class are both checkable: see [examples/bonsai-27b-audit](examples/bonsai-27b-audit/README.md).
- **The practical floor converges across stacks.** [Unsloth's GLM-5.2 dynamic "1-bit"](https://unsloth.ai/docs/models/glm-5.2) (UD-IQ1_S GGUF) ships at **223 GB ≈ 2.4 bpw effective** for the same 745 B model — the size class a sensitivity-graded MLX 2.56-bpw build already occupies — while the same team's R1-671B went to 131 GB (1.56 bpw). The floor is model-dependent; below that line the fight is quality-per-byte (recipe + error-compensation passes + DWQ), not container size. *Update 2026-07-14*: [antirez's ds4-recipe GGUF](https://huggingface.co/antirez/glm-5.2-gguf) ships GLM-5.2 at **197 GB (~2.1 bpw effective)** — routed experts ~2-bit (IQ-class up/gate + Q2_K down, imatrix-weighted), everything decision-making (attention, router, shared experts, embeddings) at Q8. The "nobody under ~220 GB" line is stale; the open question is quality-per-byte vs our E1 (215.8 GB) — see [examples/glm-5.2-ds4-vs-alis](examples/glm-5.2-ds4-vs-alis/README.md) for the head-to-head protocol.
- **Ship the traffic census with the quant.** pulsar-class streaming engines warm their expert cache from a popularity census measured on first run (`.gguf.warm` sidecar). `expert_traffic --save traffic.npz` produces a better one (per-language slices) as a build byproduct — attach it to HF releases so streaming users skip the census, and state the flat-routing caveat (top-64 = 39–57% on GLM-5.2, so hit-rate expectations should be sized accordingly).

## Experimental features & rollback (v0.1 baseline)

Three additions landed 2026-07-12, motivated by REAP / router-KD (0xSero GLM-5.2-504B). **All are opt-in and default-off — the default pipeline is byte-identical to the pre-change baseline**, preserved as branch [`backup/v0.1-pre-router-kd`](https://github.com/avlp12/alis-dwq/tree/backup/v0.1-pre-router-kd) (commit `27863a7`). Pin that branch if you need the exact prior behavior. Every experimental path announces itself with an `[EXPERIMENTAL]` banner on stderr, so any session/model log shows at a glance whether a build used them:

| Flag | Tool | What it adds | Off = |
|---|---|---|---|
| `ALIS_DWQ_TRAIN_ROUTERS=1` | DWQ (§3) | router gates train with scales/biases (router-KD) | scales/biases only |
| `--norms` | `expert_traffic` (§0) | expert output-norm saliency (REAP proxy), busy-but-weak report | frequency only |
| `--loop-probe N` | `eval_kld` (§4) | greedy degeneration probe per slice | KL/flip only |
| `ALIS_DWQ_LORA_RANK=r` | DWQ (§3) | LoRA error compensators train with scales/biases; adapters saved separately | scales/biases only |
| `ALIS_DWQ_CKA_MONITOR=1` | DWQ (§3) | per-round layerwise CKA drift report (diagnostic) | no report |
| *(new tool)* | `code_entropy` (§1a) | effective-bpw / clip pre-scan from the student alone (`--per-expert`: damage-proxy saliency) | — |
| *(new tool)* | `gen_calib` (§1) | synthetic calibration mix with expert-coverage stopping (`--chat-template`: serve-distribution calibration) | curated jsonl |
| `--permute-ffn` | `clip_quantize` (§1b) | outlier-scattering FFN-hidden permutation before requantization | min-max grouping as-is |
| `--kv-probe BITS` | `eval_kld` (§4) | self-KL of a quantized KV cache vs own FP16-KV run | no KV probe |
| *(new tool)* | `weight_forensics` | method-class fingerprints of a transformed model vs its original (Bonsai audit Tier 2) | — |

First on-device validation (2026-07-13, GLM-5.2 3-bit-expert student, 8-bit teacher, K=6): **router-KD is harmless but did not help** — same-teacher valid loss edged the baseline (0.1357 vs 0.1365) yet held-out wikitext lost by a hair (2.7820 vs 2.7774, well inside the CIs). The valid-vs-held-out inversion pattern strikes again; the baseline shipped. `--norms` and `--loop-probe` are validated in production use (see the E1 case study and the measured-reference note in §0).

The 2026-07-13 batch (LoRA compensators, CKA monitor, `code_entropy`, `gen_calib`) carries no on-device measurements yet — validate on held-out PPL/KL before putting any of it in a shipping recipe. The LoRA adapter save/load round-trip in particular must be verified against `mlx_lm.load(..., adapter_path=...)` on-device before trusting a long run to it, and the CKA monitor exists precisely because the inversion above keeps recurring — use them together.

## License

MIT
