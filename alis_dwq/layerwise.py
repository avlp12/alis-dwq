"""Layerwise DWQ as a monkeypatch over stock mlx-lm (>= 0.31).

Works without any mlx-lm fork: importing this module replaces
``mlx_lm.quant.dwq.dwq_quantize`` with a rounds-based variant that trains at
most ``ALIS_DWQ_LAYERS_PER_ROUND`` layers' quantization scales/biases at a
time (deepest first, per-round validation with rollback). Bounds the
quantized-matmul backward memory so very large students (100GB+ MoE) can be
DWQ'd on a single machine.

Once mlx-lm ships ``--layers-per-round`` natively, prefer that flag; this
module keeps working either way.

Usage:
    ALIS_DWQ_LAYERS_PER_ROUND=8 python -m alis_dwq.run \
        --model <tokenizer-source> --quantized-model <student> \
        --target-dir <targets> --mlx-path <out> ...
"""
import os
import re
import sys
import time

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

import mlx_lm.quant.dwq as D
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches

from . import gate
from . import losses

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# router/gate modules producing expert logits ("mlp.gate", "mlp.router"); the
# $ anchor excludes projection weights like "gate_proj" / "switch_mlp.gate_proj"
_ROUTER_RE = re.compile(r"(?:^|\.)(?:gate|router)$")


def _is_router(name, m):
    return (_ROUTER_RE.search(name) is not None
            and getattr(m, "weight", None) is not None
            and not _is_quantized(m))


def _is_quantized(m):
    return (
        hasattr(m, "bits")
        and hasattr(m, "group_size")
        and getattr(m, "mode", "affine") == "affine"
        and m.bits < 8
    )


def _is_lora(m):
    return getattr(m, "lora_a", None) is not None


def _wrap_lora(model, rank):
    """Wrap every quantized module in a LoRA adapter (Recover-LoRA / MiLo
    style error compensator). Returns the per-layer key suffixes wrapped."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    sufs = set()

    def visit(name, m):
        if _is_quantized(m):
            mm = _LAYER_RE.search(name)
            if mm:
                sufs.add(name.split(f"layers.{mm.group(1)}.", 1)[1])

    model.apply_to_modules(visit)
    if not sufs:
        raise RuntimeError("no quantized modules found to wrap")
    cfg = {"rank": rank, "scale": 20.0, "dropout": 0.0, "keys": sorted(sufs)}
    linear_to_lora_layers(model, len(model.layers), cfg)
    return cfg


def _save_adapters(model, cfg, out_dir):
    """Adapter file + config in mlx-lm's load_adapters format; the base
    checkpoint mlx-lm saves afterwards stays stock (wrappers are removed
    by the caller)."""
    model.freeze()
    model.apply_to_modules(
        lambda n, m: m.unfreeze(keys=[k for k in ("lora_a", "lora_b")
                                      if getattr(m, k, None) is not None],
                                recurse=False) if _is_lora(m) else None)
    flat = tree_flatten(model.trainable_parameters())
    model.freeze()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(flat))
    json.dump(
        {"fine_tune_type": "lora", "num_layers": len(model.layers),
         "lora_parameters": cfg},
        open(out / "adapter_config.json", "w"), indent=2)
    print(f"[alis-dwq] {len(flat)} adapter tensors -> {out} "
          "(load with: mlx_lm ... --adapter-path)", file=sys.stderr)


def _unwrap_lora(model):
    swaps = []
    model.apply_to_modules(
        lambda n, m: swaps.append((n, m.linear)) if _is_lora(m) and hasattr(m, "linear") else None)
    if swaps:
        model.update_modules(tree_unflatten(swaps))
    return len(swaps)


def _capture_hiddens(model, batch):
    """One forward; per-block output as (tokens, dim) float32 numpy arrays.
    Class-level __call__ patch (special methods resolve on the type), keyed
    by block identity so any decoder-layer signature passes through."""
    import numpy as np
    blocks = list(model.layers)
    outs = [None] * len(blocks)
    idx = {id(b): i for i, b in enumerate(blocks)}
    origs = {}
    try:
        for cls in {type(b) for b in blocks}:
            orig = cls.__call__

            def call(self, *a, _orig=orig, **k):
                y = _orig(self, *a, **k)
                j = idx.get(id(self))
                if j is not None:
                    outs[j] = y[0] if isinstance(y, tuple) else y
                return y

            origs[cls] = orig
            cls.__call__ = call
        mx.eval(model(batch))
    finally:
        for cls, orig in origs.items():
            cls.__call__ = orig
    return [None if o is None else
            np.array(o.astype(mx.float32), copy=True).reshape(-1, o.shape[-1])
            for o in outs]


def _cka(x, y):
    """Linear CKA between (tokens, dim) feature matrices (token-space Grams)."""
    import numpy as np
    x = (x - x.mean(0)).astype(np.float64)
    y = (y - y.mean(0)).astype(np.float64)
    kx, ky = x @ x.T, y @ y.T
    denom = np.linalg.norm(kx) * np.linalg.norm(ky)
    return float((kx * ky).sum() / denom) if denom > 0 else float("nan")


def layerwise_dwq_quantize(
    model, target_fn, opt, train_data, valid_data, batch_size, max_seq_length,
    seed, dtype: mx.Dtype = mx.bfloat16, gradient_checkpoint: bool = False,
    temperature: float = 2.0, **_ignored,
):
    K = int(os.environ.get("ALIS_DWQ_LAYERS_PER_ROUND", "8"))
    train_routers = os.environ.get("ALIS_DWQ_TRAIN_ROUTERS", "") == "1"
    lora_rank = int(os.environ.get("ALIS_DWQ_LORA_RANK", "0") or 0)
    cka_mon = os.environ.get("ALIS_DWQ_CKA_MONITOR", "") == "1"
    loss_kind, loss_topk = losses.parse(os.environ.get("ALIS_DWQ_LOSS", "kl"))
    if loss_kind != "kl" or loss_topk:
        losses.banner(loss_kind, loss_topk)
    need_labels = loss_kind == "cakld"

    # Opt-in per-slice valid gate (see alis_dwq.gate). Env pair unset ->
    # gate_cfg is None and everything below behaves exactly as before.
    # Any inconsistency raises here, before the first training step.
    gate_cfg = gate.load_gate(batch_size, max_seq_length, seed,
                              len(valid_data), target_fn)
    gate_entries, en_eps = gate_cfg if gate_cfg is not None else (None, None)
    if gate_entries is not None:
        print(f"[alis-dwq][EXPERIMENTAL] per-slice valid gate: "
              f"manifest={os.environ[gate.MANIFEST_ENV]} "
              f"EN eps={en_eps} — strict overall gate + EN ceiling "
              f"({sum(1 for e in gate_entries if e['slice'] == 'EN')} EN / "
              f"{len(gate_entries)} valid ordinals, hashes verified). Unset "
              f"{gate.MANIFEST_ENV}/{gate.EPS_ENV} for legacy behavior.",
              file=sys.stderr)

    lora_cfg = None
    if lora_rank > 0:
        print(f"[alis-dwq][EXPERIMENTAL] ALIS_DWQ_LORA_RANK={lora_rank}: LoRA "
              "error compensators (Recover-LoRA/MiLo style) train alongside "
              "scales/biases; adapters are saved SEPARATELY (base checkpoint "
              "stays stock) — load with --adapter-path and verify on-device. "
              "Unset the env var (or pin branch backup/v0.1-pre-router-kd) for "
              "scales/biases-only DWQ.", file=sys.stderr)
        try:
            lora_cfg = _wrap_lora(model, lora_rank)
            print(f"[alis-dwq] LoRA wrapped keys: {lora_cfg['keys']}", file=sys.stderr)
        except Exception as e:  # abort BEFORE training, not mid-run
            raise SystemExit(f"[alis-dwq] LoRA wrap failed: {e!r} — rerun without "
                             "ALIS_DWQ_LORA_RANK") from e
    if cka_mon:
        print("[alis-dwq][EXPERIMENTAL] ALIS_DWQ_CKA_MONITOR=1: per-round "
              "layerwise CKA drift report (diagnostic only, no behavior "
              "change) — targets the valid-vs-held-out inversion pattern.",
              file=sys.stderr)

    quant_layers, has_extras, router_names = set(), [False], []

    def scan(name, m):
        if _is_quantized(m):
            mm = _LAYER_RE.search(name)
            if mm:
                quant_layers.add(int(mm.group(1)))
            else:
                has_extras[0] = True
        elif train_routers and _is_router(name, m) and _LAYER_RE.search(name):
            router_names.append(name)

    model.apply_to_modules(scan)
    ordered = sorted(quant_layers, reverse=True)  # deepest first
    rounds = [ordered[i:i + K] for i in range(0, len(ordered), K)]
    print(f"[alis-dwq] {len(ordered)} quant layers -> {len(rounds)} rounds of {K}"
          f" (extras with round 1: {has_extras[0]})", file=sys.stderr)
    if train_routers:
        print(f"[alis-dwq][EXPERIMENTAL] ALIS_DWQ_TRAIN_ROUTERS=1: {len(router_names)} "
              "router gate modules will train alongside scales/biases (router-KD "
              "for quantization, after 0xSero's REAP recovery). Per-round rollback "
              "still applies. Unset the env var (or pin branch "
              "backup/v0.1-pre-router-kd) for the previous scales/biases-only "
              "behavior.", file=sys.stderr)
        if router_names:
            print(f"[alis-dwq] router modules: {router_names[0]} ... {router_names[-1]}",
                  file=sys.stderr)
        else:
            print("[alis-dwq][WARN] no router modules matched (pattern "
                  "'(gate|router)$' with a weight param) — flag has no effect",
                  file=sys.stderr)

    model.train()
    if gradient_checkpoint:
        grad_checkpoint(model.layers[0])

    scale = 1 / temperature

    def loss_fn(params, x, targets, lengths, labels=None):
        model.update(tree_map(lambda v: v.astype(dtype), params))
        logits = model(x)
        ids = None
        if isinstance(targets, tuple):
            targets, ids = targets
            logits = mx.take_along_axis(logits, ids, axis=-1)
        if loss_kind == "kl" and loss_topk is None:  # stock path, untouched
            per_tok = kl_div_loss(scale * logits, scale * targets)
        else:
            per_tok = losses.per_token_loss(loss_kind, loss_topk, logits,
                                            targets, scale, labels=labels,
                                            ids=ids)
        mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        return (mask * per_tok).sum() / ntoks, ntoks

    def validate(tag):
        if gate_entries is None:
            # legacy single-scalar path — byte-identical pre-gate behavior
            v_loss, v_tok = 0.0, 0
            params = model.trainable_parameters()
            for i, (batch, lengths) in enumerate(
                iterate_batches(valid_data, batch_size, max_seq_length, seed=seed)
            ):
                labels = batch[:, 1:] if need_labels else None
                batch = batch[:, :-1]
                targets = target_fn(batch, i, split="valid")
                mx.eval(targets)
                loss, ntoks = loss_fn(params, batch, targets, lengths, labels)
                mx.eval(loss, ntoks)
                v_tok += ntoks.item()
                v_loss += loss.item() * ntoks.item()
            loss = v_loss / v_tok
            print(f"[alis-dwq][valid] {tag}: {loss:.4f}", file=sys.stderr)
            return loss

        # gate path: verify per-ordinal input hash BEFORE trusting the
        # ordinal-bound target file, then accumulate token-weighted per-slice
        # losses (overall + one entry per manifest slice label)
        sums, toks = {"overall": 0.0}, {"overall": 0}
        params = model.trainable_parameters()
        seen = 0
        for i, (batch, lengths) in enumerate(
            iterate_batches(valid_data, batch_size, max_seq_length, seed=seed)
        ):
            labels = batch[:, 1:] if need_labels else None
            batch = batch[:, :-1]
            entry = gate_entries[i]
            gate.check_ordinal_input(i, entry, batch)
            targets = target_fn(batch, i, split="valid")
            mx.eval(targets)
            loss, ntoks = loss_fn(params, batch, targets, lengths, labels)
            mx.eval(loss, ntoks)
            lv, nv = float(loss.item()), int(ntoks.item())
            gate.check_loss_finite(i, lv, nv)
            sums["overall"] += lv * nv
            toks["overall"] += nv
            s = entry["slice"]
            sums[s] = sums.get(s, 0.0) + lv * nv
            toks[s] = toks.get(s, 0) + nv
            seen += 1
        if seen != len(gate_entries):
            raise ValueError("slice gate: manifest iterator count mismatch "
                             f"({seen} batches vs {len(gate_entries)} entries)")
        metrics = {k: sums[k] / toks[k] for k in sums}
        rendered = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        print(f"[alis-dwq][valid] {tag}: {rendered}", file=sys.stderr)
        return metrics

    model.freeze()
    best = init = None
    init_metrics = best_metrics = en_limit = None
    if gate_entries is None:
        best = init = validate("initial")
    else:
        init_metrics = validate("initial")
        best_metrics = dict(init_metrics)
        # EN ceiling: a round may improve overall yet still regress EN beyond
        # eps of the initial (pre-round) EN loss -> REVERT (see alis_dwq.gate)
        en_limit = init_metrics["EN"] * (1.0 + en_eps)

    cka_prev = None
    if cka_mon:
        for batch, _lengths in iterate_batches(valid_data, batch_size,
                                               max_seq_length, seed=seed):
            cka_batch = batch[:, :-1]
            cka_prev = _capture_hiddens(model, cka_batch)  # pre-training state
            break

    for r, subset in enumerate(rounds):
        model.freeze()
        sub, first = set(subset), r == 0

        def unfreeze(name, m):
            if _is_quantized(m):
                mm = _LAYER_RE.search(name)
                if (mm and int(mm.group(1)) in sub) or (mm is None and first and has_extras[0]):
                    m.unfreeze(keys=["scales", "biases"], recurse=False)
            elif train_routers and _is_router(name, m):
                mm = _LAYER_RE.search(name)
                if mm and int(mm.group(1)) in sub:
                    # only "weight": e_score_correction_bias etc. act through
                    # top-k selection, which gradients cannot see
                    m.unfreeze(keys=["weight"], recurse=False)
            elif lora_cfg is not None and _is_lora(m):
                mm = _LAYER_RE.search(name)
                if mm and int(mm.group(1)) in sub:
                    m.unfreeze(keys=[k for k in ("lora_a", "lora_b")
                                     if getattr(m, k, None) is not None],
                               recurse=False)

        model.apply_to_modules(unfreeze)
        snapshot = tree_map(lambda v: v, model.trainable_parameters())
        params = tree_map(lambda v: v.astype(mx.float32), model.trainable_parameters())
        ropt = optimizers.Adam(learning_rate=opt.learning_rate, bias_correction=True)

        def step(inputs, targets, lengths, params, labels=None):
            (loss, ntoks), grads = mx.value_and_grad(loss_fn)(
                params, inputs, targets, lengths, labels)
            return loss, ntoks, ropt.apply_gradients(grads, params)

        total, tok = 0.0, 0
        for it, (batch, lengths) in (
            pbar := tqdm(
                enumerate(iterate_batches(train_data, batch_size, max_seq_length, seed=seed)),
                total=len(train_data) // batch_size,
                desc=f"round {r + 1}/{len(rounds)} L{subset[-1]}-{subset[0]}",
            )
        ):
            labels = batch[:, 1:] if need_labels else None
            batch = batch[:, :-1]
            targets = target_fn(batch, it, split="train")
            mx.eval(targets)
            loss, ntoks, params = step(batch, targets, lengths, params, labels)
            mx.eval(loss, params)
            tok += ntoks.item()
            total += loss.item() * ntoks.item()
            if (it + 1) % 20 == 0:
                pbar.set_description(
                    f"round {r + 1}/{len(rounds)} loss={total / tok:.4f} "
                    f"peak={mx.get_peak_memory() / 1e9:.0f}GB"
                )

        model.update(tree_map(lambda v: v.astype(dtype), params))

        cka_post = None
        if cka_prev is not None:
            cka_post = _capture_hiddens(model, cka_batch)
            sims = [(_cka(a, b), i) for i, (a, b) in enumerate(zip(cka_prev, cka_post))
                    if a is not None and b is not None]
            if sims:
                worst = sorted(sims)[:3]
                print(f"[alis-dwq][cka] round {r + 1}: min "
                      + ", ".join(f"L{i}={c:.4f}" for c, i in worst)
                      + f"  (mean {sum(c for c, _ in sims)/len(sims):.4f})"
                      " — low CKA on a round valid-KL accepts = the "
                      "inversion signature", file=sys.stderr)

        rv = validate(f"round {r + 1}")
        if gate_entries is None:
            # legacy keep-best: revert only when strictly worse (tie accepts)
            if not gate.legacy_accept(rv, best):
                model.update(snapshot)
                print(f"[alis-dwq][round {r + 1}/{len(rounds)}] REVERTED"
                      f" ({rv:.4f} > best {best:.4f})", file=sys.stderr)
                # cka_prev unchanged: the rollback restored exactly that state
            else:
                best = rv
                print(f"[alis-dwq][round {r + 1}/{len(rounds)}] ACCEPTED {rv:.4f}",
                      file=sys.stderr)
                if cka_post is not None:
                    cka_prev = cka_post  # next round drifts against the kept state
        else:
            ok, reasons = gate._accept(rv, best_metrics, en_limit)
            if not ok:
                model.update(snapshot)
                print(f"[alis-dwq][round {r + 1}/{len(rounds)}] REVERTED "
                      + "; ".join(reasons), file=sys.stderr)
                # cka_prev unchanged: the rollback restored exactly that state
            else:
                best_metrics = rv
                print(f"[alis-dwq][round {r + 1}/{len(rounds)}] ACCEPTED "
                      + " ".join(f"{k}={v:.6f}" for k, v in rv.items()),
                      file=sys.stderr)
                if cka_post is not None:
                    cka_prev = cka_post  # next round drifts against the kept state

    model.freeze()
    if gate_entries is None:
        print(f"[alis-dwq] valid {init:.4f} -> {best:.4f}", file=sys.stderr)
    else:
        print(f"[alis-dwq] valid {init_metrics['overall']:.6f} -> "
              f"{best_metrics['overall']:.6f} (EN {init_metrics['EN']:.6f} -> "
              f"{best_metrics['EN']:.6f})", file=sys.stderr)
    if lora_cfg is not None:
        _save_adapters(model, lora_cfg,
                       os.environ.get("ALIS_DWQ_ADAPTER_DIR", "alis_adapters"))
        n = _unwrap_lora(model)
        print(f"[alis-dwq] {n} LoRA wrappers removed — the checkpoint mlx-lm "
              "saves next is stock; pair it with the adapter dir above",
              file=sys.stderr)


def install():
    D.dwq_quantize = layerwise_dwq_quantize


install()
