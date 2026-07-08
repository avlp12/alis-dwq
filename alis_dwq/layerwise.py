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

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx.utils import tree_map
from tqdm import tqdm

import mlx_lm.quant.dwq as D
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _is_quantized(m):
    return (
        hasattr(m, "bits")
        and hasattr(m, "group_size")
        and getattr(m, "mode", "affine") == "affine"
        and m.bits < 8
    )


def layerwise_dwq_quantize(
    model, target_fn, opt, train_data, valid_data, batch_size, max_seq_length,
    seed, dtype: mx.Dtype = mx.bfloat16, gradient_checkpoint: bool = False,
    temperature: float = 2.0, **_ignored,
):
    K = int(os.environ.get("ALIS_DWQ_LAYERS_PER_ROUND", "8"))

    quant_layers, has_extras = set(), [False]

    def scan(name, m):
        if _is_quantized(m):
            mm = _LAYER_RE.search(name)
            if mm:
                quant_layers.add(int(mm.group(1)))
            else:
                has_extras[0] = True

    model.apply_to_modules(scan)
    ordered = sorted(quant_layers, reverse=True)  # deepest first
    rounds = [ordered[i:i + K] for i in range(0, len(ordered), K)]
    print(f"[alis-dwq] {len(ordered)} quant layers -> {len(rounds)} rounds of {K}"
          f" (extras with round 1: {has_extras[0]})", file=sys.stderr)

    model.train()
    if gradient_checkpoint:
        grad_checkpoint(model.layers[0])

    scale = 1 / temperature

    def loss_fn(params, x, targets, lengths):
        model.update(tree_map(lambda v: v.astype(dtype), params))
        logits = model(x)
        if isinstance(targets, tuple):
            targets, ids = targets
            logits = mx.take_along_axis(logits, ids, axis=-1)
        losses = kl_div_loss(scale * logits, scale * targets)
        mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        return (mask * losses).sum() / ntoks, ntoks

    def validate(tag):
        v_loss, v_tok = 0.0, 0
        params = model.trainable_parameters()
        for i, (batch, lengths) in enumerate(
            iterate_batches(valid_data, batch_size, max_seq_length, seed=seed)
        ):
            batch = batch[:, :-1]
            targets = target_fn(batch, i, split="valid")
            mx.eval(targets)
            loss, ntoks = loss_fn(params, batch, targets, lengths)
            mx.eval(loss, ntoks)
            v_tok += ntoks.item()
            v_loss += loss.item() * ntoks.item()
        loss = v_loss / v_tok
        print(f"[alis-dwq][valid] {tag}: {loss:.4f}", file=sys.stderr)
        return loss

    model.freeze()
    best = init = validate("initial")

    for r, subset in enumerate(rounds):
        model.freeze()
        sub, first = set(subset), r == 0

        def unfreeze(name, m):
            if not _is_quantized(m):
                return
            mm = _LAYER_RE.search(name)
            if (mm and int(mm.group(1)) in sub) or (mm is None and first and has_extras[0]):
                m.unfreeze(keys=["scales", "biases"], recurse=False)

        model.apply_to_modules(unfreeze)
        snapshot = tree_map(lambda v: v, model.trainable_parameters())
        params = tree_map(lambda v: v.astype(mx.float32), model.trainable_parameters())
        ropt = optimizers.Adam(learning_rate=opt.learning_rate, bias_correction=True)

        def step(inputs, targets, lengths, params):
            (loss, ntoks), grads = mx.value_and_grad(loss_fn)(params, inputs, targets, lengths)
            return loss, ntoks, ropt.apply_gradients(grads, params)

        total, tok = 0.0, 0
        for it, (batch, lengths) in (
            pbar := tqdm(
                enumerate(iterate_batches(train_data, batch_size, max_seq_length, seed=seed)),
                total=len(train_data) // batch_size,
                desc=f"round {r + 1}/{len(rounds)} L{subset[-1]}-{subset[0]}",
            )
        ):
            batch = batch[:, :-1]
            targets = target_fn(batch, it, split="train")
            mx.eval(targets)
            loss, ntoks, params = step(batch, targets, lengths, params)
            mx.eval(loss, params)
            tok += ntoks.item()
            total += loss.item() * ntoks.item()
            if (it + 1) % 20 == 0:
                pbar.set_description(
                    f"round {r + 1}/{len(rounds)} loss={total / tok:.4f} "
                    f"peak={mx.get_peak_memory() / 1e9:.0f}GB"
                )

        model.update(tree_map(lambda v: v.astype(dtype), params))
        rv = validate(f"round {r + 1}")
        if rv > best:
            model.update(snapshot)
            print(f"[alis-dwq][round {r + 1}/{len(rounds)}] REVERTED"
                  f" ({rv:.4f} > best {best:.4f})", file=sys.stderr)
        else:
            best = rv
            print(f"[alis-dwq][round {r + 1}/{len(rounds)}] ACCEPTED {rv:.4f}",
                  file=sys.stderr)

    model.freeze()
    print(f"[alis-dwq] valid {init:.4f} -> {best:.4f}", file=sys.stderr)


def install():
    D.dwq_quantize = layerwise_dwq_quantize


install()
