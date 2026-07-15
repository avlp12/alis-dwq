"""Alternative distillation losses for layerwise DWQ — opt-in via ALIS_DWQ_LOSS.

Ported from Tencent AngelSlim's distill loss zoo
(angelslim/compressor/distill/loss.py, Apache-2.0) — the QAD stack that
independently converged on scales-only KD at lr 1e-6. The default ("kl")
never routes through this module: layerwise.py keeps calling mlx-lm's
kl_div_loss directly, so the v0.1 pipeline stays byte-identical. Everything
else announces itself with an [EXPERIMENTAL] banner. Judge on held-out
slice PPL/KL as always — the valid-vs-held-out inversion has been measured
three times in this repo.

  ALIS_DWQ_LOSS=kl          KL(teacher || student), T-scaled — default
  ALIS_DWQ_LOSS=rkl         KL(student || teacher) — mode-seeking
  ALIS_DWQ_LOSS=cakld       per-token blend conf*rkl + (1-conf)*fkl where
                            conf = teacher prob (T=1) on the ground-truth
                            next token (BitDistiller-style coefficient;
                            AngelSlim's per-token variant). Works with both
                            dense targets and top-k dumps (label outside
                            the dumped top-k => conf 0 => pure forward KL).
  ALIS_DWQ_LOSS=kl_top_K    forward KL renormalized over the teacher's
                            top-K logits (e.g. kl_top_1000). Dense targets
                            only: a top-k target dump is already this loss
                            (with K = the dump's k), so it refuses tuples.

python -m alis_dwq.losses   runs the numpy-reference selftest (no model).
"""
import sys

import mlx.core as mx


def parse(spec):
    """'kl' | 'rkl' | 'cakld' | 'kl_top_K' -> (kind, topk|None); raises on junk."""
    s = (spec or "kl").strip().lower()
    if s in ("kl", "rkl", "cakld"):
        return s, None
    if s.startswith("kl_top_"):
        k = int(s[len("kl_top_"):])
        if k <= 0:
            raise ValueError(f"ALIS_DWQ_LOSS top-k must be positive: {spec}")
        return "kl_top", k
    raise ValueError(
        f"ALIS_DWQ_LOSS={spec!r} not supported (kl | rkl | cakld | kl_top_K)")


def banner(kind, topk):
    print(f"[alis-dwq][EXPERIMENTAL] ALIS_DWQ_LOSS={kind}"
          + (f" (K={topk})" if topk else "")
          + ": non-default distillation loss (AngelSlim port). Unset the env "
          "var (or pin branch backup/v0.1-pre-router-kd) for the stock "
          "forward-KL loss. Judge on held-out slice PPL/KL, not train/valid.",
          file=sys.stderr)


def _kl(logp_q, logp_p):
    """Per-token KL(p || q) from log-probs."""
    return (mx.exp(logp_p) * (logp_p - logp_q)).sum(axis=-1)


def _logsm(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def per_token_loss(kind, topk, logits, targets, scale, labels=None, ids=None):
    """Per-token (B, T) loss. `logits`/`targets` are student/teacher logits
    over the same columns — full vocab, or the dump's top-k columns with
    `ids` (B, T, k) naming them. `labels` (B, T) = ground-truth next token,
    required by cakld only. `scale` = 1/temperature, matching the stock
    kl_div_loss(scale*logits, scale*targets) convention."""
    if kind == "kl_top":
        if ids is not None:
            raise ValueError(
                "kl_top_K with a top-k target dump is redundant — the dump "
                "already restricts the loss to the teacher's top-k columns; "
                "use ALIS_DWQ_LOSS=kl (or re-dump with a different --dwq-top-k)")
        kidx = mx.argpartition(-targets, kth=topk - 1, axis=-1)[..., :topk]
        targets = mx.take_along_axis(targets, kidx, axis=-1)
        logits = mx.take_along_axis(logits, kidx, axis=-1)
        return _kl(_logsm(scale * logits), _logsm(scale * targets))

    s_logp = _logsm(scale * logits)
    t_logp = _logsm(scale * targets)
    fkl = _kl(s_logp, t_logp)          # KL(teacher || student)
    if kind == "kl":
        return fkl
    bkl = _kl(t_logp, s_logp)          # KL(student || teacher)
    if kind == "rkl":
        return bkl
    if kind == "cakld":
        if labels is None:
            raise ValueError("cakld needs ground-truth labels")
        p_t = mx.softmax(targets.astype(mx.float32), axis=-1)  # conf at T=1
        if ids is not None:  # top-k dump: label may be absent -> conf 0
            conf = (p_t * (ids == labels[..., None])).sum(axis=-1)
        else:
            conf = mx.take_along_axis(p_t, labels[..., None], axis=-1).squeeze(-1)
        return conf * bkl + (1.0 - conf) * fkl
    raise ValueError(f"unknown loss kind {kind!r}")


# ------------------------------------------------------------ selftest

def _selftest():
    import numpy as np

    rng = np.random.default_rng(7)
    B, T, V, K = 2, 5, 37, 8
    s = rng.normal(size=(B, T, V)).astype(np.float32)
    t = rng.normal(size=(B, T, V)).astype(np.float32)
    labels = rng.integers(0, V, size=(B, T))
    scale = 0.5

    def np_logsm(x):
        return x - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)) \
            - x.max(-1, keepdims=True)

    def np_kl(lq, lp):
        return (np.exp(lp) * (lp - lq)).sum(-1)

    sl, tl = np_logsm(scale * s), np_logsm(scale * t)
    ref_fkl = np_kl(sl, tl)
    ref_bkl = np_kl(tl, sl)
    p1 = np.exp(np_logsm(t))
    conf = np.take_along_axis(p1, labels[..., None], axis=-1).squeeze(-1)
    ref_ca = conf * ref_bkl + (1 - conf) * ref_fkl

    ms, mt = mx.array(s), mx.array(t)
    ml = mx.array(labels)
    for kind, ref in [("kl", ref_fkl), ("rkl", ref_bkl), ("cakld", ref_ca)]:
        got = np.array(per_token_loss(kind, None, ms, mt, scale,
                                      labels=ml))
        assert np.allclose(got, ref, atol=1e-5), (kind, np.abs(got - ref).max())

    # kl == stock mlx-lm kl_div_loss path (the default equivalence)
    from mlx_lm.tuner.losses import kl_div_loss
    stock = np.array(kl_div_loss(scale * ms, scale * mt))
    assert np.allclose(stock, ref_fkl, atol=1e-5), "stock-kl mismatch"

    # kl_top_K on dense targets == full KL restricted+renormalized to top-K
    kidx = np.argsort(-t, axis=-1)[..., :K]
    ts = np.take_along_axis(t, kidx, axis=-1)
    ss = np.take_along_axis(s, kidx, axis=-1)
    ref_top = np_kl(np_logsm(scale * ss), np_logsm(scale * ts))
    got = np.array(per_token_loss("kl_top", K, ms, mt, scale))
    assert np.allclose(got, ref_top, atol=1e-5), np.abs(got - ref_top).max()

    # cakld with a top-k dump: conf falls to 0 when the label left the dump
    dump_ids = np.argsort(-t, axis=-1)[..., :K]
    ds = np.take_along_axis(s, dump_ids, axis=-1)
    dt = np.take_along_axis(t, dump_ids, axis=-1)
    got = np.array(per_token_loss(
        "cakld", None, mx.array(ds), mx.array(dt), scale,
        labels=ml, ids=mx.array(dump_ids)))
    in_dump = (dump_ids == labels[..., None]).any(-1)
    # where the label is outside the dump, cakld must equal pure forward KL
    sl_d, tl_d = np_logsm(scale * ds), np_logsm(scale * dt)
    ref_f_d = np_kl(sl_d, tl_d)
    assert np.allclose(got[~in_dump], ref_f_d[~in_dump], atol=1e-5)

    # parse/refuse behavior
    assert parse("kl_top_1000") == ("kl_top", 1000)
    for bad in ("kl_top_0", "banana"):
        try:
            parse(bad)
            raise AssertionError(f"parse({bad!r}) should have raised")
        except ValueError:
            pass
    try:
        per_token_loss("kl_top", 4, mx.array(ds), mx.array(dt), scale,
                       ids=mx.array(dump_ids))
        raise AssertionError("kl_top on a dump should have raised")
    except ValueError:
        pass

    print("[losses] selftest OK (kl / rkl / cakld / kl_top vs numpy reference; "
          "stock-kl equivalence; dump-edge cases)")


if __name__ == "__main__":
    _selftest()
