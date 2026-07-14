"""Synthetic calibration generator with expert-coverage stopping.

Two 2026-06 results justify self-generated calibration text: Recover-LoRA
(10k synthetic samples match curated data for distillation recovery) and
NVIDIA's QAD report (teacher logit distributions carry cross-domain
knowledge — single-domain data recovered near-full-data accuracy). Our own
measurement makes it urgent: GLM-5.2 routing is flat (top-64/256 carry only
39-57%), so a 145-sample mix cannot give every expert a training signal.

This tool generates {"text": ...} jsonl with the model itself, seeded from
the same EN/code/ZH corpora as eval_kld, while instrumenting routing (the
expert_traffic hooks) — and stops when expert coverage plateaus instead of
at an arbitrary sample count. Degenerate generations are dropped using the
eval_kld loop detector, so a damaged generator cannot poison the mix.

  python -m alis_dwq.gen_calib --model <teacher-or-student> --out dwq_data \
      --samples 2000 --max-tokens 400 --temp 0.9 --mix EN=0.3,code=0.25,ZH=0.45

Appends nothing: --out/train.jsonl + valid.jsonl are (over)written whole.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SLICES = {"EN": "wikitext.txt", "code": "code.txt", "ZH": "zh.txt"}


def parse_mix(s, available):
    """"EN=0.3,code=0.25,ZH=0.45" -> normalized {slice: frac} over available."""
    if not s:
        return {k: 1.0 / len(available) for k in available}
    mix = {}
    for part in s.split(","):
        k, v = part.split("=")
        if k.strip() in available:
            mix[k.strip()] = float(v)
    if not mix:
        raise SystemExit(f"[gen] --mix names none of {sorted(available)}")
    tot = sum(mix.values())
    return {k: v / tot for k, v in mix.items()}


def plateaued(history, patience=3, eps=0.005):
    """True when the last `patience` coverage gains were each < eps."""
    if len(history) < patience + 1:
        return False
    tail = history[-(patience + 1):]
    return all(b - a < eps for a, b in zip(tail, tail[1:]))


def coverage(hooked, min_tokens):
    """(mean fraction of experts seen >= min_tokens, median routed tokens per
    expert, p10). The percentiles matter more than the alive-fraction:
    min_tokens says an expert is *reachable*, not *trained* — the working
    753B reference is ~23k routed tokens per expert (NF3-hybrid's per-expert
    GPTQ Hessians)."""
    import mlx.core as mx
    fracs, all_counts = [], []
    for _, _, m in hooked:
        c = m._alis_counts
        if c is None:
            fracs.append(0.0)
            continue
        mx.eval(c)
        cv = np.array(c, copy=True)
        fracs.append(float((cv >= min_tokens).mean()))
        all_counts.append(cv)
    if not fracs:
        return float("nan"), 0.0, 0.0
    cat = np.concatenate(all_counts) if all_counts else np.zeros(1)
    return (float(np.mean(fracs)), float(np.median(cat)),
            float(np.percentile(cat, 10)))


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    from .eval_kld import loop_stats
    from .expert_traffic import _instrument

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="generator (teacher or student)")
    ap.add_argument("--out", default="dwq_data")
    ap.add_argument("--samples", type=int, default=2000, help="hard cap")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--mix", default="", help="e.g. EN=0.3,code=0.25,ZH=0.45")
    ap.add_argument("--batch", type=int, default=50, help="samples per coverage check")
    ap.add_argument("--min-tokens", type=int, default=8,
                    help="an expert counts as covered at this many routed tokens")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    print("[gen][EXPERIMENTAL] synthetic calibration generation (Recover-LoRA / "
          "QAD rationale); coverage-stopped via routing hooks. Curate your own "
          "jsonl instead for the v0.1 workflow.", file=sys.stderr)

    model, tok = load(a.model)
    mx.eval(model.parameters())
    info = getattr(mx, "device_info", None) or mx.metal.device_info
    mx.set_wired_limit(info()["max_recommended_working_set_size"])

    hooked = _instrument(model)  # counts accumulate across ALL generations
    if not hooked:
        print("[gen] dense model (no MoE hooks): coverage loop disabled, "
              "generating to --samples", file=sys.stderr)

    avail = {k: (DATA / f) for k, f in SLICES.items() if (DATA / f).exists()}
    if not avail:
        raise SystemExit(f"[gen] no seed corpora in {DATA}")
    mix = parse_mix(a.mix, set(avail))
    chunks = {}
    for k, path in avail.items():
        ids = tok.encode(path.read_text())
        step = a.prompt_tokens
        chunks[k] = [ids[i:i + step] for i in range(0, max(len(ids) - step, 1), step)]

    rng = np.random.default_rng(a.seed)
    sampler = make_sampler(temp=a.temp)
    names, probs = zip(*sorted(mix.items()))
    rows, dropped, hist = [], 0, []
    while len(rows) < a.samples:
        n_target = min(a.batch, a.samples - len(rows))
        made = 0
        while made < n_target:
            k = str(rng.choice(names, p=probs))
            prompt_ids = chunks[k][int(rng.integers(len(chunks[k])))]
            prompt = tok.decode(prompt_ids)
            text = generate(model, tok, prompt=prompt, max_tokens=a.max_tokens,
                            sampler=sampler)
            toks = tok.encode(text)
            distinct, period = loop_stats(toks) if len(toks) > 8 else (1.0, 0)
            if period or distinct < 0.3:
                dropped += 1
                if dropped > 20 and dropped > len(rows):
                    raise SystemExit("[gen] generator is producing mostly "
                                     "degenerate text — fix the model first "
                                     "(see eval_kld --loop-probe)")
                continue
            rows.append({"text": prompt + text})
            made += 1
        if hooked:
            frac, med, p10 = coverage(hooked, a.min_tokens)
            hist.append(frac)
            print(f"[gen] {len(rows)} samples, expert coverage "
                  f"(>= {a.min_tokens} tokens): {frac*100:.1f}%  "
                  f"routed tokens/expert median {med:.0f} / p10 {p10:.0f} "
                  f"(753B working reference ~23k)  "
                  f"(dropped {dropped} degenerate)", file=sys.stderr)
            if plateaued(hist, a.patience):
                print(f"[gen] coverage plateaued for {a.patience} batches — "
                      "stopping early (note: plateau means reachability "
                      "saturated, not that tokens/expert is sufficient)",
                      file=sys.stderr)
                break

    n_valid = max(1, int(len(rows) * a.valid_frac))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    perm = rng.permutation(len(rows))
    with open(out / "valid.jsonl", "w") as f:
        for i in perm[:n_valid]:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    with open(out / "train.jsonl", "w") as f:
        for i in perm[n_valid:]:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    print(f"[gen] wrote {len(rows) - n_valid} train / {n_valid} valid to {out}"
          + (f"; final coverage {hist[-1]*100:.1f}%" if hist else ""))


if __name__ == "__main__":
    main()
