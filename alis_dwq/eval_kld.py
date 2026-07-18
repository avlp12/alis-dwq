"""KL-divergence + top-1 flip-rate for quantized Hy3 builds vs the T512REF
8-bit reference (the low-bit quality ground truth; PPL can hide damage via
token-cancellation).

Fixed slice = EN wikitext / code / Chinese thirds (Tencent model: language-
specialized experts; an EN-only slice can hide expert damage).

  --model REF  --save-ref evals/ref.npz     # dump reference log-probs
  --model CAND --ref evals/ref.npz          # KL(ref||cand) + top-1 flip
"""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load

from . import power

DATA = Path(__file__).resolve().parent.parent / "data"


def set_wired():
    lim = mx.metal.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(lim)
    print(f"[wire] wired limit set to {lim/1e9:.1f} GB")


def get_tokens(tokenizer, n=3072):
    """Deterministic fixed slice: EN + code + ZH thirds (same for every build)."""
    missing = [f for f in ("wikitext.txt", "code.txt", "zh.txt")
               if not (DATA / f).exists()]
    if missing:
        raise SystemExit(
            f"[eval] missing eval corpora in {DATA}: {', '.join(missing)} — "
            "these are local files, not part of the repo (see data/README.md "
            "for what goes there and the hashes of the published runs)")
    third = n // 3
    w = tokenizer.encode((DATA / "wikitext.txt").read_text())[:third]
    c = tokenizer.encode((DATA / "code.txt").read_text())[:third]
    z = tokenizer.encode((DATA / "zh.txt").read_text())[: n - 2 * third]
    return mx.array((w + c + z)[:n])


def loop_stats(toks, max_period=32, tail=96):
    """Degeneration metrics for a greedy continuation: (distinct-4gram ratio,
    detected cycle period or 0). A cycle means the last `tail` tokens are
    fully periodic with period <= max_period — the failure mode REAP's 504B
    doubled (3.6% -> 7.2% loop rate) while holding eval parity."""
    n4 = len(toks) - 3
    distinct = len({tuple(toks[i:i + 4]) for i in range(n4)}) / n4 if n4 > 0 else 1.0
    t = toks[-tail:]
    for p in range(1, min(max_period, len(t) // 3) + 1):
        if all(t[i] == t[i + p] for i in range(len(t) - p)):
            return distinct, p
    return distinct, 0


def _sample(logits, temp, top_k, top_p):
    """Temperature sampling with optional top-k/top-p (temp 0 = greedy)."""
    if temp <= 0:
        return mx.argmax(logits, axis=-1)
    logits = (logits / temp).astype(mx.float32)
    if top_k and top_k > 0:
        thresh = mx.sort(logits, axis=-1)[..., -top_k:-top_k + 1]
        logits = mx.where(logits < thresh, -mx.array(np.inf), logits)
    if top_p < 1.0:
        srt = mx.sort(logits, axis=-1)[..., ::-1]
        probs = mx.softmax(srt, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        # nucleus: keep tokens whose preceding cumulative mass < top_p;
        # threshold = smallest kept logit (max is always kept)
        thresh = mx.min(mx.where(cum - probs < top_p, srt, srt[..., :1]),
                        axis=-1, keepdims=True)
        logits = mx.where(logits < thresh, -mx.array(np.inf), logits)
    return mx.random.categorical(logits)


def gen_continue(model, ids, n_gen, temp=0.0, top_k=0, top_p=1.0):
    """Decode n_gen tokens after prompt ids (greedy when temp=0)."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    tok = _sample(logits[:, -1], temp, top_k, top_p)[None]
    out = []
    for _ in range(n_gen):
        out.append(int(tok.item()))
        logits = model(tok, cache=cache)
        tok = _sample(logits[:, -1], temp, top_k, top_p)[None]
    return out


def loop_probe(model, tokenizer, n_gen, n_prompt=64, temp=0.0, top_k=0,
               top_p=1.0, samples=1):
    mode = ("greedy" if temp <= 0 else
            f"T={temp} top_k={top_k} top_p={top_p} x{samples} samples")
    print(f"[eval][EXPERIMENTAL] --loop-probe: degeneration probe, {mode} "
          f"({n_gen} tokens from a {n_prompt}-token prompt per slice). Omit "
          "the flag (or pin branch backup/v0.1-pre-router-kd) for the "
          "previous KL/flip-only behavior.")
    results = []
    for name, fname in [("EN", "wikitext.txt"), ("code", "code.txt"), ("ZH", "zh.txt")]:
        path = DATA / fname
        if not path.exists():
            continue
        ids = mx.array(tokenizer.encode(path.read_text())[:n_prompt])
        for s in range(max(1, samples)):
            mx.random.seed(1000 + s)
            toks = gen_continue(model, ids, n_gen, temp, top_k, top_p)
            distinct, period = loop_stats(toks)
            state = f"cycle=len{period} LOOPED" if period else "cycle=none"
            tag = f"{name}#{s}" if samples > 1 else name
            print(f"[loop] {tag:>6}: distinct4={distinct:.3f}  {state}")
            results.append({"slice": name, "sample": s, "temp": temp,
                            "distinct4": round(distinct, 4), "cycle": period})
    return results


_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(-?\d+)", re.IGNORECASE)


def extract_answer(text):
    """Last 'ANSWER: <int>' in the text; None when absent. Grading only looks
    at this marker (the probe's prompts instruct the model to end with it), so
    a thinking chain of any length cannot false-fail the run — the measured
    max_tokens=30 harness artifact (README: 'budget the thinking, grade the
    tail')."""
    m = _ANSWER_RE.findall(text)
    return int(m[-1]) if m else None


def grade_reason(text, answer):
    got = extract_answer(text)
    return got is not None and got == int(answer)


def reason_probe(model, tokenizer, path, max_tokens, temp=0.0, samples=3):
    """Long-form reasoning collapse probe (the README §4 gate caveat made
    runnable): a small fixed problem set, generated end-to-end and graded on
    the final ANSWER line. Detects the AIME-93->57 class of collapse that
    teacher-forced KL/flip and the loop probe are blind to; it does NOT rank
    builds — treat a multi-problem drop vs the previous build as a ship
    blocker, a single-problem flip as noise to investigate.

    Runs greedy always (the default regime of math harnesses, where
    Bonsai-class damage showed); pass --reason-temp to ALSO run seeded
    sampled attempts (the bonsai-audit temperature control: the two regimes
    can disagree completely)."""
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    problems = [json.loads(l) for l in open(path) if l.strip()]
    regimes = [("greedy", 0.0, 1)]
    if temp > 0:
        regimes.append((f"T={temp}", temp, max(1, samples)))
    print(f"[eval][EXPERIMENTAL] --reason-probe: {len(problems)} problems "
          f"from {path}, max_tokens={max_tokens}, regimes: "
          + ", ".join(f"{n}x{s}" for n, _, s in regimes)
          + ". Generative long-form gate (collapse detector, not a ranker); "
          "omit the flag for KL/flip-only behavior.")
    tpl = getattr(tokenizer, "chat_template", None) is not None
    if not tpl:
        print("[rsn ] no chat template — prompting raw (base model?)")
    results = []
    for prob in problems:
        prompt = prob["prompt"]
        if tpl:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        for rname, rtemp, rsamples in regimes:
            sampler = make_sampler(temp=rtemp)
            for s in range(rsamples):
                mx.random.seed(3000 + s)
                text = generate(model, tokenizer, prompt=prompt,
                                max_tokens=max_tokens, sampler=sampler)
                toks = tokenizer.encode(text)
                distinct, period = loop_stats(toks) if len(toks) > 8 else (1.0, 0)
                got = extract_answer(text)
                ok = got is not None and got == int(prob["answer"])
                tag = prob["id"] if rsamples == 1 else f"{prob['id']}#{s}"
                state = "PASS" if ok else "FAIL"
                extra = f"  cycle=len{period} LOOPED" if period else ""
                print(f"[rsn ] {tag:>10} ({rname}): {state}  got={got} "
                      f"expect={prob['answer']}  gen_toks={len(toks)}  "
                      f"distinct4={distinct:.3f}{extra}")
                results.append({
                    "id": prob["id"], "regime": rname, "sample": s,
                    "pass": ok, "got": got, "expect": int(prob["answer"]),
                    "gen_tokens": len(toks), "distinct4": round(distinct, 4),
                    "cycle": period})
    for rname, _, _ in regimes:
        rs = [r for r in results if r["regime"] == rname]
        print(f"[rsn ] {rname}: {sum(r['pass'] for r in rs)}/{len(rs)} passed")
    return results


def logprobs(model, ids, chunk=1024, kv_bits=0, kv_group_size=64):
    """Chunked forward with KV cache (harness rule #3: prefill <= 2048/chunk).
    kv_bits > 0 quantizes the cache (layers whose cache type lacks
    to_quantized — e.g. DSA/recurrent states — stay full precision)."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    if kv_bits:
        kept = 0
        conv = []
        for c in cache:
            if hasattr(c, "to_quantized"):
                conv.append(c.to_quantized(group_size=kv_group_size, bits=kv_bits))
            else:
                conv.append(c)
                kept += 1
        cache = conv
        if kept:
            print(f"[kv  ] {kept}/{len(cache)} layer caches have no quantized "
                  "form — left at full precision (probe is a lower bound)")
    outs = []
    for i in range(0, int(ids.size), chunk):
        chunk_tic = time.time()
        logits = model(ids[None, i:i + chunk], cache=cache)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        outs.append(lp)
        power.throttle(time.time() - chunk_tic)
    return mx.concatenate(outs, axis=0)


def kv_probe(model, ids, lp_fp16, bits, group_size, n):
    """Self-KL induced by a quantized KV cache vs the model's own FP16-KV
    baseline (the Bonsai-27B tolerance measurement: low-bit-weight models
    absorbed 4-bit KV 12-95x better than FP16/Q4-weight builds — if DWQ'd
    students inherit that, long-context memory drops ~4x for free)."""
    print(f"[eval][EXPERIMENTAL] --kv-probe {bits}: self-KL vs own FP16-KV "
          "baseline (per slice). Omit the flag for v0.1 behavior.")
    lp_q = logprobs(model, ids, kv_bits=bits, kv_group_size=group_size)
    T = min(lp_fp16.shape[0], lp_q.shape[0])
    a, b = lp_fp16[:T], lp_q[:T]
    kl = (mx.exp(a) * (a - b)).sum(axis=-1)
    flip = (mx.argmax(a, axis=-1) != mx.argmax(b, axis=-1))
    mx.eval(kl, flip)
    klv = kl.tolist()
    third = n // 3
    out = {"bits": bits, "group_size": group_size, "slices": {}}
    for name, s, e in [("EN", 0, third), ("code", third, 2 * third), ("ZH", 2 * third, T)]:
        seg = klv[s:e]
        if seg:
            fl = flip[s:e]
            print(f"[kv  ] {name:>4}: selfKL={sum(seg)/len(seg):.5f}  "
                  f"flip={float(fl.sum())/len(seg):.4f}")
            out["slices"][name] = {"self_kl": sum(seg) / len(seg),
                                   "flip": float(fl.sum()) / len(seg)}
    print(f"[kv  ] all : selfKL={sum(klv[:T])/T:.5f}  "
          f"flip={float(flip.sum())/T:.4f}  (kv {bits}-bit gs{group_size})")
    out["all"] = {"self_kl": sum(klv[:T]) / T, "flip": float(flip.sum()) / T}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save-ref")
    ap.add_argument("--ref")
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--loop-probe", type=int, default=0, metavar="N_GEN",
                    help="generate N tokens per slice and report "
                         "repetition/loop metrics (0 = off, v0.1 behavior)")
    ap.add_argument("--loop-temp", type=float, default=0.0,
                    help="loop-probe sampling temperature (0 = greedy); the "
                         "temperature-matched control for 'it only loops "
                         "under greedy' rebuttals")
    ap.add_argument("--loop-top-k", type=int, default=0)
    ap.add_argument("--loop-top-p", type=float, default=1.0)
    ap.add_argument("--loop-samples", type=int, default=1,
                    help="samples per slice when loop-temp > 0 (seeded)")
    ap.add_argument("--kv-probe", type=int, default=0, metavar="BITS",
                    help="also measure self-KL with a BITS-bit quantized KV "
                         "cache vs this model's own FP16-KV run (0 = off)")
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--reason-probe", nargs="?", const=str(DATA / "reason_probe.jsonl"),
                    default=None, metavar="FILE",
                    help="generative long-form reasoning gate: jsonl problems "
                         "({id, prompt, answer}), graded on the final "
                         "'ANSWER: <int>' line (default file: "
                         "data/reason_probe.jsonl)")
    ap.add_argument("--reason-max-tokens", type=int, default=2048,
                    help="generation budget per problem — must cover the "
                         "whole thinking chain (a too-small budget false-"
                         "fails: the measured max_tokens=30 harness artifact)")
    ap.add_argument("--reason-temp", type=float, default=0.0,
                    help="ALSO run seeded sampled attempts at this "
                         "temperature next to the always-on greedy pass")
    ap.add_argument("--reason-samples", type=int, default=3,
                    help="sampled attempts per problem when reason-temp > 0")
    ap.add_argument("--save-json", metavar="FILE",
                    help="also write every metric this run produced "
                         "(KL/flip per slice, loop/kv/reason probes) as JSON "
                         "— cross-build tables stop being hand-assembled")
    a = ap.parse_args()

    results = {"model": a.model, "n": a.n}

    def flush():
        if a.save_json:
            with open(a.save_json, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[save-json] {a.save_json}")

    model, tok = load(a.model)
    mx.eval(model.parameters())
    set_wired()
    print(f"[mem ] load {mx.get_active_memory()/1e9:.1f}GB / peak {mx.get_peak_memory()/1e9:.1f}GB")
    if a.loop_probe > 0:
        results["loop"] = loop_probe(model, tok, a.loop_probe, temp=a.loop_temp,
                                     top_k=a.loop_top_k, top_p=a.loop_top_p,
                                     samples=a.loop_samples)
    if a.reason_probe:
        results["reason"] = reason_probe(model, tok, a.reason_probe,
                                         a.reason_max_tokens,
                                         temp=a.reason_temp,
                                         samples=a.reason_samples)
    ids = get_tokens(tok, a.n)
    lp = logprobs(model, ids)
    if a.kv_probe > 0:
        results["kv"] = kv_probe(model, ids, lp, a.kv_probe, a.kv_group_size, a.n)

    if a.save_ref:
        np.savez(a.save_ref,
                 ids=np.array(ids.tolist(), dtype=np.int32),
                 logp=np.array(lp.astype(mx.float16)))
        print(f"[save-ref] {a.save_ref}  T={lp.shape[0]} V={lp.shape[1]}")
        flush()
        return

    ref = np.load(a.ref)
    assert np.array_equal(ref["ids"], np.array(ids.tolist(), dtype=np.int32)), \
        "token slice mismatch vs reference — tokenizer/data changed"
    rp = mx.array(ref["logp"].astype(np.float32))
    T = min(rp.shape[0], lp.shape[0])
    rp, cp = rp[:T], lp[:T]
    kl = (mx.exp(rp) * (rp - cp)).sum(axis=-1)
    flip = (mx.argmax(rp, axis=-1) != mx.argmax(cp, axis=-1))
    mx.eval(kl, flip)
    klv = kl.tolist()
    n = len(klv)
    mean = sum(klv) / n
    se = (sum((v - mean) ** 2 for v in klv) / (n - 1) / n) ** 0.5
    # per-slice breakdown (EN / code / ZH thirds)
    third = a.n // 3
    results["kld"] = {"ref": a.ref, "slices": {},
                      "overall": {"kl": mean, "se": se,
                                  "flip": float(flip.sum()) / T, "T": T}}
    for name, s, e in [("EN", 0, third), ("code", third, 2 * third), ("ZH", 2 * third, T)]:
        seg = klv[s:e]
        fl = flip[s:e]
        if seg:
            print(f"[KLD ] {name:>4}: KL={sum(seg)/len(seg):.5f}  flip={float(fl.sum())/len(seg):.4f}")
            results["kld"]["slices"][name] = {
                "kl": sum(seg) / len(seg), "flip": float(fl.sum()) / len(seg)}
    print(f"[KLD ] {a.model}\n       KL(ref||cand)={mean:.5f}±{se:.5f}  "
          f"top1_flip={float(flip.sum())/T:.4f}  T={T}")
    flush()


if __name__ == "__main__":
    main()
