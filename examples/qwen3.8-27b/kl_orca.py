"""OrcaRouter 프로토콜 재현 — WikiText-2 · ctx 1024 · Mean KLD / KLD p95 / Top-1.

우리 `eval_corpus/en.txt` 는 이미 WikiText-2 다(첫 문서 "= Robert Boulter ="). 따라서
그들과의 프로토콜 차이는 코퍼스가 아니라 **컨텍스트 길이(2048→1024)와 미보고 지표(p95)**
뿐이다. 그들 자로 다시 재서 같은 저울에 올린다.

그들 표기 "WikiText-2, 1,024 tokens" 는 컨텍스트 1024 로 읽었다. 1024 토큰 **총량**으로
읽을 수도 있어(그러면 창 하나뿐이라 그들 오차막대가 설명된다), 단일-창 값도 함께 낸다.
"""
import argparse, json, os, sys
for _f in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_f, "mlx_lm")): sys.path.insert(0, _f); break
import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm.utils import load

CORPUS = os.path.expanduser("~/qwen38/eval_corpus/en.txt")
CTX = int(os.environ.get("CTX", 1024))

def logp(model, w):
    o = model(mx.array([w])); o = o[0] if isinstance(o, tuple) else o
    return mnn.log_softmax(o[0].astype(mx.float32), axis=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref, tok = load(a.ref)
    ids = tok.encode(open(CORPUS).read())
    n = (len(ids) // CTX) * CTX
    wins = [ids[i:i+CTX] for i in range(0, n, CTX)]
    print(f"WikiText-2 · ctx {CTX} · 창 {len(wins)} · 토큰 {n}", flush=True)

    strict = os.environ.get("KL_STRICT", "1") == "1"
    def _load(t):
        try: return load(t)[0]
        except ValueError as e:
            if strict or "Missing" not in str(e): raise
            print(f"[warn] {os.path.basename(t)}: 비-엄격 로드", flush=True)
            from mlx_lm.utils import load_model
            from pathlib import Path
            r = load_model(Path(t), lazy=False, strict=False)
            return r[0] if isinstance(r, tuple) else r
    models = [(t, _load(t)) for t in a.targets]

    acc = {t: {"kl": [], "hit": 0, "n": 0, "first": None} for t, _ in models}
    for wi, w in enumerate(wins):
        rl = logp(ref, w); p = mx.exp(rl); ra = mx.argmax(rl, axis=-1)
        for t, m in models:
            tl = logp(m, w)
            kl = np.array(mx.sum(p * (rl - tl), axis=-1))
            acc[t]["kl"].append(kl)
            acc[t]["hit"] += int(mx.sum(mx.argmax(tl, axis=-1) == ra))
            acc[t]["n"] += kl.shape[0]
            if wi == 0: acc[t]["first"] = (float(kl.mean()), float(np.percentile(kl, 95)))
            del tl
        del rl, p, ra; mx.clear_cache()

    res = {}
    print(f"\n{'빌드':<26}{'Mean KLD':>11}{'KLD p95':>11}{'Top-1':>9}   (단일창 mean/p95)", flush=True)
    for t, _ in models:
        k = np.concatenate(acc[t]["kl"])
        r = {"mean_kld": float(k.mean()), "kld_p95": float(np.percentile(k, 95)),
             "top1_pct": 100.0 * acc[t]["hit"] / acc[t]["n"], "n_tokens": acc[t]["n"],
             "first_window": acc[t]["first"], "ctx": CTX}
        res[os.path.basename(t)] = r
        print(f"{os.path.basename(t):<26}{r['mean_kld']:>11.5f}{r['kld_p95']:>11.5f}"
              f"{r['top1_pct']:>8.2f}%   ({acc[t]['first'][0]:.5f} / {acc[t]['first'][1]:.5f})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("ORCA-PROTO-DONE", flush=True)

main()
