"""두 빌드를 한 프로세스에서 같은 창으로 재는 대응표본 KL.

kl_eval.py 는 요약만 저장해 사후 대응표본 검정이 안 된다. 여기서는 bf16 참조를
상주시키고 타깃 두 개를 번갈아 돌려 **블록별 KL 차이**를 남긴다 — 같은 창·같은
순서이므로 창-간 분산이 상쇄되고, 차이의 표준오차가 주변 SE 보다 훨씬 작아진다.
"""
import argparse, json, os, sys
for _f in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_f, "mlx_lm")): sys.path.insert(0, _f); break
import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm.utils import load

CORPUS = os.path.expanduser("~/qwen38/eval_corpus")
CTX, BLOCK = 2048, 512

def windows(tok, path):
    ids = tok.encode(open(path).read())
    n = (len(ids) // CTX) * CTX
    return [ids[i:i+CTX] for i in range(0, n, CTX)]

def logprobs(model, w):
    x = mx.array([w])
    out = model(x)
    if isinstance(out, tuple): out = out[0]
    return out[0].astype(mx.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default="en,ko,code")
    a = ap.parse_args()

    # 참조 로짓을 쌓으면 슬라이스당 수십 GB 다(창×2048×248320×4B). 대신 모든 모델을
    # 동시에 상주시키고 **창마다 교대**한다 — 그래야 같은 창·같은 순서라는 대응표본
    # 전제가 성립하고 메모리도 창 하나치만 든다.
    ref, tok = load(a.ref)
    # 외부 빌드는 MTP 헤드가 없을 수 있다(예: orcarouter 판). MTP 는 평문 전방 패스에
    # 참여하지 않으므로 비-엄격 로드로 재도 텍스트 KL 은 유효하다 — 다만 **로드가
    # 조용히 다른 것을 채우지 않았는지** 결과값 자체로 확인해야 한다(기대 범위 밖이면 의심).
    strict = os.environ.get("KL_STRICT", "1") == "1"
    def _load(t):
        try:
            return load(t)[0]
        except ValueError as e:
            if strict or "Missing" not in str(e):
                raise
            print(f"[warn] {os.path.basename(t)}: 비-엄격 로드 ({str(e).splitlines()[0][:60]})", flush=True)
            from mlx_lm.utils import load_model
            from pathlib import Path
            r = load_model(Path(t), lazy=False, strict=False)
            return r[0] if isinstance(r, tuple) else r
    models = [(t, _load(t)) for t in a.targets]
    print(f"[paired] 참조 + 타깃 {len(models)} 상주", flush=True)

    per_block = {t: {} for t in a.targets}
    for tag in a.slices.split(","):
        wins = windows(tok, f"{CORPUS}/{tag}.txt")
        acc = {t: [] for t in a.targets}
        for wi, w in enumerate(wins):
            rl = mnn.log_softmax(logprobs(ref, w), axis=-1)
            p = mx.exp(rl)
            for t, m in models:
                tl = mnn.log_softmax(logprobs(m, w), axis=-1)
                kl = np.array(mx.sum(p * (rl - tl), axis=-1))
                n = len(kl) - len(kl) % BLOCK
                for i in range(0, n, BLOCK):
                    acc[t].append(float(kl[i:i+BLOCK].mean()))
            del rl, p
            mx.clear_cache()
        for t in a.targets:
            per_block[t][tag] = acc[t]
            print(f"[{tag}] {os.path.basename(t)}: KL {np.mean(acc[t]):.5f} "
                  f"({len(acc[t])}블록)", flush=True)

    base = a.targets[0]
    out = {"per_block": per_block, "paired": {}}
    for t in a.targets[1:]:
        out["paired"][t] = {}
        for tag in per_block[t]:
            d = np.array(per_block[t][tag]) - np.array(per_block[base][tag])
            se = d.std(ddof=1) / np.sqrt(len(d))
            tt = d.mean() / se if se > 0 else float("inf")
            out["paired"][t][tag] = {"mean_diff": float(d.mean()), "paired_se": float(se),
                                     "t": float(tt),
                                     "rel": float(d.mean() / np.mean(per_block[base][tag])),
                                     "n": len(d)}
            print(f"[대응 {tag}] {os.path.basename(t)} vs {os.path.basename(base)}: "
                  f"ΔKL {d.mean():+.5f} ± {se:.5f}  t={tt:+.1f}  "
                  f"{d.mean()/np.mean(per_block[base][tag]):+.1%}", flush=True)
    json.dump(out, open(a.out, "w"))
    print("KL-PAIRED-DONE", flush=True)


main()
