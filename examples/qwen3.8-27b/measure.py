"""Qwen3.8-27B MLX 빌드 측정 — 카드 수치 산출.

측정: ①크기/bpw ②디코드 tok/s(웜, 3프롬프트) ③프리필 tok/s ④피크 메모리
     ⑤품질: bf16 대비 teacher-forced NLL·top-1 일치(KO/EN/코드 프로브)
사용: python3 measure.py <빌드경로> [--ref <bf16경로>] --out <json>
"""
import argparse
import glob
import json
import os
import sys
import time

# 포크를 명시적으로 앞세운다. 이걸 안 하면 설치된 스톡 mlx-lm 이 잡히는데,
# 스톡 qwen3_5.sanitize 는 `has_mtp_weights` 를 raw-HF 판별자로 써서 **이미 시프트된**
# norm 가중치에 +1.0 을 한 번 더 얹는다(γ 0.944→1.944). 증상은 크래시가 아니라
# nll 1.7→17 의 조용한 붕괴이고, 하필 MTP 를 보존한 빌드만 골라서 망가져 보인다.
for _fork in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_fork, "mlx_lm")):
        sys.path.insert(0, _fork)
        break

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step

# 조용히 스톡으로 되돌아가면 수치가 전부 무의미해지므로 여기서 크게 실패한다.
_used = os.path.dirname(mlx_lm.__file__)
if "site-packages" in _used:
    raise SystemExit(f"스톡 mlx-lm 이 잡혔다({_used}) — 포크 경로를 확인하라")
print(f"[measure] mlx_lm = {_used}", file=sys.stderr)

PROMPTS = [
    ("ko", "한국의 전통 건축양식과 현대 건축의 조화에 대해 자세히 설명해줘."),
    ("en", "Explain how speculative decoding accelerates LLM inference."),
    ("code", "Write a Python function that merges two sorted lists."),
]
PROBES = [
    ("ko", "조선 시대의 과학 기술은 세종 대에 절정을 이루었다. 장영실이 제작한 자격루는 "
           "물의 흐름을 이용해 시각을 자동으로 알리는 장치였고, 앙부일구는 그림자의 방향과 "
           "길이로 시각과 절기를 동시에 읽을 수 있는 해시계였다. 측우기는 세계 최초의 "
           "표준화된 강우량 측정 기구로, 전국 각지의 강우 기록을 중앙에서 취합해 농정에 "
           "활용했다. 훈민정음의 창제는 음운학적 분석에 기반한 문자 체계로 평가받는다."),
    ("en", "The transformer architecture replaced recurrence with self-attention, letting "
           "every token attend to all previous tokens through scaled dot-product attention. "
           "This enabled parallel training and better long-range dependency modeling. Later "
           "work on sparse mixtures of experts scaled these models to trillions of parameters "
           "while keeping inference cost proportional to the active parameters alone."),
    ("code", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
             "    pivot = arr[len(arr) // 2]\n"
             "    left = [x for x in arr if x < pivot]\n"
             "    mid = [x for x in arr if x == pivot]\n"
             "    right = [x for x in arr if x > pivot]\n"
             "    return quicksort(left) + mid + quicksort(right)\n"),
]


def dir_size_gb(p):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(p, "*.safetensors"))) / 2**30


def logprobs(model, ids):
    lg = model(ids[None])[0]
    return (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--ref", default=None, help="bf16 원본(품질 대조용)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    res = {"build": a.build, "size_gb": round(dir_size_gb(a.build), 2)}
    cfg = json.load(open(os.path.join(a.build, "config.json")))
    q = cfg.get("quantization", {})
    res["bits"] = q.get("bits")
    res["group_size"] = q.get("group_size")

    model, tok = load(a.build, lazy=False)
    res["peak_mem_gb"] = round(mx.get_peak_memory() / 2**30, 1)

    # 디코드 속도(웜업 후 3프롬프트 평균)
    speeds = {}
    for tag, p in PROMPTS:
        ids = mx.array(tok.encode(p))
        t0 = None
        n = 0
        for i, _ in enumerate(generate_step(ids, model, max_tokens=96)):
            if i == 8:
                t0 = time.time(); n = 0
            n += 1
        speeds[tag] = round((n - 1) / (time.time() - t0), 2)
    res["decode_tok_s"] = speeds
    res["decode_avg"] = round(sum(speeds.values()) / len(speeds), 2)

    # 프리필(2048 토큰, 청크 512)
    long_ids = mx.array((tok.encode(PROBES[0][1] + " ") * 40)[:2048])
    from mlx_lm.models import cache as cache_mod
    c = cache_mod.make_prompt_cache(model)
    t0 = time.time()
    y = long_ids
    while y.size > 0:
        n = min(512, y.size)
        model(y[:n][None], cache=c)
        mx.eval([x.state for x in c])
        y = y[n:]
    res["prefill_tok_s"] = round(2048 / (time.time() - t0), 1)

    # 품질 프로브(자기 로짓 저장 → ref와 비교는 별도 실행에서)
    probe = {}
    for tag, text in PROBES:
        ids = mx.array(tok.encode(text))
        lp = logprobs(model, ids)
        nll = -mx.take_along_axis(lp[:-1], ids[1:, None], axis=-1).mean()
        top1 = mx.argmax(lp, axis=-1)
        mx.eval(nll, top1)
        probe[tag] = {"nll": round(float(nll.item()), 4),
                      "top1": [int(x) for x in top1.tolist()]}
    res["probe"] = probe
    res["peak_mem_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "probe"},
                     indent=2, ensure_ascii=False), flush=True)
    print("MEASURE-DONE", flush=True)


if __name__ == "__main__":
    main()
