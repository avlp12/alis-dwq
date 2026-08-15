#!/usr/bin/env python3
"""KV 캐시 양자화 실측 — 하이브리드(48 linear + 16 full-attn)에서의 실제 이득 측정.

측정 항목: 장문맥 디코드 속도 / 피크 메모리 / 품질(top-1 일치).
[I219] 큐-일괄 타이밍: 반복마다 mx.eval 하면 ~250µs 동기 바닥이 깔려 측정이 오염된다.
[I220] 제너레이터-시작 타이머는 prefill을 디코드에 혼입시킨다 → 첫 토큰 이후로 타이머 시작.
"""
import argparse, json, os, sys, time

# measure.py 와 같은 이유로 포크를 고정한다 — 스톡 mlx-lm 은 MTP 보존 빌드의 norm 을
# 이중 시프트해 조용히 망가뜨린다([CA79], 상류 PR #1735).
for _f in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_f, "mlx_lm")):
        sys.path.insert(0, _f); break
import mlx.core as mx
import mlx_lm
if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다 — 포크 경로 확인")
from mlx_lm.utils import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--out", required=True)
ap.add_argument("--ctx", type=int, default=16384, help="프리필 길이(장문맥일수록 KV 비중↑)")
ap.add_argument("--gen", type=int, default=128)
args = ap.parse_args()

model, tok = load(args.model)
mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

# 재현 가능한 장문 프롬프트: 어휘 전체를 순환하는 결정적 토큰열(내용 무관, 길이가 관건)
base = tok.encode("The quick brown fox jumps over the lazy dog. " * 64)
ids = (base * (args.ctx // len(base) + 1))[: args.ctx]
prompt = mx.array(ids)

results = {}
for kv_bits in (None, 8, 4):
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)

    t_pre = time.perf_counter()
    steps = generate_step(
        prompt, model, max_tokens=args.gen, prompt_cache=cache,
        kv_bits=kv_bits, kv_group_size=64, quantized_kv_start=0,
        prefill_step_size=2048,
    )
    toks, t_first, t0 = [], None, None
    for i, (t, _) in enumerate(steps):
        if i == 0:
            mx.eval(t)                      # 첫 토큰 = 프리필 완료 시점
            t_first = time.perf_counter()
            t0 = t_first
        toks.append(t)
    mx.eval(toks)                            # [I219] 나머지는 한 번에 flush
    toks = [int(t.item()) if hasattr(t, "item") else int(t) for t in toks]
    t_end = time.perf_counter()

    n_dec = len(toks) - 1
    results[str(kv_bits)] = {
        "kv_bits": kv_bits,
        "prefill_tok_s": args.ctx / (t_first - t_pre),
        "decode_tok_s": n_dec / (t_end - t0),
        "peak_gb": mx.get_peak_memory() / 1e9,
        "top1": [int(t) for t in toks],
    }
    r = results[str(kv_bits)]
    print(f"kv_bits={str(kv_bits):>4s}  프리필 {r['prefill_tok_s']:6.0f}  "
          f"디코드 {r['decode_tok_s']:6.2f}  피크 {r['peak_gb']:6.2f}GB", flush=True)
    del cache

ref = results["None"]["top1"]
for k, r in results.items():
    agree = sum(a == b for a, b in zip(ref, r["top1"])) / len(ref) * 100
    r["top1_agree_vs_bf16kv"] = agree
    print(f"kv_bits={k:>4s}  top-1 일치 {agree:5.1f}%")

json.dump({"model": args.model, "ctx": args.ctx, "gen": args.gen, "results": results},
          open(args.out, "w"), indent=1)
print("KV-MEASURE-DONE", args.out)
