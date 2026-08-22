import sys, os
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
"""공개본 대 개선본 디코드/프리필 속도 — 같은 프로세스에서 교대.

혼합 정밀은 텐서마다 다른 커널을 타므로 속도 대가가 있을 수 있다. 품질 이득이
속도 손실을 넘는지 확인하기 전에는 게시하지 않는다.
번갈아 재서(A B A B …) 열·표류가 한쪽에만 얹히지 않게 한다.
"""
import json, time
FORK = os.path.expanduser(os.environ.get("FORK", "~/glm5.2/mlx-lm"))
sys.path.insert(0, FORK)
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import generate_step
from mlx_lm.models import cache as cache_mod

PROMPTS = [("en", "Explain the theory of relativity in simple terms."),
           ("ko", "인공지능의 역사를 간단히 설명해줘."),
           ("code", "Write a Python function that merges two sorted lists.")]
ROUNDS = int(os.environ.get("ROUNDS", 3))
targets = sys.argv[1:-1]; OUT = sys.argv[-1]

def measure(path):
    model, tok = load(path, lazy=False)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    dec = {}
    for tag, p in PROMPTS:
        ids = mx.array(tok.encode(p)); t0 = None; n = 0
        for i, _ in enumerate(generate_step(ids, model, max_tokens=96)):
            if i == 8: t0 = time.time(); n = 0
            n += 1
        dec[tag] = (n - 1) / (time.time() - t0)
    long_ids = mx.array((tok.encode(PROMPTS[0][1] + " ") * 60)[:2048])
    c = cache_mod.make_prompt_cache(model)
    t0 = time.time(); y = long_ids
    while y.size > 0:
        k = min(512, y.size); model(y[:k][None], cache=c)
        mx.eval([x.state for x in c]); y = y[k:]
    pf = 2048 / (time.time() - t0)
    peak = mx.get_peak_memory() / 2**30
    del model; mx.clear_cache(); mx.reset_peak_memory()
    return dec, pf, peak

acc = {t: {"dec": {k: [] for k, _ in PROMPTS}, "pf": [], "peak": 0.0} for t in targets}
for r in range(ROUNDS):
    for t in targets:                      # 교대
        dec, pf, peak = measure(t)
        for k, v in dec.items(): acc[t]["dec"][k].append(v)
        acc[t]["pf"].append(pf); acc[t]["peak"] = max(acc[t]["peak"], peak)
        print(f"  [r{r+1}] {os.path.basename(t):<16} decode "
              + " ".join(f"{k} {v:.2f}" for k, v in dec.items())
              + f" · prefill {pf:.1f} · peak {peak:.1f} GiB", flush=True)

res = {}
print(f"\n{'빌드':<18}{'decode en':>11}{'ko':>9}{'code':>9}{'평균':>9}{'prefill':>10}{'peak GiB':>10}")
for t in targets:
    d = {k: max(v) for k, v in acc[t]["dec"].items()}      # 최고값(노이즈 하방 제거)
    avg = sum(d.values()) / len(d); pf = max(acc[t]["pf"])
    res[os.path.basename(t)] = {"decode": d, "decode_avg": avg, "prefill": pf,
                                "peak_gib": acc[t]["peak"], "rounds": ROUNDS}
    print(f"{os.path.basename(t):<18}{d['en']:>11.2f}{d['ko']:>9.2f}{d['code']:>9.2f}"
          f"{avg:>9.2f}{pf:>10.1f}{acc[t]['peak']:>10.1f}")
if len(targets) == 2:
    a, b = [res[os.path.basename(t)] for t in targets]
    print(f"\n{os.path.basename(targets[1])} vs {os.path.basename(targets[0])}: "
          f"decode {b['decode_avg']/a['decode_avg']-1:+.1%} · prefill {b['prefill']/a['prefill']-1:+.1%}")
json.dump(res, open(OUT, "w"), indent=1)
print("SPEED-DONE", flush=True)
