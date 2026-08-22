"""측정된 손상(measured damage) 민감도 — imatrix 도 그래디언트도 쓰지 않는다.

1원칙: 우리가 최소화하려는 것은 **출력 KL** 이고, 텐서 t 의 기여는
    S_t = E‖ G_t · Δ_t x ‖²      (G_t = ∂z/∂y_t, 하류 이득)
로 분해된다. imatrix 는 이 중 **입력 쪽(x 통계)만** 준다. 우리 배낭은 암묵적으로
G_t 가 모든 t 에서 같다고 가정했고, 그 가정이 두 곳에서 깨진다:
  ① 블록마다 RMSNorm 이 절대 스케일을 지우므로 활성이 큰 층은 하류 이득이 작다
     — 원시 E[x²] 가중은 같은 것을 두 번 센다.
  ② lm_head 는 하류가 아예 없다(G=I). 오차가 로짓에 직행한다. 입력 통계로는
     원리적으로 볼 수 없다.
G_t 를 얻는 정공법은 역전파지만 GatedDeltaNet 에 VJP 가 없다([CA80]).

그래서 **곱을 직접 잰다**: 그룹 g 만 4bit 로 양자화하고 나머지는 bf16 으로 둔 채
KL 을 측정한다. 이것은 대리지표가 아니라 정의 그 자체이며, 전방 패스만 있으면 되므로
아키텍처를 가리지 않는다.
"""
import json, os, re, sys, time
FORK = os.environ.get("FORK", os.path.expanduser("~/glm5.2/mlx-lm"))
sys.path.insert(0, FORK)
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load

SRC = os.path.expanduser("~/qwen38/src")
CORPUS = os.path.expanduser("~/qwen38/eval_corpus")
CTX = 2048
NWIN = int(os.environ.get("NWIN", 3))
BITS = [int(b) for b in os.environ.get("BITS", "4").split(",")]
# BASE 가 설정되면 **한계 가치**를 잰다: 나머지 전부를 BASE 비트로 두고 그룹 g 만
# BITS 비트로 올린 뒤 KL 을 잰다. 고립 손상(BASE 없음, 나머지 bf16)은 가법성을
# 가정하는데, 실측(K팔)이 그 가정을 깼다 — 결정 그대로를 재는 쪽이 옳다.
BASE = os.environ.get("BASE")
BASE = int(BASE) if BASE else None
# 8bit 빌드에는 승격할 비트가 없다(어파인 집합 {2,3,4,5,6,8}). 남는 손잡이는 group_size
# 로, g64→g32 는 스케일을 두 배 촘촘히 해 오차를 줄이고 bpw 를 +0.5 늘린다.
# PROMO_GS 가 설정되면 '그룹 크기 승격'의 한계 가치를 잰다.
PROMO_GS = os.environ.get("PROMO_GS")
PROMO_GS = int(PROMO_GS) if PROMO_GS else None
GS = 64
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "damage.json")
P = lambda *a: print(*a, flush=True)

def group_of(path):
    """텐서 경로 → 프로브 그룹. 층은 8개 버킷으로 묶어 프로브 수를 통제한다."""
    if "visual" in path: return None
    if "lm_head" in path: return "lm_head"
    if "embed_tokens" in path: return "embed_tokens"
    m = re.search(r"layers\.(\d+)\.(.+)$", path)
    if not m:
        return "mtp.fc" if "mtp.fc" in path else None
    L, rest = int(m.group(1)), m.group(2)
    if "mtp" in path: return f"mtp.{rest}"
    return f"{rest}@{L // 8}"

model, tok = load(SRC, lazy=False)
P(f"[probe] mlx_lm={__import__('mlx_lm').__file__}")
paths = [p for p, mod in model.named_modules() if hasattr(mod, "to_quantized")
         and mod.weight.shape[-1] % GS == 0]
groups = {}
for p in paths:
    g = group_of(p)
    if g: groups.setdefault(g, []).append(p)
P(f"[probe] 양자화가능 {len(paths)} · 그룹 {len(groups)}")

def windows(tag, n):
    ids = tok.encode(open(f"{CORPUS}/{tag}.txt").read())
    return [ids[i * CTX:(i + 1) * CTX] for i in range(n)]
wins = []
for tag in ("en", "ko", "code"):
    wins += windows(tag, NWIN)
P(f"[probe] 창 {len(wins)}")

def logp(w):
    out = model(mx.array([w]))
    if isinstance(out, tuple): out = out[0]
    return nn.log_softmax(out[0].astype(mx.float32), axis=-1)

ref = []
t0 = time.perf_counter()
for w in wins:
    ref.append(mx.array(logp(w)))
    mx.eval(ref[-1]); mx.clear_cache()
P(f"[probe] 참조 로짓 {time.perf_counter()-t0:.0f}s")

def kl_now():
    tot, n = 0.0, 0
    for w, rl in zip(wins, ref):
        tl = logp(w)
        p = mx.exp(rl)
        k = float(mx.sum(p * (rl - tl)) / rl.shape[0])
        tot += k; n += 1
        mx.clear_cache()
    return tot / n

orig = model.leaf_modules()
res = {}
for i, (g, plist) in enumerate(sorted(groups.items())):
    row = {"paths": len(plist),
           "numel": int(sum(dict(model.named_modules())[p].weight.size for p in plist))}
    for b in BITS:
        sel = set(plist)
        if BASE is None:
            nn.quantize(model, group_size=GS, bits=b,
                        class_predicate=lambda path, m, _s=sel: path in _s)
        else:
            allq = set(paths)
            gs_hi = PROMO_GS or GS
            nn.quantize(
                model, group_size=GS, bits=BASE,
                class_predicate=lambda path, m, _s=sel, _a=allq, _g=gs_hi: (
                    {"group_size": _g, "bits": b, "mode": "affine"} if path in _s
                    else (path in _a)))
        row[f"kl@{b}"] = kl_now()
        model.update_modules(orig)          # 원본 리프로 복원 (awq.py 와 같은 관용)
    res[g] = row
    P(f"  [{i+1}/{len(groups)}] {g:<28} n={row['numel']/1e6:>7.1f}M "
      + " ".join(f"kl@{b}={row[f'kl@{b}']:.5f}" for b in BITS))
    json.dump(res, open(OUT, "w"), indent=0)
P("PROBE-DONE")
