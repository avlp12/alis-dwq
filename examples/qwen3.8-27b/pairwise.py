import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
"""저장된 블록값으로 **임의의 두 빌드**를 대응표본 비교한다.

kl_paired.py 는 첫 인자를 기준으로 나머지를 비교하므로, 네 빌드를 한 번에 넘기면
원하지 않는 짝(예: 8bit 를 6bit 기준으로)만 나온다. 블록값이 전부 저장되므로
재실행 없이 올바른 짝을 계산할 수 있다 — 같은 창·같은 순서라 대응표본 전제는 그대로다.
"""
import json, os, numpy as np
d = json.load(open(sys.argv[1]))["per_block"]
keys = {os.path.basename(k): k for k in d}
pairs = [(sys.argv[i], sys.argv[i+1]) for i in range(2, len(sys.argv), 2)]
for base, tgt in pairs:
    b, t = d[keys[base]], d[keys[tgt]]
    print(f"\n{tgt} vs {base}")
    tot_b, tot_t = [], []
    for sl in b:
        x, y = np.array(b[sl]), np.array(t[sl])
        diff = y - x
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        tt = diff.mean() / se if se > 0 else float("nan")
        print(f"  {sl:<5} {x.mean():.5f} → {y.mean():.5f}  ΔKL {diff.mean():+.5f} ± {se:.5f} "
              f"· t={tt:+.1f} · {diff.mean()/x.mean():+.1%}")
        tot_b.append(x); tot_t.append(y)
    X, Y = np.concatenate(tot_b), np.concatenate(tot_t)
    dd = Y - X; se = dd.std(ddof=1)/np.sqrt(len(dd))
    print(f"  합산  {X.mean():.5f} → {Y.mean():.5f}  ΔKL {dd.mean():+.5f} ± {se:.5f} "
          f"· t={dd.mean()/se:+.1f} · {dd.mean()/X.mean():+.1%}")
