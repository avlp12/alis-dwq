"""디코드-모드 KL 번들 계산: A(융합 OFF) vs B(프로덕션) 스텝 로짓 대조.

주의: 그리디 경로가 ULP로 분기하면 분기 이후 로짓은 다른 문맥 조건부라 비교 무의미 —
공통 그리디 접두(argmax 일치 구간)까지만 KL을 집계하고 분기 위치를 별도 보고한다.
파일 페어링: 두 디렉토리 모두 요청 순서대로 타임스탬프 정렬 → i번째끼리 대응.
"""
import glob
import os
import sys

import numpy as np


def load_dir(d):
    fs = sorted(glob.glob(os.path.join(d, "steps_*.npz")))
    return [np.load(f)["logits"].astype(np.float32) for f in fs]


def softmax_log(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(-1, keepdims=True))


def main(da, db):
    A, B = load_dir(da), load_dir(db)
    assert len(A) == len(B), (len(A), len(B))
    print(f"프롬프트 {len(A)}쌍")
    tot_kl, tot_n, tot_flip_at = [], 0, []
    for i, (a, b) in enumerate(zip(A, B)):
        n = min(len(a), len(b))
        am_a = a[:n].argmax(-1)
        am_b = b[:n].argmax(-1)
        div = np.nonzero(am_a != am_b)[0]
        L = int(div[0]) if len(div) else n          # 공통 접두 길이(분기 스텝 제외 전 구간)
        La = L if L > 0 else 0
        if La == 0:
            print(f"  p{i+1}: 첫 스텝부터 분기 — KL 산출 불가(비교 불능)")
            continue
        la = softmax_log(a[:La])
        lb = softmax_log(b[:La])
        p = np.exp(la)
        kl = (p * (la - lb)).sum(-1)                # [La]
        top2gap = np.sort(a[:La], axis=-1)
        gap = top2gap[:, -1] - top2gap[:, -2]
        print(f"  p{i+1}: 공통접두 {La}/{n}스텝 · KL 평균 {kl.mean():.3e} · 최대 {kl.max():.3e} nats"
              + (f" · 분기@{L}(top-2 갭 {gap[-1]:.3f})" if len(div) else " · 완주 무분기"))
        tot_kl.append(kl)
        tot_n += La
        if len(div):
            tot_flip_at.append(L / n)
    if tot_kl:
        allkl = np.concatenate(tot_kl)
        print(f"종합: {tot_n}스텝 · KL 평균 {allkl.mean():.3e} · p99 {np.percentile(allkl, 99):.3e} "
              f"· 최대 {allkl.max():.3e} nats")
        print("판정 참고: 캠페인 전체 KL(교사 대비)=0.2253 nats — 융합 드리프트가 그 1/100"
              " 이하(≤2e-3)면 무시 가능 급.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
