#!/usr/bin/env python3
"""Qwen3.8-27B MLX 빌드 대조표 — 멀티모달·MTP 보존판 기준.

각 m_*.json 은 자기 완결적이다(빌드 디렉터리를 다시 읽지 않는다) — 박스마다
경로가 달라 예전 판본은 표 전체가 FileNotFoundError 로 죽었다.
"""
import json, os
import numpy as np

ROWS = [
    ("m_bf16.json",   "bf16 (기준)",   "—"),
    ("m_q8v.json",    "8bit",          "보존"),
    ("m_q6v.json",    "6bit",          "보존"),
    ("m_q4v.json",    "4bit",          "보존"),
    ("m_q6awq2.json", "6bit AWQ",      "보존"),
    ("m_q4awq2.json", "4bit AWQ",      "보존"),
]

ref = json.load(open("m_bf16.json"))
hdr = (f"{'빌드':13s} {'멀티모달':6s} {'GB':>6s} {'디코드':>7s} {'프리필':>7s} "
       f"{'피크':>6s} {'ko-nll':>7s} | {'ko':>5s} {'en':>5s} {'code':>5s}")
print(hdr)
print("─" * len(hdr))

for f, lbl, mm in ROWS:
    if not os.path.exists(f):
        print(f"{lbl:13s} {mm:6s} {'(미측정)':>6s}")
        continue
    d = json.load(open(f))
    if f == "m_bf16.json":
        ag = ["100.0"] * 3
    else:
        ag = [
            f"{np.mean(np.array(ref['probe'][t]['top1']) == np.array(d['probe'][t]['top1'])) * 100:.1f}"
            for t in ("ko", "en", "code")
        ]
    print(f"{lbl:13s} {mm:6s} {d['size_gb']:6.1f} {d['decode_avg']:7.1f} "
          f"{d['prefill_tok_s']:7.0f} {d['peak_mem_gb']:6.1f} "
          f"{d['probe']['ko']['nll']:7.3f} | {ag[0]:>5s} {ag[1]:>5s} {ag[2]:>5s}")

# nll 이 기준선 대비 터무니없이 크면 하네스가 스톡 mlx-lm 을 잡은 것이다(norm 이중 시프트).
bad = [lbl for f, lbl, _ in ROWS if os.path.exists(f)
       and json.load(open(f))["probe"]["ko"]["nll"] > ref["probe"]["ko"]["nll"] * 3]
if bad:
    print(f"\n⚠ nll 이상: {bad} — 포크가 아닌 스톡 mlx-lm 으로 측정됐는지 확인하라")
