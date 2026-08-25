"""[Round6e] 서빙-충실(prefill+teacher-forced decode) 하이브리드 hidden 전량 재캡처.

배경: gate3(gate3_head_ab.py, 로그 ~/dsv4flash/align/logs/gate3_head_ab.log)가
"실제 서빙에 배치된 6c 드래프트 헤드"에 h_pre(384토큰 일괄 프리필) vs
h_dec(320토큰 벌크 프리필 + 320..383 1토큰씩 교사강제 디코드)를 각각 먹여
d1 예측을 직접 비교 — cross-agree 78.6%, acc_dec가 acc_pre 대비 -19.8pp로
확정 열화. r6c 훈련 데이터(dump_hidden_tp2_corpus.py)는 384토큰 **일괄
프리필** h만 담아 서빙이 실제로 만드는 디코드-h 분포를 드래프트 헤드가 한
번도 보지 못했다 — 이 격차가 -19.8pp의 원인. 이 스크립트는 그 서빙-충실
h를 코퍼스 전량(697윈도우)에 대해 재캡처해 6e 재훈련 데이터를 만든다.

하이브리드 h 구성(윈도우당 [1,384,4,D]):
  위치 0..319  = 320토큰 벌크 프리필의 raw hidden (h_pre_prefix)
  위치 320..383 = 그 캐시 위에서 320..383을 1토큰씩 교사강제 디코드한
                  raw hidden (h_dec_tail)
이게 실서빙이 실제로 각 시퀀스 위치에서 만드는 h 그대로다 — 앞부분은
프롬프트 벌크 프리필, 뒷부분은 신규 생성 토큰의 캐시-위 디코드.
(gate2/gate3의 "384 일괄 프리필 h_pre 전체"와는 다름 — 그건 참조/비교용
이었고, 6e 훈련 입력은 이 하이브리드다.)

boilerplate 계보: model load/shard/wsdpa 무력화는 dump_hidden_tp2_corpus.py
와 100% 동일. build_corpus/코퍼스 리스트/seed 7은 gate2/gate3/r6c와 100%
동일 인라인 재현 — 이래야 windows[i]가 r6c 캐시의 같은 인덱스와 같은
윈도우가 되어 교차검증(앞 320 위치 rel 비교)이 성립한다. 320+64 프리필/
디코드 캡처 프로토콜 자체는 gate2_serving_faithful.py/gate3_head_ab.py와
동일. 드래프트 헤드(6c)는 이 스크립트에서 로드하지 않는다 — 순수 백본
hidden 캡처만 한다(훈련은 train_align.py가 별도로 담당).

재개 가능 설계: 윈도우마다 rank 0의 로컬(공유 아님 — /Users/Shared/tp2는
양쪽 박스에 개별 존재하는 디렉터리이지 NFS 마운트가 아님, ssh로 확인됨)
h_{i}.safetensors + ids_{i}.json 존재 여부로 스킵을 결정한다. 단
model.model() 호출은 내부에서 send/recv/all_gather를 하므로 **양 랭크가
매 윈도우 함께 호출하거나 함께 건너뛰어야** 한다 — rank 0만 스킵하면
rank 1(엡실론)이 그 윈도우의 forward를 계속 기다려 행(hang)한다. 그래서
스킵 여부는 rank 0가 결정한 뒤 mx.distributed.all_sum으로 전 랭크에
브로드캐스트한다(rank 0가 1을 보내면 합이 1, 나머지는 항상 0을 보냄 —
world=2 고정 전제, dspark_tp4_common.py의 all_gather 브로드캐스트 관례와
동일 계열). 저장은 임시파일→os.replace 원자적 치환 — 오늘 있었던 일시적
Metal 오류로 중간에 죽으면 부분기록 파일이 남을 수 있는데, 원자적 치환은
그 시나리오에서 최종 파일 자체가 아예 안 생기게 해 재개 시 안전하게
재계산된다.
"""
import os, sys, json, random, time
import mlx.core as mx

random.seed(7)


def build_corpus(tok, seq_len, files):
    windows = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        ids = tok.encode(txt)
        for i in range(0, max(0, len(ids) - seq_len - 4), seq_len):
            windows.append(ids[i:i + seq_len + 3])
    random.shuffle(windows)
    return windows


def broadcast_skip(local_skip, group):
    """rank 0의 스킵 결정을 전 랭크에 브로드캐스트(world=2 전제: rank 0만
    1을 기여, 나머지는 항상 0 — 합=1이면 스킵)."""
    v = mx.array([1 if local_skip else 0], dtype=mx.int32)
    v = mx.distributed.all_sum(v, group=group)
    mx.eval(v)
    return bool(int(v.item()))


group = mx.distributed.init()
rank, world = group.rank(), group.size()
assert world == 2, f"world={world} != 2"

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
apply_deepseek_v4_patch()
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
assert apply_mlx_lm_mtp_patch()
set_mtp_active(True); set_mtp_depth(1)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = os.path.expanduser("~/dsv4flash/mlx4bit")
CORPUS_LIST = os.path.expanduser("~/dsv4flash/align/corpus_onpolicy_c_portable.txt")
SEQ_LEN = 384
PREFILL_LEN = 320                    # 서빙-충실 벌크 프리필 길이
DECODE_LEN = SEQ_LEN - PREFILL_LEN   # 64 — "신규 생성분" 1토큰씩 구간
OUT_DIR = "/Users/Shared/tp2/exp_chain/r6e_h"


def h_path(i):
    return os.path.join(OUT_DIR, f"h_{i}.safetensors")


def h_path_tmp(i):
    # mx.save_safetensors는 파일명이 정확히 ".safetensors"로 끝나지 않으면
    # 자동으로 ".safetensors"를 덧붙인다(실측 확인됨: "x.safetensors.tmp" →
    # "x.safetensors.tmp.safetensors" 로 저장돼 이후 os.replace가
    # FileNotFoundError). 그래서 임시 파일명도 ".safetensors"로 끝나게 만든다.
    return os.path.join(OUT_DIR, f".tmp_h_{i}.safetensors")


def ids_path(i):
    return os.path.join(OUT_DIR, f"ids_{i}.json")


t0 = time.monotonic()
model, tok = load(MODEL, lazy=True)
assert hasattr(model, "shard")
model.shard(group)
for l in model.model.layers:
    mx.eval(l.parameters()); mx.synchronize()
mx.eval(model.parameters()); mx.synchronize()
mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

# dump_hidden_tp2_corpus.py / gate2 / gate3와 동일한 wsdpa 무력화 — 반복
# forward에서 GPU Timeout 재현 이력 있음.
_patched = []
for _n in ("mlx_lm.models.deepseek_v4", "mlx_lm.models.deepseek_v4_mtp",
           "omlx.patches.deepseek_v4.wsdpa_attention"):
    _m = sys.modules.get(_n)
    if _m is not None and hasattr(_m, "wsdpa_prefill"):
        _m.wsdpa_prefill = lambda *a, **k: None
        if hasattr(_m, "wsdpa_topk_prefill"):
            _m.wsdpa_topk_prefill = lambda *a, **k: None
        _patched.append(_n)

if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[r6e-dump] load+shard {time.monotonic()-t0:.1f}s · wsdpa 무력화: {_patched}", flush=True)

files = [os.path.expanduser(l.strip()) for l in open(CORPUS_LIST) if l.strip()]
windows_full = build_corpus(tok, SEQ_LEN, files)
N_TOTAL = len(windows_full)
LIMIT = int(os.environ.get("R6E_LIMIT", "0")) or N_TOTAL
windows = windows_full[:LIMIT]
if rank == 0:
    print(f"[r6e-dump] 윈도우 {len(windows)}개 (limit={LIMIT}, 전체 {N_TOTAL}) · "
          f"prefill={PREFILL_LEN} decode={DECODE_LEN} · out={OUT_DIR}", flush=True)

computed = skipped = 0
compute_time_sum = 0.0
t1 = time.monotonic()
for i, w in enumerate(windows):
    local_skip = False
    if rank == 0:
        local_skip = os.path.exists(h_path(i)) and os.path.exists(ids_path(i))
    skip = broadcast_skip(local_skip, group)

    if skip:
        skipped += 1
    else:
        tw0 = time.monotonic()
        ids_384 = mx.array([w[:SEQ_LEN]])
        cache = make_prompt_cache(model)

        # 1) 서빙-충실 벌크 프리필 — 앞 320토큰, 한 번에
        out_320, h_320 = model.model(ids_384[:, :PREFILL_LEN], cache, return_raw_hidden=True)
        mx.eval(out_320, h_320)

        # 2) 320..383을 그 캐시 위에서 1토큰씩 교사강제 디코드
        h_steps = []
        for t in range(PREFILL_LEN, SEQ_LEN):
            ids_t = ids_384[:, t:t + 1]
            out_t, h_t = model.model(ids_t, cache, return_raw_hidden=True)
            mx.eval(out_t, h_t)
            h_steps.append(h_t)
        h_dec = mx.concatenate(h_steps, axis=1)
        mx.eval(h_dec)

        # 3) 하이브리드 h: 0..319=프리필, 320..383=디코드
        h_hybrid = mx.concatenate([h_320, h_dec], axis=1)
        mx.eval(h_hybrid)

        if rank == 0:
            tmp_h = h_path_tmp(i)
            mx.save_safetensors(tmp_h, {"h": h_hybrid.astype(mx.float32)})
            os.replace(tmp_h, h_path(i))
            tmp_ids = ids_path(i) + ".tmp"
            with open(tmp_ids, "w") as f:
                json.dump(w, f)
            os.replace(tmp_ids, ids_path(i))

        compute_time_sum += time.monotonic() - tw0
        computed += 1

    if rank == 0 and (i + 1) % 10 == 0:
        avg = compute_time_sum / computed if computed else float("nan")
        remaining = len(windows) - (i + 1)
        eta_min = (avg * remaining / 60.0) if computed else float("nan")
        print(
            f"[r6e-dump] 진행 {i+1}/{len(windows)} (계산 {computed}·건너뜀 {skipped}) · "
            f"평균 {avg:.2f}s/윈도우(계산분) · 이번 실행 잔여 예상 {eta_min:.1f}분",
            flush=True,
        )

if rank == 0:
    total_s = time.monotonic() - t1
    avg = compute_time_sum / computed if computed else float("nan")
    print(
        f"[r6e-dump] 완료: {len(windows)}개 처리(계산 {computed}·건너뜀 {skipped}) · "
        f"전체 {total_s:.1f}s · 평균 {avg:.2f}s/윈도우(계산분)",
        flush=True,
    )
    if computed:
        proj_min = avg * N_TOTAL / 60.0
        print(f"[r6e-dump] {N_TOTAL}윈도우 전량 예상시간: {proj_min:.1f}분 "
              f"(평균 {avg:.2f}s/윈도우 × {N_TOTAL})", flush=True)
    print("[r6e-dump-pass]", flush=True)
