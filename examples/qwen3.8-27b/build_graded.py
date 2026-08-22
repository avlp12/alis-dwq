"""배분표(JSON)를 읽어 MLX 혼합정밀 빌드를 만든다."""
import json, os, sys

# 포크 경로를 한 박스 기준으로 하드코딩하면 안 된다. epsilon 에는 GLM-5.2 시절의
# 낡은 ~/glm5.2/mlx-lm 이 남아 있어서, 이 줄이 PYTHONPATH 를 가리고 MTP 포트도
# passthrough_patterns 도 없는 트리를 로드했다 — 결과물은 MTP 미양자화 + 비전 타워
# 333텐서 소실이었고 로그는 침묵했다([I178]/[I189]).
#
# 규칙: ① PYTHONPATH 를 존중하고 ② 로드된 트리가 **의도한 트리인지 단언**한다.
_FORK = os.environ.get("FORK")
if _FORK:
    sys.path.insert(0, os.path.expanduser(_FORK))
elif not any(os.path.isdir(os.path.join(p, "mlx_lm")) for p in sys.path if p):
    for _c in ("~/glm5.2/mlx-lm", "~/mlx-lm-fork"):
        if os.path.isdir(os.path.expanduser(_c) + "/mlx_lm"):
            sys.path.insert(0, os.path.expanduser(_c)); break

import mlx_lm
from mlx_lm.convert import convert
from mlx_lm.models import qwen3_5 as _m

print(f"[fork] mlx_lm = {mlx_lm.__file__}", flush=True)
if not getattr(_m.Model, "passthrough_patterns", None):
    raise SystemExit(
        f"이 mlx_lm 트리에는 passthrough_patterns 가 없다 ({_m.__file__}). "
        "낡은 포크를 로드했다는 뜻이고, 그대로 빌드하면 비전 타워와 MTP 가 조용히 사라진다. "
        "FORK=<정본 경로> 로 지정하라."
    )

D = os.path.dirname(os.path.abspath(__file__))
alloc_path, out = sys.argv[1], sys.argv[2]
# 값은 정수(비트) 또는 {"bits": N, "group_size": G}
raw = {}
for _k, _v in json.load(open(alloc_path)).items():
    _k = _k.replace(".weight", "")
    raw[_k] = {"bits": int(_v)} if isinstance(_v, (int, float)) else {
        "bits": int(_v["bits"]), "group_size": int(_v.get("group_size", 0)) or None}
# 체크포인트 이름(model.language_model.layers…)과 MLX 모듈 경로
# (language_model.model.layers…)는 접두 순서가 다르다. 꼬리로 색인해 둘 다 흡수한다.
alloc = dict(raw)
for k, v in raw.items():
    parts = k.split(".")
    for i in range(len(parts)):
        alloc.setdefault(".".join(parts[i:]), v)
DEFAULT = int(os.environ.get("DEFAULT_BITS", 4))
GS = 64
hits = {"hit": 0, "miss": 0}

def pred(path, module):
    b = alloc.get(path)
    if b is None:
        parts = path.split(".")
        for i in range(len(parts)):
            cand = ".".join(parts[i:])
            if cand in alloc: b = alloc[cand]; break
    if b is None:
        hits["miss"] += 1; b = {"bits": DEFAULT}
    else:
        hits["hit"] += 1
    if isinstance(b, dict):
        gs = b.get("group_size") or GS
        if module.weight.shape[-1] % gs != 0: gs = GS
        return {"group_size": gs, "bits": b["bits"], "mode": "affine"}
    return {"group_size": GS, "bits": int(b), "mode": "affine"}

convert(hf_path=os.path.expanduser("~/qwen38/src"), mlx_path=out,
        quantize=True, q_group_size=GS, q_bits=DEFAULT, quant_predicate=pred)
print(f"[build] 배분 적중 {hits['hit']} · 미적중(기본 {DEFAULT}bit) {hits['miss']}", flush=True)

# 발화 검증: 텐서가 조용히 사라지지 않았는지 산출물로 확인한다. 크기로만 잡혔던 사고가 있다.
_idx = json.load(open(f"{out}/model.safetensors.index.json"))["weight_map"]
_n_vis = len([k for k in _idx if "visual" in k])
_n_mtp = len([k for k in _idx if "mtp" in k])
print(f"[verify] 텐서 {len(_idx)} · visual {_n_vis} · mtp {_n_mtp}", flush=True)
if _n_vis == 0 or _n_mtp == 0:
    raise SystemExit(f"✗ 산출물에 타워가 없다 (visual {_n_vis}, mtp {_n_mtp}) — 게시 불가")
print("BUILD-DONE", flush=True)
