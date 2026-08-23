import sys, time, re
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

model_path = sys.argv[1]; target_tokens = int(sys.argv[2]); gen_tokens = int(sys.argv[3])
model, tok = load(model_path)
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

# 장문 소스: 로컬 기술 문서 연결
src = ""
for f in ["/Users/gesicht/qwen38/DSPARK_FINDINGS.md", "/Users/gesicht/qwen38_alis_mlx/docs/LEDGER.md"]:
    try: src += open(f, encoding="utf-8", errors="ignore").read() + "\n"
    except Exception: pass
words = src.split()
prompt_doc = " ".join(words)
msgs = [{"role":"user","content": f"Read the following engineering notes and summarize the single most important lesson in English.\n\n{prompt_doc}"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
# 목표 길이로 절단(문서 부분에서)
while len(ids) > target_tokens:
    words = words[: int(len(words) * 0.9)]
    prompt_doc = " ".join(words)
    msgs = [{"role":"user","content": f"Read the following engineering notes and summarize the single most important lesson in English.\n\n{prompt_doc}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)

out, ptps, gtps = "", 0, 0
for r in stream_generate(model, tok, ids, max_tokens=gen_tokens, sampler=make_sampler(0.0)):
    out += r.text
    ptps, gtps = r.prompt_tps, r.generation_tps
cjk = len(re.findall(r'[一-鿿぀-ヿ가-힯]', out))
print(f"PROMPT_TPS {ptps:.1f} GEN_TPS {gtps:.2f} PROMPT_TOKENS {len(ids)} CJK_SLIPS {cjk}")
print("OUT:", out[:400].replace("\n", " "))
