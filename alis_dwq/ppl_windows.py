"""Full-window non-overlapping PPL — the llama.cpp-comparable number.

llama.cpp's `llama-perplexity` evaluates full non-overlapping context
windows; this repo's other PPL numbers are *strided* (ctx 2048 / stride
1024) and are NOT comparable across harnesses. For a cross-engine
head-to-head (examples/glm-5.2-ds4-vs-alis) recompute both sides under one
windowing: run `llama-perplexity -c 512` on the GGUF and this on the MLX
build, over byte-identical corpus files.

  python -m alis_dwq.ppl_windows --model <mlx-dir> --window 512 \
      --text data/wikitext.txt data/code.txt data/zh.txt
"""
import argparse
import math

import mlx.core as mx
from mlx_lm import load


def window_ppl(model, tok, path, window):
    ids = tok.encode(open(path).read())
    n_win = len(ids) // window
    tot_nll, tot_tok = 0.0, 0
    for w in range(n_win):
        chunk = mx.array(ids[w * window:(w + 1) * window])[None]
        logits = model(chunk[:, :-1]).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = chunk[:, 1:]
        nll = -mx.take_along_axis(lp, tgt[..., None], axis=-1).sum()
        mx.eval(nll)
        tot_nll += nll.item()
        tot_tok += tgt.size
    return math.exp(tot_nll / tot_tok), n_win, tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--text", nargs="+", required=True)
    a = ap.parse_args()

    model, tok = load(a.model)
    mx.eval(model.parameters())
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    for path in a.text:
        ppl, n_win, n_tok = window_ppl(model, tok, path, a.window)
        print(f"[ppl-w{a.window}] {path}: ppl={ppl:.4f}  "
              f"({n_win} windows, {n_tok} scored tokens)")


if __name__ == "__main__":
    main()
