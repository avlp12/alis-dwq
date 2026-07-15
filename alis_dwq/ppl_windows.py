"""Non-overlapping-window PPL mirroring llama.cpp's `llama-perplexity`.

Verified against tools/perplexity/perplexity.cpp (master, 2026-07): text is
tokenized once with no special tokens, each window's token 0 is replaced by
BOS when the tokenizer defines one, and NLL is scored only for predicted
token positions >= window/2 + 1 (n_ctx-1-first targets = 255 at -c 512) —
the first half of each window is context. This repo's other PPL numbers are
*strided* (ctx 2048 / stride 1024) and are NOT comparable across harnesses.

For a cross-engine head-to-head (examples/glm-5.2-ds4-vs-alis) run
`llama-perplexity -c 512` on the GGUF and this on the MLX build over
byte-identical corpus files. Residual comparability caveat: identical bytes
do not guarantee identical token IDs across the GGUF and HF tokenizers —
spot-check counts per slice before quoting a cross-engine delta.

  python -m alis_dwq.ppl_windows --model <mlx-dir> --window 512 \
      --text data/wikitext.txt data/code.txt data/zh.txt
"""
import argparse
import math

import mlx.core as mx
from mlx_lm import load


def window_ppl(model, tok, path, window):
    """Mirror llama-perplexity's chunk scoring exactly (verified against
    tools/perplexity/perplexity.cpp): raw tokenization (no special tokens),
    each chunk's token 0 replaced by BOS when the tokenizer defines one, and
    NLL scored only for token positions >= window/2 — the first half is
    context. Scoring all 511 targets instead reads systematically worse
    (low-context positions) and is NOT comparable across harnesses."""
    with open(path) as f:
        text = f.read()
    ids = tok.encode(text, add_special_tokens=False)
    bos = getattr(tok, "bos_token_id", None)
    first = window // 2
    n_win = len(ids) // window
    tot_nll, tot_tok = 0.0, 0
    for w in range(n_win):
        chunk_ids = list(ids[w * window:(w + 1) * window])
        if bos is not None:
            chunk_ids[0] = bos
        chunk = mx.array(chunk_ids)[None]
        logits = model(chunk[:, :-1]).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = chunk[:, 1:]
        # llama.cpp keeps logits for positions >= first and scores
        # n_ctx-1-first targets (predicted token positions first+1..n_ctx-1);
        # target index j here predicts token position j+1, so j >= first
        nll = -mx.take_along_axis(lp[:, first:], tgt[:, first:, None],
                                  axis=-1).sum()
        mx.eval(nll)
        tot_nll += nll.item()
        tot_tok += int(tgt.shape[1]) - first
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
