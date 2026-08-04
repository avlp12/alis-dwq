# Kimi-K3 2.8T → 2.10 bpw, and the decode campaign that followed

Case study for the largest build in this repo: **Kimi K3 (2.8T-param multimodal MoE,
93 layers = 69 KDA linear-attention + 24 MLA, 896 experts top-16)** quantized to
**2.096 bpw** (1.5625 bpw ternary codebook for 82k experts + mxfp4 promoted tier +
6-bit dense) and served on **two M3 Ultra 512GB Macs** over expert parallelism.

- Weights + runtime + install guide: [avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw](https://huggingface.co/avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw)
- Full lesson set (porting, campaign mechanics, kernels, DWQ-v3, pitfalls):
  [docs/kimi-k3-mlx-port.md](../../docs/kimi-k3-mlx-port.md) — this directory holds the
  **raw receipts** referenced there.

## Quality (certified)

| Measure | Value |
|---|---|
| KL vs fp teacher (48 windows) | **0.2253 nats** (−35% vs the 211 GB larger v2 build) |
| top-1 flip vs teacher | 13.77% |
| PPL wikitext / korean | 2.078 / 3.311 |
| Decode-path fusion drift (this dir) | **1.1e-3 nats** ≈ 1/200 of the quantization's own KL |

## Serving speed: 2.31 → 6.0 tok/s on unchanged weights

Every adopted lever is bit-exact or ULP-bounded with a killswitch; every rejected lever
is recorded with numbers. Highlights (full narrative in the docs file):

| Lever | Verdict | e2e |
|---|---|---|
| session KV + lazy-graph collectives + Metal fast-synch | adopted | 2.31 → 4.6 |
| ternary-codebook decode kernels (split-K → row-parallel v4) | adopted, bit-identical | → 5.4 |
| KDA projection packing (6 QMM → 1) + fused glue kernels (T≤4) | adopted, bit-exact decode | → 5.75 |
| fused MoE router (top-16-of-896, one dispatch) | adopted, set+weights bit-identical | → 5.9 |
| SiTU-at-store + shared-experts packing | adopted, bit-exact | → **6.0** |
| attention/dense tensor-parallel, buffer-budget raise, f_b in-kernel | **rejected** (measured neutral/regression) | — |
| top-k 16→12 | **rejected for quality** (flat router tail — see below) | (+5% if opted in) |
| speculative decoding (n-gram PLD) | break-even at k=1; verify premium halved to 1.28×, drafter is the binding constraint | ±0 |

## Novel data in this directory

- **`router_mass_45k.npz` + summary** — the first published router probability-mass profile
  for a k≥16 fine-grained MoE (45k layer-samples, 4 domains). Key finding: the tail is
  *flat* (ranks 13-16 carry ~3% each), the opposite of coarse-MoE folklore — fine-grained
  designs are per-expert fragile, so top-k reduction needs a real code eval, not PPL.
- **`kl_decode_bundle_results.txt` + script** — decode-mode logit-drift certification of the
  fused serving stack. Method caveat that cost us almost a wasted run: prefill-based KL is
  *vacuous* for decode-gated fusions; you must dump per-step decode logits.

## Reproduce

The serving runtime (fusion kernels, EP harness, web chat, 2-box launcher) ships in the
HF repo with an install guide (including a note addressed to AI coding agents). The
quantization pipeline is this repo's `alis_dwq` + the campaign notes in the docs file.
