# data/ — eval + calibration-seed corpora (local files, not in the repo)

`eval_kld`, `ppl_windows` (via its instructions), `expert_traffic`, and
`gen_calib` read three plain-text files from this directory:

| file | slice | what goes here |
|---|---|---|
| `wikitext.txt` | EN | raw wikitext-style English prose |
| `code.txt` | code | source code, mixed languages |
| `zh.txt` | ZH | Chinese prose |

Any UTF-8 text works — but **every number these tools print depends on the
exact bytes**, so slice evals are only comparable across builds (and across
machines) when the files are identical. Hash them (`shasum -a 256`) and
record the hashes next to your results, the way the ds4 head-to-head did
(`examples/glm-5.2-ds4-vs-alis`: corpus hashes `1b00a74f` / `ccdb70ea` /
`82db86d9`, first 8 hex chars, for wikitext/code/zh respectively — all
published alis-dwq numbers used those files).

The files are gitignored (`data/*.txt`): they are corpora, not code, and the
repo previously shipped broken machine-local symlinks here instead — if you
cloned before v0.2, delete those.

House rule reminder (README §1): keep calibration and evaluation corpora
disjoint — gating a ship on wikitext PPL after calibrating on wikitext-like
data overfits the quant to the gate.

`reason_probe.jsonl` (committed) is different: it is the fixed problem set
for `eval_kld --reason-probe` — original problems with brute-force-verified
integer answers. Keep these problems out of `gen_calib` seed corpora
(calibration/eval disjointness applies to the probe too).
