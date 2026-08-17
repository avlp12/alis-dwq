"""Language-slice override for the fixed EN / code / <lang> eval trio.

Every measuring tool (eval_kld, expert_traffic, gen_calib) uses the same
deterministic three slices so numbers are comparable across builds. The EN and
code slices are universal; the third slice is the *target-language* slice —
ZH for the Tencent/GLM family the repo grew up on. For a model family with a
different target language (e.g. Korean for Motif-3), set

    ALIS_DWQ_LANG_SLICE="KO:/abs/path/ko.txt"        # LABEL:PATH

to swap only that third slice. A bare filename resolves against the repo
data/ dir. Unset -> stock ZH behavior, byte-identical numbers. The label is
purely cosmetic (report rows); the path may differ per invocation, e.g. a
small frozen slice for eval_kld but a larger disjoint pool as gen_calib seeds.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def lang_slice():
    """-> (label, Path) for the third slice of the EN/code/<lang> trio."""
    spec = os.environ.get("ALIS_DWQ_LANG_SLICE", "")
    if not spec:
        return "ZH", DATA / "zh.txt"
    label, sep, path = spec.partition(":")
    if not sep or not label.strip() or not path.strip():
        raise SystemExit(f"[slices] bad ALIS_DWQ_LANG_SLICE {spec!r} — want LABEL:PATH")
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = DATA / path
    if not p.exists():
        raise SystemExit(f"[slices] ALIS_DWQ_LANG_SLICE file missing: {p}")
    print(f"[slices][EXPERIMENTAL] language slice override: {label} -> {p} "
          "(unset ALIS_DWQ_LANG_SLICE for the stock EN/code/ZH trio)",
          file=sys.stderr)
    return label.strip(), p
