"""Pure helpers of eval_kld: loop detection and reason-probe grading."""
from alis_dwq.eval_kld import extract_answer, grade_reason, loop_stats


def test_loop_stats_detects_cycle():
    toks = list(range(50)) + [7, 8] * 60
    distinct, period = loop_stats(toks)
    assert period == 2
    assert distinct < 0.5


def test_loop_stats_clean_sequence():
    distinct, period = loop_stats(list(range(200)))
    assert period == 0
    assert distinct == 1.0


def test_extract_answer_last_marker_wins():
    text = ("Let me think. ANSWER: 12 — wait, that was wrong.\n"
            "Recomputing gives 405.\nANSWER: 405")
    assert extract_answer(text) == 405


def test_extract_answer_case_and_negatives():
    assert extract_answer("blah\nanswer:  -12") == -12
    assert extract_answer("no marker at all, though 42 appears") is None


def test_extract_answer_formatting_tolerance():
    # thousands grouping and markdown emphasis are formatting, not wrongness
    assert extract_answer("ANSWER: 52,432") == 52432
    assert extract_answer("ANSWER: **12,000**") == 12000
    assert extract_answer("ANSWER: `233`") == 233
    # prose after a bare number still parses the number itself
    assert extract_answer("ANSWER: 52, which concludes the proof") == 52


def test_grade_reason():
    assert grade_reason("steps...\nANSWER: 233", 233)
    assert not grade_reason("steps...\nANSWER: 234", 233)
    assert not grade_reason("ran out of budget mid-thought", 233)
