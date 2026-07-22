import math
import re
import unittest

from alis_dwq.layerwise import _format_validation_metric


_NUMBER = r"([0-9.eE+-]+)"
_INITIAL_RE = re.compile(rf"\[alis-dwq\]\[valid\] initial: {_NUMBER}")
_ACCEPTED_RE = re.compile(rf"\[alis-dwq\]\[round (\d+)/(\d+)\] ACCEPTED {_NUMBER}")
_REVERTED_RE = re.compile(
    rf"\[alis-dwq\]\[round (\d+)/(\d+)\] REVERTED "
    rf"\({_NUMBER} > best {_NUMBER}\)"
)
_FINAL_RE = re.compile(rf"\[alis-dwq\] valid {_NUMBER} -> {_NUMBER}")


class LayerwiseMetricLogTests(unittest.TestCase):
    def test_validation_metric_round_trips_binary64(self):
        values = (
            0.0,
            0.1687,
            math.nextafter(0.1687, math.inf),
            math.nextafter(0.1687, -math.inf),
            float.fromhex("0x1.fffffffffffffp-1"),
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(float(_format_validation_metric(value)), value)

    def test_reverted_line_preserves_raw_ordering_with_existing_regex(self):
        best = 0.1687
        attempted = math.nextafter(best, math.inf)
        self.assertEqual(f"{attempted:.4f}", f"{best:.4f}")

        line = (
            "[alis-dwq][round 48/48] REVERTED "
            f"({_format_validation_metric(attempted)} > best "
            f"{_format_validation_metric(best)})"
        )
        match = _REVERTED_RE.fullmatch(line)
        self.assertIsNotNone(match)
        self.assertGreater(float(match.group(3)), float(match.group(4)))

    def test_all_receipt_lines_keep_the_existing_numeric_contract(self):
        initial = 0.2
        accepted = math.nextafter(initial, -math.inf)
        lines = (
            (
                _INITIAL_RE,
                f"[alis-dwq][valid] initial: {_format_validation_metric(initial)}",
            ),
            (
                _ACCEPTED_RE,
                f"[alis-dwq][round 1/48] ACCEPTED "
                f"{_format_validation_metric(accepted)}",
            ),
            (
                _FINAL_RE,
                f"[alis-dwq] valid {_format_validation_metric(initial)} -> "
                f"{_format_validation_metric(accepted)}",
            ),
        )
        for regex, line in lines:
            with self.subTest(line=line):
                self.assertIsNotNone(regex.fullmatch(line))


if __name__ == "__main__":
    unittest.main()
