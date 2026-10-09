import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mastermind"))

from mastermind import Mastermind  # noqa: E402


def _game(code, colors=6):
    mm = Mastermind(colors=colors, holes=len(code))
    mm.code = list(code)
    mm.code_counts = __import__("collections").Counter(code)
    return mm


def test_exact_match_on_highest_colour_is_not_negative_near():
    assert _game([6, 6, 1, 2]).grade([6, 6, 1, 2]) == (4, 0)
    assert _game([6, 1, 2, 3]).grade([6, 4, 4, 4]) == (1, 0)


def test_highest_colour_counts_toward_near():
    assert _game([1, 2, 3, 6]).grade([6, 1, 2, 3]) == (0, 4)


def test_near_respects_multiplicity():
    # code has one 5; guess has three: only one can be a near/exact.
    assert _game([5, 1, 2, 3]).grade([4, 5, 5, 5]) == (0, 1)
