"""The codemaker's reply, and the Fallout 3 terminal filter."""

import random
from collections import Counter
from itertools import product

from mastermind.fallout3 import Terminal
from mastermind.mastermind import Mastermind, score


def _game(code, colors=6):
    mm = Mastermind(colors=colors, holes=len(code))
    mm.code = list(code)
    return mm


def test_exact_match_on_highest_color_is_not_negative_near():
    assert _game([6, 6, 1, 2]).grade([6, 6, 1, 2]) == (4, 0)
    assert _game([6, 1, 2, 3]).grade([6, 4, 4, 4]) == (1, 0)


def test_highest_color_counts_toward_near():
    assert _game([1, 2, 3, 6]).grade([6, 1, 2, 3]) == (0, 4)


def test_near_respects_multiplicity():
    # code has one 5; guess has three: only one can be a near/exact.
    assert _game([5, 1, 2, 3]).grade([4, 5, 5, 5]) == (0, 1)


def test_score_agrees_with_counting_by_multiset_intersection():
    codes = list(product(range(1, 7), repeat=4))
    rng = random.Random(1)
    for _ in range(20_000):
        code, guess = rng.choice(codes), rng.choice(codes)
        exact = sum(c == g for c, g in zip(code, guess, strict=True))
        common = sum((Counter(code) & Counter(guess)).values())
        assert score(code, guess) == (exact, common - exact) == score(guess, code)


def test_fallout_terminal_keeps_words_with_that_many_letters_in_place():
    words = ["doctrines", "batteries", "scorpions", "generated", "occasions"]
    t = Terminal(words)
    assert t.letters_common("BATTERIES", "DOCTRINES") == 3  # T, E and S line up
    t.attempted("doctrines", 3)
    assert t.words == ["BATTERIES", "OCCASIONS"]
