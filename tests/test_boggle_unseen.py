"""Boggle: estimators of unseen words (boggle/unseen.py) and exact chances of words
(boggle/chance.py)."""

import json
import random
from fractions import Fraction
from itertools import permutations
from math import prod
from pathlib import Path

import numpy as np
import pytest

from boggle import anneal, chance, unseen
from boggle.board import CUBES, load_words, possible, word_faces

HERE = Path(__file__).parent.parent / "boggle"
needs_cc = pytest.mark.skipif(anneal.compiler_missing(), reason="bound.c needs a C compiler")


def test_estimators_on_counts_worked_by_hand():
    # Boards per word: two words on one board each, one on two, one on three, one on four.
    f = unseen.frequency_counts([1, 1, 2, 3, 0, 4])
    assert f == {"S_obs": 5, "U": 11, "Q1": 2, "Q2": 1, "Q3": 1, "Q4": 1}
    t = 10
    assert unseen.good_turing(5, 2, 11) == pytest.approx(5 / (1 - 2 / 11))
    assert unseen.chao2(5, 2, 1, t) == pytest.approx(5 + 0.9 * 4 / 2)  # 6.8
    assert unseen.chao2(5, 2, 0, t) == pytest.approx(5 + 0.9 * 2 * 1 / 2)  # no doubles
    # iChao2: 6.8 + (7/40)(Q3/Q4) max(2 - (7/18) Q2 Q3 / Q4, 0)
    assert unseen.ichao2(5, 2, 1, 1, 1, t) == pytest.approx(6.8 + 7 / 40 * (2 - 7 / 18))
    assert unseen.jackknife2(5, 2, 1, t) == pytest.approx(5 + 2 * 17 / 10 - 64 / 90)
    lo, hi = unseen.chao2_interval(5, 2, 1, t)
    assert 5 <= lo < 6.8 < hi


def test_chao2_is_close_when_every_word_is_equally_likely():
    # Chao's bound is nearly unbiased when the rare words have equal chances (Chao and
    # Colwell 2017, section 2.3): 2,000 words, each on a board with chance 1/1,000.
    rng = np.random.default_rng(1)
    counts = rng.binomial(1000, 0.001, size=2000)
    estimate = unseen.estimates(counts, 1000)["chao2"]
    assert (counts > 0).sum() < 1400 and abs(estimate / 2000 - 1) < 0.05


def test_committed_unseen_results_are_what_the_script_writes():
    committed = json.loads((HERE / "results" / "unseen.json").read_text())
    possible_words = unseen.possible_words()
    assert committed["truth"] == len(possible_words) == 160_719
    found = unseen.board_words(processes=2)
    rows = unseen.summary(found, possible_words, chance.word_bounds(sorted(possible_words)),
                          chance.short_chances(possible_words))  # fmt: skip
    assert rows == committed
    last = committed["rows"][-1]
    assert (last["boards"], last["S_obs"], last["Q1"]) == (15_000, 41_171, 11_018)
    # Every estimate is far below the truth, and no interval reaches it.
    for row in committed["rows"]:
        for k in unseen.ESTIMATORS:
            assert row[k]["high"] < 0.4 * committed["truth"]


def test_path_counts_match_the_published_sequence():
    # OEIS A236690: simple directed paths on the n by n king graph, single slots included.
    assert [sum(chance.path_counts(chance.grid_adjacent(n, n))) for n in (2, 3)] == [64, 10_305]
    assert sum(chance.PATHS_4X4) == 12_029_640
    assert chance.path_counts(chance.grid_adjacent(2, 2)) == [0, 4, 12, 24, 24]


def test_placements_match_a_sum_over_every_choice_of_cubes():
    for word, cubes in (
        ("toe", CUBES),
        ("quote", CUBES[:6] + CUBES[12:15]),
        ("settee", CUBES[8:14]),
    ):
        faces = word_faces(word)
        brute = sum(
            prod(cubes[c].count(f) for c, f in zip(choice, faces, strict=True))
            for choice in permutations(range(len(cubes)), len(faces))
        )
        assert chance.placements(faces, cubes) == brute, word
    assert chance.placements(word_faces("toe")) == 592


@needs_cc
def test_c_placements_agree_with_the_exact_count():
    words = sorted(w for w in load_words(HERE / "enable1.txt") if len(w) >= 3 and possible(w))
    sample = random.Random(3).sample(words, 300) + ["replastering", "quinquennia"]
    fast = chance.c_placements([word_faces(w) for w in sample])
    for w, x in zip(sample, fast, strict=True):
        exact = chance.placements(word_faces(w))
        assert x == pytest.approx(int(exact), rel=1e-12), w


SMALL = [CUBES[i] for i in (4, 12, 9, 13, 10, 8, 0, 1, 2)]  # cubes carrying t, o and e


def test_word_chance_matches_every_shake_of_four_cubes():
    for word in ("toe", "tee", "tot", "eye", "tet", "too"):
        assert chance.word_chance(word, SMALL[:4], 2, 2) == chance.shake_chance(
            word, SMALL[:4], 2, 2
        )
    assert chance.word_chance("toe", SMALL[:4], 2, 2) == Fraction(8, 81)


def test_word_chance_matches_a_search_of_every_coloring():
    for word in ("toe", "tee", "tot", "tit", "eye", "too", "eet"):
        assert chance.word_chance(word, SMALL[:6], 2, 3) == chance.coloring_chance(
            word, SMALL[:6], 2, 3
        )
    assert chance.word_chance("toe", SMALL, 3, 3) == chance.coloring_chance("toe", SMALL, 3, 3)


def test_committed_chances_are_what_the_script_writes(tmp_path):
    chance.write(tmp_path / "chances.json")
    assert (tmp_path / "chances.json").read_bytes() == (
        HERE / "results" / "chances.json"
    ).read_bytes()
    result = json.loads((HERE / "results" / "chances.json").read_text())
    assert result["most likely word"]["word"] == "toe"
    assert Fraction(result["most likely word"]["chance"]) == chance.word_chance("toe")
    # Every longer word's chance is at most its bound, and the largest bound is below toe's chance.
    assert (
        result["largest bound, four or more faces"]["bound"] < result["most likely word"]["value"]
    )
    for w, v in result["short words"].items():
        assert Fraction(v["chance"]) <= Fraction(chance.expected_paths(w)), w


def test_readme_and_note_quote_the_results():
    u = json.loads((HERE / "results" / "unseen.json").read_text())
    c = json.loads((HERE / "results" / "chances.json").read_text())
    text = (HERE / "README.md").read_text() + (HERE / "results" / "unseen.md").read_text()
    last = u["rows"][-1]
    for key in unseen.ESTIMATORS:
        assert f"{last[key]['estimate']:,.0f}" in text, key
    ex = u["explanation"]
    assert f"{ex['Chao2 unseen words']:,.0f}" in text and f"{ex['unseen possible words']:,}" in text
    assert c["most likely word"]["chance"] in text
    for share, boards in c["boards needed for a share of the words, at least"].items():
        assert f"{boards:,}" in text, share
    check = u["short words against the sample"]
    assert f"{check['words']} " in text and f"{check['outside the 95% interval']} " in text
