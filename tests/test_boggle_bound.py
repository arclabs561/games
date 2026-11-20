"""Boggle: branch and bound over classes of boards (boggle/bound.c, boggle/bound.py)."""

import json
import random
from itertools import product
from pathlib import Path

import pytest

from boggle import anneal, bound
from boggle.board import get_words, load_words, make_trie, points

HERE = Path(__file__).parent.parent / "boggle"
pytestmark = pytest.mark.skipif(anneal.compiler_missing(), reason="bound.c needs a C compiler")
ENABLE1, ENABLE2K = bound.LISTS["enable1"], bound.LISTS["enable2k"]


def test_bounds_match_vanderkams_published_class():
    # hybrid-boggle README: "max=9359, sum=106383" for this class with ENABLE2K.
    cells = "lnrsy aeiou chkmpt chkmpt aeiou lnrsy lnrsy aeiou bdfgjvwxz".split()
    assert bound.c_bounds(ENABLE2K, 3, 3, cells) == (9359, 106383)


def test_scores_agree_with_anneal_and_the_published_best_board():
    assert bound.c_score(ENABLE1, 3, 3, "streaedlp") == (545, 267)
    assert bound.c_score(ENABLE2K, 3, 3, "streaedlp") == (545, 267)
    rng = random.Random(2)
    for _ in range(40):
        letters = "".join(rng.choice("aeeinorstlqdxz") for _ in range(9))
        assert bound.c_score(ENABLE1, 3, 3, letters) == anneal.c_score(letters), letters


def test_bounds_hold_on_small_classes():
    # Every board of a random class of up to 512 boards scores at most both bounds, and the
    # union bound of a single board is its score.
    trie = make_trie(w for w in load_words(ENABLE1) if len(w) >= 3)
    rng = random.Random(4)
    for _ in range(6):
        cells = ["".join(sorted(set(rng.choices("aeilnorst", k=2)))) for _ in range(9)]
        best = 0
        for letters in product(*cells):
            board = [["qu" if c == "q" else c for c in letters[i : i + 3]] for i in (0, 3, 6)]
            best = max(best, sum(map(points, get_words(board, trie))))
        mx, un = bound.c_bounds(ENABLE1, 3, 3, cells)
        assert best <= mx and best <= un, cells
    assert bound.c_bounds(ENABLE1, 3, 3, list("streaedlp"))[1] == 545


def test_search_finds_what_scoring_every_board_finds():
    # All 26^4 boards of 2 by 2, and the 5^9 boards of 3 by 3 over five letters.
    found, _ = bound.parse(bound.solve(ENABLE1, 2, 2, 15))
    assert {bound.canonical(b, 2, 2): p for b, p in found.items()} == bound.exhaust(
        ENABLE1, 2, "abcdefghijklmnopqrstuvwxyz", 15
    )
    found, counts = bound.parse(bound.solve(ENABLE1, 3, 3, 200, "ae rst"))
    every = bound.exhaust(ENABLE1, 3, "aerst", 200)
    assert {bound.canonical(b, 3, 3): p for b, p in found.items()} == every
    assert len(every) > 50 and counts["pruned"] > 0


def test_canonical_is_the_least_image():
    assert bound.canonical("streaedlp", 3, 3) == anneal.canonical("streaedlp")
    assert (
        bound.canonical("abcdef", 2, 3) == "abcdef" and bound.canonical("fedcba", 2, 3) == "abcdef"
    )


def test_committed_summary_is_what_the_script_writes():
    paths = sorted((HERE / "results" / "bound-3x3").glob("bound-*.txt"))
    committed = json.loads((HERE / "results" / "best-3x3.json").read_text())
    assert bound.summary(paths) == committed
    for name in ("enable1", "enable2k"):
        assert committed[name]["slices"] == 20
        assert committed[name]["best"]["board"] == bound.canonical("streaedlp", 3, 3)
    assert committed["enable2k"]["boards found"] == 57  # as hybrid-boggle's README reports
