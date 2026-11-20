"""Tests for codenames/calibration.py, and checks of the committed results it wrote."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from codenames.calibration import (
    Picks,
    agreement_bounds,
    ece,
    ece_replicates,
    exact_repeats,
    parse_picks,
    plackett_luce,
    reliability,
    shared_pairs,
    shared_words,
    turn_scores,
)
from codenames.stats import cluster_draws

RESULTS = Path(__file__).parent.parent / "codenames" / "results"


def test_turn_scores_by_hand():
    q = np.array([0.5, 0.3, 0.2])
    s = turn_scores(q, 1)
    assert s["loglik"] == pytest.approx(math.log(0.3))
    assert s["confidence"] == 0.5 and s["hit"] == 0.0
    assert s["brier_top"] == pytest.approx(0.25)
    assert s["brier_all"] == pytest.approx(0.25 + 0.49 + 0.04)


def test_ece_by_hand_and_perfect_calibration():
    # bin 0.5-0.6 holds two turns at 0.55, one hit: gap 0.05; bin 0.9-1 one hit at 0.95: 0.05
    assert ece([0.55, 0.55, 0.95], [1, 0, 1]) == pytest.approx((2 * 0.05 + 0.05) / 3)
    assert ece([0.25] * 4, [1, 0, 0, 0]) == pytest.approx(0)
    rows = reliability([0.55, 0.55, 0.95], [1, 0, 1])
    assert [r["turns"] for r in rows] == [2, 1] and rows[0]["hits"] == 0.5


def test_ece_replicates_equal_weighted_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.1, 1, 40)
    hit = (rng.uniform(size=40) < conf).astype(float)
    games = [f"g{i % 7}" for i in range(40)]
    index, counts = cluster_draws(games, draws=50)
    reps = ece_replicates(conf, hit, index, counts)
    for r in range(5):
        w = counts[r][index]
        assert reps[r] == pytest.approx(ece(conf, hit, weights=w))


def test_plackett_luce_against_every_order():
    w = np.array([3.0, 1.0, 2.0, 0.5])
    total = sum(math.exp(plackett_luce(w, list(p))) for p in itertools.permutations(range(4)))
    assert total == pytest.approx(1.0)
    assert plackett_luce(w, [0]) == pytest.approx(math.log(3 / 6.5))
    assert plackett_luce(w, [0, 2]) == pytest.approx(math.log(3 / 6.5) + math.log(2 / 3.5))


def test_agreement_bounds_hold_for_every_distribution():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.dirichlet(np.ones(rng.integers(2, 8)))
        a = float((q**2).sum())
        b = agreement_bounds(a)
        assert b["lower"] <= q.max() + 1e-12 and q.max() <= b["upper"] + 1e-12


def pick(clue, words, picks, game, guesser):
    return Picks(clue, tuple(words.split()), tuple(picks), game, "test", guesser)


def test_pairs_and_repeats():
    a = pick("sea", "ship wave fish tree", ["wave"], "g1", "p1")
    b = pick("sea", "wave fish car moon", ["fish"], "g2", "p2")
    c = pick("sea", "ship wave", ["wave"], "g1", "p3")  # same game as a
    d = pick("sea", "car moon", ["car"], "g3", "p4")  # a's pick is not on its board
    e = pick("sea", "ship wave fish tree", ["wave"], "g4", "p5")  # repeats a
    turns = (a, b, c, d, e)
    assert shared_pairs(turns) == [(0, 1), (0, 4), (1, 4), (2, 4)]
    assert shared_words(a, b) == ("wave", "fish")
    assert exact_repeats(turns) == {
        "items_with_two_or_more_rows": 1,
        "pairs": 1,
        "pairs_agreeing": 1,
    }
    assert parse_picks("king, unicorn") == ("king", "unicorn")


def committed():
    return json.loads((RESULTS / "guesser-calibration.json").read_text())


def test_committed_calibration_reproduces_c1_and_its_own_bins():
    cal = committed()
    c1 = json.loads((RESULTS / "c1.json").read_text())
    for model, v in c1["untrained"].items():
        assert cal["guessers"][model.split("/")[-1]]["hit"]["mean"] == pytest.approx(
            v["top1"]["mean"], abs=1e-12
        )
    for model, v in c1["maps"].items():
        mine = cal["guessers"][f"{model.split('/')[-1]} + map"]["hit"]["mean"]
        assert mine == pytest.approx(np.mean(v["top1_by_seed"]), abs=1e-12)
    for name, g in cal["guessers"].items():
        if "reliability" in g:  # ECE from its own bins
            n = sum(b["turns"] for b in g["reliability"])
            gap = sum(b["turns"] * abs(b["hits"] - b["confidence"]) for b in g["reliability"])
            assert gap / n == pytest.approx(g["ece"]["mean"]), name
            assert n == g["turns"] == cal["rows"]["test_one_pick"] == 551
    people = cal["people"]
    assert people["best_top1_bounds"]["upper"] == pytest.approx(
        math.sqrt(people["agreement"]["mean"])
    )
    rows = cal["rows"]
    assert rows["one_pick"] == 6088 and rows["one_pick"] + rows["several_picks"] == rows["all_kept"]


def test_committed_calibration_holds_no_strings_from_the_data():
    allowed_keys = set()

    def leaves(x, out):
        if isinstance(x, dict):
            for k, v in x.items():
                allowed_keys.add(k)
                leaves(v, out)
        elif isinstance(x, list):
            for v in x:
                leaves(v, out)
        elif isinstance(x, str):
            out.append(x)
        return out

    strings = leaves(committed(), [])
    words = set((RESULTS.parent / "words.txt").read_text().split())
    assert not [s for s in strings if s in words]
    assert all(len(s.split()) > 1 or "/" in s or s in {"bare", "word", "query"} for s in strings)
