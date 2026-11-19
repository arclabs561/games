"""Yahtzee: the fast solver against brute force, closed forms and the published values."""

import csv
from fractions import Fraction
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from yahtzee import policy
from yahtzee.brute import brute_force
from yahtzee.rules import Rules, score
from yahtzee.simulate import deal, play
from yahtzee.solver import reachable_count, solve, turn

HERE = Path(__file__).parent.parent / "yahtzee"
SMALL = [
    Rules(dice=2, faces=3, rolls=2, categories=("aces", "twos", "threes", "chance"), threshold=8),
    Rules(
        dice=3,
        faces=4,
        rolls=2,
        categories=("fours", "small_straight", "yahtzee", "three_kind"),
        threshold=8,
    ),
    Rules(dice=4, faces=3, rolls=2, categories=("threes", "four_kind", "yahtzee"), threshold=9),
    Rules(
        dice=5, faces=2, rolls=2, categories=("aces", "twos", "full_house", "yahtzee"), threshold=9
    ),
] + [
    # Free choice, forced placement and no Joker give three different values here.
    Rules(
        dice=3,
        faces=3,
        rolls=2,
        categories=("twos", "threes", "full_house", "yahtzee", "chance"),
        threshold=5,
        joker=j,
    )
    for j in ("free", "forced", "none")
]


def start_value(rules: Rules) -> float:
    t, values = solve(rules)
    return float(values[t.index(0, 0, 0)])


def test_scores():
    assert score((2, 3, 4, 5, 5), "small_straight") == 30
    assert score((1, 2, 3, 4, 5), "small_straight") == 30  # a large straight is also a small one
    assert score((3, 3, 3, 5, 5), "three_kind") == 19
    assert score((3, 3, 3, 3, 5), "full_house") == 0
    assert score((6,) * 5, "four_kind") == 30


@pytest.mark.parametrize(
    "rules", SMALL, ids=lambda r: f"{r.dice}d{r.faces}-{len(r.categories)}-{r.joker}"
)
def test_solver_matches_brute_force(rules):
    assert start_value(rules) == pytest.approx(float(brute_force(rules)), abs=1e-10)


def test_joker_rules_differ_in_the_small_game():
    values = [start_value(r) for r in SMALL[-3:]]
    assert values[0] > values[1] > values[2]


def yahtzee_chance(rolls: int = 3) -> Fraction:
    """The chance of five of a kind, keeping the largest group after every roll."""

    @cache
    def after(best: int, left: int) -> Fraction:
        if best == 5 or left == 0:
            return Fraction(best == 5)
        total = Fraction(0)
        rerolls = list(product(range(6), repeat=5 - best))  # face 0 is the kept one
        for new in rerolls:
            others = max((new.count(v) for v in range(1, 6)), default=0)
            total += after(max(best + new.count(0), others), left - 1)
        return total / len(rerolls)

    first = list(product(range(6), repeat=5))
    return sum((after(max(r.count(v) for v in r), rolls - 1) for r in first), Fraction(0)) / len(
        first
    )


def test_one_box_games():
    p = yahtzee_chance()
    assert start_value(Rules(categories=("yahtzee",))) == pytest.approx(50 * float(p), abs=1e-12)
    # Chance alone: keep a die showing 5 or 6 after the first roll, 4 or more after the second.
    assert start_value(Rules(categories=("chance",))) == pytest.approx(70 / 3, abs=1e-12)


def test_chance_distribution_is_five_independent_dice():
    die = np.array([0, 1, 1, 1, 3, 6, 6]) / 18  # final face of one die: 1/18, ..., 1/3, 1/3
    expected = np.array([1.0])
    for _ in range(5):
        expected = np.convolve(expected, die)
    t, values = solve(Rules(categories=("chance",)))
    dist = policy.distribution(policy.graph(t, values))
    assert np.allclose(dist, expected[: len(dist)], atol=1e-15) and len(dist) == 31


def test_endgame_closed_form():
    # Only Aces open and an upper total of 62: keep every 1. Each die is a 1 at the end with
    # chance 91/216, and the bonus needs one 1 among up to 15 dice.
    rules = Rules()
    t, values = solve(rules, stop=12)
    mask = (1 << 13) - 1 - t.bit["aces"]
    assert values[t.index(mask, 62, 0)] == pytest.approx(
        5 * 91 / 216 + 35 * (1 - (5 / 6) ** 15), abs=1e-12
    )
    assert values[t.index((1 << 13) - 1 - t.bit["yahtzee"], 0, 0)] == pytest.approx(
        50 * float(yahtzee_chance()), abs=1e-12
    )


@pytest.mark.parametrize("rules", SMALL[-3:], ids=["free", "forced", "none"])
def test_distribution_and_simulation_agree_with_value(rules):
    t, values = solve(rules)
    g = policy.graph(t, values)
    dist = policy.distribution(g)
    value = values[t.index(0, 0, 0)]
    assert dist.sum() == pytest.approx(1, abs=1e-12)
    assert policy.summary(dist)["mean"] == pytest.approx(value, abs=1e-9)
    s = g.stats
    assert s["points"].sum() + s["bonus"][0] + rules.upper_bonus * s["upper_bonus"][
        0
    ] == pytest.approx(value, abs=1e-9)
    games = play(t, values, deal(t, 20_000, seed=7))
    assert abs(games.mean() - value) < 4 * games.std() / np.sqrt(len(games))
    empirical = np.bincount(games, minlength=len(dist)) / len(games)
    assert np.abs(np.cumsum(empirical) - np.cumsum(dist)).max() < 0.02  # Kolmogorov distance


def test_greedy_is_worse_and_simulates_to_its_exact_value():
    rules = SMALL[-3]
    t, greedy_values = solve(rules, anchor=lambda s: np.zeros(len(s)))
    value = policy.summary(policy.distribution(policy.graph(t, greedy_values)))["mean"]
    assert value < start_value(rules) - 1
    games = play(t, greedy_values, deal(t, 20_000, seed=8))
    assert abs(games.mean() - value) < 4 * games.std() / np.sqrt(len(games))


def test_reachable_states():
    assert reachable_count() == 536_448  # Verhoeff's count of the states that need computing


@pytest.fixture(scope="module")
def official():
    return solve(Rules())


def test_published_values(official):
    t, values = official
    assert values[t.index(0, 0, 0)] == pytest.approx(254.5896, abs=5e-5)  # Verhoeff; Glenn: 254.59
    tr = turn(t, values, np.array([t.index(0, 0, 0)]), choices=True)
    k0, k1 = tr.keep_values[0][0], tr.keep_values[1][0]
    k, b = t.keep_index, t.bit
    # Verhoeff's "dilemmas resolved", to two decimals.
    assert round(k0[k[(6, 6, 6)]], 2) == 265.12
    assert round(25 + values[t.index(b["full_house"], 0, 0)], 2) == 253.91
    for keep, published in (
        ((3, 4), 245.17),
        ((1, 1), 245.14),
        ((4,), 244.96),
        ((3,), 244.74),
        ((), 244.55),
        ((6,), 244.52),
    ):
        assert round(k1[k[keep]], 2) == published
    assert round(24 + values[t.index(b["sixes"], 24, 0)], 2) == 268.23
    assert round(25 + values[t.index(b["four_kind"], 0, 0)], 2) == 260.54
    assert round(50 + values[t.index(b["yahtzee"], 0, 1)], 2) == 320.84


def test_committed_openings_are_current(official):
    t, values = official
    tr = turn(t, values, np.array([t.index(0, 0, 0)]), choices=True)
    rows = list(csv.DictReader(open(HERE / "openings.csv")))
    assert len(rows) == 252 and sum(int(r["ways"]) for r in rows) == 6**5
    for i, r in enumerate(rows):
        assert r["roll"] == "".join(map(str, t.rolls[i]))
        assert r["keep"] == ("".join(map(str, t.keeps[tr.keeps[0][0, i]])) or "-")
        assert float(r["expected"]) == pytest.approx(tr.levels[0][0, i], abs=5e-5)
    mean = sum(int(r["ways"]) * tr.levels[0][0, i] for i, r in enumerate(rows)) / 6**5
    assert mean == pytest.approx(values[t.index(0, 0, 0)], abs=1e-9)


def test_committed_distribution(official):
    t, values = official
    rows = list(csv.DictReader(open(HERE / "distribution.csv")))
    x = np.array([int(r["score"]) for r in rows])
    p = np.array([float(r["probability"]) for r in rows])
    assert p.sum() == pytest.approx(1, abs=1e-12)
    assert x @ p == pytest.approx(values[t.index(0, 0, 0)], abs=1e-9)
    assert np.sqrt(((x - x @ p) ** 2) @ p) == pytest.approx(59.6117, abs=5e-5)  # Verhoeff
    assert (x.min(), x.max()) == (
        12,
        1575,
    )  # Verhoeff's least score of optimal play; the most possible
