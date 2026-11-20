"""Playing for a target score or against an opponent, and head to head against the cheat sheet."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tests.test_yahtzee import SMALL
from yahtzee import policy, target, versus
from yahtzee.brute import brute_force
from yahtzee.rules import OFFICIAL
from yahtzee.solver import Tables, best_keep, final_rolls, solve

RESULTS = Path(__file__).parent.parent / "yahtzee" / "results"
FAST = SMALL[:3] + SMALL[4:5]  # the small games that brute force settles in seconds


@pytest.mark.parametrize("rules", FAST, ids=lambda r: "-".join(r.categories))
def test_reach_matches_brute_force(rules):
    f = target.reach(rules)
    top = int(np.nonzero(f > 0)[0][-1])
    for t in sorted({1, top // 3, top // 2, top, top + 1}):
        assert f[t] == pytest.approx(
            float(brute_force(rules, lambda s, t=t: Fraction(s >= t))), abs=1e-12
        )


@pytest.mark.parametrize("rules", FAST, ids=lambda r: "-".join(r.categories))
def test_respond_matches_brute_force(rules):
    t, values = solve(rules)
    opponent = policy.distribution(policy.graph(t, values))
    top = len(opponent)
    for tie in (0.0, 0.5):

        def utility(s: int, tie: float = tie) -> Fraction:
            below = sum((Fraction(x) for x in opponent[: min(s, top)]), Fraction(0))
            return below + (Fraction(tie) * Fraction(opponent[s]) if s < top else 0)

        value = target.respond(opponent, tie, rules)
        assert value == pytest.approx(float(brute_force(rules, utility)), abs=1e-12)


@pytest.mark.parametrize("rules", SMALL, ids=lambda r: f"{'-'.join(r.categories)}-{r.joker}")
def test_columns_left_out_change_nothing(rules):
    f = target.reach(rules)
    full = target.solve(rules, target.reach_end(len(f)), len(f), "reach", window=False)
    assert np.abs(f - full).max() < 1e-14
    t, values = solve(rules)
    opponent = policy.distribution(policy.graph(t, values))
    a = target.respond(opponent, 0.5, rules)
    top = len(opponent)
    end = target.respond_end(opponent, 0.5)
    b = target.solve(rules, end, top + 1, "respond", window=False)[top]
    assert a == pytest.approx(b, abs=1e-14) and not np.isnan(a)


def test_more_workers_give_the_same_vector():
    rules = SMALL[4]
    assert np.array_equal(target.reach(rules), target.reach(rules, workers=2))


def test_keeps_one_die_at_a_time_match_the_matrix():
    t = Tables(OFFICIAL)
    k = target.Keeps(t)
    x = np.random.default_rng(0).random((len(t.rolls), 7))
    avg = k.average(x)
    assert np.allclose(avg, t.P @ x, rtol=0, atol=1e-15)
    assert np.allclose(k.best(avg.copy()), best_keep(t, (t.P @ x).T).T, rtol=0, atol=1e-15)


@pytest.mark.parametrize("rules", SMALL, ids=lambda r: f"{'-'.join(r.categories)}-{r.joker}")
def test_box_moves_match_the_solver(rules):
    t, values = solve(rules)
    for filled in range(len(rules.categories)):
        states = t.states(filled)
        best, _ = final_rolls(t, values, states)
        mine = np.full(best.shape, -np.inf)
        for allowed, points, nxt in target.box_moves(t, states):
            v = np.where(allowed, points + values[np.where(allowed, nxt, 0)], -np.inf)
            mine = np.maximum(mine, v)
        assert np.allclose(mine, best, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rules", SMALL, ids=lambda r: f"{'-'.join(r.categories)}-{r.joker}")
def test_bounds_shrink_by_at_least_the_points(rules):
    t = Tables(rules)
    for filled in range(len(rules.categories)):
        states = t.states(filled)
        rest, so_far = target.bounds(t, states)
        for allowed, points, nxt in target.box_moves(t, states):
            r2, s2 = target.bounds(t, np.where(allowed, nxt, 0).ravel())
            r2, s2 = r2.reshape(allowed.shape), s2.reshape(allowed.shape)
            assert (~allowed | (rest[:, None] >= points + r2)).all()
            assert (~allowed | (s2 >= so_far[:, None] + points)).all()
    t = Tables(OFFICIAL)
    assert target.bounds(t, np.array([t.index(0, 0, 0)]))[0][0] == target.OFFICIAL_WIDTH - 2


def test_duel_and_margin_by_hand():
    a, b = np.array([0.5, 0.5]), np.array([0.25, 0.5, 0.25])  # A in {0, 1}, B in {0, 1, 2}
    d = versus.duel(a, b)
    assert d == pytest.approx({"win": 0.125, "tie": 0.375, "loss": 0.5})
    m, first = versus.margin(a, b)
    assert first == -2 and m == pytest.approx([0.125, 0.375, 0.375, 0.125])


def committed_versus() -> dict:
    return json.loads((RESULTS / "versus.json").read_text())


def test_committed_head_to_head():
    data = committed_versus()
    sheet = np.array(data["sheet_distribution"])
    cards = json.loads((RESULTS / "cards.json").read_text())["cheat"]
    assert sheet.sum() == pytest.approx(1, abs=1e-12)
    assert data["sheet_mean"] == pytest.approx(cards["score"], abs=1e-9)
    summary = policy.summary(sheet)
    assert summary["median"] == cards["summary"]["median"]
    assert summary["sd"] == pytest.approx(cards["summary"]["sd"], abs=1e-9)
    best = target.optimal_distribution()
    assert data["optimal_against_sheet"] == versus.duel(best, sheet)
    m, first = versus.margin(best, sheet)
    assert data["margin"] == versus.summary(m, first)
    assert data["margin"]["mean"] == pytest.approx(254.5896094820 - cards["score"], abs=1e-8)
    sim = data["simulated"]
    for key in ("win", "tie"):
        lo, hi = sim[f"{key}_interval"]
        assert lo <= data["optimal_against_sheet"][key] <= hi
