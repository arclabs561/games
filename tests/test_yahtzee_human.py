"""Yahtzee strategy cards: the exact evaluator against the solver and simulation, and results."""

import json
from pathlib import Path

import numpy as np
import pytest

from yahtzee import advice, cards
from yahtzee.glenn import GLENN
from yahtzee.human import (
    BASE,
    Box,
    Keep,
    Strategy,
    card_values,
    decision_key,
    expected_score,
    moves,
    play,
)
from yahtzee.learn import from_json, to_json
from yahtzee.rules import OFFICIAL
from yahtzee.solver import Tables, solve

HERE = Path(__file__).parent.parent / "yahtzee"
# A card that reads many of the tests, so that the checks below exercise them.
BUSY = Strategy(
    keep_rules=(
        *BASE.keep_rules,
        Keep(("one roll left", "FH open", "a pair or more"), "pairs"),
        Keep(("on pace for the bonus", "the most common face is 2 or less"), "fives"),
        Keep(("Yahtzee box holds 50", "nine or more boxes open"), "most"),
    ),
    box_rules=(
        Box(("the bonus is made", "it scores"), "four_kind"),
        Box(("three or more of a kind", "its upper box is open"), "face"),
        Box(("the dice total 25 or more", "six or fewer boxes open"), "chance"),
    ),
    weight=1.5,
    default="most open",
)


@pytest.fixture(scope="module")
def t():
    return Tables(OFFICIAL)


def test_card_values_match_the_solver_following_the_card(t):
    """Two programs: the solver with every choice fixed, and the card's own backward pass."""
    _, a = solve(OFFICIAL, rule=BUSY, stop=10)
    _, b = card_values(BUSY, t, stop=10)
    s = np.concatenate([t.states(k) for k in (10, 11, 12)])
    assert np.abs(a[s] - b[s]).max() < 1e-9


def test_forward_and_backward_agree(t):
    _, values = card_values(BUSY, t)
    assert expected_score(BUSY, t) == pytest.approx(values[t.index(0, 0, 0)], abs=1e-9)


def test_moves_depend_only_on_the_decision_key(t):
    """Working out each key once gives the moves of every state with that key."""
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(t.states(6), 400, replace=False))
    assert len(np.unique(decision_key(t, s))) < len(s)
    a, b = moves(BUSY, t, s), moves(BUSY, t, s, dedupe=False)
    for x, y in zip(
        [*a.keeps, a.box, a.gain, a.next], [*b.keeps, b.box, b.gain, b.next], strict=True
    ):
        assert (x == y).all()


@pytest.mark.parametrize("card", [BUSY, GLENN], ids=["busy", "glenn"])
def test_simulation_agrees_with_the_exact_score(t, card):
    """Seeded games scored by the rules module, not the solver's tables."""
    scores = play(card, t, 3000, seed=2)
    exact = expected_score(card, t)
    assert abs(scores.mean() - exact) < 3 * scores.std() / np.sqrt(len(scores))


def test_json_round_trip():
    assert from_json(to_json(BUSY)) == BUSY


def test_committed_results(t):
    data = json.loads((HERE / "results" / "cards.json").read_text())
    assert data["glenn"] == pytest.approx(expected_score(GLENN, t), abs=1e-9)
    sheet = from_json(data["cheat"]["card"])
    assert data["cheat"]["score"] == pytest.approx(expected_score(sheet, t), abs=1e-9)
    for row in data["frontier"][:4]:
        assert row["score"] == pytest.approx(expected_score(from_json(row["card"]), t), abs=1e-9)
    assert [r["size"] for r in data["frontier"]] == sorted(r["size"] for r in data["frontier"])


def test_cheat_sheet_in_readme():
    data = json.loads((HERE / "results" / "cards.json").read_text())
    text = (HERE / "README.md").read_text()
    for line in cards.sheet_lines(from_json(data["cheat"]["card"])):
        assert line in text, line


def test_committed_advice_matches_one_recomputation():
    rows = json.loads((HERE / "results" / "advice.json").read_text())
    names = [a.name for a in advice.ADVICE]
    assert [r["name"] for r in rows["cost"]] == names
    i = names.index("A forced zero goes in the upper section")
    assert advice.cost(i)["value"] == pytest.approx(rows["cost"][i]["value"], abs=1e-9)
