"""Hanabi with six colors: both solvers against brute force, and the committed results."""

import json
import random
from pathlib import Path

import pytest

from hanabi import dfs_solver as sb
from hanabi import hat_gap
from hanabi.rules import Rules, card_name, full_deck, make_card, seeded_deck
from hanabi.solver import brute_winnable, replay, root_certificates, winnable
from hanabi.upper_bound import classify, multiset_orders

RESULTS = Path(__file__).parent.parent / "hanabi" / "results"

SMALL_SIX = [
    Rules(players=2, colors=6, counts=(1,), hand_size=1, max_hints=1, lives=1),
    Rules(players=2, colors=6, counts=(1, 1), hand_size=1, max_hints=1, lives=1),
    Rules(players=3, colors=6, counts=(1, 1), hand_size=1, max_hints=1, lives=1),
]


def test_six_colors_deck_and_names():
    rules = Rules(colors=6)
    deck = full_deck(rules)
    assert len(deck) == 60 and rules.max_score == 30
    assert card_name(make_card(5, 5, rules), rules) == "P5"
    assert sorted(seeded_deck(rules, 7)) == deck


@pytest.mark.parametrize("k", range(len(SMALL_SIX)))
def test_both_solvers_agree_with_brute_force_with_six_colors(k):
    rules = SMALL_SIX[k]
    if k == 0:
        decks = list(multiset_orders(full_deck(rules)))  # all 720 orders
    else:
        rng = random.Random(f"six {k}")
        decks = []
        for _ in range(30):
            d = full_deck(rules)
            rng.shuffle(d)
            decks.append(d)
    for deck in decks:
        brute = brute_winnable(rules, deck)
        ok, line = winnable(rules, deck)
        assert ok == brute, deck
        if ok:
            assert replay(rules, deck, line) == rules.max_score
        b, actions = sb.solve(rules, deck)
        assert b == brute, deck
        if b:
            assert sb.replay_actions(rules, deck, actions) == rules.max_score
        if root_certificates(rules, deck):
            assert not brute


def test_a_full_six_color_deal_is_classified_by_both_solvers():
    r = classify(Rules(players=3, colors=6), 0, solver="both")
    assert r["verdict"] == r["verdict_b"] == "won"


def test_committed_six_color_lost_deals_hold_their_certificates():
    data = json.loads((RESULTS / "six-suits.json").read_text())
    cols = data["columns"]
    for row in data["deals"]:
        r = dict(zip(cols, row, strict=True))
        if r["certificates"]:
            rules = Rules(players=r["players"], colors=6)
            assert set(r["certificates"]) <= set(
                root_certificates(rules, seeded_deck(rules, r["seed"]))
            )
    for p, c in data["summary"]["hat bot"].items():
        assert c["below the ceiling"], p


def test_hat_gap_scores_replay():
    data = json.loads((RESULTS / "hat-gap.json").read_text())
    for w in (4, 8):
        stored = data["widths"][str(w)]["scores"][:40]
        assert "".join(chr(65 + s) for s in hat_gap.scores(3, w, 40)) == stored
    assert data["widths"]["4"]["mean"] == 22.2538  # the README's 22.254, game by game


def test_readme_table_is_the_committed_summary():
    from hanabi.six_suits import readme_rows

    readme = (RESULTS.parent / "README.md").read_text()
    note = (RESULTS / "six-suits.md").read_text()
    summary = json.loads((RESULTS / "six-suits.json").read_text())["summary"]["all seeds"]
    for line in readme_rows(summary):
        assert line in readme and line in note, line
