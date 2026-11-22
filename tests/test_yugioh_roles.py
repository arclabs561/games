"""Two-role cards, hand traps and the Normal Summon: the exact sum against brute force."""

from fractions import Fraction
from pathlib import Path

import pytest

from yugioh import odds, roles, second
from yugioh.odds import FIRST, SECOND, Deck
from yugioh.roles import group, need

FOLDER = Path(__file__).parent.parent / "yugioh"

SMALL = [
    (
        group(2, "starter", summon=True),
        group(1, "starter", "extender"),
        group(2, "extender"),
        group(2, "hand trap"),
        group(1, "starter", "hand trap"),
    ),
    (
        group(3, "starter", "extender", summon=True),
        group(2, "extender", summon=True),
        group(1, "starter"),
        group(2, "hand trap"),
        group(1, "brick"),
    ),
]
NEEDS = [
    need("starter"),
    need("starter", "extender"),
    need("starter", "extender", summons=1),
    need("starter", "starter|extender", summons=1),
    need("starter", "hand trap"),
    need("starter", "extender", "hand trap", summons=1),
    need("hand trap", "hand trap"),
    need("starter|hand trap"),
]


@pytest.mark.parametrize("deck", SMALL, ids=["a", "b"])
@pytest.mark.parametrize("total", [12, 14])
def test_exact_sum_matches_brute_force(deck, total):
    for n in (FIRST, SECOND):
        for want in NEEDS:
            assert roles.p_need(deck, n, want, total) == roles.p_brute(deck, n, want, total)


def test_by_hand():
    # Four cards, one a starter and an extender, three blank; two drawn. One card cannot be
    # both, so the need fails; counted twice, it holds whenever that card is drawn: 3/6.
    deck = (group(1, "starter", "extender"),)
    assert roles.p_need(deck, 2, need("starter", "extender"), 4) == 0
    assert roles.p_naive(deck, 2, ("starter", "extender"), 4) == Fraction(1, 2)
    # Two such cards in four: both must be drawn, 1 hand of 6.
    deck = (group(2, "starter", "extender"),)
    assert roles.p_need(deck, 2, need("starter", "extender"), 4) == Fraction(1, 6)
    # Two starters that need the Normal Summon and one extender that does too, in three cards,
    # all drawn: no pair uses it only once.
    deck = (group(2, "starter", summon=True), group(1, "extender", summon=True))
    assert roles.p_need(deck, 3, need("starter", "extender", summons=1), 3) == 0
    assert roles.p_need(deck, 3, need("starter", "extender"), 3) == 1


def test_one_role_per_card_gives_the_odds_values():
    for deck in (Deck(40, 12, 9, 3), Deck(45, 15, 6, 0), Deck(40, 9, 12, 2)):
        groups = tuple(
            group(k, r) for r, k in zip(odds.ROLES[:3], deck.counts()[:3], strict=True) if k
        )
        for n in (FIRST, SECOND):
            for want, old in (
                (need("starter"), odds.starter),
                (need("starter", "extender"), odds.starter_and_extender),
                (need("starter", "starter|extender"), odds.two_engine),
            ):
                assert roles.p_need(groups, n, want, deck.size) == odds.p_hand(deck, n, old)
            assert roles.p_naive(
                groups, n, ("starter", "extender"), deck.size
            ) == odds.p_starter_and_extender(deck, n)


def test_example_deck_variations_keep_their_totals():
    for traps in (0, 12):
        for kw in ({}, {"both": 3}, {"starters_ns": 8, "extenders_ns": 3}):
            assert roles.size(second.example(traps, **kw)) == 24 + traps
    deck = second.example(traps=0, both=12)
    # Every starter also extends: the strict need becomes a starter and another engine card.
    assert roles.p_need(deck, FIRST, second.STRICT, 40) == odds.p_hand(
        Deck(40, 12, 9, 3), FIRST, odds.two_engine
    )


def test_small_simulation_agrees():
    for _, deck, n, want in second.CHECKS:
        score = roles.p_going_second if n == SECOND else roles.p_need
        p = score(deck, n, want, second.SIZE)
        hits = roles.simulate(deck, n, want, second.SIZE, 4000, seed=2, going_second=n == SECOND)
        lo, hi = odds.wilson(hits, 4000, z=3.29)  # 99.9%: four rows, one seed
        assert lo <= p <= hi


def test_committed_tables_are_current():
    assert (FOLDER / "results" / "second.md").read_text() == second.exact_tables()


def test_committed_simulation_matches_exact_values():
    text = (FOLDER / "results" / "second_simulation.md").read_text()
    assert f"Seed {second.SEED}, {second.TRIALS:,} hands" in text
    body = [
        line for line in text.splitlines() if line.startswith("| ") and "Exact inside" not in line
    ]
    assert len(body) == len(second.CHECKS)
    for line, (label, deck, n, want) in zip(body, second.CHECKS, strict=True):
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[0] == label and cells[-1] == "yes"
        score = roles.p_going_second if n == SECOND else roles.p_need
        assert cells[1] == odds_pct(score(deck, n, want, second.SIZE))


def odds_pct(p: Fraction) -> str:
    return f"{float(p) * 100:.4f}%"


def test_late_card_can_start_but_cannot_interrupt():
    from itertools import permutations

    deck = (group(1, "hand trap"), group(1, "hand trap", "starter"), group(2, "starter"))
    cards = roles.physical(deck, 5)
    orders = list(permutations(cards, 3))

    def on_time(order):
        return any(
            t and "hand trap" in t.roles and s and "starter" in s.roles
            for i, t in enumerate(order[:2])
            for j, s in enumerate(order)
            if i != j
        )

    assert roles.p_going_second(deck, 3, second.STARTER_AND_TRAP, 5) == Fraction(
        sum(on_time(order) for order in orders), len(orders)
    )
    assert roles.p_going_second(second.plain(12, 12), 6, second.STARTER_AND_TRAP, 40) == Fraction(
        7981, 10545
    )
