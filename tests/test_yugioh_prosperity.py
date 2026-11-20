"""Pot of Prosperity (yugioh/prosperity.py): exact sums against brute force and simulation."""

from fractions import Fraction
from pathlib import Path

import pytest

from yugioh import prosperity as pros
from yugioh.draws import Engine, p_engine
from yugioh.odds import FIRST, SECOND, Deck, starter, starter_and_extender, wilson


@pytest.mark.parametrize(
    ("counts", "n", "look"),
    [
        ((2, 1, 1, 1, 1, 1), 3, 3),
        ((1, 2, 1, 2, 1, 1), 3, 2),
        ((2, 2, 0, 1, 1, 1), 2, 4),
        ((1, 1, 1, 3, 1, 1), 3, 3),
        ((1, 1, 0, 3, 1, 0), 2, 3),
    ],
)
@pytest.mark.parametrize("need", [starter, starter_and_extender])
def test_exact_equals_every_order_of_a_small_deck(counts, n, look, need):
    s, e, b, o, _, d2 = counts
    deck = Deck(sum(counts), s, e, b)
    assert pros.p_prosperity(deck, n, need, look, d2) == pros.p_brute(counts, n, look, need)


def test_by_hand():
    # Deck S, O, Prosperity; hand of 1. The hand holds S (1/3), O (1/3, fails), or
    # Prosperity (1/3), which reveals both other cards and takes S. So 2/3.
    assert pros.p_prosperity(Deck(3, 1), 1, starter, look=2) == Fraction(2, 3)
    # Revealing only 1 finds S half the time: 1/3 + 1/3 * 1/2.
    assert pros.p_prosperity(Deck(3, 1), 1, starter, look=1) == Fraction(1, 2)


def test_alone_it_is_one_generic_looker():
    for n in (FIRST, SECOND):
        for need in (starter, starter_and_extender):
            assert pros.p_prosperity(pros.EXAMPLE, n, need, 6) == p_engine(
                pros.EXAMPLE, Engine(lookers=1, look=6), n, need
            )


def test_the_no_draw_rule_costs_something():
    real = pros.p_prosperity(pros.EXAMPLE, FIRST, starter, 6, 3)
    generic = p_engine(pros.EXAMPLE, Engine(draw2=3, lookers=1, look=6), FIRST, starter)
    assert real < generic
    best = pros.p_prosperity(pros.EXAMPLE, FIRST, starter_and_extender, 6, 3)
    for policy in ("prosperity", "draw2"):
        assert pros.p_prosperity(pros.EXAMPLE, FIRST, starter_and_extender, 6, 3, policy) <= best


@pytest.mark.parametrize("preferred", ["prosperity", "draw2"])
def test_simulation_agrees(preferred):
    trials = 20_000
    hits = pros.simulate(pros.EXAMPLE, FIRST, starter, 6, 3, preferred, trials, seed=2)
    lo, hi = wilson(hits, trials)
    assert lo <= pros.p_prosperity(pros.EXAMPLE, FIRST, starter, 6, 3, preferred) <= hi


def test_committed_results_are_current():
    path = Path(pros.__file__).parent / "results" / "prosperity.md"
    assert path.read_text() == pros.report()
