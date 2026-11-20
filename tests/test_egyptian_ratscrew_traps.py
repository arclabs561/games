"""The census of small decks with a trapped position (egyptian_ratscrew/traps.py)."""

import json
from collections import Counter
from itertools import combinations_with_replacement

import pytest

from egyptian_ratscrew import traps
from egyptian_ratscrew.game import CHANCES
from egyptian_ratscrew.termination import decks_with_traps


def deck_type(deck) -> tuple:
    """The type of a deck: its royals, and the sorted copy counts of its number ranks."""
    numbers = Counter(c for c in deck if c not in CHANCES)
    royals = tuple(sorted((c for c in deck if c in CHANCES), key=list(CHANCES).index))
    return royals, tuple(sorted(numbers.values(), reverse=True))


@pytest.mark.parametrize("cards", [2, 3, 4, 5])
def test_deck_types_against_every_deck(cards):
    ranks = (2, 3, 4, 5, 6, 7, 8, 9, 10, *CHANCES)
    every = {
        deck_type(d)
        for d in combinations_with_replacement(ranks, cards)
        if any(c in CHANCES for c in d) and max(Counter(d).values()) <= 4
    }
    listed = [deck_type(d) for d in traps.deck_types(cards)]
    assert len(listed) == len(set(listed)) == len(every)
    assert set(listed) == every


def test_partitions():
    assert list(traps.partitions(5)) == [
        (4, 1),
        (3, 2),
        (3, 1, 1),
        (2, 2, 1),
        (2, 1, 1, 1),
        (1,) * 5,
    ]


def test_small_census_matches_the_committed_file():
    record = {(r["cards"], r["players"]): r for r in json.loads(traps.RESULTS.read_text())}
    for cards in (4, 5, 6):
        assert traps.census(cards, 2) == record[cards, 2]


def test_six_cards_agree_with_the_older_search():
    older = {" ".join(map(str, d)) for d in decks_with_traps(6, 2)}
    record = {(r["cards"], r["players"]): r for r in json.loads(traps.RESULTS.read_text())}
    found = {t["deck"] for t in record[6, 2]["with_traps"]}
    # The older search uses the ranks 2 to 5 only; every deck type it can name agrees.
    assert {deck_type(d.split()) for d in older} == {deck_type(d.split()) for d in found}


def test_committed_census_is_consistent():
    rows = json.loads(traps.RESULTS.read_text())
    for r in rows:
        assert r["deck_types"] == len(traps.deck_types(r["cards"]))
        assert r["all_traps_slap_free"]
        assert all(t["trapped_positions"] > 0 for t in r["with_traps"])


def test_a_deal_that_never_ends_though_slaps_happen():
    # Every position this deal can reach, whoever wins each slap, is still being played.
    assert traps.never_ends(traps.NEVER_ENDS) == (745, 16)
    # The six-card trap never offers a slap; a deal that can end is not reported.
    assert traps.never_ends(((("J", 2, 3)), (2, "J", 3)))[1] == 0
    assert traps.never_ends(((2, "J"), (3, "Q"))) is None


def test_the_deal_never_ends_in_the_rules_engine():
    import random

    from egyptian_ratscrew.game import play_game

    rng = random.Random(3)
    games = [play_game([1, 1], rng, max_cards=5_000, hands=traps.NEVER_ENDS) for _ in range(50)]
    assert all(g.winner is None for g in games)


def test_components_and_closure_by_hand():
    # 0 -> 1 -> 2 -> 1, 0 -> 3 -> 3: closed classes {1, 2} and {3}.
    edges = [[(None, 1), (None, 3)], [(None, 2)], [(None, 1)], [(None, 3)]]
    comps = sorted(sorted(c) for c in traps.components([0, 1, 2, 3], edges))
    assert comps == [[0], [1, 2], [3]]
