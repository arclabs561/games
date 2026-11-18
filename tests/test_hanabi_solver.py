"""Hanabi: the open-game solver, its bounds, and the exact counts."""

from collections import Counter

import pytest

from hanabi.rules import Rules, card_name, full_deck, seeded_deck
from hanabi.solver import brute_winnable, replay, root_certificates, winnable
from hanabi.upper_bound import deals, multiset_orders, total_orders, verdict

TINY = [
    Rules(players=2, colors=2, counts=(2, 1), hand_size=1, max_hints=2, lives=1),
    Rules(players=2, colors=2, counts=(2, 1), hand_size=2, max_hints=1, lives=2),
    Rules(players=3, colors=2, counts=(2, 1), hand_size=1, max_hints=1, lives=2),
    Rules(
        players=1,
        colors=1,
        counts=(3, 2, 2, 1),
        hand_size=2,
        max_hints=0,
        lives=1,
        discard_when_full=True,
    ),
]


@pytest.mark.parametrize("rules", TINY)
def test_fast_solver_agrees_with_brute_force_on_every_deal(rules):
    for deck in multiset_orders(full_deck(rules)):
        brute = brute_winnable(rules, deck)
        fast, line = winnable(rules, deck)
        assert fast == brute == winnable(rules, deck, bounds=False)[0], deck
        if fast:
            assert replay(rules, deck, line) == rules.max_score
        if root_certificates(rules, deck):
            assert not brute
        for target in range(1, rules.max_score):
            assert winnable(rules, deck, target=target)[0] == brute_winnable(rules, deck, target)


def test_deal_weights_add_up():
    for rules in TINY + [Rules(players=2, colors=1, hand_size=3)]:
        assert sum(w for _, w in deals(rules)) == total_orders(rules)
    assert total_orders(Rules(players=1, colors=1)) == 75600


def test_van_den_bergh_hand_5():
    """van den Bergh (2015), Appendix A: 0.8778 of one-color decks for hand size 5."""
    rules = Rules(players=1, colors=1, hand_size=5, max_hints=0, lives=1, discard_when_full=True)
    won = sum(w for deck, w in deals(rules) if winnable(rules, deck)[0])
    assert won == 66360 and round(won / 75600, 4) == 0.8778


def test_all_ones_at_the_bottom_is_lost_by_time():
    rules = Rules(players=5)
    deck = sorted(full_deck(rules), key=lambda c: (c % 5 == 0, c))
    assert root_certificates(rules, deck)[0] == "time"


def test_a_full_deal_is_won_and_replayed():
    rules = Rules(players=3)
    deck = seeded_deck(rules, 0)
    assert verdict(rules, deck, budget=50000) == ("won", [])


def test_a_two_player_deal_lost_in_the_pooled_game():
    rules = Rules(players=2)
    deck = seeded_deck(rules, 161)
    assert "pooled" in root_certificates(rules, deck)
    assert Counter(card_name(c, rules) for c in deck)["W5"] == 1
