"""Egyptian Ratscrew: exact challenge odds, simulation checks, and Beggar-my-neighbour records."""

import random
from collections import Counter
from fractions import Fraction
from itertools import permutations
from math import comb

from egyptian_ratscrew.game import (
    CHANCES,
    DECK,
    ROYALS,
    answer_chance,
    challenger_keeps,
    deal,
    first_challenge,
    play,
    simulate_challenge,
    slappable,
)


def hand(notation):
    """A hand in the usual Beggar-my-neighbour notation: '-' for any card that is not a royal."""
    return [card if card in CHANCES else 2 for card in notation]


def test_deal_shares_the_deck():
    hands = deal(3, random.Random(1))
    assert Counter(c for h in hands for c in h) == Counter(DECK)
    assert sorted(map(len, hands)) == [17, 17, 18]


def test_slaps():
    assert slappable([5, 5]) and slappable(["K", 2, "K"]) and not slappable(["K", 2, 3, "K"])


def test_answer_chance_by_counting_hands():
    for royal in ROYALS:
        k = CHANCES[royal]
        assert answer_chance(k) == Fraction(comb(51, k) - comb(36, k), comb(51, k))


def test_challenger_keeps_matches_every_order_of_a_small_deck():
    def by_enumeration(chances, cards):
        orders = set(permutations(cards))
        keeps = 0
        for order in orders:
            owed, challenger = chances, 0
            for card in order:
                if card in CHANCES:
                    challenger, owed = 1 - challenger, CHANCES[card]
                else:
                    owed -= 1
                    if owed == 0:
                        break
            keeps += challenger == 0
        return Fraction(keeps, len(orders))

    assert by_enumeration(2, ["J", "Q", "A", 2, 2, 2, 2]) == challenger_keeps(2, (1, 1, 0, 1), 4)
    assert by_enumeration(3, ["J", "J", "K", 2, 2, 2, 2, 2]) == challenger_keeps(3, (2, 0, 1, 0), 5)
    assert by_enumeration(4, ["Q", "K", "A", 3, 3, 3]) == challenger_keeps(4, (0, 1, 1, 1), 3)


def test_first_challenge_agrees_with_simulation():
    rng, n = random.Random(1), 20_000
    for royal in ROYALS:
        p = float(first_challenge(royal))
        observed = sum(simulate_challenge(royal, rng) for _ in range(n)) / n
        assert abs(observed - p) < 4 * (p * (1 - p) / n) ** 0.5


def test_beggar_my_neighbour_records():
    """With slaps off the game is Beggar-my-neighbour: two record deals end after exactly the
    published number of cards, and the first deal known to loop never ends."""
    records = [
        ("---AJ--Q---------QAKQJJ-QK", "-----A----KJ-K--------A---", 8344),  # Nessler, 2022
        ("--A-Q--J--J---Q--AJ-K---K-", "-J-------Q------A--A--QKK-", 7225),  # Wu, 2012
    ]
    for first, second, cards in records:
        laid, winner = play(
            [1, 1], random.Random(0), hands=[hand(first), hand(second)], slaps=False
        )
        assert laid == cards and winner is not None
    endless = [hand("---K---Q-KQAJ-----AAJ--J--"), hand("----------Q----KQ-J-----KA")]
    assert play([1, 1], random.Random(0), hands=endless, slaps=False) == (100_000, None)


def test_games_are_seeded_and_finish():
    first = [play([0.5, 0.5], random.Random(7)) for _ in range(3)]
    assert first == [play([0.5, 0.5], random.Random(7)) for _ in range(3)]
    rng = random.Random(3)
    assert all(winner is not None for _, winner in (play([0.25] * 4, rng) for _ in range(200)))


def test_a_player_who_wins_every_slap_usually_wins():
    rng = random.Random(5)
    wins = sum(play([1, 0], rng)[1] == 0 for _ in range(300))
    assert wins > 290
