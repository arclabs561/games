"""Egyptian Ratscrew: exact challenge odds, simulation checks, and Beggar-my-neighbour records."""

import random
from collections import Counter
from fractions import Fraction
from math import comb

from egyptian_ratscrew.game import (
    CHANCES,
    DECK,
    ROYALS,
    Draw,
    answer_chance,
    challenger_keeps,
    challenger_keeps_by_orders,
    deal,
    first_challenge,
    play,
    play_game,
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
    assert challenger_keeps_by_orders(2, ["J", "Q", "A", 2, 2, 2, 2]) == challenger_keeps(
        2, (1, 1, 0, 1), 4
    )
    assert challenger_keeps_by_orders(3, ["J", "J", "K", 2, 2, 2, 2, 2]) == challenger_keeps(
        3, (2, 0, 1, 0), 5
    )
    assert challenger_keeps_by_orders(4, ["Q", "K", "A", 3, 3, 3]) == challenger_keeps(
        4, (0, 1, 1, 1), 3
    )


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


def test_tricks_match_the_beggar_my_neighbour_checker():
    """Cards and tricks for four record deals, as BeggarMyPython (the checker linked from
    Richard Mann's list) counts them. Mann's list gives 713 tricks for Manasse's deal and 6,914
    cards for Collins's; the checker and this code give 712 and 6,913."""
    deals = [
        ("---AJ--Q---------QAKQJJ-QK", "-----A----KJ-K--------A---", 8344, 1164),
        ("--A-Q--J--J---Q--AJ-K---K-", "-J-------Q------A--A--QKK-", 7225, 1016),
        ("A-QK------Q----KA-----J---", "-JAK----A--Q----J---QJ--K-", 6913, 960),
        ("Q--J----K--K-J---Q---A---A", "---K-K--JA-QA--J-----Q----", 5104, 712),
    ]
    for first, second, cards, tricks in deals:
        game = play_game([1, 1], random.Random(0), hands=[hand(first), hand(second)], slaps=False)
        assert (game.cards, game.tricks) == (cards, tricks)


def test_one_trick_deals():
    """Truman Collins lists four deals that end in one trick, and they do. (His page labels
    their winners the other way round from his 960-trick deal, so the winners are not checked.)"""
    deals = [
        ("0000000K00Q000J00Q00JQK000", "00000000K0AA0AQJA000000KJ0"),
        ("0000000000000KA00J0JJ0K00Q", "0000000000000Q00AKKA0AQJQ0"),
        ("0000000000KA00Q00KAAQ0J00Q", "000000000J00J00KA0Q0K000J0"),
        ("0000000000A0Q00JAKKQ00A0J0", "000000000000Q0KK0J0J0Q0A00"),
    ]
    for first, second in deals:
        game = play_game([1, 1], random.Random(0), hands=[hand(first), hand(second)], slaps=False)
        assert game.tricks == 1


def test_play_is_play_game_without_misslaps():
    for seed in range(5):
        game = play_game([0.6, 0.4], random.Random(seed))
        assert play([0.6, 0.4], random.Random(seed)) == (game.cards, game.winner)
        assert game.paid == 0
    assert play_game([0.6, 0.4], random.Random(1), misslaps=[0, 0]) == play_game(
        [0.6, 0.4], random.Random(1)
    )


def test_games_with_misslaps_end():
    """With heavy misslapping, games of two, three and four players still end."""
    rng = random.Random(11)
    for players in (2, 3, 4):
        for _ in range(100):
            game = play_game([1 / players] * players, rng, misslaps=[0.2] * players)
            assert game.winner is not None and game.paid > 0


def test_misslap_pays_one_card_under_the_pile():
    """Player 1 misslaps after every card. Player 0 lays the 2, and player 1 pays the 3 for
    misslapping. Player 1 lays the 6, their last card, and misslaps with nothing to pay, which
    puts them out. Player 0 holds every card not on the table and wins after two cards, with
    one card paid and one pile left to pick up."""
    game = play_game([0.5, 0.5], random.Random(0), hands=[[2, 4, 5], [3, 6]], misslaps=[0, 1])
    assert game == (2, 0, 1, 1)


def test_a_careless_player_loses_more():
    rng = random.Random(4)
    careful = sum(play_game([0.5, 0.5], rng, misslaps=[0, 0.1]).winner == 0 for _ in range(400))
    assert careful > 300


def test_challenge_by_hand_on_three_cards():
    # A jack is laid; unseen are one queen and two other cards. The next card is another
    # card (2/3), and the challenger keeps the pile, or the queen (1/3), and then nobody can
    # answer it from two plain cards. So the challenger keeps it with chance 2/3.
    assert challenger_keeps(1, (0, 1, 0, 0), 2) == Fraction(2, 3)
    assert challenger_keeps_by_orders(1, ["Q", 2, 2]) == Fraction(2, 3)


def test_committed_svgs_are_what_the_script_writes(tmp_path):
    from egyptian_ratscrew import figures

    figures.main(tmp_path)
    for name in figures.FIGURES:
        assert (tmp_path / name).read_bytes() == (figures.HERE / "figures" / name).read_bytes(), (
            name
        )


def test_card_cap_includes_the_last_card():
    from egyptian_ratscrew.termination import play_from, start

    for hands, cap, expected in [
        ([[2], []], 0, (0, 0)),
        ([[2], [3]], 0, (0, None)),
        ([[2], [3]], 1, (1, 1)),
        ([[2], [3]], 2, (1, 1)),
    ]:
        shares = [0.5, 0.5]
        assert play(shares, random.Random(0), cap, hands) == expected
        assert play_from(start(hands), shares, random.Random(0), cap) == expected


def test_penalty_burns_the_last_card_with_no_claim_on_the_pile():
    result = play_game([0.5, 0.5], random.Random(0), hands=[[2], [3]], misslaps=[0, 1])
    assert isinstance(result, Draw)
    assert (result.cards, result.winner, result.paid) == (1, None, 1)
