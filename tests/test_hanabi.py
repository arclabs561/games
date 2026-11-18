"""Hanabi: the rules engine and the information strategy."""

import pytest

from hanabi import information as info
from hanabi.rules import (
    Drand48,
    Rules,
    discard,
    full_deck,
    hansim_decks,
    hint,
    legal,
    legal_actions,
    make_card,
    new_game,
    play,
    play_game,
    seeded_deck,
    step,
)

R2 = Rules(players=2)


def test_deck():
    deck = full_deck(R2)
    assert len(deck) == 50 and len(set(deck)) == 25
    assert deck.count(make_card(0, 1, R2)) == 3 and deck.count(make_card(4, 5, R2)) == 1
    assert sorted(seeded_deck(R2, 7)) == deck and seeded_deck(R2, 7) == seeded_deck(R2, 7)


def test_hand_sizes():
    for p, h in [(2, 5), (3, 5), (4, 4), (5, 4)]:
        s = new_game(Rules(players=p), full_deck(Rules(players=p)))
        assert [len(x) for x in s.hands] == [h] * p
        assert s.deck_left == 50 - p * h


def test_hints_and_discards():
    s = new_game(R2, full_deck(R2))  # player 1 holds R3 R3 R4 R4 R5
    assert not legal(s, hint(1, "rank", 1))  # touches no card
    assert not legal(s, hint(0, "rank", 1))  # no hints to yourself
    assert not legal(s, discard(0))  # all 8 tokens are on the table
    assert legal(new_game(Rules(players=2, discard_when_full=True), full_deck(R2)), discard(0))
    e = step(s, hint(1, "rank", 4))
    assert e.touched == (2, 3) and s.hints == 7
    assert s.knowledge[1][2][1] == 0b01000 and s.knowledge[1][0][1] == 0b10111
    assert legal(s, discard(0))


def test_play_and_misplay():
    s = new_game(R2, full_deck(R2))  # player 0 holds R1 R1 R1 R2 R2
    step(s, play(0))
    assert s.board[0] == 1 and s.hands[0][-1] == full_deck(R2)[10]
    step(s, play(0))  # player 1 plays R3 too early
    assert s.lives == 2 and len(s.discards) == 1


def test_three_misplays_score_zero():
    s = new_game(R2, full_deck(R2))
    step(s, play(0))  # R1
    while not s.over:  # player 1 misplays R3, R3, R4 while player 0 hints red
        step(s, play(0) if s.current == 1 else hint(1, "color", 0))
    assert s.over and s.lives == 0 and s.score == 0


def test_a_played_five_returns_a_token():
    deck = full_deck(R2)
    s = new_game(R2, deck)
    s.board[0], s.hints = 4, 3
    s.hands[0][0] = make_card(0, 5, R2)
    step(s, play(0))
    assert s.board[0] == 5 and s.hints == 4


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_one_more_turn_each_after_the_last_card(players):
    rules = Rules(players=players)

    def choose(state):
        acts = legal_actions(state)
        return next((a for a in acts if a.kind == "discard"), acts[-1])

    s = new_game(rules, seeded_deck(rules, 1))
    last_draw = -1
    while not s.over:
        step(s, choose(s))
        if s.deck_left == 0 and last_draw < 0:
            last_draw = s.turn
    assert s.turn - last_draw == players
    assert play_game(rules, seeded_deck(rules, 1), choose).turn == s.turn


def test_drand48_matches_the_c_library():
    rng = Drand48(100)  # values printed by glibc's drand48 after srand48(100)
    assert [rng() for _ in range(3)] == [
        0.25105890266513953,
        0.20894840485149757,
        0.9409279099583152,
    ]


# --- the information strategy ------------------------------------------------------------


def test_partition_matches_figure_6():
    """Cox et al., Figures 4b and 6, with colors in the paper's order blue, green, red,
    white, yellow; the 1s are dead."""
    rules = Rules(players=5)
    team = info.InfoTeam(rules)
    blue, green, red, white = 0, 1, 2, 3
    table = {
        blue: [2, 3, 4],
        green: [1, 2, 3, 4, 5],
        red: [1, 2, 4, 5],
        white: [1, 3, 4, 5],
    }
    mask = 0
    for c, ranks in table.items():
        for r in ranks:
            mask |= 1 << make_card(c, r, rules)
    dead, _ = team.board_masks([1, 1, 1, 1, 1])
    part = team.partition(mask, dead)
    figure_6 = {
        blue: {2: 1, 3: 4, 4: 7},
        green: {1: 0, 2: 2, 3: 5, 4: 7, 5: 7},
        red: {1: 0, 2: 3, 4: 7, 5: 7},
        white: {1: 0, 3: 6, 4: 7, 5: 7},
    }
    for c, row in figure_6.items():
        for r, k in row.items():
            assert part[make_card(c, r, rules)] == k


def test_hint_encoding_round_trips():
    for p in (3, 4, 5):
        rules = Rules(players=p)
        team = info.InfoTeam(rules)
        s = new_game(rules, seeded_deck(rules, 3))
        team.start(s)
        for giver in range(p):
            s.current = giver
            for number in range(team.hats):
                a = team.encode(giver, number, s)
                assert legal(s, a) and team.decode(giver, a) == number


# HanSim's scores for its first 100 games (srand48(100), five players), last digit of 20 to 25.
HANSIM_FIRST_100 = "5555555555555553345455555554545555555554545455552555545554345525554525555544554555555555455255355545"


def test_information_strategy_matches_hansim_game_by_game():
    rules = Rules(players=5, discard_when_full=True)
    decks = hansim_decks(100)
    scores = [info.play_info_game(rules, next(decks)).score for _ in range(100)]
    assert "".join(str(s - 20) for s in scores) == HANSIM_FIRST_100


@pytest.mark.parametrize("players", [3, 4, 5])
def test_information_strategy_never_misplays(players):
    rules = Rules(players=players)
    for seed in range(30):
        s = info.play_info_game(rules, seeded_deck(rules, seed), hansim=False)
        assert s.lives == 3 and s.score >= 15
