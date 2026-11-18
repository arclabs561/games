"""Hanabi: the text interface for language-model players."""

from hanabi.llm import (
    COLOR_WORDS,
    move_text,
    observation,
    parse_action,
    play_text_game,
    possible_values,
)
from hanabi.rules import (
    Rules,
    color_of,
    full_deck,
    hint,
    legal_actions,
    new_game,
    play,
    seeded_deck,
)

R3 = Rules(players=3)


def test_every_legal_move_round_trips():
    s = new_game(R3, seeded_deck(R3, 5))
    for a in legal_actions(s):
        assert parse_action(f"I think...\n{move_text(a, R3)}", s) == a


def test_parse_takes_the_last_move_and_rejects_illegal_ones():
    s = new_game(R3, seeded_deck(R3, 5))
    assert parse_action("PLAY 1 looks risky, so: discard 2", s) is None  # 8 tokens: no discard
    assert parse_action("DISCARD 2 or maybe PLAY 3", s) == play(2)
    c = color_of(s.hands[1][0], R3)
    assert parse_action(f"hint 1 {COLOR_WORDS[c]}", s) == hint(1, "color", c)
    assert parse_action("no move here", s) is None
    assert parse_action("PLAY 9", s) is None


def test_observation_hides_own_cards():
    rules = Rules(players=2)
    s = new_game(rules, full_deck(rules))  # player 0 holds R1 R1 R1 R2 R2
    text = observation(s, 0, [])
    assert "Player 1 holds (oldest first): R3 R3 R4 R4 R5." in text
    assert "R1 R1" not in text and "Legal moves: PLAY 1," in text


def test_deduction_counts_visible_cards():
    rules = Rules(players=2)
    s = new_game(rules, full_deck(rules))
    poss = possible_values(s, 1)  # player 1 sees three R1s, so no card of theirs is an R1
    assert all(0 not in vals for vals in poss)
    assert "could be" in observation(s, 1, [], deduction=True)


def test_a_scripted_player_finishes_a_game():
    def first_discard_or_hint(prompt):
        moves = prompt.split("Legal moves: ")[1].split(".")[0].split(", ")
        return next((m for m in moves if m.startswith("DISCARD")), moves[-1])

    state, misses = play_text_game(R3, seeded_deck(R3, 2), first_discard_or_hint)
    assert state.over and misses == 0
    state, misses = play_text_game(R3, seeded_deck(R3, 2), lambda prompt: "?")
    assert state.over and misses == state.turn
