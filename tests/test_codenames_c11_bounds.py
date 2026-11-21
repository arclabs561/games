"""Check the bounds against every possible named-word order on small boards."""

import json
from itertools import permutations

import pytest

from codenames import c4, c11, c11_bounds
from codenames.game import Board


@pytest.mark.parametrize("kinds", ["TTONA", "AOTNT", "NTTTO", "TTTAN"])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_short_reply_bounds_match_every_named_order(kinds, count):
    board = Board(("ant", "bat", "cat", "dog", "eel"), tuple(kinds))
    possible = {}
    for length in range(1, len(board.words) + 1):
        for words in permutations(board.words, length):
            # Repetitions, unrelated words, and substrings never add a new named pick.
            reply = f"baton: {', '.join(words)}. {words[0].upper()}!"
            parsed = c4.read_ranking(reply, board.words)
            padded = c4.play_turn(board, parsed["ranking"], count)
            named = c4.play_turn(board, parsed["ranking"][: parsed["named"]], count)
            possible.setdefault(padded, set()).add(named[1])
    short = c11_bounds.short_reply_values(board, count)
    assert short.keys() <= possible.keys()
    for historical, scores in possible.items():
        assert {historical[1], *short.get(historical, ())} == scores


def test_trooper_example_matches_the_published_board_and_score():
    rows = json.loads(c11.LISTS.read_text())["boards"]
    row = next(r for r in rows if (r["seed"], r["board"]) == (c11.TEST, 896))
    assert row["strict"]["P"] == "trooper/2"
    guesses = c11.read_guesses(c11.GUESSES.read_text())
    cell = guesses["P", "gpt-oss-120b", c11.TEST, 0][896]
    assert c11.read_cell(cell) == (1, -8.0, False)

    board = c4.boards(c11.TEST, 897)[896]
    for reply, corrected_value in [("SOLDIER", 1), ("SOLDIER, ROOM", -8)]:
        parsed = c4.read_ranking(reply, board.words)
        assert c4.play_turn(board, parsed["ranking"], 2) == (1, -8)
        named = parsed["ranking"][: parsed["named"]]
        assert c4.play_turn(board, named, 2) == (1, corrected_value)


def test_committed_primary_bounds():
    rows = [r for r in json.loads(c11.LISTS.read_text())["boards"] if r["seed"] == c11.TEST]
    bounds = c11_bounds.primary_bounds(rows, c11.read_guesses(c11.GUESSES.read_text()))
    assert bounds == {
        "kept": 998,
        "shared_prompt_boards": 138,
        "variable_boards": 379,
        "h_a1": {"lower": -0.241483, "historical": -0.023046, "upper": 0.210922},
        "h_a2": {"lower": -0.463312, "historical": -0.244876, "upper": -0.010908},
    }
