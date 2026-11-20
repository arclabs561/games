"""C4's amendment of 2026-11-20: the giving part read under the rulebook's clue rule."""

import pytest

from codenames import c4strict
from codenames.game import Board

WORDS = ("england", "horseshoe", "stone", "river", "piano", "moon", "cat", "bank", "tree",
         "glass", "rock", "star", "ship", "nail", "fork", "lamp", "bell", "card", "coat",
         "drum", "egg", "fan", "gold", "hat", "ice")  # fmt: skip


def board() -> Board:
    kinds = ("red",) * 9 + ("blue",) * 8 + ("neutral",) * 7 + ("assassin",)
    return Board(WORDS, kinds)


def test_three_hand_made_replies():
    b = board()
    # A substring that the rulebook allows ("land" for ENGLAND): C4 rejected it, the
    # strict check calls it illegal, and only the level beyond substrings shows it legal.
    land = c4strict.classify("LAND 2", b)
    assert (land["c4"], land["level"], land["beyond_substring"]) == (
        "board word",
        "illegal",
        "legal",
    )
    # A plural of a board word: C4 rejected it as a substring; it is an inflection too.
    stones = c4strict.classify("stones 2", b)
    assert (stones["c4"], stones["level"], stones["beyond_substring"]) == (
        "board word", "illegal", "illegal",
    )  # fmt: skip
    # A part of a compound board word in another form: C4 accepted it, the rulebook does not.
    shoes = c4strict.classify("Snowshoe 3", b)
    assert shoes["c4"] == "accepted" and shoes["level"] == "illegal"
    assert not shoes["strict_legal"]


def test_replies_with_no_word_and_legal_ones():
    b = board()
    assert c4strict.classify("", b)["level"] == "no word"
    assert c4strict.classify("I cannot decide", b)["c4"] == "unparsed"
    ok = c4strict.classify("music 2", b)
    assert ok["c4"] == "accepted" and ok["level"] == "legal" and ok["strict_legal"]
    assert ok["count"] == 2


def test_committed_tables_add_up():
    import json

    data = json.loads((c4strict.PLAYED.parent / "c4-strict.json").read_text())
    for seeds in data["levels"].values():
        for seed, t in seeds.items():
            assert sum(t["c4_by_level"].values()) == t["replies"] == c4strict.N[int(seed)]
            assert t["accepted_strictly"] <= t["accepted_by_c4"]
            assert sum(t["level_share"].values()) == pytest.approx(1)
    assert data["h_give"]["h_give_strict"]["model"] == c4strict.SELECTED


def test_played_levels_are_what_the_module_writes():
    import json

    assert json.loads(c4strict.PLAYED.read_text()) == json.loads(
        json.dumps(c4strict.played(), sort_keys=True)
    )
