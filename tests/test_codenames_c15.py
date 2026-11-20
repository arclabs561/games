"""Plan C15: the changed boards and the reading of replies (no model calls)."""

import numpy as np

from codenames import c15
from codenames.contamination import ZIPF_WINDOW, arms, swap_pool
from codenames.legality import strict_legal
from codenames.words import zipf

WORDS = ("piano", "moon", "bank", "tree", "glass", "rock", "star", "ship", "nail", "fork",
         "lamp", "bell", "card", "coat", "drum", "egg", "fan", "gold", "hat", "ice")  # fmt: skip


def test_swapped_board_keeps_clue_and_pick_and_matches_frequencies():
    a = arms("music", WORDS, "drum", 7)
    o, s, w = a["O"], a["S"], a["W"]
    assert o.words == WORDS and sorted(s.words) == sorted(WORDS) and s.words != WORDS
    assert w.clue == o.clue == "music" and w.picked == "drum"
    assert w.words.index("drum") == s.words.index("drum")
    assert len(set(w.words)) == len(WORDS)
    for old, new in zip(s.words, w.words, strict=True):
        if old == "drum":
            continue
        assert new not in WORDS
        assert abs(zipf(new) - zipf(old)) <= ZIPF_WINDOW + 1e-9
        assert strict_legal(new, ("music",))
    assert w.widened == 0


def test_arms_are_seeded_by_the_turn():
    assert arms("music", WORDS, "drum", 7) == arms("music", WORDS, "drum", 7)
    assert arms("music", WORDS, "drum", 7)["W"] != arms("music", WORDS, "drum", 8)["W"]


def test_pool_is_sorted_nouns_of_three_to_nine_letters():
    pool = swap_pool()
    assert all(3 <= len(w) <= 9 and w.isalpha() for w, _ in pool)
    assert [f for _, f in pool] == sorted(f for _, f in pool)


def test_cells():
    a = arms("music", WORDS, "drum", 7)["O"]

    def rec(reply, finish="stop"):
        return {"reply": reply, "usage": {}, "finish": finish, "pin_held": True}

    assert c15.cell(rec("drum\npiano"), a) == "1"
    assert c15.cell(rec("piano\ndrum"), a) == "0"
    assert c15.cell(rec("drum", "length"), a) == "H"
    assert c15.cell(rec(""), a) == "e"
    assert c15.cell(rec("no idea"), a) == "n"
    assert c15.cell(None, a) == "x"
    w = arms("music", WORDS, "drum", 7)["W"]
    gone = next(x for x in WORDS if x not in w.words)
    assert c15.named_original(rec(f"drum\n{gone}"), a, w) == "1"
    assert c15.named_original(rec("drum"), a, w) == "0"


def test_difference_in_differences_drops_unanswered_turns():
    model = {"O": np.array([1, 1, np.nan, 0.0]), "W": np.array([0, 1, 1, 0.0]), "S": np.zeros(4)}
    ctrl = {"O": np.array([1, 0, 1, 0.0]), "W": np.array([1, 0, 0, 0.0]), "S": np.zeros(4)}
    out = c15.did(model, ctrl, ["a", "a", "b", "b"], "W")
    assert out["turns_dropped"] == 1 and out["n"] == 3
    assert out["mean"] == (1 + 0 + 0) / 3


def test_read_lines():
    g = c15.read_lines("gpt-oss-120b O 0 10e\ngpt-oss-120b W-orig 0 0-1\n")
    assert list(c15.hits(g[("gpt-oss-120b", "O", 0)])[:2]) == [1.0, 0.0]
    assert np.isnan(c15.hits(g[("gpt-oss-120b", "O", 0)])[2])
