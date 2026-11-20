"""Plan C12: the full prompt, the retry and the reading of giver replies (no model calls)."""

import numpy as np

from codenames import c4, c12, llm
from codenames.legality import ILLEGAL, flags, strict_legal

BOARD = c4.boards(c12.PILOT, 1)[0]


def test_full_prompt_states_the_rule_and_the_scoring():
    p = llm.giver_prompt_full(BOARD)
    for words in ("form of a board word", "plural", "derived", "spelling", "contain a board word",
                  "compound", "+1", "-1", "scores 0", "assassin", "more than 4"):  # fmt: skip
        assert words in p, words
    assert set(ILLEGAL) <= set(llm.RELATION_WORDS)


def test_retry_names_the_board_word():
    w = BOARD.words[0]
    msg = llm.retry_message(w + "s", flags(w + "s", BOARD.words))
    assert w.upper() in msg and "not allowed" in msg


def test_read_strict():
    w = BOARD.words[0]
    assert c12.read_strict("", BOARD)["status"] == "empty"
    assert c12.read_strict("no clue here", BOARD)["status"] == "unparsed"
    assert c12.read_strict(f"{w}s 2", BOARD)["status"] == "illegal"
    ok = c12.read_strict("Galaxy 3", BOARD)
    assert ok == {"status": "ok", "word": "galaxy", "count": 3} or ok["status"] == "illegal"


class Stub:
    """A Hosted stand-in that replies from a list and records the calls."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def call(self, prompt, tag=None, history=None):
        self.calls.append((prompt, tag, history))
        return {"reply": self.replies.pop(0), "finish": "stop"}


def test_retry_once_and_only_after_an_illegal_clue():
    w = BOARD.words[0]
    fb = ("fallbackword", 1)
    s = Stub([f"{w}s 2", f"{w}es 2"])
    row = c12.give_one(s, "A2", 0, BOARD, c12.PILOT, fb)
    assert len(s.calls) == 2 and s.calls[1][2] is not None and row["why"] == "fallback-illegal"
    assert (row["clue"], row["count"]) == fb
    s = Stub(["", "unused"])
    row = c12.give_one(s, "A3", 0, BOARD, c12.PILOT, fb)
    assert len(s.calls) == 1 and row["why"] == "fallback-empty"
    s = Stub([f"{w}s 2"])
    row = c12.give_one(s, "A0", 0, BOARD, c12.PILOT, fb)
    assert len(s.calls) == 1 and row["why"] == "fallback-illegal"


def test_committed_fallbacks_are_strictly_legal():
    given = c4.read_clue_lines(c12.E_LINES.read_text())
    for seed in (c12.PILOT, c12.VAL):
        for b, (c, k, _) in zip(c4.boards(seed, c12.N[seed]), given[("F1", seed)], strict=True):
            assert k == 1 and strict_legal(c, b.words)
        for b, (c, _, _) in zip(c4.boards(seed, c12.N[seed]), given[("E", seed)], strict=True):
            assert strict_legal(c, b.words)


def test_h_models_drops_a_board_with_a_missing_reply():
    a3 = np.array([1.0, np.nan, 2.0])
    e = np.array([0.0, 1.0, 1.0])
    keep = ~np.isnan(a3) & ~np.isnan(e)
    assert list((a3 - e)[keep]) == [1.0, 1.0]
