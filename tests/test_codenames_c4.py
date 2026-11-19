"""Tests for codenames/hosted.py and codenames/c4.py, with a stub in place of the router."""

import json

import pytest

from codenames import c4
from codenames.game import Board
from codenames.hosted import (
    Hosted,
    Meter,
    Model,
    OverBudgetError,
    call_cost,
    parse_prices,
    visible,
)
from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM


class Stub:
    """Answers like the router: a reply text, and usage with a cost in dollars."""

    def __init__(self, reply="ocean 2", cost=1e-4, fail=0, usage=None):
        self.reply, self.cost, self.fail, self.usage = reply, cost, fail, usage
        self.sent = []

    def chat_completion(self, messages, model, extra_body=None, **params):
        self.sent.append((messages, model, extra_body, params))
        if self.fail:
            self.fail -= 1
            raise ConnectionError("try again")
        reply = self.reply(messages[0]["content"]) if callable(self.reply) else self.reply
        usage = self.usage or {"prompt_tokens": 200, "completion_tokens": 3, "cost": self.cost}
        return {
            "model": model,
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": usage,
        }


MODEL = Model("org/model", "provider", {"reasoning_effort": "low"})


def test_meter_stops_before_the_cap():
    meter = Meter(cap=0.01)
    hosted = Hosted(MODEL, meter, client=Stub(cost=0.003))
    replies = []
    with pytest.raises(OverBudgetError):
        for _ in range(10):
            replies.append(hosted("prompt"))
    # Each call reserves max(0.002, twice the dearest call) = 0.006 after the first.
    assert meter.spent <= 0.01
    assert len(replies) == meter.calls == 2
    assert meter.reserved == pytest.approx(0)


def test_retries_then_logs_every_field(tmp_path):
    log = tmp_path / "calls.jsonl"
    stub = Stub(fail=2)
    hosted = Hosted(MODEL, Meter(1.0), log, max_tokens=64, client=stub, sleep=lambda s: None)
    assert hosted("the prompt") == "ocean 2"
    rec = json.loads(log.read_text())
    assert rec["attempts"] == 3 and len(rec["errors"]) == 2
    assert rec["params"] == {
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 0,
        "extra_body": {"reasoning_effort": "low"},
    }
    assert rec["prompt"] == "the prompt" and rec["cost"] == 1e-4
    assert stub.sent[-1][2] == {"reasoning_effort": "low"}


def test_a_call_that_always_fails_returns_empty_and_costs_nothing(tmp_path):
    meter = Meter(1.0)
    hosted = Hosted(MODEL, meter, tmp_path / "c.jsonl", client=Stub(fail=9), sleep=lambda s: None)
    assert hosted("p") == ""
    assert meter.spent == 0 and meter.calls == 1


def test_cost_without_a_reported_cost_is_overcounted():
    assert call_cost({"cost": 2e-5}) == (2e-5, "reported")
    cost, source = call_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0})
    assert source == "fallback price" and cost == 3.0


def test_think_blocks_are_removed():
    assert visible("<think>sea, wave</think>\nOCEAN 2") == "OCEAN 2"
    assert visible("<think>never closed") == ""


BOARD = Board(
    ("whale", "ship", "sail", "bank", "pen", "nurse", "bomb"),
    (TEAM, TEAM, TEAM, OPPONENT, NEUTRAL, NEUTRAL, ASSASSIN),
)


def test_clue_replies_are_read_with_a_reason_when_they_fail():
    assert c4.read_clue("**OCEAN** 3", BOARD) == {"clue": "ocean", "count": 3, "why": None}
    assert c4.read_clue("", BOARD)["why"] == "empty"
    assert c4.read_clue("ocean", BOARD)["why"] == "unparsed"
    assert c4.read_clue("ship 2", BOARD)["why"] == "board word"
    assert c4.read_clue("sea 0", BOARD)["why"] == "count below 1"


def test_rankings_count_the_board_words_named():
    r = c4.read_ranking("Ship\nsail\nthe ship", BOARD.words)
    assert r["named"] == 2 and r["ranking"][:2] == ["ship", "sail"]
    assert len(r["ranking"]) == len(BOARD.words)


def test_pilot_end_to_end(tmp_path, monkeypatch):
    def reply(prompt):
        if prompt.startswith("You are the spymaster"):
            return "maritime 2"
        return "\n".join(prompt.split("Board words:\n")[1].split()[:2])

    stubs = {}

    def hosted(model, meter, log, max_tokens):
        stub = stubs.setdefault(model.repo, Stub(reply))
        return Hosted(model, meter, log, max_tokens, client=stub)

    monkeypatch.setattr(c4, "Hosted", hosted)
    models = ["gpt-oss-120b", "qwen3-32b"]
    summary = c4.pilot(models, seed=1, n=4, meter=Meter(1.0), out=tmp_path, repeats=2)
    for name in models:
        s = summary["models"][name]
        assert s["give"]["clues"] == 4 and s["guess"]["turns"] == 4
        assert s["guess"]["named_mean"] == 2 and s["repeat"] == {
            "prompts": 2,
            "times": 2,
            "sent": 4,
            "changed": 0,
        }
        assert s["calls"]["calls"] == 4 + 4 + 4 and s["calls"]["failed"] == 0
        assert s["calls"]["cost"] == pytest.approx(12e-4)
    assert summary["spent"] == pytest.approx(24e-4)
    given = json.loads((tmp_path / "give-qwen3-32b-1.json").read_text())
    assert [r["clue"] for r in given["rows"]] == ["maritime"] * 4


def test_boards_follow_the_study_deal():
    a, b = c4.boards(4004, 3), c4.boards(4004, 3)
    assert [x.words for x in a] == [x.words for x in b]
    assert a[0].words != c4.boards(4005, 1)[0].words


def test_prices_are_read_from_the_router_listing():
    listing = {
        "data": [
            {
                "id": "org/model",
                "providers": [
                    {"provider": "a", "pricing": {"input": 0.1, "output": 0.5}},
                    {"provider": "b"},
                ],
            }
        ]
    }
    prices = parse_prices(listing)
    assert prices == {"org/model": {"a": (0.1, 0.5)}}
    usage = {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000}
    assert call_cost(usage, prices["org/model"]["a"]) == (pytest.approx(0.7), "listed price")


def test_clue_lines_round_trip():
    given = {
        7: {
            "m": [
                {"clue": "ocean", "count": 2},
                {"clue": None, "why": "board word"},
                {"stopped": True},
            ]
        }
    }
    text = c4.clue_lines(given)
    assert text == "m 7 ocean/2 -/board-word -/stopped\n"
    assert c4.read_clue_lines(text) == {
        ("m", 7): [("ocean", 2, None), (None, None, "board-word"), (None, None, "stopped")]
    }


def test_guess_lines_mark_hits_misses_and_unparsed_replies():
    rows = [
        {"named": 3, "hit": 1},
        {"named": 2, "hit": 0},
        {"named": 0, "hit": 1},
        {"named": 0, "hit": 0},
        {"stopped": True},
    ]
    assert c4.guess_lines({"val": {"m": rows}}) == "m val 10ba-\n"


def test_guess_reads_the_first_board_word(tmp_path, monkeypatch):
    from codenames import split
    from codenames.norms import Guess

    turns = [
        split.Turn("test", 0, "g1", "a", "b", Guess("sea", BOARD.words, "whale")),
        split.Turn("test", 1, "g1", "a", "b", Guess("money", BOARD.words, "bank")),
    ]
    monkeypatch.setattr(split, "turns", lambda name: turns)

    def hosted(model, meter, log, max_tokens):
        return Hosted(model, meter, log, max_tokens, client=Stub("Whale\nship"))

    monkeypatch.setattr(c4, "Hosted", hosted)
    got = c4.guess(["llama-3.3-70b"], "val", Meter(1.0), tmp_path)["llama-3.3-70b"]
    assert [r["hit"] for r in got] == [1, 0] and [r["named"] for r in got] == [2, 2]
    assert (
        "for 1 word(s)"
        in json.loads((tmp_path / "calls-llama-3.3-70b.jsonl").read_text().splitlines()[0])[
            "prompt"
        ]
    )


def test_bootstrap_and_holm():
    from codenames.c4score import bootstrap, holm

    b = bootstrap([1.0] * 10)
    assert b["mean"] == 1 and b["ci"] == [1.0, 1.0] and b["p"] == 0
    b = bootstrap([1.0, -1.0] * 50)
    assert b["mean"] == 0 and b["p"] == 1
    assert holm({"a": 0.01, "b": 0.04}) == {"a": True, "b": True}
    assert holm({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}
    assert holm({"a": 0.001, "b": 0.2}) == {"a": True, "b": False}


def test_a_turn_stops_at_the_first_wrong_word():
    assert c4.play_turn(BOARD, ["whale", "ship", "bank"], 3) == (2, 1.0)
    assert c4.play_turn(BOARD, ["whale", "pen"], 3) == (1, 1.0)
    # The assassin costs the team words still to find: 3 before the turn.
    assert c4.play_turn(BOARD, ["whale", "bomb"], 2) == (1, -2.0)
    assert c4.play_turn(BOARD, ["whale", "ship", "sail", "bank", "pen"], 9) == (3, 2.0)


def test_crossguess_scores_pair_with_the_embedding_giver():
    from codenames.c4score import score_crossguess

    text = "embeddinggemma-2 x 1 2/2 1/1\nm x 1 1/1 0/-3\nembeddinggemma-2 m 1 1/1 1/1\n"
    out = score_crossguess(text)
    assert out["pairs"]["m -> x"]["value"]["mean"] == -1
    assert out["givers"]["embeddinggemma-2"]["value"]["mean"] == pytest.approx(1.25)
    assert out["givers"]["m"]["minus_embeddinggemma_same_guessers"]["mean"] == -2.5
