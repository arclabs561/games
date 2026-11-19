"""Hanabi: experiment H2's runner, cost meter and report, with a scripted model."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from hanabi.h2 import (
    bootstrap,
    ceiling,
    complete_prefix,
    main,
    paired,
    parse_seeds,
    play_game,
    run_games,
    wilson,
)
from hanabi.hosted import CapError, Hosted, Meter, Price, ProviderError, listed_price
from hanabi.llm import play_text_game
from hanabi.rules import Rules, seeded_deck

MODELS = [
    {
        "id": "org/model",
        "providers": [
            {"provider": "slow", "pricing": {"input": 1.0, "output": 2.0}},
            {"provider": "nopricing"},
        ],
    }
]


def first_discard_or_last(prompt: str) -> str:
    moves = prompt.split("Legal moves: ")[1].split(".")[0].split(", ")
    return "Thinking.\n" + next((m for m in moves if m.startswith("DISCARD")), moves[-1])


class FakeClient:
    """Answers like an Inference Providers client. `fail` makes the first n calls raise;
    `reply` maps a prompt to the reply text."""

    def __init__(self, reply=first_discard_or_last, fail=0, finish="stop"):
        self.reply, self.fail, self.finish, self.calls = reply, fail, finish, 0

    def chat_completion(self, messages, model, **params):
        self.calls += 1
        if self.calls <= self.fail:
            raise ConnectionError("provider down")
        prompt = messages[0]["content"]
        return SimpleNamespace(
            id=f"call-{self.calls}",
            model=model,
            usage=SimpleNamespace(prompt_tokens=len(prompt) // 2, completion_tokens=10),
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish,
                    message=SimpleNamespace(content=self.reply(prompt), reasoning="r"),
                )
            ],
        )


def test_play_game_matches_play_text_game():
    for players, seed, deduction in [(2, 0, True), (2, 4, False), (3, 1, True)]:
        rules = Rules(players=players)
        deck = seeded_deck(rules, seed)
        ref, misses = play_text_game(rules, deck, first_discard_or_last, deduction)
        state, record = play_game(rules, deck, first_discard_or_last, deduction)
        assert (state.score, state.lives, state.turn, state.board) == (
            ref.score,
            ref.lives,
            ref.turn,
            ref.board,
        )
        assert record["fallbacks"] == misses == 0 and len(record["moves"]) == ref.turn


def test_unreadable_replies_fall_back_like_play_text_game():
    rules = Rules(players=2)
    deck = seeded_deck(rules, 3)
    ref, misses = play_text_game(rules, deck, lambda p: "no idea")
    state, record = play_game(rules, deck, lambda p: "no idea", deduction=False)
    assert record["fallbacks"] == misses == ref.turn == state.turn
    assert record["fallback_turns"] == list(range(state.turn))


def test_price_lookup_and_cost():
    price = listed_price(MODELS, "org/model", "slow")
    assert price == Price(1.0, 2.0) and price.cost(1_000_000, 500_000) == 2.0
    with pytest.raises(LookupError):
        listed_price(MODELS, "org/model", "nopricing")
    with pytest.raises(LookupError):
        listed_price(MODELS, "org/other", "slow")


def test_meter_never_lets_spending_pass_the_cap():
    meter = Meter(Price(1.0, 1.0), cap=0.0101, max_call=0.004)
    meter.reserve()
    meter.reserve()  # two calls in flight could cost 0.008
    with pytest.raises(CapError):
        meter.reserve()  # a third could reach 0.012
    meter.settle(1000, 1000)  # 0.002 spent, one call in flight
    meter.reserve()  # 0.002 + 2 * 0.004 = 0.010, within the cap
    assert meter.spent == pytest.approx(0.002) and meter.in_flight == 2
    with pytest.raises(CapError):
        meter.reserve()
    meter.release()
    assert meter.in_flight == 1 and meter.calls == 1


def test_hosted_retries_three_times_then_gives_up():
    meter = Meter(Price(1.0, 1.0), cap=1.0, max_call=0.01)
    log = []
    fake = FakeClient(reply=lambda p: "PLAY 1", fail=3)
    ok = Hosted("org/model", "slow", meter, client=fake, log=log.append, wait=0)
    assert ok("prompt") == "PLAY 1"
    assert log[0]["attempt"] == 3 and len(log[0]["errors"]) == 3
    assert log[0]["params"] == {"max_tokens": 512, "temperature": 0, "seed": 0}
    bad = Hosted("org/model", "slow", meter, client=FakeClient(fail=4), wait=0)
    with pytest.raises(ProviderError):
        bad("prompt")
    assert meter.in_flight == 0 and meter.calls == 1


def test_reasoning_effort_is_sent():
    meter = Meter(Price(1.0, 1.0), cap=1.0, max_call=0.01)
    h = Hosted("m", "p", meter, reasoning_effort="medium", client=FakeClient())
    assert h.params["extra_body"] == {"reasoning_effort": "medium"}


def test_seeds_prefix_and_statistics():
    assert parse_seeds("0-3,7") == [0, 1, 2, 3, 7]
    rows = [{"seed": s, "status": st} for s, st in [(2, "complete"), (0, "complete"), (1, "x")]]
    assert [r["seed"] for r in complete_prefix(rows)] == [0]
    assert bootstrap([3.0] * 10) == (3.0, 3.0)
    lo, hi = bootstrap(list(range(30)))
    assert lo < 14.5 < hi and bootstrap(list(range(30))) == (lo, hi)  # seeded
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    d = paired({0: 1.0, 1: 2.0, 5: 9.0}, {0: 2.0, 1: 4.0})
    assert d["deals"] == 2 and d["mean"] == 1.5


def test_ceiling_of_a_deal_lost_only_by_turn_order():
    # from the README: one color, two players, hands of two; R1 R1 | R4 R5 | R1 R3 R3 R2 R2 R4
    rules = Rules(players=2, colors=1, hand_size=2)
    best, v = ceiling(rules, [0, 0, 3, 4, 0, 2, 2, 1, 1, 3])
    assert v == "lost" and best < rules.max_score
    assert ceiling(Rules(players=2), seeded_deck(Rules(players=2), 0)) == (25, "won")


def args_for(out, **kw):
    a = dict(
        model="org/model",
        provider="slow",
        players=2,
        seeds="0-2",
        deduction="on",
        max_tokens=512,
        reasoning_effort=None,
        cap=10.0,
        workers=2,
        stop_after=60,
        hard_stop=60,
        label="test",
        retry_wait=0,
        out=str(out),
    )
    return Namespace(**(a | kw))


def test_run_and_report_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    run_games(args_for(runs / "on"), MODELS, FakeClient())
    run_games(args_for(runs / "off", deduction="off"), MODELS, FakeClient(finish="length"))
    run_games(args_for(runs / "p3", players=3, seeds="0-1"), MODELS, FakeClient())
    # another reply cap, on other deals: never paired with the runs above
    run_games(args_for(runs / "big", seeds="5", max_tokens=4096), MODELS, FakeClient())
    summary = json.loads((runs / "on" / "summary.json").read_text())
    assert summary["complete_prefix"] == 3 and summary["stopped_at_seed"] is None
    games = [json.loads(x) for x in (runs / "on" / "games.jsonl").read_text().splitlines()]
    calls = (runs / "on" / "calls" / "seed-0.jsonl").read_text().splitlines()
    assert len(calls) == games[0]["turns"] == games[0]["calls"]
    assert summary["spent"] == pytest.approx(sum(g["cost"] for g in games))
    out = tmp_path / "h2.json"
    main(["report", "--runs", str(runs), "--out", str(out)])
    result = json.loads(out.read_text())
    arms = {a["run"]: a for a in result["arms"]}
    assert arms["on"]["gap_mean"] == pytest.approx(25 - arms["on"]["score_mean"])
    assert arms["off"]["truncated_share_of_calls"] == 1.0
    assert arms["p3"]["information_strategy"]["deals"] == 2
    assert len(result["deduction_effect"]) == 1
    effect = result["deduction_effect"][0]
    assert effect["deals"] == 3 and effect["mean"] == pytest.approx(
        arms["off"]["gap_mean"] - arms["on"]["gap_mean"]
    )


def test_a_run_stops_at_its_cap_and_keeps_a_prefix(tmp_path):
    # each call costs about $0.001 at these prices, and a game takes about 70 calls
    run_games(args_for(tmp_path, seeds="0-3", cap=0.15, workers=1), MODELS, FakeClient())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["spent"] <= 0.15
    assert 0 < summary["complete_prefix"] < 4 and summary["stopped_at_seed"] is not None
    assert "stopped" in summary["statuses"]


def test_a_run_stops_at_its_time_limit(tmp_path):
    run_games(args_for(tmp_path, seeds="0-1", hard_stop=0, workers=1), MODELS, FakeClient())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["statuses"] == {"stopped": 2} and summary["complete_prefix"] == 0


def test_provider_failure_replays_the_game_once(tmp_path):
    class Flaky(FakeClient):
        def chat_completion(self, messages, model, **params):
            if self.calls == 5:  # the sixth call and its three retries all fail
                self.fail = self.calls + 4
            return super().chat_completion(messages, model, **params)

    run_games(args_for(tmp_path, seeds="0", workers=1), MODELS, Flaky())
    row = json.loads((tmp_path / "scores.jsonl").read_text())
    assert row["status"] == "complete" and row["attempts"] == 2
    assert (tmp_path / "calls" / "seed-0-replay1.jsonl").exists()
