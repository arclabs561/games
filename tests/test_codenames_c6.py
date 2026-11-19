"""Tests for codenames/actions.py, codenames/rerank.py and codenames/c6score.py, with toy
vectors and a stub in place of the router."""

import json
import random

import numpy as np
import pytest

from codenames import actions, c6score, rerank
from codenames.c4score import bootstrap
from codenames.embed import normalize
from codenames.game import deal
from codenames.hosted import Hosted, Meter
from codenames.llm import giver_prompt
from codenames.players import Space, embedding_weigher
from codenames.words import legal

CLUES = ["ocean", "river", "money", "space"]


def toy_space(seed=0):
    """Vectors where each 'clueN' word sits next to a fixed set of board words (as in
    test_codenames.py)."""
    rng = np.random.default_rng(seed)
    board = [f"w{i}" for i in range(30)]
    vecs = {w: rng.normal(size=16) for w in board}
    clues = []
    for j in range(40):
        group = rng.choice(30, size=3, replace=False)
        vecs[f"clue{j}"] = sum(vecs[board[g]] for g in group) + 0.3 * rng.normal(size=16)
        clues.append(f"clue{j}")
    words = list(vecs)
    m = normalize(np.array([vecs[w] for w in words]))
    return Space("toy", m, m, words), tuple(board), tuple(clues)


class Stub:
    """Answers like the router, with the reply a function of the prompt."""

    def __init__(self, reply):
        self.reply = reply

    def chat_completion(self, messages, model, extra_body=None, **params):
        usage = {"prompt_tokens": 500, "completion_tokens": 3, "cost": 1e-4}
        content = self.reply(messages[0]["content"])
        return {"model": model, "choices": [{"message": {"content": content}}], "usage": usage}


def toy_board(seed=3):
    sp, words, clues = toy_space()
    return sp, deal(words, random.Random(seed)), clues


def test_action_list_has_120_legal_actions_in_clue_order():
    sp, board, clues = toy_board()
    acts = actions.legal_actions(board, sp, 0.05, pool=clues)
    assert len(acts) == 120 and len(set(acts)) == 120
    assert [n for _, n in acts[:8]] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert all(legal(c, board.unrevealed()) for c, _ in acts)
    actions.check(board, acts, [sp])
    # The same list as rl.actions, which C7 also reads.
    from codenames.rl import actions as rl_actions

    assert acts == rl_actions(sp, 0.05, board, clues, 30)[0]


def test_action_lines_round_trip():
    lists = [["a", "b"], ["c", "d"]]
    text = actions.action_lines(7, lists) + actions.action_lines(8, lists[:1])
    assert actions.read_action_lines(text) == {7: lists, 8: lists[:1]}


def test_values_match_a_loop_and_give_the_baselines():
    sp, board, clues = toy_board()
    acts = actions.legal_actions(board, sp, 0.05, pool=clues)
    weigh = embedding_weigher(sp, 0.05)
    fast = actions.values(weigh, board, acts)
    slow = actions.values_by_loop(weigh, board, acts)
    assert fast == pytest.approx(slow, abs=1e-9)
    # Uniform is the mean over every action; first choice and oracle are argmaxes.
    assert fast.mean() == pytest.approx(sum(slow) / len(slow))
    assert int(np.argmax(fast)) == max(range(len(slow)), key=lambda i: (slow[i], -i))


def test_prompt_lists_the_clues_after_the_giver_prompt():
    _, board, _ = toy_board()
    p = rerank.rerank_prompt(board, CLUES)
    assert p.startswith(giver_prompt(board))
    assert "\n1. ocean\n2. river\n3. money\n4. space\n" in p
    assert p.endswith("a number from 1 to 4.")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("ocean 2", ("ocean", 2, None)),
        ("RIVER 3", ("river", 3, None)),
        ("**Money** 1", ("money", 1, None)),
        ("2. space 4", ("space", 4, None)),
        ("I pick ocean, 2 words.", ("ocean", 2, None)),
        ("<|channel>thought\n<channel|>river 2", ("river", 2, None)),
        ("lake 2", (None, None, "unlisted")),
        ("ocean", (None, None, "unlisted")),
        ("ocean 5", (None, None, "count out of range")),
        ("ocean 0", (None, None, "count out of range")),
        ("  ", (None, None, "empty")),
        ("lake 2 or river 1", ("river", 1, None)),
    ],
)
def test_read_choice(reply, expected):
    r = rerank.read_choice(reply, CLUES)
    assert (r["clue"], r["count"], r["why"]) == expected


def test_action_index_follows_expand():
    acts = actions.expand(CLUES)
    for i, (c, n) in enumerate(acts):
        assert rerank.action_index({"clue": c, "count": n}, CLUES) == i
    assert rerank.action_index({"clue": None, "count": None}, CLUES) is None


def test_choice_lines_round_trip():
    rows = [
        {"clue": "ocean", "count": 2, "why": None},
        {"clue": None, "count": None, "why": "count out of range"},
        {"clue": None, "count": None, "why": "unlisted"},
    ]
    text = rerank.choice_lines({6005: {"m": rows}})
    assert text == "m 6005 ocean/2 -/count-out-of-range -/unlisted\n"
    assert rerank.read_choice_lines(text) == {("m", 6005): rows}


def test_choose_end_to_end(tmp_path, monkeypatch):
    def reply(prompt):
        listed = prompt.split("from this list:\n")[1].split("\n\n")[0].splitlines()
        return listed[1].split(". ")[1] + " 3"  # the second clue, count 3

    def hosted(model, meter, log, max_tokens):
        return Hosted(model, meter, log, max_tokens, client=Stub(reply))

    monkeypatch.setattr(rerank, "Hosted", hosted)
    letters = "abcdefghijklmnopqrstuvwxyz"
    lists = [
        [f"clue{'abcd'[i]}{letters[j // 26]}{letters[j % 26]}" for j in range(30)] for i in range(4)
    ]
    chosen = rerank.choose(["gpt-oss-120b", "gemma-4-31b"], 1, 4, lists, Meter(1.0), tmp_path)
    for rows in chosen.values():
        assert [(r["clue"], r["count"]) for r in rows] == [(c[1], 3) for c in lists]
    rec = json.loads((tmp_path / "choose-gemma-4-31b-1.json").read_text())
    assert rec["repo"] == "google/gemma-4-31B-it" and len(rec["rows"]) == 4
    bs = rerank.c4.boards(1, 2)
    rep = rerank.repeat("gemma-4-31b", 1, bs, lists[:2], chosen["gemma-4-31b"], Meter(1), tmp_path)
    assert rep == {"prompts": 2, "times": 2, "sent": 4, "changed": 0, "changed_action": 0}


def test_six_choosers_in_the_plan_order():
    assert rerank.PLAN == (
        "gpt-oss-120b",
        "llama-3.3-70b",
        "qwen3-32b",
        "deepseek-v3.2",
        "qwen3.8-27b",
        "gemma-4-31b",
    )
    assert rerank.MODELS["gpt-oss-120b"].extra == {"reasoning_effort": "low"}
    assert rerank.tokens("gpt-oss-120b") == 1024 and rerank.tokens("qwen3-32b") == 64


def test_ratio_bootstrap_uses_the_same_resamples():
    rng = np.random.default_rng(1)
    d = rng.normal(0.1, 1, 50)
    r = c6score.ratio_bootstrap(d, np.ones(50))
    b = bootstrap(d)
    assert r["mean"] == pytest.approx(b["mean"]) and r["ci"] == pytest.approx(b["ci"])


def test_summary_of_hand_made_boards():
    # Two boards, three measures (strangers, human, EAT) per arm.
    def row(first, oracle_s, oracle_h, pick, uniform):
        return {
            "value": {"first": first, "oracle_s": oracle_s, "oracle_h": oracle_h, "m": pick},
            "uniform": uniform,
            "actions": {
                "first": ["a", 1],
                "oracle_s": ["b", 2],
                "oracle_h": ["c", 3],
                "m": ["a", 1] if pick == first else ["d", 4],
            },
            "rank": {"m": 1 if pick == first else 7},
            "fallback": {},
        }

    rows = [
        row([0.5, 0.4, 0.0], [1.5, 0.6, 0.0], [0.7, 1.4, 0.0], [1.0, 0.9, 0.1], [0.1, 0.2, 0.0]),
        row([0.2, 0.2, 0.0], [1.2, 0.3, 0.0], [0.3, 0.8, 0.0], [0.2, 0.2, 0.0], [0.0, 0.0, 0.0]),
    ]
    s = c6score.summarize(rows, {"m": None})
    m = s["arms"]["m"]
    assert m["strangers_minus_first"]["mean"] == pytest.approx(0.25)
    assert m["strangers_gap_closed"]["mean"] == pytest.approx(0.25 / 1.0)
    assert m["human_gap_closed"]["mean"] == pytest.approx(0.25 / 0.8)
    assert m["is_first"] == 0.5 and m["rank_mean"] == 4 and m["distinct_clues"] == 2
    assert s["oracle_gap"]["strangers"]["mean"] == pytest.approx(1.0)
    assert s["arms"]["uniform"]["strangers"]["mean"] == pytest.approx(0.05)
