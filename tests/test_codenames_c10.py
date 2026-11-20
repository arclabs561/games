"""Tests for codenames/listen.py and the parts of codenames/c10score.py that need no word
vectors, with toy vectors and a stub in place of the router."""

import itertools
import json
import random

import numpy as np
import pytest

from codenames import c4, listen
from codenames.actions import expand
from codenames.actions import values as action_values
from codenames.c10score import read_rankings
from codenames.embed import normalize
from codenames.game import Board, deal
from codenames.hosted import Hosted, Meter
from codenames.llm import guesser_prompt
from codenames.players import Space, embedding_weigher

WORDS = ("apple", "river", "bank", "cloud", "stone", "piano")


def ranking(reply: str, words=WORDS) -> list[str]:
    return c4.read_ranking(reply, words)["ranking"]


@pytest.mark.parametrize(
    ("reply", "prefix", "cell"),
    [
        # Two words of its own, then the rest in the prompt's order.
        ("bank\nstone\napple\nriver\ncloud\npiano", ["bank", "stone"], "ce"),
        # The whole board in the prompt's order: no prefix.
        ("apple\nriver\nbank\ncloud\nstone\npiano", [], "="),
        # One word, then nothing: the unnamed words follow in the prompt's order.
        ("Cloud is my guess.", ["cloud"], "d"),
        # A first word that is also first in the prompt joins the tail.
        ("apple\nbank\nriver\ncloud\nstone\npiano", ["apple", "bank"], "ac"),
        # A reversed board: everything but the last word is the prefix.
        (
            "piano stone cloud bank river apple",
            ["piano", "stone", "cloud", "bank", "river"],
            "fedcb",
        ),
        # No board word.
        ("I would guess orange, then lemon.", [], "0"),
    ],
)
def test_tail_rule_on_hand_made_replies(reply, prefix, cell):
    assert listen.informative_prefix(ranking(reply), WORDS) == prefix
    assert listen.prefix_by_loop(ranking(reply), WORDS) == prefix
    assert listen.cell(reply, WORDS) == cell
    assert listen.read_cell(cell, WORDS) == prefix


def test_empty_failed_and_stopped_replies_have_marks():
    assert listen.cell("", WORDS) == listen.FAILED
    assert listen.cell("   \n", WORDS) == listen.FAILED
    assert listen.cell(None, WORDS) == listen.STOPPED
    for mark in listen.MARKS:
        assert listen.read_cell(mark, WORDS) == []
        assert (listen.ranks([], WORDS) == 0).all()


def test_fast_prefix_matches_the_loop_on_every_ranking_of_five_words():
    words = WORDS[:5]
    for perm in itertools.permutations(words):
        fast = listen.informative_prefix(list(perm), words)
        assert fast == listen.prefix_by_loop(list(perm), words)
        rest = list(perm[len(fast) :])
        assert rest == [w for w in words if w in rest]


def test_ranks_and_weights():
    r = listen.ranks(["bank", "stone"], WORDS)
    assert r.tolist() == [2, 2, 0, 2, 1, 2]
    w = listen.listener_weights(r, 0.5)
    assert w.tolist() == pytest.approx(
        [np.exp(-4), np.exp(-4), 1, np.exp(-4), np.exp(-2), np.exp(-4)]
    )


def toy_space(seed=0):
    """Vectors where each 'clueN' word sits next to a fixed set of board words."""
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


def toy_board(seed=3):
    sp, words, clues = toy_space()
    return sp, deal(words, random.Random(seed)), clues


def random_prefixes(board: Board, clues, seed=0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    words = list(board.unrevealed())
    return {c: rng.sample(words, rng.randint(0, 6)) for c in clues}


def test_listener_values_match_a_loop():
    _, board, clues = toy_board()
    kept = list(clues[:6])
    prefixes = random_prefixes(board, kept)
    for temp in (0.3, 1.0, 4.0):
        fast = listen.listener_values(board, kept, prefixes, temp)
        slow = listen.listener_values_by_loop(board, kept, prefixes, temp)
        assert fast.shape == (24,)
        assert fast == pytest.approx(slow, abs=1e-12)


def test_fit_recovers_a_known_temperature():
    rng = np.random.default_rng(1)
    true = 1.7
    turns = []
    for _ in range(4000):
        p = int(rng.integers(0, 8))
        r = np.array([min(i, p) for i in rng.permutation(20)], dtype=float)
        w = listen.listener_weights(r, true)
        turns.append((r, int(rng.choice(20, p=w / w.sum()))))
    fit = listen.fit_temperature(turns)
    assert fit["T"] == pytest.approx(true, rel=0.1)
    # The log-likelihood at the fitted T is at least that at the true one.
    assert fit["loglik_per_turn"] >= listen.log_likelihood(turns, true) - 1e-12


def test_log_likelihood_by_hand():
    r = np.array([0.0, 1.0, 1.0])
    expected = np.log(1 / (1 + 2 * np.exp(-1)))
    assert listen.log_likelihood([(r, 0)], 1.0) == pytest.approx(expected)


def test_prefilter_keeps_the_first_choice_and_breaks_ties_by_list_order():
    sp, board, clues = toy_board()
    from codenames.actions import legal_actions

    acts = legal_actions(board, sp, 0.05, pool=clues)
    clue_list = list(dict.fromkeys(c for c, _ in acts))
    v = action_values(embedding_weigher(sp, 0.05), board, acts)
    first = acts[int(np.argmax(v))][0]
    for k in (1, 10, 20):
        kept = listen.prefilter_values(v, clue_list, k)
        assert len(kept) == k and first in kept and kept[0] == first
    ties = np.zeros(12)
    assert listen.prefilter_values(ties, ["a", "b", "c"], 2) == ["a", "b"]
    best = np.array([0, 0, 0, 1.0, 0, 2.0, 0, 0, 0, 0, 0, 2.0])
    assert listen.prefilter_values(best, ["a", "b", "c"], 3) == ["b", "c", "a"]


def test_kept_index_points_into_the_full_list():
    clues = ["a", "b", "c", "d"]
    idx = listen.kept_index(clues, ["c", "a"])
    assert [expand(clues)[i] for i in idx] == expand(["c", "a"])


def test_population_pick_against_a_loop_and_the_mixture_at_its_ends():
    rng = np.random.default_rng(2)
    for _ in range(50):
        s = rng.normal(size=(7, 40))
        lv = rng.normal(size=40)
        for held in range(7):
            members = [g for g in range(7) if g != held]
            best, best_v = None, -np.inf
            for j in range(40):
                v = sum(s[g, j] for g in members) / 6
                if v > best_v:
                    best, best_v = j, v
            assert listen.pick_population(s, held) == best
            assert listen.pick_mixture(lv, s, held, 0.0) == best
            assert listen.pick_mixture(lv, s, held, 1.0) == int(np.argmax(lv))


def test_argmax_takes_the_first_of_equal_values():
    assert listen.argmax_first([1.0, 3.0, 3.0, 2.0]) == 1


class Stub:
    """Answers like the router; the reply is a function of the prompt."""

    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def chat_completion(self, messages, model, extra_body=None, **params):
        prompt = messages[0]["content"]
        self.sent.append(prompt)
        usage = {"prompt_tokens": 200, "completion_tokens": 30}
        content = self.reply(prompt, len(self.sent))
        return {
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": usage,
        }


def board_words_of(prompt: str) -> list[str]:
    return prompt.split("Board words:\n")[1].splitlines()


def test_listen_job_end_to_end(tmp_path, monkeypatch):
    """Two boards, three clues each, the first board repeated; the first call comes back
    empty and is sent again at the end."""
    bs = c4.boards(77, 2)
    lists = tmp_path / "lists.txt"
    lists.write_text("77 0 sun moon star\n77 1 tree leaf root\n")
    monkeypatch.setattr(listen, "LISTS", lists)

    def reply(prompt, n):
        if n == 1:
            return ""
        words = board_words_of(prompt)
        return "\n".join([words[3], words[0], *words[1:3], *words[4:]])

    stub = Stub(reply)

    def hosted(model, meter, log, max_tokens):
        return Hosted(model, meter, log, max_tokens, client=stub, sleep=lambda s: None)

    monkeypatch.setattr(listen, "Hosted", hosted)
    listen.listen(77, range(0, 2), 3, 1, Meter(1.0), tmp_path, workers=1)
    assert len(stub.sent) == 2 * 3 + 3 + 3 + 1
    assert stub.sent[0] == guesser_prompt("sun", 1, bs[0].unrevealed())
    rk = read_rankings((tmp_path / "rankings.txt").read_text())
    assert sorted(rk) == [(77, 0, 0), (77, 0, 1), (77, 0, 2), (77, 1, 0)]
    # Word 3, then words 0, 1, 2: word 0 starts the increasing tail.
    assert all(c == "d" for cells in rk.values() for c in cells)
    summary = json.loads((tmp_path / "listen.json").read_text())
    assert summary["retried"] == 1 and summary["cells"] == {"prefix": 12}
    rows = json.loads((tmp_path / "listen-77-0-2.json").read_text())["rows"]
    assert sum(r["retry"] for r in rows) == 1


def test_speak_job_reads_the_choice_from_the_kept_clues(tmp_path, monkeypatch):
    lists = tmp_path / "lists.txt"
    lists.write_text("78 0 sun moon star\n78 1 tree leaf root\n")
    monkeypatch.setattr(listen, "LISTS", lists)

    def reply(prompt, n):
        return "moon 2" if "1. sun" in prompt else "root 5"

    def hosted(model, meter, log, max_tokens):
        return Hosted(model, meter, log, max_tokens, client=Stub(reply))

    monkeypatch.setattr(listen, "Hosted", hosted)
    listen.speak(78, 2, 2, Meter(1.0), tmp_path, workers=1)
    text = (tmp_path / "choices.txt").read_text()
    # K = 2 keeps sun and moon, tree and leaf: root is not listed on board 1.
    assert text == "speaker 78 moon/2 -/unlisted\n"


def test_crossguess_sends_a_shared_clue_once(tmp_path, monkeypatch):
    lines = "a 5 sun/2 moon/1\nb 5 sun/2 star/1\nc 5 sun/1 moon/1\n"
    stub = Stub(lambda prompt, n: "\n".join(board_words_of(prompt)))

    def hosted(model, meter, log, max_tokens):
        return Hosted(model, meter, log, max_tokens, client=stub)

    monkeypatch.setattr(c4, "Hosted", hosted)
    text = c4.crossguess(["llama-3.3-70b"], lines, 5, Meter(1.0), tmp_path, distinct=True)
    # Board 0: sun/2, sun/1; board 1: moon/1, star/1.
    assert len(stub.sent) == 4
    again = c4.crossguess(["llama-3.3-70b"], lines, 5, Meter(1.0), tmp_path)
    assert len(stub.sent) == 4 + 6 and text == again


def test_rankings_lines_round_trip():
    text = "9005 0 0 ab = 0\n9005 0 1 ab ! ?\n"
    assert read_rankings(text) == {(9005, 0, 0): ["ab", "=", "0"], (9005, 0, 1): ["ab", "!", "?"]}


def test_committed_lists_are_the_prefiltered_action_lists():
    """c10-listen.txt holds 20 of the 30 clues of c10-actions.txt on every board, for
    the plan's seeds and board counts."""
    from codenames.actions import read_action_lines

    full = read_action_lines(listen.ACTIONS.read_text())
    kept = read_action_lines(listen.LISTS.read_text())
    assert {s: len(v) for s, v in kept.items()} == {9999: 5, 9005: 100, 9004: 300}
    for seed, rows in kept.items():
        for k, f in zip(rows, full[seed], strict=True):
            assert len(k) == 20 and len(set(k)) == 20 and set(k) <= set(f)
