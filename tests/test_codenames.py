"""Tests for codenames/: exact turn odds, the rules, the players, and the data parsers.

Nothing here downloads a model: the players run on small random vectors.
"""

import json
import random
import struct

import numpy as np
import pytest

from codenames.embed import normalize, read_safetensors, static_encoder
from codenames.game import Board, Clue, deal, play
from codenames.luce import (
    ASSASSIN,
    NEUTRAL,
    OPPONENT,
    TEAM,
    best_count,
    luce_weights,
    turn_odds,
    turn_odds_brute,
)
from codenames.make_words import collides, derived
from codenames.norms import PATTERN, held_out
from codenames.players import (
    Space,
    embedding_guesser,
    embedding_weigher,
    expected_giver,
    expected_value,
    norm_weigher,
    threshold_giver,
)
from codenames.words import legal

KINDS = (TEAM, TEAM, TEAM, OPPONENT, NEUTRAL, NEUTRAL, ASSASSIN)


def test_recursion_matches_every_ordered_sequence():
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = rng.random(len(KINDS)) + 0.05
        fast, slow = turn_odds(w, KINDS, 3), turn_odds_brute(w, KINDS, 3)
        assert np.allclose(fast.all_team, slow.all_team)
        for k in fast.first_wrong:
            assert np.allclose(fast.first_wrong[k], slow.first_wrong[k])


def test_one_pick_is_the_team_share_of_the_weight():
    w = np.array([3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    odds = turn_odds(w, KINDS, 1)
    assert odds.all_team[1, 0] == pytest.approx(5 / 10)
    assert odds.first_wrong[ASSASSIN][1, 0] == pytest.approx(1 / 10)


def test_outcomes_of_a_turn_add_up_to_one():
    # Each turn of up to n picks ends in exactly one way: n team words, or a first wrong pick.
    w = np.random.default_rng(2).random((5, len(KINDS)))
    odds = turn_odds(w, KINDS, 3)
    total = odds.all_team[3] + sum(odds.stopped_by(k, 3) for k in (OPPONENT, NEUTRAL, ASSASSIN))
    assert np.allclose(total, 1)


def test_recursion_matches_simulation():
    w = np.array([2.0, 1.5, 0.5, 1.0, 1.0, 0.7, 0.3])
    odds = turn_odds(w, KINDS, 2)
    rng = random.Random(0)
    n, both = 100_000, 0
    for _ in range(n):
        first = rng.choices(range(7), weights=w.tolist())[0]
        rest = [i for i in range(7) if i != first]
        second = rng.choices(rest, weights=w[rest].tolist())[0]
        both += KINDS[first] == TEAM and KINDS[second] == TEAM
    assert both / n == pytest.approx(odds.all_team[2, 0], abs=0.006)  # about 4 sigma


def test_low_temperature_is_greedy():
    sims = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.0])  # order: T T O ...
    odds = turn_odds(luce_weights(sims, 0.001), KINDS, 3)
    assert odds.all_team[2, 0] == pytest.approx(1)
    assert odds.first_wrong[OPPONENT][3, 0] == pytest.approx(1)


def test_best_count_stops_before_a_likely_assassin():
    sims = np.array([0.9, 0.9, 0.5, 0.1, 0.1, 0.1, 0.5])  # third team word ties the assassin
    odds = turn_odds(luce_weights(sims, 0.01), KINDS, 3)
    n, _ = best_count(odds, {OPPONENT: 1, NEUTRAL: 0, ASSASSIN: 3})
    assert n[0] == 2


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


def test_twins_never_pick_a_wrong_word_with_the_threshold_giver():
    # Kim et al. (2019): a giver paired with a guesser that shares its vectors wins every
    # game. With the threshold giver, the clue's targets are the guesser's top choices.
    sp, board_words, clues = toy_space()
    give = threshold_giver(sp, t=-1.0, candidates=clues)
    guess = embedding_guesser(sp)
    for seed in range(5):
        board = deal(board_words, random.Random(seed))
        result = play(board, give, guess)
        for clue, picks in result.clues:
            assert all(board.kind(w) == TEAM for w in picks[: clue.count - 1])


def test_expected_giver_gives_a_legal_clue_with_its_best_count():
    sp, board_words, clues = toy_space()
    board = deal(board_words, random.Random(3))
    give = expected_giver(sp, tau=0.05, beta=0.0, candidates=clues)
    clue = give(board)
    assert clue.word in clues and 1 <= clue.count <= 4
    weigh = embedding_weigher(sp, 0.05)
    planned = give.plan(board)[2]
    assert expected_value(weigh, board, clue) == pytest.approx(planned, abs=1e-6)


def test_game_rules():
    words = ("a", "b", "c", "d") + tuple(f"x{i}" for i in range(21))
    kinds = (TEAM,) * 9 + (OPPONENT,) * 8 + (NEUTRAL,) * 7 + (ASSASSIN,)
    board = Board(words, kinds)

    def picks_in_order(clue, count, ws):
        return list(ws)

    result = play(board, lambda b: Clue("z", 25), picks_in_order)
    assert result.won and result.turns == 1  # the nine team words come first
    board = Board(tuple(reversed(words)), tuple(reversed(kinds)))
    result = play(board, lambda b: Clue("z", 25), picks_in_order)
    assert not result.won and result.turns == 1  # the assassin is first


def test_norm_weights_and_parsing():
    w = norm_weigher({"sea": {"water": 5, "fish": 2}}, eps=1.0)("sea", ("water", "fish", "car"))
    assert list(w) == [6.0, 3.0, 1.0]
    m = PATTERN.search("remaining: ['spot', 'march'], hint: sandwich")
    assert m and m[2] == "sandwich"
    share = sum(held_out(f"cue{i}") for i in range(10_000)) / 10_000
    assert 0.18 < share < 0.22


def test_legal_clues():
    board = ("bank", "river", "ice")
    assert legal("money", board)
    assert not legal("banks", board) and not legal("ice", board) and not legal("rice", board)


def test_word_list_helpers():
    assert collides("cat", "cats") and not collides("cat", "catalog")
    assert derived("falls", {"fall": 1}, {}) and not derived("fall", {"fall": 1}, {})


def test_static_encoder_and_safetensors(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel({"[UNK]": 0, "sea": 1, "fish": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tmp_path / "tok.json"))
    table = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float16)
    header = json.dumps(
        {"embedding.weight": {"dtype": "F16", "shape": [3, 2], "data_offsets": [0, 12]}}
    )
    blob = struct.pack("<Q", len(header)) + header.encode() + table.tobytes()
    (tmp_path / "t.safetensors").write_bytes(blob)
    loaded = read_safetensors(tmp_path / "t.safetensors")["embedding.weight"]
    assert np.array_equal(loaded, table)
    enc = static_encoder(loaded.astype(np.float32), tmp_path / "tok.json")
    v = enc(["sea fish", "sea"])
    assert np.allclose(v[0], [2**-0.5, 2**-0.5]) and np.allclose(v[1], [1, 0])


def test_infonce_gradient_matches_finite_differences():
    from codenames.train import loss_and_grad

    rng = np.random.default_rng(4)
    e = normalize(rng.normal(size=(30, 5)))
    clue = rng.integers(0, 30, size=6)
    cand = rng.integers(0, 30, size=(6, 25))
    weight = rng.random(6) + 0.1
    a = np.eye(5) + 0.1 * rng.normal(size=(5, 5))
    _, grad = loss_and_grad(a, e, e, clue, cand, weight, l2=0.5, n_total=12)
    h = 1e-6
    for i, j in [(0, 0), (1, 3), (4, 2)]:
        da = np.zeros_like(a)
        da[i, j] = h
        up, _ = loss_and_grad(a + da, e, e, clue, cand, weight, 0.5, 12)
        down, _ = loss_and_grad(a - da, e, e, clue, cand, weight, 0.5, 12)
        assert (up - down) / (2 * h) == pytest.approx(grad[i, j], rel=1e-5, abs=1e-8)


def test_llm_replies_are_parsed_leniently():
    from codenames.llm import llm_guesser, parse_clue

    board = Board(("bank", "river", "ice"), (TEAM, OPPONENT, ASSASSIN))
    assert parse_clue("Clue: MONEY, 2", board) == Clue("money", 2)
    assert parse_clue("banks 1", board) is None  # contains a board word
    guess = llm_guesser(lambda prompt: "1. River\n2. bank\n")
    assert guess("water", 1, ("bank", "river", "ice")) == ["river", "bank", "ice"]


def test_rlvr_reward_and_ensemble_giver():
    from codenames.players import EnsembleGiver
    from codenames.rl import rewards_rlvr

    sp, board_words, clues = toy_space()
    board = deal(board_words, random.Random(5))
    weigh = embedding_weigher(sp, 0.05)
    words = board.unrevealed()
    top = int(np.argmax(weigh(clues[0], words)))
    r = rewards_rlvr(weigh, board, [(clues[0], 1)])[0]
    assert r == pytest.approx(1 / 9 if board.kind(words[top]) == TEAM else -1 / 16)
    other, _, _ = toy_space(seed=1)
    give = EnsembleGiver([(sp, 0.05), (other, 0.05)], beta=0.0, candidates=clues)
    clue = give(board)
    assert legal(clue.word, words) and 1 <= clue.count <= 4


@pytest.mark.parametrize("scheme", ["published", "game"])
def test_calibrate_stops_rather_than_drop_models(monkeypatch, scheme):
    # Without the remote vectors, models() lists only the local four. The committed
    # files hold all eight, so calibrate must stop before it computes anything.
    from codenames import study

    monkeypatch.setattr(study, "models", lambda: list(study.LOCAL))
    monkeypatch.setattr(study, "all_words", lambda: pytest.fail("calibrate did work"))
    with pytest.raises(SystemExit, match="outputs/codenames-embed"):
        study.calibrate(scheme)
