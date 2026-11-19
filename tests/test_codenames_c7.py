"""Tests for plan C7: codenames/rlvr.py (the reward and the GRPO job) and codenames/c7.py
(the tables, the feature policy and the two-way bootstrap).

Nothing is downloaded: the guessers run on small random vectors. The last test runs the
GRPO loop on a tiny random model and is skipped where trl is not installed (CI installs
every group but codenames-gpu).
"""

import itertools
import json
import random
import string

import numpy as np
import pytest

import codenames.c7 as c7
import codenames.rlvr as rlvr
from codenames.embed import normalize
from codenames.game import Board, Clue, deal
from codenames.players import Space, embedding_weigher, expected_value
from codenames.rl import rewards


def name(i: int) -> str:
    """Alphabetic names, since the parser reads a clue as letters: aa, ab, ..."""
    return "".join(string.ascii_lowercase[d] for d in (i // 26, i % 26))


def toy_space(seed=0, n_board=30, n_clues=50):
    rng = np.random.default_rng(seed)
    board = [f"w{name(i)}" for i in range(n_board)]
    vecs = {w: rng.normal(size=16) for w in board}
    clues = []
    for j in range(n_clues):
        group = rng.choice(n_board, size=3, replace=False)
        c = f"q{name(j)}"
        vecs[c] = sum(vecs[board[g]] for g in group) + 0.3 * rng.normal(size=16)
        clues.append(c)
    words = list(vecs)
    m = normalize(np.array([vecs[w] for w in words]))
    return Space(f"toy{seed}", m, m, words), tuple(board), tuple(clues)


@pytest.fixture
def toy():
    giver, board_words, clues = toy_space(0)
    scorers = {f"g{s}": embedding_weigher(toy_space(s)[0], 0.05) for s in (1, 2, 3)}
    boards = [deal(board_words, random.Random(i)) for i in range(4)]
    return giver, scorers, boards, clues


# The parser and the reward.
CLUES = ["ocean", "river", "money", "bank"]


@pytest.mark.parametrize(
    ("reply", "action"),
    [
        ("ocean 3", 2),
        ("RIVER 1", 4),
        ("Clue: money, 4", 11),
        ("1. bank 2", 13),
        ("ocean 0", None),  # a count outside 1 to 4
        ("ocean 5", None),
        ("water 2", None),  # not in the list
        ("ocean", None),  # no count
        ("", None),
        ("water 2 then river 3", 6),  # the first listed clue with a number after it
        ("ocean 3\nI choose: bank 1", 12),  # the last such line decides (C6)
    ],
)
def test_read_choice(reply, action):
    assert rlvr.read_choice(reply, CLUES) == action


def test_a_truncated_reply_has_no_action():
    assert rlvr.parse("ocean 3", CLUES, truncated=True) == {
        "clue": None,
        "count": None,
        "why": "truncated",
        "action": None,
    }
    assert rlvr.reward("ocean 3", CLUES, np.ones(16), truncated=True) == rlvr.INVALID


def test_reward_is_the_listed_value_or_minus_one():
    values = np.arange(16, dtype=float) / 10
    assert rlvr.reward("money 2", CLUES, values) == pytest.approx(0.9)
    assert rlvr.reward("money 9", CLUES, values) == rlvr.INVALID == -1.0


def test_prompt_lists_every_clue_and_the_board():
    board = Board(("bank", "river", "ice"), ("T", "O", "A"))
    text = rlvr.prompt(board, ["money", "water"])
    assert "- bank: yours" in text and "- ice: ASSASSIN" in text
    assert "1. money\n2. water" in text and text.endswith("a number from 1 to 4.")


def test_training_boards_are_fresh_for_each_step():
    assert list(rlvr.step_boards(0)) == list(range(16))
    assert list(rlvr.step_boards(2)) == list(range(32, 48))
    a, b = c7.train_boards(2), c7.train_boards(3)
    assert [x.words for x in a] == [x.words for x in b[:32]]
    assert len({x.words for x in b}) == 48
    assert b[16].words == c7.deal_boards(7001, 16)[0].words


# The tables.
def test_table_values_are_exact_expected_turn_values(toy):
    giver, scorers, boards, pool = toy
    data = c7.build(boards, (giver, 0.05), scorers, pool)
    assert data["clues"].shape == (4, 30) and data["values"].shape == (4, 120, 3)
    for b, board in enumerate(boards):
        clues = [str(c) for c in data["clues"][b]]
        assert len(set(clues)) == 30
        acts = [(c, n) for c in clues for n in range(1, 5)]
        for j, w in enumerate(scorers.values()):
            assert np.allclose(data["values"][b, :, j], rewards(w, board, acts), atol=1e-6)
        # Against the independent single-clue computation, for a few actions.
        mean = data["values"][b].mean(axis=1)
        for a in (0, 7, 61, 119):
            c, n = acts[a]
            want = np.mean([expected_value(w, board, Clue(c, n)) for w in scorers.values()])
            reply = f"{c} {n}"
            assert rlvr.reward(reply, clues, mean) == pytest.approx(want, abs=1e-5)


def test_table_round_trip_and_checksum(toy, tmp_path):
    giver, scorers, boards, pool = toy
    data = c7.build(boards, (giver, 0.05), scorers, pool)
    path = tmp_path / "t.npz"
    np.savez_compressed(
        path,
        train_words=data["words"],
        train_kinds=data["kinds"],
        train_clues=data["clues"],
        train_values=data["values"],
        val_words=data["words"][:2],
        val_kinds=data["kinds"][:2],
        val_clues=data["clues"][:2],
    )
    t = rlvr.read_table(str(path), c7.sha256(path))
    assert len(t["train"]) == 4 and len(t["val"]) == 2 and t["val"].values is None
    assert t["train"].board(1).words == boards[1].words
    assert t["train"].board(1).kinds == boards[1].kinds
    with pytest.raises(ValueError, match="sha256"):
        rlvr.read_table(str(path), "0" * 64)


def test_feature_policy_runs_on_the_table_layout(toy):
    giver, scorers, boards, pool = toy
    part = c7.build(boards, (giver, 0.05), scorers, pool)
    data = {
        "train_phi": part["phi"],
        "train_values": part["values"],
        "training_guessers": list(scorers),
        "test_phi": part["phi"],
    }
    out = c7.feature_policy(data, seed=1, steps=20)
    assert len(out["picks"]["test"]) == 4
    assert all(0 <= a < 120 for a in out["picks"]["test"])


# The two-way bootstrap.
def test_two_way_bootstrap_matches_exact_enumeration():
    d = np.array([[0.3, -0.2, 0.5, 0.1], [0.0, 0.4, -0.1, 0.2]])
    exact = c7.two_way_exact(d)
    assert sum(exact.values()) == pytest.approx(1.0)
    support = np.array(sorted(exact))
    probs = np.array([exact[x] for x in support])
    assert support @ probs == pytest.approx(d.mean())
    cs, cb = c7.two_way_draws(2, 4, draws=200_000, seed=0)
    reps = c7.two_way(d, cs, cb)
    assert reps.mean() == pytest.approx(d.mean(), abs=2e-3)
    assert reps.var() == pytest.approx(((support - d.mean()) ** 2) @ probs, rel=0.02)
    for x in np.percentile(support, [10, 50, 90]):
        assert (reps <= x + 1e-12).mean() == pytest.approx(
            probs[support <= x + 1e-12].sum(), abs=0.005
        )


def test_two_way_exact_by_brute_force():
    # Every ordered draw of seeds and of boards, each with probability 1 / n^n.
    d = np.array([[1.0, 2.0, 4.0], [0.0, 8.0, 16.0]])
    brute: dict[float, float] = {}
    for s in itertools.product(range(2), repeat=2):
        for b in itertools.product(range(3), repeat=3):
            key = round(float(np.mean([d[i, j] for i in s for j in b])), 12)
            brute[key] = brute.get(key, 0.0) + 1 / (4 * 27)
    exact = c7.two_way_exact(d)
    assert exact.keys() == brute.keys()
    assert all(exact[k] == pytest.approx(brute[k]) for k in brute)


def test_score_pairs_seeds_and_uses_the_fallback(tmp_path):
    rng = np.random.default_rng(0)
    names = [
        *c7.HELD_OUT,
        *c7.TRAINING,
        c7.GIVER,
        "eat",
        *(f"potion-map/{s}" for s in c7.MAP_SEEDS),
    ]
    values = rng.normal(size=(5, 120, len(names))).astype(np.float32)
    clues = np.array([[f"{w}{name(i)}" for i in range(30)] for w in "abcde"])
    data = {"guessers": names, "test_values": values, "test_clues": clues}
    first = values[:, :, names.index(c7.GIVER)].argmax(axis=1)
    raw = tmp_path / "raw"
    runs = tmp_path / "runs"
    runs.mkdir()
    for seed in (1, 2):
        run = raw / f"lm-{seed}"
        run.mkdir(parents=True)
        trained = [
            {"action": None, "why": "unlisted", "tokens": 3, "reply": "no"}
            if b == 0
            else {"action": seed + b, "why": None, "tokens": 5, "reply": "x"}
            for b in range(5)
        ]
        untrained = [{"action": None, "why": "unlisted", "tokens": 2, "reply": "?"}] * 5
        (run / "picks.json").write_text(json.dumps({"trained": trained, "untrained": untrained}))
        (run / "config.json").write_text(json.dumps({"seed": seed, "eval": "test"}))
        (run / "timings.json").write_text("{}")
        curve = [
            {"step": i, "mean_reward": i, "invalid": 0.5, "trl": {"entropy": 1}} for i in range(25)
        ]
        (run / "curves.json").write_text(json.dumps(curve))
        (runs / f"lm-{seed}.json").write_text(json.dumps(c7.compact(run)))
        feature = {"seed": seed, "picks": {"test": [7] * 5}}
        (runs / f"feature-{seed}.json").write_text(json.dumps(feature))
    out = c7.score(data, runs)
    held = [names.index(g) for g in c7.HELD_OUT]
    want = []
    for seed in (1, 2):
        a = [first[0], *(seed + b for b in range(1, 5))]
        lm = values[np.arange(5), a][:, held].mean(axis=1)
        want.append(lm - values[:, 7, held].mean(axis=1))
    assert out["seeds"] == [1, 2]
    assert out["difference"]["held_out"]["mean"] == pytest.approx(np.mean(want), abs=1e-6)
    assert out["replies"] == {
        "1": {"unlisted": 1, "read": 4},
        "2": {"unlisted": 1, "read": 4},
        "untrained": {"unlisted": 5},
    }
    assert out["behavior"]["arm2"]["1"]["mean_count"] == 4  # action 7 is clue 1, count 4
    assert out["behavior"]["untrained"]["first_choice"] == 1
    curves = out["runs"]["1"]["curves_every_20_steps"]
    assert [c["steps"] for c in curves] == [[0, 19], [20, 24]] and curves[1]["mean_reward"] == 22


# The GRPO loop, on a tiny random model (needs the codenames-gpu group).
def test_grpo_loop_trains_the_adapter(toy, tmp_path, monkeypatch):
    pytest.importorskip("trl")
    safetensors = pytest.importorskip("safetensors.numpy")
    giver, scorers, _, pool = toy
    board_words = toy_space(0)[1]
    boards = [deal(board_words, random.Random(i)) for i in range(32)]
    data = c7.build(boards, (giver, 0.05), scorers, pool)
    path = tmp_path / "t.npz"
    np.savez_compressed(
        path,
        **{f"train_{k}": data[k] for k in ("words", "kinds", "clues", "values")},
        **{f"val_{k}": data[k][:3] for k in ("words", "kinds", "clues")},
    )
    # A random model almost never names a listed clue, so every reward would be -1 and
    # every advantage 0. This reward varies with the reply, so the adapter must move.
    monkeypatch.setattr(rlvr, "reward", lambda reply, clues, values, cut: float(len(reply) % 5))
    out = tmp_path / "out"
    rlvr.main(["--tiny", "--steps", "2", "--eval", "val", "--table", str(path), "--out", str(out)])
    # --stop-after 0 stops after the first step and still writes everything.
    early = tmp_path / "early"
    argv = ["--tiny", "--steps", "2", "--eval", "val", "--table", str(path), "--out", str(early)]
    rlvr.main([*argv, "--stop-after", "0"])
    assert json.loads((early / "config.json").read_text())["steps_done"] == 1
    assert len(json.loads((early / "picks.json").read_text())["trained"]) == 3
    at = tmp_path / "at"
    argv = ["--tiny", "--steps", "2", "--eval", "val", "--table", str(path), "--out", str(at)]
    rlvr.main([*argv, "--stop-at-step", "1"])
    log = json.loads((at / "trl_log.json").read_text())
    assert json.loads((at / "config.json").read_text())["steps_done"] == 1
    assert log[0]["learning_rate"] == pytest.approx(2e-5)  # the schedule spans 2 steps
    # A random model's rewards are all -1: no spread, so --stop-flat 1 ends at step 1.
    flat = tmp_path / "flat"
    argv = ["--tiny", "--steps", "2", "--eval", "val", "--table", str(path), "--out", str(flat)]
    monkeypatch.setattr(rlvr, "reward", lambda reply, clues, values, cut: -1.0)
    rlvr.main([*argv, "--stop-flat", "1", "--min-steps", "1"])
    assert json.loads((flat / "config.json").read_text())["steps_done"] == 1
    curves = json.loads((out / "curves.json").read_text())
    assert [c["step"] for c in curves] == [0, 1] and all(c["replies"] == 128 for c in curves)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["boards_seen"] == [0, 31, 32]
    picks = json.loads((out / "picks.json").read_text())
    assert len(picks["untrained"]) == len(picks["trained"]) == 3
    weights = safetensors.load_file(str(out / "adapter" / "adapter_model.safetensors"))
    lora_b = [v for k, v in weights.items() if "lora_B" in k]
    assert lora_b and any(np.abs(v).max() > 0 for v in lora_b)  # B starts at zero
    assert not any("visual" in k for k in weights)
    summary = c7.compact(out)
    assert len(summary["trained"]) == 3 and [c[0] for c in summary["curve"]] == [0, 1]
