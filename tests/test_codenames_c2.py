"""Tests for codenames/c2.py and the parts of codenames/lora.py that run without torch.

The human data is replaced by small synthetic turns, so nothing is downloaded.
"""

import math
import random
import time

import numpy as np
import pytest

import codenames.c2 as c2
import codenames.evaluate as evaluate
import codenames.lora as lora
from codenames.embed import TEMPLATES, normalize
from codenames.norms import Guess, held_out
from codenames.players import Space


def toy_turns(seed: int, n: int, vocab: list[str], unique: bool = False) -> tuple[Guess, ...]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        clue, *words = rng.sample(vocab, 1 + rng.randint(8, 12))
        out.append(Guess(f"{clue}x{i}" if unique else clue, tuple(words), rng.choice(words)))
    return tuple(out)


@pytest.fixture
def toy(monkeypatch):
    vocab = [f"w{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    sp = Space(
        "toy", normalize(rng.normal(size=(40, 8))), normalize(rng.normal(size=(40, 8))), vocab
    )
    splits = {s: toy_turns(i, 60, vocab) for i, s in enumerate(("train", "val", "test"))}

    def read(split, scheme="published"):
        return splits[split]

    monkeypatch.setattr(evaluate, "cultural_guesses", read)
    monkeypatch.setattr(c2, "cultural_guesses", read)
    return sp, splits


def test_log_probs_are_a_distribution_over_the_board():
    s = np.array([0.3, -0.1, 0.2, 0.25])
    total = sum(math.exp(c2.log_probs([(s, k)], 0.05)[0]) for k in range(4))
    assert total == pytest.approx(1.0)
    assert c2.hits([(s, 0), (s, 2)]) == [1, 0]


def test_score_agrees_with_evaluate(toy):
    # c2.score computes per turn what evaluate.guess_scores computes in aggregate.
    sp, splits = toy
    out = c2.score({"bare": sp}, splits, "test")
    ref = evaluate.guess_scores(sp, "test")
    assert out["chosen"]["tau"] == ref["tau"]
    assert out["top1"] == pytest.approx(ref["top1"])
    assert out["loglik"] == pytest.approx(ref["loglik"])
    assert [t["clue"] for t in out["turns"]] == [g.clue for g in splits["test"]]
    assert out["chosen"]["val_loglik"] == pytest.approx(
        evaluate.guess_loglik(sp, "val", ref["tau"])
    )


def test_score_picks_the_template_by_validation(toy):
    sp, splits = toy
    rng = np.random.default_rng(1)
    noise = Space("noise", normalize(rng.normal(size=sp.clue_vecs.shape)), sp.word_vecs, sp.words)
    out = c2.score({"bare": sp, "query": noise}, splits, "val")
    best = max(out["settings"], key=lambda r: r["val_loglik"])
    assert out["chosen"] == best and out["split"] == "val"
    assert "test" not in out and out["n"] == len(splits["val"])


def test_published_scheme_reads_the_published_split(toy):
    _, splits = toy
    assert c2.guesses("val") == splits["val"]


def fake_eat():
    cues = [f"cue{i}" for i in range(200)]
    return {c: {"rare": 1, "often": 7, c: 9} for c in cues}


def test_training_rows(monkeypatch):
    board = [f"b{i}" for i in range(30)]
    turns = toy_turns(0, 40, board, unique=True)
    monkeypatch.setattr(lora, "guesses", lambda split, scheme: turns)
    monkeypatch.setattr(lora, "eat_counts", fake_eat)
    monkeypatch.setattr(lora, "board_words", lambda: tuple(board))
    cultural = lora.rows(1, "cultural")
    assert len(cultural) == len(turns)  # every toy board has at least 8 other words
    by_clue = {g.clue: g for g in turns}
    for r in cultural:
        clue = r["anchor"].rsplit("Query: ", 1)[1]
        assert r["anchor"] == TEMPLATES["query"].format(clue)
        g = by_clue[clue]
        assert r["positive"] == g.picked
        negs = [r[f"negative_{i}"] for i in range(1, 8)]
        assert len(set(negs)) == 7 and set(negs) <= set(g.words) - {g.picked}
    both = lora.rows(1, "cultural+eat")
    eat = [r for r in both if r["anchor"].rsplit("Query: ", 1)[1].startswith("cue")]
    kept = [c for c in fake_eat() if not held_out(c)]
    assert len(eat) == len(kept)  # only "often" (7 people); not "rare", not the cue itself
    assert all(r["positive"] == "often" for r in eat)
    assert lora.rows(1, "cultural") == cultural  # seeded


def test_container_start_is_in_the_past():
    start = lora.container_start()
    if start is not None:
        assert 0 < start <= time.time()


def test_summary_keeps_hits_in_order():
    turns = [{"clue": "a", "picked": "x", "hit": 1, "logp": -0.1}]
    turns.append({"clue": "b", "picked": "y", "hit": 0, "logp": -2.0})
    result = {"config": {"seed": 1}, "before": {"top1": 0.5, "turns": turns}}
    s = lora.summary(result)
    assert s["before"]["hits"] == "10" and "turns" not in s["before"] and "after" not in s
    assert s["before"]["turns_sha256"] == c2.turn_digest([("a", "x"), ("b", "y")])
    assert result["before"]["turns"] == turns  # the full result is not changed


def test_committed_c2_results_match_the_run_summaries():
    # results/c2.json is computed from results/c2-runs/ and C1's map results; recompute
    # its point estimates from those files.
    import json
    from pathlib import Path

    res = Path(c2.__file__).parent / "results"
    out = json.loads((res / "c2.json").read_text())
    runs = {
        r: json.loads((res / "c2-runs" / f"{r}.json").read_text()) for r in c2.TEST_RUNS["lora"]
    }
    lora_hits = np.mean([c2.as_hits(s["after"]["hits"]) for s in runs.values()], axis=0)
    maps = json.loads((res / "train-potion-game.json").read_text())["runs"].values()
    potion = np.mean([c2.as_hits(r["guess"]["hits"]) for r in maps], axis=0)
    assert out["arms"]["lora"]["top1"]["mean"] == pytest.approx(lora_hits.mean())
    assert out["primary"]["mean"] == pytest.approx((lora_hits - potion).mean())
    assert len(lora_hits) == out["turns"] == 551
    lo, hi = out["primary"]["ci"]
    assert lo < out["primary"]["mean"] < hi
