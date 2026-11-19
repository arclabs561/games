"""Tests for codenames/distill.py, the teacher sources of codenames/lora.py, and
codenames/c8.py. The human data is replaced by small synthetic turns, and the router by a
stub, so nothing is downloaded and no model is called."""

import json
import random
from collections import Counter

import numpy as np
import pytest

import codenames.c8 as c8
import codenames.distill as distill
import codenames.lora as lora
from codenames.c4 import MODELS
from codenames.hosted import Hosted, Meter
from codenames.llm import guesser_prompt
from codenames.norms import Guess
from codenames.words import legal


def toy(seed: int, n: int, vocab: list[str]) -> tuple[Guess, ...]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        words = tuple(rng.sample(vocab, rng.randint(9, 14)))
        out.append(Guess(f"clue{letters('', 50)[i]}", words, rng.choice(words)))
    return tuple(out)


def letters(prefix: str, n: int) -> list[str]:
    """n alphabetic words (replies are parsed into alphabetic tokens)."""
    abc = "abcdefghijklmnopqrstuvwxyz"
    return [prefix + abc[i // 26] + abc[i % 26] for i in range(n)]


TRAIN = toy(0, 50, letters("t", 30))
HELD = toy(1, 20, letters("v", 30))  # val and test boards: other words


def test_board_vocab_is_the_words_of_the_given_boards():
    assert distill.board_vocab(TRAIN) == tuple(sorted({w for g in TRAIN for w in g.words}))
    assert not set(distill.board_vocab(TRAIN)) & set(distill.board_vocab(HELD))


def test_resampled_boards_keep_size_and_draw_only_training_words():
    vocab = distill.board_vocab(TRAIN)
    out = distill.resample_boards(TRAIN, vocab, k=4, seed=8000)
    assert len(out) == 4 * len(TRAIN)
    assert [(t.turn, t.copy) for t in out] == [(i, j) for i in range(len(TRAIN)) for j in range(4)]
    held = set(distill.board_vocab(HELD))
    replaced = total = 0
    for t in out:
        g = TRAIN[t.turn]
        assert t.kind == "resampled" and t.clue == g.clue
        assert len(t.words) == len(g.words) == len(set(t.words))
        assert set(t.words) <= set(vocab) and not set(t.words) & held
        new = [w for w, old in zip(t.words, g.words, strict=True) if w != old]
        assert all(legal(t.clue, (w,)) for w in new)
        replaced += len(new)
        total += len(t.words)
    # each word is replaced with probability 1/2 (a replacement can draw the word that was
    # there only if it left the board earlier, which cannot happen: it is never re-added)
    assert 0.45 < replaced / total < 0.55
    assert distill.resample_boards(TRAIN, vocab, 4, 8000) == out  # seeded
    assert distill.resample_boards(TRAIN, vocab, 4, 8001) != out


def test_a_replacement_never_breaks_the_clue_rule():
    g = Guess("sea", ("a1", "a2", "a3"), "a1")
    vocab = ["seat", "sea", "undersea", "b1"]  # only b1 is legal with the clue "sea"
    for seed in range(20):
        board = distill.resample_board(g.clue, g.words, vocab, random.Random(seed), p=1.0)
        assert set(board) <= {"a1", "a2", "a3", "b1"} and len(set(board)) == 3


class Stub:
    """A router that names the board words in reverse order, or nothing for some clues."""

    def __init__(self):
        self.sent = 0

    def chat_completion(self, messages, model, extra_body=None, **params):
        self.sent += 1
        prompt = messages[0]["content"]
        words = prompt.split("Board words:\n")[1].split("\n")
        reply = "no idea" if "clueah'" in prompt else "\n".join(reversed(words))
        return {
            "model": model,
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 1e-5},
        }


def label_rows():
    hosted = Hosted(MODELS["gpt-oss-120b"], Meter(1.0), client=Stub())
    todo = distill.resample_boards(TRAIN, distill.board_vocab(TRAIN), 4) + distill.relabeled(TRAIN)
    records = [hosted.call(guesser_prompt(t.clue, 1, t.words)) for t in todo]
    return todo, records, distill.teacher_rows(todo, records)


def test_teacher_rows_and_agreement():
    todo, records, rows = label_rows()
    assert len(rows) == 5 * len(TRAIN)
    for t, r in zip(todo, rows, strict=True):
        if t.clue == "clueah":
            assert r["label"] is None and r["why"] == "named no board word"
        else:
            assert r["label"] == t.words[-1] and r["named"] == len(t.words)
    hits = distill.agreement(rows, TRAIN)
    expect = "".join(
        "-" if g.clue == "clueah" else str(int(g.words[-1] == g.picked)) for g in TRAIN
    )
    assert hits == expect
    stopped = distill.teacher_rows(todo[:2], [None, records[1]])
    assert stopped[0]["why"] == "stopped" and stopped[1]["label"] == rows[1]["label"]


def test_repeat_changes_counts_turns_not_replies():
    first = [{"words": ["a", "b"], "label": "a", "reply": "a\nb"}] * 3
    same = {"reply": "a\nb"}
    other_order = {"reply": "b\na"}
    other_text = {"reply": "a\nb\n"}
    out = distill.repeat_changes(first, [[same, other_text, None], [same, other_order, same]])
    assert out == {"turns": 3, "times": 2, "sent": 5, "label_changed": 1, "reply_changed": 1}


def test_lora_rows_with_teacher_sources(monkeypatch):
    monkeypatch.setattr(lora, "guesses", lambda split, scheme: TRAIN)
    _, _, teacher = label_rows()
    resampled = [r for r in teacher if r["kind"] == "resampled" and r["label"]]
    relabeled = [r for r in teacher if r["kind"] == "relabeled" and r["label"]]
    assert len(resampled) == 4 * (len(TRAIN) - 1) and len(relabeled) == len(TRAIN) - 1
    human = lora.rows(1, "cultural", "game")
    both = lora.rows(1, "cultural+teacher", "game", teacher)
    only = lora.rows(1, "teacher", "game", teacher)
    rel = lora.rows(1, "relabeled", "game", teacher)
    assert (len(human), len(both), len(only), len(rel)) == (50, 50 + 196, 196, 49)
    count = Counter((r["anchor"], r["positive"]) for r in both)
    assert count == Counter((r["anchor"], r["positive"]) for r in human + only)
    by_clue = {g.clue: g for g in TRAIN}
    for r in rel:
        g = by_clue[r["anchor"].rsplit("Query: ", 1)[1]]
        negs = {r[f"negative_{i}"] for i in range(1, 8)}
        assert r["positive"] == g.words[-1] and len(negs) == 7
        assert negs <= set(g.words) - {r["positive"]}
    assert lora.rows(1, "cultural+teacher", "game", teacher) == both  # seeded
    bad = [dict(r, clue="other") if r["kind"] == "relabeled" else r for r in teacher]
    with pytest.raises(AssertionError):
        lora.rows(1, "relabeled", "game", bad)


def test_load_teacher_checks_the_digest(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps({"kind": "resampled", "label": "a"}) + "\n")
    rows, digest = lora.load_teacher(str(path))
    assert rows == [{"kind": "resampled", "label": "a"}] and len(digest) == 64
    assert lora.load_teacher(str(path), digest)[1] == digest
    with pytest.raises(SystemExit):
        lora.load_teacher(str(path), "0" * 64)


def test_gpt_oss_hits_read_c4_lines():
    lines = ["llama-3.3-70b test 0000", "gpt-oss-120b val 1111", "gpt-oss-120b test 10ba"]
    assert c8.teacher_test_hits(lines).tolist() == [1, 0, 1, 0]


def test_lr_choice_and_stopping_rule():
    val = {"0.0001": -1.60, "0.0003": -1.55}
    assert c8.choose_lr(val, c2_val=-1.52) == (0.0003, True)  # below C2: the arm failed
    assert c8.choose_lr({"0.0001": -1.40, "0.0003": -1.45}, c2_val=-1.52) == (0.0001, False)
    assert c8.choose_lr({"0.0001": -1.40, "0.0003": -1.40}, c2_val=-1.52) == (0.0001, False)


def test_committed_c8_results_match_the_run_summaries():
    out = c8.load_results()
    if out is None:
        pytest.skip("results/c8.json is written after the runs")
    summaries = c8.read_runs(c8.RESULTS / "c8-runs")
    for arm, runs in c8.ARMS.items():
        if arm not in out["arms"] or not all(r in summaries for r in runs):
            continue
        hits = np.mean([c8.as_hits(summaries[r]["after"]["hits"]) for r in runs], axis=0)
        assert out["arms"][arm]["top1"]["mean"] == pytest.approx(hits.mean())
    if "primary" in out:
        lo, hi = out["primary"]["ci"]
        assert lo <= out["primary"]["mean"] <= hi


def test_label_job_writes_its_files(monkeypatch, tmp_path):
    import argparse

    import codenames.c4 as c4
    import codenames.split as split
    from codenames.split import Turn

    turns = tuple(Turn("train", i, f"g{i // 5}", "a", "b", g) for i, g in enumerate(TRAIN))
    stub = Stub()
    monkeypatch.setattr(split, "turns", lambda name: turns)
    monkeypatch.setattr(c4, "price_models", lambda models, out: None)
    monkeypatch.setattr(Hosted, "connect", lambda self: stub)
    args = argparse.Namespace(
        seed=8000, limit=0, repeats=10, workers=4, cap=1.0, stop_after=10, out=tmp_path
    )
    distill.label(args)
    rows = [json.loads(line) for line in (tmp_path / "teacher.jsonl").read_text().splitlines()]
    assert len(rows) == 5 * len(TRAIN) and stub.sent == len(rows) + 2 * 10
    summary = json.loads((tmp_path / "label.json").read_text())
    assert summary["turns"]["resampled"] == {
        "sent": 200, "labeled": 196, "named_none": 4, "stopped": 0, "truncated": 0
    }  # fmt: skip
    assert summary["train_hits"] == distill.agreement(rows, TRAIN)
    assert summary["repeats"]["label_changed"] == summary["repeats"]["reply_changed"] == 0
    assert summary["teacher_sha256"] == distill.sha256(tmp_path / "teacher.jsonl")
    assert summary["spent"] == pytest.approx(1e-5 * stub.sent)
    # the job reads its own output back as lora does
    teacher, digest = lora.load_teacher(str(tmp_path / "teacher.jsonl"), summary["teacher_sha256"])
    assert teacher == rows


def test_compact_summary_keeps_what_the_analysis_reads():
    full = {
        "config": {"seed": 1, "sources": "teacher"},
        "before": {"hits": "00", "top1": 0.0},
        "after": {"hits": "10", "top1": 0.5, "loglik": -1.0, "settings": [], "split": "test"},
    }
    text = c8.compact_text(full)
    assert text.endswith("\n") and "\n" not in text[:-1]
    small = json.loads(text)
    assert small == {"config": full["config"], "after": {k: full["after"][k] for k in
                     ("hits", "top1", "loglik", "split")}}  # fmt: skip
    assert c8.compact_text(small) == text  # compacting a compact summary changes nothing
