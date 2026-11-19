"""Tests for codenames/c3.py: the slices, the interquartile mean, the stratified bootstrap
(against an exact enumeration of every resample), and the analysis on synthetic runs."""

import itertools
import json

import numpy as np
import pytest

import codenames.c3 as c3


def test_slices_cover_every_task_and_seed_once():
    got = [c3.slice_of(n) for n in range(70)]
    assert len(set(got)) == 70
    assert {g for g, _, _ in got} == set(c3.GIVERS)
    for g, m, s in got:
        assert m != g and m in c3.MODELS and s in c3.SEEDS
    assert c3.slice_of(0) == ("potion", "wordllama", 1)
    assert c3.slice_of(69) == ("Qwen/Qwen3-Embedding-0.6B", "google/embeddinggemma-2", 5)
    # The two exploratory tasks of the README are among them.
    assert ("potion", "bge") in c3.tasks()
    assert ("Qwen/Qwen3-Embedding-0.6B", "google/embeddinggemma-2") in c3.tasks()


def test_iqm():
    assert c3.iqm(np.arange(8.0)) == pytest.approx(np.mean([2, 3, 4, 5]))
    x = np.array([100.0, -50.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # cut 1 from each end
    assert c3.iqm(x) == pytest.approx(np.mean([1, 2, 3, 4, 5]))
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    for n in (4, 7, 70, 71):
        y = rng.normal(size=(3, n))
        assert np.allclose(c3.iqm(y), stats.trim_mean(y, 0.25, axis=1))


def exact_moments(d: np.ndarray) -> tuple[float, float]:
    """Mean and variance of the stratified bootstrap statistic for one task, by listing
    every ordered draw of boards and of seeds."""
    s, b = d.shape
    vals = []
    for boards in itertools.product(range(b), repeat=b):
        for seeds in itertools.product(range(s), repeat=s):
            vals.append(d[np.ix_(seeds, boards)].mean())
    v = np.array(vals)  # every ordered draw has the same probability
    return float(v.mean()), float(v.var())


def test_bootstrap_matches_exact_enumeration():
    d = np.array([[0.0, 1.0, 3.0], [2.0, -1.0, 0.5]])
    draws = c3.Draws(3, [2], draws=200_000, seed=3)
    reps = c3.task_reps(d, draws, 0)
    mean, var = exact_moments(d)
    assert reps.mean() == pytest.approx(mean, abs=0.01)
    assert reps.var() == pytest.approx(var, rel=0.02)


def test_boards_are_shared_across_tasks():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(1, 50))
    draws = c3.Draws(50, [1, 1], draws=500)
    s = c3.summarize([d, -d], draws)
    # With one board draw for both tasks, the two tasks cancel in every resample.
    assert s["mean_over_tasks"]["ci"] == [0.0, 0.0]
    assert s["per_task"][0]["ci"] == [-x for x in reversed(s["per_task"][1]["ci"])]


def fake_runs(folder, drop=()):
    rng = np.random.default_rng(7)
    start = {m: rng.normal(size=200).round(4).tolist() for m in (*c3.MODELS, "eat")}
    for n in range(70):
        g, m, s = c3.slice_of(n)
        if (g, m, s) in drop:
            continue
        per = {p: {x: rng.normal(size=200).round(4).tolist() for x in start} for p in c3.POLICIES}
        per["start"] = start
        per["population"][m] = (np.array(per["self"][m]) + 0.25).round(4).tolist()
        theta = {p: rng.normal(size=7).tolist() for p in c3.POLICIES}
        theta["self"] = [float(s + c3.GIVERS.index(g))] * 7
        run = {
            "config": {
                "giver": g,
                "held_out": m,
                "seed": s,
                "test_seed": c3.TEST_SEED,
                "train_boards": 300,
                "test_boards": 200,
                "shortlist": 30,
                "steps": 400,
                "lr": 0.05,
            },
            "slice": n,
            "theta": theta,
            "per_board": per,
        }
        sub = folder / f"seed-{n}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"{c3.run_name(g, m, s)}.json").write_text(json.dumps(run))
        (sub / "manifest.json").write_text("{}")


def test_analysis_on_synthetic_runs(tmp_path):
    fake_runs(tmp_path)
    r = c3.analyze(tmp_path)
    assert r["missing_runs"] == []
    assert all(r["checks"].values())
    p = r["primary"]["mean_over_tasks"]
    assert p["mean"] == pytest.approx(0.25, abs=1e-4)
    assert p["ci"][0] == pytest.approx(0.25, abs=1e-3)
    assert r["primary"]["share_of_runs_above_0"]["mean"] == 1.0
    assert len(r["theta"]["runs"]) == 70
    assert len(r["per_board"]["held_out"]["self"]) == 14


def test_analysis_with_a_missing_run(tmp_path):
    fake_runs(tmp_path, drop={("potion", "bge", 3)})
    r = c3.analyze(tmp_path)
    assert r["missing_runs"] == ["potion__bge__3"]
    assert r["tasks"][2]["seeds"] == [1, 2, 4, 5]
    assert r["primary"]["runs"] == 69
