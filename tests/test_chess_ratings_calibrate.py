"""Chess: the calibration of the rating from one game, on synthetic players whose moves come from
a known model of move choice by rating."""

import itertools
import json
import math

import duckdb
import numpy as np
import pytest

from chess_ratings import calibrate as cal
from chess_ratings import calibration, estimate, games
from chess_ratings.calibrate import GRID, OLD, Arm, Record

OPTIONS = 4


def synthetic(n: int, seed: int, shift: float = 0.0, repeat: int = 1, moves: int = 30):
    """Players with true ratings spread over 600 to 2600. Each move has four options whose
    chances are a softmax linear in the rating; a player rated t moves as the model says a player
    rated t + shift would. With repeat k, each distinct move is copied k times, so the moves are
    dependent and the likelihood is k times too sharp."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = float(rng.uniform(600, 2600))
        rows = []
        for _ in range(moves // repeat):
            a = rng.normal(0, 1, OPTIONS)
            b = rng.normal(0, 1.2, OPTIONS)
            logits = a[None, :] + b[None, :] * (GRID[:, None] - 1500) / 400
            logp = logits - cal.logsumexp(logits, axis=1)[:, None]  # (grid, options)
            chance = np.exp(a + b * (t + shift - 1500) / 400)
            k = rng.choice(OPTIONS, p=chance / chance.sum())
            rows += [logp[:, k]] * repeat
        logp = np.array(rows)
        out.append(
            Record(
                rating=t,
                band=estimate.band(t),
                game=f"g{i // 2}",
                fold=i % 2,
                time_control="180+0",
                logp=logp,
                ply=np.arange(len(rows), dtype=float) * 2,
                clock=np.linspace(1, 0.3, len(rows)),
                month_change=0.0,
            )
        )
    return out


def test_old_arm_reproduces_the_old_estimate():
    r = synthetic(1, seed=1)[0]
    old = estimate.estimate(r.logp[:, OLD])
    e = Arm("F0", old_grid=True).evaluate(r)
    assert e["estimate"] == pytest.approx(old["posterior_mean"])
    assert (e["lo"], e["hi"]) == pytest.approx((old["posterior_lo"], old["posterior_hi"]))


def test_a_correct_model_is_calibrated_and_unbiased():
    recs = synthetic(400, seed=2)
    f1 = Arm("F1")
    assert 0.92 <= cal.coverage(f1, recs) <= 0.98
    s = cal.sandwich(recs)
    assert 0.8 < s["tau_centered"] < 1.25
    mid = [r for r in recs if 1000 <= r.rating < 2200]
    bias = np.mean([f1.evaluate(r)["estimate"] - r.rating for r in mid])
    assert abs(bias) < 40
    per_move = cal.mean_scores(recs, lambda r, j: "all", seed=1)["all"]
    lo, hi = per_move["interval"]
    assert lo < 0 < hi


def test_dependent_moves_are_found_and_tempered():
    recs = synthetic(400, seed=3, repeat=3)
    f1 = Arm("F1")
    assert cal.coverage(f1, recs) < 0.85
    assert 0.2 < cal.sandwich(recs)["tau_centered"] < 0.5  # about 1/3
    tau = cal.fit_tau(f1, recs)
    assert 0.2 < tau < 0.5
    assert abs(cal.coverage(cal.replace(f1, tau=tau), recs) - 0.95) < 0.02


def test_a_shifted_scale_is_mapped_back():
    recs = synthetic(600, seed=4, shift=200)
    f1 = Arm("F1")
    mid = [r for r in recs if 1000 <= r.rating < 2200]
    bias = np.mean([f1.evaluate(r)["estimate"] - r.rating for r in mid])
    assert 150 < bias < 250
    rows = cal.best_rating_by_bin(recs)
    middle = [x for x in rows if 1000 <= x["bin"] < 2200]
    assert np.mean([x["best_rating"] - x["mean_rating"] for x in middle]) == pytest.approx(
        200, abs=40
    )
    f5 = cal.fit_arm("F5", recs)
    fresh = [r for r in synthetic(400, seed=5, shift=200) if 1000 <= r.rating < 2200]
    assert abs(np.mean([f5.evaluate(r)["estimate"] - r.rating for r in fresh])) < 40
    f6 = cal.fit_arm("F6", recs)
    assert abs(np.mean([f6.evaluate(r)["estimate"] - r.rating for r in fresh])) < 40


def test_cross_validation_prefers_the_map_when_the_scale_is_shifted():
    recs = synthetic(300, seed=6, shift=250)
    cv = cal.cross_validate(recs, names=("F0", "F1", "F5"))
    assert cv["F5"]["log_score"] > cv["F1"]["log_score"]
    assert cal.choose(cv) == "F5"


def test_mixtures_of_zero_width_change_nothing():
    r = synthetic(1, seed=7)[0]
    assert np.allclose(cal.game_mixture(r.curve, 0), r.curve)
    assert np.allclose(cal.move_mixture(r.logp, 0), r.curve)
    k = cal.kernel(150)
    assert np.allclose(k.sum(axis=1), 1)
    wide = Arm("F3", game_sigma=200).evaluate(r)
    narrow = Arm("F1").evaluate(r)
    assert wide["hi"] - wide["lo"] > narrow["hi"] - narrow["lo"]


def test_isotonic_matches_brute_force():
    """Against the best nondecreasing fit over a small grid of values, found by brute force."""
    rng = np.random.default_rng(8)
    levels = np.arange(0, 5)
    for _ in range(20):
        y = rng.integers(0, 5, 5).astype(float)
        w = rng.integers(1, 4, 5).astype(float)
        fit = cal.isotonic(y, w)
        assert np.all(np.diff(fit) >= -1e-12)
        best = min(
            (c for c in itertools.product(levels, repeat=5) if np.all(np.diff(c) >= 0)),
            key=lambda c: float(np.sum(w * (y - np.array(c)) ** 2)),
        )
        # the true fit is no worse than the best on integer levels
        assert np.sum(w * (y - fit) ** 2) <= np.sum(w * (y - np.array(best)) ** 2) + 1e-9


def test_maps_invert():
    h = (np.array([600.0, 1000, 2000]), np.array([800.0, 1150, 2050]))
    x = np.array([400.0, 700, 1500, 2500])
    assert np.allclose(cal.invert_map(h, cal.apply_map(h, x)), x)
    assert cal.apply_map(h, np.array([3000.0]))[0] == 3050


def test_log_score_and_summaries():
    w = np.zeros(len(GRID))
    w[GRID == 1500] = 1.0
    mean, lo, hi = cal.summarize(w, GRID)
    assert mean == 1500 and lo == pytest.approx(1452.5) and hi == pytest.approx(1547.5)
    inside = cal.log_score(w, GRID, 1520)
    outside = cal.log_score(w, GRID, 2000)
    assert inside == pytest.approx(math.log((1 - cal.FLOOR) + cal.FLOOR * 100 / 2700))
    assert outside == pytest.approx(math.log(cal.FLOOR * 100 / 2700))


def test_interval_helpers():
    assert cal.wilson(95, 100) == pytest.approx((0.8882, 0.9785), abs=1e-4)
    assert cal.mcnemar_one_sided(0, 0) == 1.0
    assert cal.mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert cal.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    recs = synthetic(20, seed=9)
    (lo, hi), _ = cal.game_bootstrap(recs, lambda i: float(len(i)), seed=1, resamples=200)
    assert lo == hi == 20  # games of two player-games each, resampled whole


def test_prior_from_counts():
    counts = {"180+0": {"1500": 100, "1525": 100}, "other": {"800": 10}}
    prior = cal.prior_from_counts(counts)
    for p in prior.values():
        assert np.exp(p).sum() == pytest.approx(1)
    assert GRID[np.argmax(prior["180+0"])] == 1500 and GRID[np.argmax(prior["other"])] == 800


def test_arm_round_trips_through_json():
    recs = synthetic(200, seed=10, shift=150)
    arm = cal.replace(cal.fit_arm("F7", recs), prior=cal.prior_from_counts({"180+0": {"1500": 5}}))
    back = calibration.arm_from_json(calibration.arm_to_json(arm))
    r = recs[0]
    assert back.evaluate(r) == pytest.approx(arm.evaluate(r))


def test_pilot_and_test_steps_run_end_to_end():
    strat = synthetic(120, seed=11, shift=100)
    pop = synthetic(60, seed=12, shift=100)
    counts = {"180+0": {str(b): 10 for b in range(600, 2600, 25)}}
    result, fit, cases = calibration.pilot(strat, pop, counts, log=lambda m: None)
    assert result["chosen"] in cal.ARMS and "true rating" in cases
    from chess_ratings.job import rounded

    json.dumps(rounded(result, 3))  # the job writes these as JSON
    fit = json.loads(json.dumps(rounded(fit, 6)))
    held = synthetic(80, seed=13, shift=100)
    for r in held:
        r.extra["top"] = np.zeros(r.logp.shape[0])
    out = calibration.test(held, pop, fit)
    json.dumps(rounded(out, 4))
    p = out["primary"]
    assert p["player_games"] == 80 and set(p["holm"]) == {"bias", "coverage"}


def parquet(tmp_path, rows):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t (Event VARCHAR, Site VARCHAR, White VARCHAR, Black VARCHAR,"
        " WhiteElo SMALLINT, BlackElo SMALLINT, WhiteRatingDiff SMALLINT, BlackRatingDiff SMALLINT,"
        " UTCDate DATE, UTCTime TIME, TimeControl VARCHAR, Termination VARCHAR, Result VARCHAR)"
    )
    for r in rows:
        con.execute("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", r)
    f = str(tmp_path / "train-00003-of-00009.parquet")
    con.execute(f"COPY t TO '{f}' (FORMAT parquet)")
    return f


def test_population_sample_month_and_counts_sql(tmp_path):
    rows = [
        (
            "Rated Blitz game",
            f"s{i}",
            f"w{i % 3}",
            "b",
            400 + 30 * i,
            410 + 30 * i,
            5,
            -5,
            "2025-09-02",
            f"00:00:{i:02d}",
            "180+0",
            "Normal",
            "1-0",
        )
        for i in range(40)
    ]
    rows.append(
        (
            "Rated Blitz game",
            "late",
            "w0",
            "b",
            1700,
            1700,
            5,
            -5,
            "2025-09-03",
            "00:00:00",
            "300+0",
            "Normal",
            "1-0",
        )
    )
    rows.append(
        (
            "Rated Blitz game",
            "new",
            "w1",
            "b",
            1500,
            1500,
            90,
            -90,
            "2025-09-03",
            "00:00:01",
            "300+0",
            "Normal",
            "1-0",
        )
    )
    f = parquet(tmp_path, rows)
    con = duckdb.connect()
    got = con.execute(games.population_sample_sql([f], 10, "s", estimate.BANDS)).fetchall()
    assert len(got) == 10 and "new" not in {g[0] for g in got}
    assert {g[5] for g in got} <= {b for lo, hi in estimate.BANDS for b in [f"{lo}-{hi}"]} | {
        "other"
    }
    assert all(g[-2].endswith("train-00003-of-00009.parquet") for g in got)
    month = dict(
        (n, (k, last)) for n, k, last in con.execute(games.month_sql([f], ["w0"])).fetchall()
    )
    assert month == {"w0": (15, 1700)}
    counts = con.execute(games.rating_counts_sql([f])).fetchall()
    assert sum(n for _, _, n in counts) == 2 * 41


class FakeMaia:
    """Uniform over legal moves, except that e2e4 is likelier at higher ratings."""

    def probabilities(self, queries):
        out = []
        for q in queries:
            moves = [m.uci() for m in q.board.legal_moves]
            w = {m: 1.0 for m in moves}
            if "e2e4" in w:
                w["e2e4"] = q.rating / 500
            total = sum(w.values())
            out.append({m: v / total for m, v in w.items()})
        return out


def test_job_records_round_trip(tmp_path):
    from chess_ratings import job

    text = (
        "1. e4 { [%clk 0:02:50] } 1... e5 { [%clk 0:02:00] } 2. Nf3 { [%clk 0:02:30] } "
        "2... Nc6 { [%clk 0:02:00] } 1-0"
    )
    game = {
        "plies": games.plies(text, 180),
        "WhiteElo": 1200,
        "BlackElo": 1250,
        "band": "1000-1400",
        "rank": 3,
        "filename": "/data/train-00033-of-00062.parquet",
        "TimeControl": "180+0",
        "White": "w",
        "Black": "b",
    }
    meta, arrays = job.calib_records(FakeMaia(), [game], GRID.astype(int))
    assert [m["sans"] for m in meta] == [["e4", "Nf3"], ["e5", "Nc6"]]
    assert meta[0]["file"] == 33 and arrays[0].shape == (2, 3 + len(GRID))
    for m in meta:
        m.pop("name")
    (tmp_path / "records.json").write_text(json.dumps(meta))
    np.savez_compressed(tmp_path / "records.npz", *arrays)
    recs = job.load_calib(tmp_path, "stratified")
    white = recs[0]
    assert white.fold == 1 and white.rating == 1200 and white.game == recs[1].game
    assert np.allclose(white.clock, [1.0, 170 / 180]) and np.allclose(white.ply, [0, 2])
    # e4 is likelier at higher ratings, so White's first move scores positive
    assert cal.scores(white.logp, 1200)[0] > 0
    assert np.allclose(recs[1].clock, [1.0, 120 / 180])
