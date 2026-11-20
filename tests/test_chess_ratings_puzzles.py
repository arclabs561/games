"""Plan R1: the puzzle split, positions, features, the crossing rule and the isotonic fit."""

import math

import numpy as np
import pytest

from chess_ratings import puzzles as pz

# Hand-made puzzles. The first move is the opponent's; the solver answers.
SCHOLAR = (
    "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
    "g8f6 h5f7",
)
TWO_MOVES = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "g1f3 b8c6 f1c4 g8f6",
)


def test_scholars_mate_is_a_one_move_mate():
    board, solver = pz.line(*SCHOLAR)
    assert [m.uci() for m in solver] == ["h5f7"]
    f = pz.board_features(*SCHOLAR)
    assert f["solver_moves"] == 1 and f["ends_in_mate"] == 1.0
    assert f["m1_capture"] == 1.0 and f["m1_check"] == 1.0 and math.isnan(f["m2_check"])
    assert f["white_to_move"] == 1.0 and f["legal_moves"] == board.legal_moves.count()


def test_two_move_puzzle_positions_and_history():
    pos = pz.solver_positions(*TWO_MOVES)
    assert [m for _, m in pos] == ["b8c6", "g8f6"]
    hist, _ = pos[1]
    # The FEN, then g1f3, b8c6 and f1c4 played: four boards, the current one last.
    assert len(hist) == 4 and hist[-1].fen().startswith("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2")
    assert pz.board_features(*TWO_MOVES)["white_to_move"] == 0.0
    assert pz.board_features(*TWO_MOVES)["solver_moves"] == 2


def brute_crossing(log_p: np.ndarray, grid=pz.GRID) -> float:
    """The crossing on a fine grid of the piecewise-linear logit, for checking r_half."""
    p = np.exp(log_p)
    logit = np.log(p) - np.log1p(-p)
    xs = np.linspace(grid[0], grid[-1], 26_001)
    ys = np.interp(xs, grid, logit)
    if ys[0] > 0:
        return float(grid[0])
    hit = np.flatnonzero(ys >= 0)
    return float(xs[hit[0]]) if len(hit) else float(grid[-1])


def test_r_half_matches_a_fine_grid():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = np.sort(rng.uniform(0.01, 0.99, len(pz.GRID)))
        assert pz.r_half(np.log(p)) == pytest.approx(brute_crossing(np.log(p)), abs=0.2)


def test_r_half_at_the_ends():
    assert pz.r_half(np.log(np.full(len(pz.GRID), 0.9))) == 400.0
    assert pz.r_half(np.log(np.full(len(pz.GRID), 0.1))) == 3000.0  # never crosses
    assert pz.r_half(np.zeros(len(pz.GRID))) == 400.0  # P = 1 everywhere


def test_isotonic_against_brute_force():
    """On small data, the fit is the nondecreasing step function with least squared error."""
    rng = np.random.default_rng(1)
    x = np.arange(8, dtype=float)
    y = rng.normal(0, 1, 8)
    fit = pz.isotonic(x, y)(x)
    assert np.all(np.diff(fit) >= -1e-12)
    # A brute-force check: no small nondecreasing perturbation lowers the error.
    err = np.sum((fit - y) ** 2)
    for _ in range(2000):
        z = np.sort(fit + rng.normal(0, 0.05, 8))
        assert np.sum((z - y) ** 2) >= err - 1e-9


def test_split_is_by_hash_and_rule():
    rows = [
        {"PuzzleId": f"p{i}", "NbPlays": 600 if i % 3 else 100, "RatingDeviation": 80}
        for i in range(300)
    ]
    old = dict(pz.SIZES)
    try:
        pz.SIZES.update(train=50, val=10, test=20)
        sets = pz.assign(rows)
    finally:
        pz.SIZES.update(old)
    assert [len(sets[k]) for k in ("train", "val", "test")] == [50, 10, 20]
    keys = [pz.order_key(r["PuzzleId"]) for r in sets["train"]]
    assert keys == sorted(keys)
    assert all(r["NbPlays"] == 100 for r in sets["few"])
    assert not {r["PuzzleId"] for r in sets["train"]} & {r["PuzzleId"] for r in sets["test"]}


def test_paired_mse_and_metrics():
    y = np.array([1000.0, 1500, 2000, 2500])
    a, b = y + 10, y + 100
    d = pz.paired_mse(a, b, y, resamples=200)
    assert d["mean"] == 100 - 10000 and d["ci"][1] < 0
    assert pz.metrics(a, y)["rmse"] == 10 and pz.metrics(a, y)["spearman"] == pytest.approx(1)
    assert list(pz.ranks(np.array([3.0, 1, 3, 2]))) == [2.5, 0, 2.5, 1]
