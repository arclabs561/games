"""Plan R1 (experiments/plans/R1.md): does Maia-3 help predict a Lichess puzzle's rating?

A puzzle is a position (FEN) and a line of moves (Moves): the first move is the
opponent's, which sets the puzzle, and the solver's moves are the 2nd, 4th, and so on.
Its rating is the Glicko-2 rating at which solvers succeed about half the time.

**Data.** The Hugging Face dataset `Lichess/chess-puzzles` (tagged cc0-1.0). A puzzle
qualifies when NbPlays >= 500 and RatingDeviation <= 100. Qualifying puzzles are ordered by
md5(SALT || PuzzleId): the first 100,000 train, the next 10,000 validate, the next 20,000
test. The *few plays* set is the first 5,000 puzzles in the same order with NbPlays from 50
to 200 and no condition on RD. Models see the FEN and Moves only.

**Predictors.**
- The Maia-3 curve: for each rating R on 400, 500, ..., 3000 and each solver move k (at
  most the first 8), p_k(R) is Maia3-5M's probability of the solution move with both
  players given R and the puzzle's own moves as history. P(R) = prod_k p_k(R).
- Z: R_half, where P crosses 1/2, by linear interpolation of the logit of P between grid
  points; 400 if P(400) > 1/2 and 3000 if P(3000) < 1/2.
- Z+: isotonic regression of the rating on R_half, fitted on the training puzzles.
- B: LightGBM on board and solution features (`board_features`).
- B+M: B's features plus log P at the 27 grid points, log p_k(1500) for the first three
  solver moves, and R_half.

    python -m chess_ratings.puzzles split --out OUT             # the split, on a runner
    python -m chess_ratings.puzzles curves --shard i/n --split-dir DIR --out OUT
    uv run python -m chess_ratings.puzzles fit --runs DIR       # here
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = RESULTS / "puzzles.json"
PLAN = "experiments/plans/R1.md"
DATASET = "Lichess/chess-puzzles"
SALT = "chess-puzzles-r1-2026-11-20"
GRID = tuple(range(400, 3001, 100))
MAX_SOLVER_MOVES = 8
SIZES = {"train": 100_000, "val": 10_000, "test": 20_000}
FEW = 5_000
MIN_PLAYS, MAX_RD = 500, 100
FEW_PLAYS = (50, 200)
PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


# The split.


def order_key(puzzle_id: str) -> str:
    return hashlib.md5((SALT + puzzle_id).encode()).hexdigest()


def qualifies(nb_plays: int, rd: int) -> bool:
    return nb_plays >= MIN_PLAYS and rd <= MAX_RD


def few_plays(nb_plays: int) -> bool:
    return FEW_PLAYS[0] <= nb_plays <= FEW_PLAYS[1]


def assign(rows: list[dict]) -> dict[str, list[dict]]:
    """The four sets, each in hash order. rows: dicts with PuzzleId, NbPlays, RD."""
    ordered = sorted(rows, key=lambda r: order_key(r["PuzzleId"]))
    good = [r for r in ordered if qualifies(r["NbPlays"], r["RatingDeviation"])]
    out, start = {}, 0
    for name, size in SIZES.items():
        out[name] = good[start : start + size]
        start += size
    out["few"] = [r for r in ordered if few_plays(r["NbPlays"])][:FEW]
    return out


# Positions and features.


def line(fen: str, moves: str) -> tuple[chess.Board, list[chess.Move]]:
    """The board after the opponent's first move, and the solver's moves in order."""
    board = chess.Board(fen)
    ms = [chess.Move.from_uci(m) for m in moves.split()]
    board.push(ms[0])
    return board, ms[1::2]


def solver_positions(fen: str, moves: str) -> list[tuple[tuple[chess.Board, ...], str]]:
    """For each of the first MAX_SOLVER_MOVES solver moves: the boards from the FEN up to the
    position before it (the history Maia-3 sees, at most eight), and the move in UCI."""
    board = chess.Board(fen)
    ms = [chess.Move.from_uci(m) for m in moves.split()]
    boards = [board.copy(stack=False)]
    out = []
    for i, m in enumerate(ms):
        if i % 2 == 1:
            if len(out) == MAX_SOLVER_MOVES:
                break
            out.append((tuple(boards[-8:]), m.uci()))
        board.push(m)
        boards.append(board.copy(stack=False))
    return out


def board_features(fen: str, moves: str) -> dict[str, float]:
    """Features of the board and the solution, with no model."""
    board, solver = line(fen, moves)
    f: dict[str, float] = {"solver_moves": len(solver)}
    for color, name in ((chess.WHITE, "own"), (chess.BLACK, "opp")):
        side = board.turn if name == "own" else not board.turn
        for p in PIECES:
            f[f"{name}_{chess.piece_name(p)}"] = len(board.pieces(p, side))
        del color
    own = sum(VALUES[p] * f[f"own_{chess.piece_name(p)}"] for p in PIECES)
    opp = sum(VALUES[p] * f[f"opp_{chess.piece_name(p)}"] for p in PIECES)
    f["material_balance"] = own - opp
    f["white_to_move"] = float(board.turn == chess.WHITE)
    f["legal_moves"] = board.legal_moves.count()
    b = board.copy()
    all_moves = [chess.Move.from_uci(m) for m in moves.split()][1:]
    for i in range(3):
        k = 2 * i
        if k < len(all_moves):
            m = all_moves[k]
            f[f"m{i + 1}_capture"] = float(b.is_capture(m))
            f[f"m{i + 1}_check"] = float(b.gives_check(m))
            f[f"m{i + 1}_promotion"] = float(m.promotion is not None)
            f[f"m{i + 1}_castling"] = float(b.is_castling(m))
            b.push(m)
            if k + 1 < len(all_moves):
                b.push(all_moves[k + 1])
        else:
            for x in ("capture", "check", "promotion", "castling"):
                f[f"m{i + 1}_{x}"] = math.nan
    end = chess.Board(fen)
    for m in moves.split():
        end.push_uci(m)
    f["ends_in_mate"] = float(end.is_checkmate())
    return f


# The curve.


def log_solve(logp: np.ndarray) -> np.ndarray:
    """log P(R) on the grid from log p_k(R), shape (moves, grid)."""
    return logp.sum(axis=0)


def r_half(log_p: np.ndarray, grid=GRID) -> float:
    """Where P crosses 1/2, reading the grid upward: linear interpolation of logit(P)
    between the last grid point below 1/2 and the first at or above it; the grid's first
    point if P starts above 1/2, its last if P never reaches 1/2."""
    lp = np.minimum(np.asarray(log_p, dtype=float), -1e-12)
    logit = lp - np.log(-np.expm1(lp))
    if logit[0] > 0:
        return float(grid[0])
    for i in range(1, len(grid)):
        if logit[i] >= 0:
            t = -logit[i - 1] / (logit[i] - logit[i - 1])
            return float(grid[i - 1] + t * (grid[i] - grid[i - 1]))
    return float(grid[-1])


def model_features(logp: np.ndarray) -> dict[str, float]:
    """B+M's extra features from one puzzle's (moves, grid) array of log p_k(R)."""
    lp = log_solve(logp)
    f = {f"logP_{r}": float(v) for r, v in zip(GRID, lp, strict=True)}
    j = GRID.index(1500)
    for k in range(3):
        f[f"logp{k + 1}_1500"] = float(logp[k, j]) if k < len(logp) else math.nan
    f["r_half"] = r_half(lp)
    return f


def isotonic(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators: the nondecreasing fit of y on x with least squared error.
    Returns a function that interpolates linearly between the fitted values at the
    distinct training x (constant past the ends), as scikit-learn's IsotonicRegression."""
    xs_all = np.asarray(x, dtype=float)
    ys_all = np.asarray(y, dtype=float)
    ux, inv = np.unique(xs_all, return_inverse=True)
    w0 = np.bincount(inv).astype(float)
    y0 = np.bincount(inv, weights=ys_all) / w0
    level, weight, size = [], [], []
    for yi, wi in zip(y0, w0, strict=True):
        level.append(yi)
        weight.append(wi)
        size.append(1)
        while len(level) > 1 and level[-2] > level[-1]:
            w = weight[-2] + weight[-1]
            level[-2] = (level[-2] * weight[-2] + level[-1] * weight[-1]) / w
            weight[-2] = w
            size[-2] += size[-1]
            level.pop(), weight.pop(), size.pop()
    fitted = np.repeat(level, size)

    def predict(z: np.ndarray) -> np.ndarray:
        return np.interp(z, ux, fitted)

    return predict


# The curves job (on a runner or a T4).


def read_dataset(path: str | None = None) -> list[dict]:
    """Every puzzle's id, FEN, moves, rating, RD and plays, from the dataset's Parquet."""
    import duckdb

    if path is None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.parquet"])
    q = (
        "SELECT PuzzleId, FEN, Moves, Rating, RatingDeviation, NbPlays "
        f"FROM read_parquet('{path}/data/*.parquet')"
    )
    cols = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "NbPlays"]
    rel = duckdb.sql(q)
    assert rel is not None
    return [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]


def dataset_revision() -> str | None:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(DATASET).sha


def curves(rows: list[dict], maia, batch: int = 1024) -> tuple[list[np.ndarray], dict]:
    """log p_k(R) for every puzzle, shape (solver moves, grid) each."""
    from chess_ratings.maia import Query

    out, t0, queries_n = [], time.time(), 0
    for r in rows:
        pos = solver_positions(r["FEN"], r["Moves"])
        qs = [Query(boards, g, g) for boards, _ in pos for g in GRID]
        probs = maia.probabilities(qs, batch=batch)
        queries_n += len(qs)
        lp = np.array(
            [[math.log(max(probs[k * len(GRID) + j].get(mv, 0.0), 1e-300)) for j in range(len(GRID))]
             for k, (_, mv) in enumerate(pos)]
        )  # fmt: skip
        out.append(lp)
    secs = time.time() - t0
    return out, {
        "queries": queries_n,
        "seconds": secs,
        "queries_per_second": queries_n / secs if secs else None,
    }


# Fitting and scoring (here).


def feature_rows(rows: list[dict], logps: list[np.ndarray] | None) -> tuple[list[str], np.ndarray]:
    feats = []
    for i, r in enumerate(rows):
        f = board_features(r["FEN"], r["Moves"])
        if logps is not None:
            f |= model_features(logps[i])
        feats.append(f)
    names = sorted(feats[0])
    return names, np.array([[f.get(n, math.nan) for n in names] for f in feats], dtype=float)


def fit_lgbm(xtr, ytr, xva, yva) -> tuple[object, dict]:
    import lightgbm as lgb

    best = None
    for leaves in (31, 127):
        m = lgb.LGBMRegressor(
            learning_rate=0.05, num_leaves=leaves, n_estimators=5000, random_state=0,
            deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1,
        )  # fmt: skip
        m.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        mse = float(np.mean((m.predict(xva) - yva) ** 2))
        if best is None or mse < best[1]["val_mse"]:
            best = (m, {"num_leaves": leaves, "val_mse": mse, "trees": int(m.best_iteration_)})
    assert best is not None
    return best


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties given their average rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    return {
        "mse": float(np.mean(e**2)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "spearman": spearman(pred, y),
        "n": len(y),
    }


def paired_mse(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, resamples: int = 10_000, seed: int = 0
) -> dict:
    """Mean of (a - y)^2 - (b - y)^2, with a bootstrap over puzzles."""
    d = (a - y) ** 2 - (b - y) ** 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (resamples, len(d)))
    reps = d[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    sd = float(np.std(d, ddof=1))
    return {
        "mean": float(d.mean()),
        "ci": [float(lo), float(hi)],
        "relative_to_b_mse": float(d.mean() / np.mean((b - y) ** 2)),
        "sd": sd,
        "detectable_effect": 2.80 * sd / math.sqrt(len(d)),
        "n": len(d),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["split", "curves", "fit"])
    parser.add_argument("--shard", default="1/1", help="i/n, or i with --shards n")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--sets", default="train,val,test,few")
    parser.add_argument("--limit", type=int, help="curves: first N puzzles of each set (pilot)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "curves":
        from chess_ratings import puzzle_job

        puzzle_job.run_curves(args)
        return
    if args.what == "split":
        from chess_ratings import puzzle_job

        puzzle_job.run_split(args)
        return
    from chess_ratings import puzzle_fit

    puzzle_fit.run(args.runs, args.out or OUT)


if __name__ == "__main__":
    main()
