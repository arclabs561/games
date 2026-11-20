"""Plan R1's jobs: the split of the Lichess puzzles and Maia-3's curves, on a GitHub Actions
runner (or a T4), for `chess_ratings.puzzles`.

Each job downloads the dataset's Parquet files and recomputes the split, so shards do not
depend on one another. `curves` installs torch (CPU) and the maia3 package at the commit
`maia.py` pins into the running environment, and downloads Maia3-5M, before it starts.

    uv run --group chess_ratings python -m chess_ratings.puzzles split --out OUT
    uv run --group chess_ratings python -m chess_ratings.puzzles curves --shard 3/20 --out OUT
    uv run --group chess_ratings python -m chess_ratings.puzzles curves --sets val --limit 500 \\
        --out OUT                                                        # the pilot

Outputs: `split.json` (counts and the dataset revision), and per shard `curves-<i>.npz`
(puzzle ids, solver-move counts and log p_k(R) on the grid, float32), `rows-<i>.jsonl.gz`
(the puzzles' id, FEN, moves, rating, deviation and plays; CC0) and `curves-<i>.json` (speed,
device, checkpoint digest, and the check against the authors' UCI engine).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from chess_ratings import puzzles as pz

MAIA_REPO, MAIA_FILE = "UofTCSSLab/Maia3-5M", "maia3-5m.pt"
SET_ORDER = ("train", "val", "test", "few")


def split_rows() -> tuple[dict[str, list[dict]], dict]:
    rows = pz.read_dataset()
    sets = pz.assign(rows)
    meta = {
        "puzzles": len(rows),
        "qualifying": sum(pz.qualifies(r["NbPlays"], r["RatingDeviation"]) for r in rows),
        "few_plays_candidates": sum(pz.few_plays(r["NbPlays"]) for r in rows),
        "sizes": {k: len(v) for k, v in sets.items()},
        "revision": pz.dataset_revision(),
        "salt": pz.SALT,
    }
    return sets, meta


def run_split(args) -> None:
    sets, meta = split_rows()
    args.out.mkdir(parents=True, exist_ok=True)
    first = {k: [r["PuzzleId"] for r in v[:3]] for k, v in sets.items()}
    meta["first_ids"] = first
    solver = [len(pz.line(r["FEN"], r["Moves"])[1]) for r in sets["val"][:2000]]
    meta["val_mean_solver_moves_first_2000"] = float(np.mean(solver))
    (args.out / "split.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def install_maia() -> None:
    from chess_ratings.maia import MAIA3_COMMIT

    uv = "uv"
    base = [uv, "pip", "install", "-q", "--python", sys.executable]
    subprocess.run(
        [*base, "--index-url", "https://download.pytorch.org/whl/cpu", "torch"], check=True
    )
    subprocess.run(
        [*base, f"maia3 @ git+https://github.com/CSSLab/maia3@{MAIA3_COMMIT}"], check=True
    )


def load_maia():
    import importlib

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MAIA_REPO, MAIA_FILE)
    importlib.invalidate_caches()
    from chess_ratings.maia import Maia

    device = os.environ.get("MAIA_DEVICE", "cpu")
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Maia(path, threads=os.cpu_count(), device=device), {
        "checkpoint_sha256": sha,
        "device": device,
    }


def uci_check(maia, rows: list[dict]) -> list[dict]:
    """The authors' UCI engine against this module's probabilities, on the first solver
    position of three puzzles (no history, as the engine reads a single position)."""
    import chess

    out = []
    for r in rows[:3]:
        board, _ = pz.line(r["FEN"], r["Moves"])
        top = maia.check_against_uci(chess.Board(board.fen()), 1500)
        out.append({"puzzle": r["PuzzleId"], "moves": [[m, round(a, 6), round(b, 6)] for m, a, b in top],
                    "max_abs_diff": max(abs(a - b) for _, a, b in top)})  # fmt: skip
    return out


def run_curves(args) -> None:
    sets, meta = split_rows()
    wanted = args.sets.split(",")
    rows = []
    for k in SET_ORDER:
        if k in wanted:
            part = sets[k][args.offset :]
            rows += [{**r, "set": k} for r in (part[: args.limit] if args.limit else part)]
    i, n = map(int, args.shard.split("/")) if "/" in args.shard else (int(args.shard), args.shards)
    mine = rows[i - 1 :: n]
    install_maia()
    maia, info = load_maia()
    logps, speed = pz.curves(mine, maia)
    args.out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(x) for x in logps])
    flat = np.concatenate([x for x in logps if len(x)]).astype(np.float32)
    np.savez_compressed(
        args.out / f"curves-{i}.npz",
        ids=np.array([r["PuzzleId"] for r in mine]),
        lengths=lengths,
        logp=flat,
        grid=np.array(pz.GRID),
    )
    with gzip.open(args.out / f"rows-{i}.jsonl.gz", "wt") as f:
        for r in mine:
            f.write(json.dumps(r) + "\n")
    record = {
        "plan": pz.PLAN,
        "shard": args.shard,
        "sets": wanted,
        "limit": args.limit,
        "puzzles": len(mine),
        "split": meta,
        **info,
        **speed,
        "solver_moves_mean": float(lengths.mean()) if len(lengths) else None,
        "cpus": os.cpu_count(),
        "platform": platform.platform(),
        "uci_check": uci_check(maia, mine) if i == 1 else None,
    }
    (args.out / f"curves-{i}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "split"}, indent=1))
