"""The Hugging Face job: Maia-3, Stockfish and the Lichess database, where they can be downloaded.

The shell that builds this repository cannot reach Hugging Face, so everything that needs the
model or the games runs here, on a CPU job, and writes JSON summaries to OUT. The job mounts the
September 2025 Parquet files of Lichess/standard-chess-games at DATA and the Maia3-5M model
repository at MODEL. Maia-3 trained on Lichess blitz games from January 2023 to July 2025, so
these games are held out from its training.

    python -m chess_ratings.job pilot|trajectories|whatwould|estimate|common|middlegame \
        --data /data --model /maia --out OUT

Parts and files (their numbers in the month's 62 files, which are in time order):
- pilot: files 30 and 31; design choices are made on these games only.
- trajectories: files 0 to 9, new players' rating records.
- estimate and middlegame: file 40, a stratified sample of 50 games per band.
- common: file 50, the openings of its first 200,000 rated blitz games.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import chess
import chess.engine
import duckdb
import numpy as np

from chess_ratings import blunders, estimate, games, trajectories
from chess_ratings.maia import Maia, Query, history
from chess_ratings.whatwould import distribution, run

SALT = "chess-ratings-2026-11-20"
BANDS = estimate.BANDS
STOCKFISH = "/usr/games/stockfish"


START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:7.0f} s] {message}", flush=True)


def files(data: str, numbers) -> list[str]:
    return [f"{data}/train-{i:05d}-of-00062.parquet" for i in numbers]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect():
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    return con


def write(out: Path, name: str, obj) -> None:
    text = json.dumps(obj, indent=1, sort_keys=True)
    (out / name).write_text(text + "\n")
    print(f"=== {name}\n{text}\n=== end {name}", flush=True)


def load_maia(model: str) -> tuple[Maia, dict]:
    path = Path(model) / "maia3-5m.pt"
    return Maia(str(path), threads=8), {"checkpoint_sha256": sha256(path)}


def sample(data: str, numbers, per_band: int, salt: str = SALT) -> list[dict]:
    """The stratified sample of games, parsed, keeping games of at least 40 plies."""
    con = connect()
    sql = games.blitz_sample_sql(files(data, numbers), 3 * per_band, salt, BANDS)
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    chosen = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    log(f"sample: {len(chosen)} candidate games")
    texts = dict(
        con.execute(
            games.movetext_sql(files(data, numbers), [g["Site"] for g in chosen])
        ).fetchall()
    )
    log("sample: moves fetched")
    kept: dict[str, list[dict]] = {}
    for game in chosen:
        if len(kept.setdefault(game["band"], [])) >= per_band:
            continue
        text = texts.get(game["Site"], "")
        if "clk" not in text:
            continue
        ply_list = games.plies(text, games.base_seconds(game["TimeControl"]))
        if len(ply_list) >= 40:
            game["plies"] = ply_list
            kept[game["band"]].append(game)
    return [g for band in sorted(kept) for g in kept[band]]


def player_queries(game: dict, white: bool, grid, use_history: bool):
    """The likelihood queries of one player's moves: each kept ply at every grid rating."""
    rows = [p for p in game["plies"] if p.white == white and not p.pressed]
    queries = []
    for p in rows:
        boards = p.boards if use_history else p.boards[-1:]
        queries.extend(Query(boards, r, r) for r in grid)
    return rows, queries


def likelihoods(maia: Maia, sample_games: list[dict], grid, use_history: bool) -> list[dict]:
    """Per player-game: true rating, band, and the (moves, ratings) log-probability array."""
    out = []
    for game in sample_games:
        for white in (True, False):
            rows, queries = player_queries(game, white, grid, use_history)
            if not rows:
                continue
            probs = maia.probabilities(queries)
            logp = np.array(
                [
                    [np.log(max(probs[i * len(grid) + j][p.move], 1e-12)) for j in range(len(grid))]
                    for i, p in enumerate(rows)
                ]
            )
            out.append(
                {
                    "rating": game["WhiteElo"] if white else game["BlackElo"],
                    "opponent": game["BlackElo"] if white else game["WhiteElo"],
                    "band": game["band"],
                    "game": game["rank"],
                    "white": white,
                    "logp": logp,
                }
            )
    return out


def move_matching(maia: Maia, sample_games: list[dict], use_history: bool) -> dict:
    """Top-1 agreement at the players' true ratings, on Maia-3's protocol: plies 10 and later,
    before a player first has under 30 seconds."""
    hits, total, by_band = 0, 0, {}
    for game in sample_games:
        rows = [p for p in game["plies"] if p.ply >= 10 and not p.pressed]
        queries = []
        for p in rows:
            own, opp = (
                (game["WhiteElo"], game["BlackElo"])
                if p.white
                else (game["BlackElo"], game["WhiteElo"])
            )
            queries.append(Query(p.boards if use_history else p.boards[-1:], own, opp))
        for p, pr in zip(rows, maia.probabilities(queries), strict=True):
            hit = max(pr, key=lambda m: pr[m]) == p.move
            hits += hit
            total += 1
            b = by_band.setdefault(game["band"], [0, 0])
            b[0] += hit
            b[1] += 1
    return {"hits": hits, "positions": total, "by_band": by_band}


def summarize_estimates(records: list[dict], grid) -> dict:
    """Errors by band and by number of moves, interval coverage, and two baselines."""
    rows = []
    for r in records:
        e = estimate.estimate(r["logp"], grid)
        prefix = {}
        for k in (5, 10, 20, 30, 40):
            if r["logp"].shape[0] >= k:
                prefix[str(k)] = estimate.estimate(r["logp"][:k], grid)["posterior_mean"]
        rows.append(
            {k: r[k] for k in ("rating", "opponent", "band", "game", "white")}
            | e
            | {"prefix": prefix}
        )
    truth = np.array([r["rating"] for r in rows], dtype=float)
    est = np.array([r["estimate"] for r in rows])
    mean_rating = float(truth.mean())
    out: dict = {
        "player_games": len(rows),
        "mae": float(np.abs(est - truth).mean()),
        "mae_interval": estimate.bootstrap_mean(np.abs(est - truth), seed=7),
        "baseline_mean_rating_mae": float(np.abs(truth - mean_rating).mean()),
        "posterior_mean_mae": float(
            np.abs(np.array([r["posterior_mean"] for r in rows]) - truth).mean()
        ),
        "posterior_mean_mae_interval": estimate.bootstrap_mean(
            np.abs(np.array([r["posterior_mean"] for r in rows]) - truth), seed=10
        ),
        "posterior_mean_error": float(
            (np.array([r["posterior_mean"] for r in rows]) - truth).mean()
        ),
        "coverage_likelihood_interval": float(
            np.mean([r["lo"] <= r["rating"] <= r["hi"] for r in rows])
        ),
        "coverage_posterior_interval": float(
            np.mean([r["posterior_lo"] <= r["rating"] <= r["posterior_hi"] for r in rows])
        ),
        "median_posterior_width": float(
            np.median([r["posterior_hi"] - r["posterior_lo"] for r in rows])
        ),
        "median_interval_width": float(np.median([r["hi"] - r["lo"] for r in rows])),
        "median_moves": float(np.median([r["moves"] for r in rows])),
    }
    by_band = {}
    for b in sorted({r["band"] for r in rows}):
        sel = [r for r in rows if r["band"] == b]
        err = np.array([r["estimate"] - r["rating"] for r in sel])
        perr = np.array([r["posterior_mean"] - r["rating"] for r in sel])
        by_band[b] = {
            "n": len(sel),
            "mae": float(np.abs(err).mean()),
            "mae_interval": estimate.bootstrap_mean(np.abs(err), seed=8),
            "mean_error": float(err.mean()),
            "coverage": float(np.mean([r["lo"] <= r["rating"] <= r["hi"] for r in sel])),
            "posterior_mae": float(np.abs(perr).mean()),
            "posterior_mae_interval": estimate.bootstrap_mean(np.abs(perr), seed=9),
            "posterior_mean_error": float(perr.mean()),
            "posterior_coverage": float(
                np.mean([r["posterior_lo"] <= r["rating"] <= r["posterior_hi"] for r in sel])
            ),
        }
    out["by_band"] = by_band
    by_moves = {}
    for k in ("5", "10", "20", "30", "40"):
        sel = [r for r in rows if k in r["prefix"]]
        err = np.array([r["prefix"][k] - r["rating"] for r in sel])
        full = np.array([r["posterior_mean"] - r["rating"] for r in sel])
        by_moves[k] = {
            "n": len(sel),
            "posterior_mae": float(np.abs(err).mean()),
            "posterior_mae_all_moves_same_players": float(np.abs(full).mean()),
        }
    out["by_moves"] = by_moves
    out["rows"] = [{k: v for k, v in r.items() if k != "prefix"} for r in rows]
    return out


def part_pilot(args, out: Path) -> None:
    info: dict = {"python": sys.version, "platform": platform.platform()}
    con = connect()
    for i in (0, 1, 30, 61):
        f = files(args.data, [i])[0]
        q = f"SELECT min(UTCDate), max(UTCDate), count(*) FROM read_parquet('{f}')"
        lo, hi, n = con.execute(q).fetchone()
        info[f"file_{i}"] = [str(lo), str(hi), n]
    events = con.execute(
        f"SELECT Event, count(*) c FROM read_parquet('{files(args.data, [30])[0]}') "
        "GROUP BY Event ORDER BY c DESC LIMIT 8"
    ).fetchall()
    info["events"] = [[e, c] for e, c in events]
    log(f"pilot: files {info}")
    maia, meta = load_maia(args.model)
    log("pilot: model loaded")
    info |= meta
    checks = []
    for moves in ([], ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"], ["d2d4", "g8f6", "c2c4"]):
        board = chess.Board()
        for m in moves:
            board.push_uci(m)
        checks.append(maia.check_against_uci(board, 1500))
    info["uci_check"] = checks
    board = chess.Board()
    queries = [Query(history(board), r, r) for r in range(600, 2601, 100)] * 50
    t = time.time()
    maia.probabilities(queries)
    info["queries_per_second"] = len(queries) / (time.time() - t)
    log(f"pilot: {info['queries_per_second']:.0f} queries per second; uci check {checks}")
    pilot = sample(args.data, [30, 31], per_band=8)
    info["pilot_games"] = len(pilot)
    for use_history in (True, False):
        key = "history" if use_history else "no_history"
        t = time.time()
        info[f"move_matching_{key}"] = move_matching(maia, pilot, use_history)
        recs = likelihoods(maia, pilot, estimate.GRID, use_history)
        s = summarize_estimates(recs, estimate.GRID)
        s.pop("rows")
        info[f"estimate_{key}"] = s
        info[f"seconds_{key}"] = time.time() - t
        log(f"pilot: {key} done")
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH) as engine:
        info["stockfish"] = engine.id.get("name")
        board = chess.Board()
        for m in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"]:
            board.push_uci(m)
        t = time.time()
        evals = blunders.evaluate_moves(engine, board, nodes=100_000)
        info["stockfish_seconds_one_position"] = time.time() - t
        info["scholar_blunders"] = sorted(blunders.blunder_moves(evals))
    write(out, "pilot.json", info)


def part_trajectories(args, out: Path) -> None:
    con = connect()
    t = time.time()
    rows = con.execute(trajectories.player_games_sql(files(args.data, range(10)))).fetchall()
    summary = trajectories.summarize(rows)
    summary["files"] = "Lichess/standard-chess-games, 2025-09, files 0 to 9"
    summary["seconds"] = time.time() - t
    write(out, "trajectories.json", summary)


SCHOLAR = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"]
TWO_KNIGHTS = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"]


def part_whatwould(args, out: Path) -> None:
    maia, meta = load_maia(args.model)
    ratings = (800, 1100, 1500, 1900, 2300)
    texts, dists = [], {}
    for moves in (SCHOLAR, TWO_KNIGHTS):
        cmd = (
            "uv run python -m chess_ratings.whatwould --checkpoint maia3-5m.pt --moves "
            + " ".join(moves)
        )
        texts.append("$ " + cmd + "\n" + run(maia, None, moves, ratings, 3))
        board = chess.Board()
        for m in moves:
            board.push_uci(m)
        probs = distribution(maia, board, range(600, 2601, 100), use_history=True)
        dists[" ".join(moves)] = {str(r): p for r, p in probs.items()}
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    cmd = f'uv run python -m chess_ratings.whatwould --checkpoint maia3-5m.pt "{fen}"'
    texts.append("$ " + cmd + "\n" + run(maia, fen, [], ratings, 3))
    (out / "whatwould.txt").write_text("\n\n".join(texts) + "\n")
    print("\n\n".join(texts), flush=True)
    write(out, "whatwould.json", meta | {"distributions": dists})


def part_estimate(args, out: Path) -> None:
    maia, meta = load_maia(args.model)
    test = sample(args.data, [40], per_band=args.per_band)
    t = time.time()
    matching = move_matching(maia, test, use_history=args.history)
    recs = likelihoods(maia, test, estimate.GRID, use_history=args.history)
    summary = summarize_estimates(recs, estimate.GRID)
    summary |= meta | {
        "move_matching": matching,
        "history": args.history,
        "games": len(test),
        "seconds": time.time() - t,
        "files": "Lichess/standard-chess-games, 2025-09, file 40",
        "salt": SALT,
    }
    np.savez_compressed(out / "logp.npz", *[r["logp"] for r in recs])
    write(out, "estimate.json", summary)


_local = local()


def engine() -> chess.engine.SimpleEngine:
    if not hasattr(_local, "engine"):
        _local.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        _local.engine.configure({"Threads": 1, "Hash": 64})
    return _local.engine


def evaluate_all(boards: list[chess.Board], nodes: int) -> list[dict[str, blunders.Eval]]:
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda b: blunders.evaluate_moves(engine(), b, nodes), boards))


def maia_blunder_chance(maia: Maia, boards, bad: list[set[str]], ratings) -> list[dict[int, float]]:
    """Maia-3's probability, at each rating, of a blunder in each position (current board only)."""
    queries = [Query((b,), r, r) for b in boards for r in ratings]
    probs = maia.probabilities(queries)
    out = []
    for i, moves in enumerate(bad):
        out.append(
            {
                r: sum(probs[i * len(ratings) + j].get(m, 0.0) for m in moves)
                for j, r in enumerate(ratings)
            }
        )
    return out


def part_common(args, out: Path) -> None:
    con = connect()
    f = files(args.data, [50])[0]
    rows = con.execute(
        f"SELECT WhiteElo, BlackElo, movetext FROM read_parquet('{f}') "
        "WHERE Event LIKE 'Rated Blitz%' LIMIT 200000"
    ).fetchall()
    seen: dict[str, list[tuple[str, int]]] = {}
    boards: dict[str, chess.Board] = {}
    for white_elo, black_elo, text in rows:
        board = chess.Board()
        try:
            for i, san in enumerate(games.opening(text, 15)):
                move = board.parse_san(san)
                if 6 <= i <= 14:
                    key = board.epd()
                    boards.setdefault(key, board.copy(stack=False))
                    rating = white_elo if board.turn == chess.WHITE else black_elo
                    seen.setdefault(key, []).append((move.uci(), rating))
                board.push(move)
        except ValueError:
            continue
    top = sorted(seen, key=lambda k: (-len(seen[k]), k))[: args.positions]
    evals = evaluate_all([boards[k] for k in top], nodes=args.nodes)
    bad = [blunders.blunder_moves(e) for e in evals]
    maia, meta = load_maia(args.model)
    ratings = tuple(range(600, 2601, 100))
    predicted = maia_blunder_chance(maia, [boards[k] for k in top], bad, ratings)
    positions = []
    for k, b, p in zip(top, bad, predicted, strict=True):
        occurrences = seen[k]
        observed = {}
        for lo, hi in BANDS:
            sel = [m for m, r in occurrences if lo <= r < hi]
            exp_sum = sum(
                p[min(max(int(round(r / 100) * 100), 600), 2600)]
                for m, r in occurrences
                if lo <= r < hi
            )
            observed[f"{lo}-{hi}"] = {
                "n": len(sel),
                "blunders": sum(m in b for m in sel),
                "maia_expected": exp_sum,
            }
        positions.append(
            {
                "epd": k,
                "count": len(occurrences),
                "blunders": sorted(b),
                "maia": {str(r): v for r, v in p.items()},
                "observed": observed,
            }
        )
    write(
        out,
        "common.json",
        meta
        | {
            "games": len(rows),
            "nodes": args.nodes,
            "positions": positions,
            "files": "Lichess/standard-chess-games, 2025-09, file 50",
        },
    )


def part_middlegame(args, out: Path) -> None:
    test = sample(args.data, [40], per_band=args.per_band)
    chosen = []
    for game in test:
        for white in (True, False):
            rows = [
                p for p in game["plies"] if p.white == white and 20 <= p.ply < 60 and not p.pressed
            ]
            if not rows:
                continue
            h = hashlib.md5(f"{SALT}{game['rank']}{game['band']}{white}".encode()).hexdigest()
            p = rows[int(h, 16) % len(rows)]
            rating = game["WhiteElo"] if white else game["BlackElo"]
            chosen.append((p, rating, game["band"]))
    evals = evaluate_all([p.boards[-1] for p, _, _ in chosen], nodes=args.nodes)
    bad = [blunders.blunder_moves(e) for e in evals]
    maia, meta = load_maia(args.model)
    queries = [Query(p.boards if args.history else p.boards[-1:], r, r) for p, r, _ in chosen]
    probs = maia.probabilities(queries)
    rows_out = []
    for (p, rating, band), b, pr in zip(chosen, bad, probs, strict=True):
        rows_out.append(
            {
                "rating": rating,
                "band": band,
                "played_blunder": p.move in b,
                "maia_chance": sum(pr.get(m, 0.0) for m in b),
                "blunder_moves": len(b),
                "legal_moves": len(pr),
            }
        )
    by_band = {}
    for lo, hi in BANDS:
        key = f"{lo}-{hi}"
        sel = [r for r in rows_out if r["band"] == key]
        by_band[key] = {
            "n": len(sel),
            "observed": sum(r["played_blunder"] for r in sel),
            "maia_expected": sum(r["maia_chance"] for r in sel),
        }
    write(
        out, "middlegame.json", meta | {"nodes": args.nodes, "by_band": by_band, "rows": rows_out}
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part")
    parser.add_argument("--data", default="/data")
    parser.add_argument("--model", default="/maia")
    parser.add_argument("--out", default="/tmp/out")
    parser.add_argument("--per-band", type=int, default=50)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", default=None, help="upload OUT to runs/chess/RUN in the dataset")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {
        "pilot": part_pilot,
        "trajectories": part_trajectories,
        "whatwould": part_whatwould,
        "estimate": part_estimate,
        "common": part_common,
        "middlegame": part_middlegame,
    }
    for name in args.part.split(","):
        log(f"part {name} starts")
        parts[name](args, out)
        log(f"part {name} done")
    if args.run:
        upload(out, args.run)


def upload(out: Path, run: str) -> None:
    """Store OUT, with the commit and the running time, in the private runs dataset."""
    hub = __import__("huggingface_hub")
    manifest = {"commit": os.environ.get("COMMIT"), "seconds": time.time() - START, "run": run}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    hub.HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
        folder_path=str(out),
        path_in_repo=f"runs/chess/{run}",
        repo_id="arclabs561/games-runs",
        repo_type="dataset",
        commit_message=f"chess/{run}",
    )
    log(f"uploaded to runs/chess/{run}")


if __name__ == "__main__":
    main()
