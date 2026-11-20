"""Chess: parsing, sampling SQL, trajectories, the rating estimate, blunders and the CLI table."""

import math
from pathlib import Path

import chess
import duckdb
import numpy as np
import pytest

from chess_ratings import blunders, estimate, games, trajectories
from chess_ratings.blunders import Eval, is_blunder, winning_chances
from chess_ratings.glicko import NEW, Glicko, lichess_game
from chess_ratings.whatwould import table

SCHOLAR = (
    "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Qh5 { [%clk 0:02:58] } "
    "2... Nc6 { [%clk 0:00:25] } 3. Bc4 { [%clk 0:02:50] } 3... Nf6?? { [%clk 0:00:20] } "
    "4. Qxf7# { [%clk 0:02:49] } 1-0"
)


def test_plies_track_history_and_time_pressure():
    ps = games.plies(SCHOLAR, 180)
    assert [p.move for p in ps] == ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    assert [len(p.boards) for p in ps] == [1, 2, 3, 4, 5, 6, 7]
    # Black's clock read 25 s after 2... Nc6, so Black is in time pressure from 3... Nf6 on
    assert [p.pressed for p in ps] == [False] * 5 + [True, False]
    assert ps[5].boards[-1].fen().startswith("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3")


def test_opening_words():
    assert games.opening(SCHOLAR, 4) == ["e4", "e5", "Qh5", "Nc6"]
    assert games.opening(SCHOLAR, 99)[-1] == "Qxf7#"
    assert games.base_seconds("180+2") == 180


def synthetic_parquet(path: Path, rows: list[dict]) -> str:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t (Event VARCHAR, Site VARCHAR, White VARCHAR, Black VARCHAR, Result VARCHAR,"
        " WhiteElo SMALLINT, BlackElo SMALLINT, WhiteRatingDiff SMALLINT, BlackRatingDiff SMALLINT,"
        " UTCDate DATE, UTCTime TIME, TimeControl VARCHAR, Termination VARCHAR, movetext VARCHAR)"
    )
    for r in rows:
        con.execute(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", list(r.values())
        )
    f = str(path / "games.parquet")
    con.execute(f"COPY t TO '{f}' (FORMAT parquet)")
    return f


def row(i, white, black, we, be, wd, bd, result, event="Rated Blitz game"):
    return {
        "Event": event,
        "Site": f"site{i}",
        "White": white,
        "Black": black,
        "Result": result,
        "WhiteElo": we,
        "BlackElo": be,
        "WhiteRatingDiff": wd,
        "BlackRatingDiff": bd,
        "UTCDate": "2025-09-01",
        "UTCTime": f"00:{i // 60:02d}:{i % 60:02d}",
        "TimeControl": "180+0",
        "Termination": "Normal",
        "movetext": SCHOLAR,
    }


def test_sample_sql_stratifies_and_filters(tmp_path):
    rows = [row(i, "a", "b", 700 + 20 * i, 700 + 20 * i, 5, -5, "1-0") for i in range(80)]
    rows.append(row(99, "a", "b", 1500, 1500, 40, -40, "1-0"))  # a provisional player
    rows.append(row(98, "a", "b", 1500, 1500, 5, -5, "1-0", event="Rated Rapid game"))
    f = synthetic_parquet(tmp_path, rows)
    got = duckdb.connect().execute(games.blitz_sample_sql([f], 3, "s", estimate.BANDS)).fetchall()
    bands = [g[5] for g in got]
    assert sorted(set(bands)) == ["1000-1400", "1400-1800", "1800-2200", "2200-2700", "600-1000"]
    assert all(bands.count(b) == 3 for b in set(bands))
    assert "site99" not in {g[0] for g in got} and "site98" not in {g[0] for g in got}


def test_trajectories_follow_new_players(tmp_path):
    rows, me, i = [], NEW, 0
    for k in range(70):  # a new player beats everyone for 8 games, then wins every other game
        opp = Glicko(me.rating, 60.0, 0.06)
        s = 1.0 if k < 8 or k % 2 == 0 else 0.0
        after, _ = lichess_game(me, opp, s)
        rows.append(
            row(
                i,
                "new",
                f"o{k}",
                int(me.rating),
                int(opp.rating),
                int(after.rating) - int(me.rating),
                0,
                "1-0" if s else "0-1",
            )
        )
        me, i = after, i + 1
    rows.append(row(i, "old", "x", 1500, 1500, 6, -6, "1-0"))  # not a new player
    f = synthetic_parquet(tmp_path, rows)
    got = duckdb.connect().execute(trajectories.player_games_sql([f])).fetchall()
    assert {r[0] for r in got} == {"new"}
    s = trajectories.summarize(got)
    assert s["players"] == 1 and s["players_with_more_than_60_games"] == 1
    assert s["games_in_record"][50] == 70
    first = s["first_game_below_110"]["quartiles"][1]
    assert first == 10  # the model's 10 games at opponent RD 60
    implied = [d["median"] for d in s["implied_deviation"][:12]]
    path = [Glicko(1500, 500).deviation]
    me = NEW
    for k in range(12):
        opp = Glicko(me.rating, 60.0, 0.06)
        me, _ = lichess_game(me, opp, 1.0 if k < 8 or k % 2 == 0 else 0.0)
        path.append(me.deviation)
    # integer ratings in the record put the implied deviation within a few points
    assert all(abs(a - b) < 4 for a, b in zip(implied, path[1:], strict=True))


def test_estimate_on_a_known_likelihood():
    grid = estimate.GRID
    xs = np.array(grid, dtype=float)
    # 20 moves whose log-likelihoods sum to -(x - 1430)^2 / (2 * 150^2), a peak at 1430
    per_move = -((xs - 1430) ** 2) / (2 * 150**2) / 20
    e = estimate.estimate(np.tile(per_move, (20, 1)))
    assert abs(e["estimate"] - 1430) < 1e-9  # a parabola is refined exactly
    half = 150 * math.sqrt(2 * estimate.DROP)
    assert abs(e["lo"] - (1430 - half)) < 15 and abs(e["hi"] - (1430 + half)) < 15
    assert abs(e["posterior_mean"] - 1430) < 1
    flat = estimate.estimate(np.zeros((5, len(grid))))
    assert (flat["lo"], flat["hi"]) == (600.0, 2600.0)


def test_bands():
    assert estimate.band(999) == "600-1000" and estimate.band(2650) == "2200-2700"
    lo, hi = estimate.bootstrap_mean(np.arange(100.0), seed=1)
    assert lo < 49.5 < hi


def test_winning_chances_and_blunder_rule():
    assert winning_chances(0) == 0
    assert winning_chances(10_000) == pytest.approx(1) and winning_chances(
        -10_000
    ) == pytest.approx(-1)
    assert winning_chances(100) == pytest.approx(2 / (1 + math.exp(-0.368208)) - 1)
    assert is_blunder(Eval(cp=50), Eval(cp=-300))  # 0.09 down to -0.50
    assert not is_blunder(Eval(cp=50), Eval(cp=-100))
    assert is_blunder(Eval(cp=-600), Eval(mate=-3))  # a mate is now unavoidable
    assert not is_blunder(Eval(cp=-800), Eval(mate=-3))
    assert is_blunder(Eval(mate=2), Eval(cp=300)) and not is_blunder(Eval(mate=2), Eval(cp=900))
    assert is_blunder(Eval(mate=2), Eval(mate=-1)) and not is_blunder(Eval(mate=2), Eval(mate=3))
    evals = {"a": Eval(mate=1), "b": Eval(cp=600), "c": Eval(cp=-50), "d": Eval(cp=900)}
    assert blunders.blunder_moves(evals) == {"b", "c"}  # letting mate go at +900 is a mistake


def test_cli_table():
    board = chess.Board()
    probs = {
        800: {"e2e4": 0.5, "d2d4": 0.2, "g1f3": 0.1, "a2a3": 0.2},
        1500: {"e2e4": 0.4, "d2d4": 0.4, "g1f3": 0.15, "a2a3": 0.05},
    }
    text = table(board, probs, top=2)
    lines = text.splitlines()
    assert lines[0].split() == ["move", "800", "1500"]
    assert [ln.split()[0] for ln in lines[1:]] == ["d4", "e4", "a3", "other"]
    assert lines[-1].split()[1:] == ["10.0%", "15.0%"]
