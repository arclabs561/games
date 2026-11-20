"""Real rating trajectories of new Lichess blitz players, summarized as aggregates.

A **new player** is one whose first rated blitz game in the files is played at rating exactly
1500 and moves that rating by at least 100 points, which only a deviation near 500 does. A
player's games are put in time order and kept up to the first game whose rating before the game
does not equal the previous rating plus its change (a gap in the record).

From each game's rating change, the opponent's rating and the result, the deviation after the
game can be backed out of Lichess's update, assuming the opponent's deviation is 60:
change = 173.7178 phi'^2 g (score - E) on the rating scale, so phi' follows. Draws, and changes
of fewer than 3 points, are skipped, since rounding dominates them.

Only aggregates leave this module: counts, medians and quantiles by game number.
"""

from math import exp, sqrt

import numpy as np

from chess_ratings.glicko import PROVISIONAL, REGULATOR, SCALE, WHITE_ADVANTAGE, g

ASSUMED_OPPONENT_RD = 60.0
GAMES = 60


def player_games_sql(files: list[str]) -> str:
    """One row per player and rated blitz game, in time order within each player."""
    listed = ", ".join(f"'{f}'" for f in files)
    return f"""
        WITH g AS (
            SELECT White, Black, WhiteElo, BlackElo, WhiteRatingDiff, BlackRatingDiff, Result,
                   UTCDate, UTCTime, Site
            FROM read_parquet([{listed}])
            WHERE Event LIKE 'Rated Blitz%' AND WhiteRatingDiff IS NOT NULL
        ), p AS (
            SELECT White AS player, WhiteElo AS elo, WhiteRatingDiff AS diff, BlackElo AS opp,
                   CASE Result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 ELSE 0.5 END AS score,
                   1 AS white, UTCDate AS d, UTCTime AS t, Site FROM g
            UNION ALL
            SELECT Black, BlackElo, BlackRatingDiff, WhiteElo,
                   CASE Result WHEN '1-0' THEN 0.0 WHEN '0-1' THEN 1.0 ELSE 0.5 END,
                   0, UTCDate, UTCTime, Site FROM g
        ), firsts AS (
            SELECT player FROM (
                SELECT player, elo, diff,
                       row_number() OVER (PARTITION BY player ORDER BY d, t, Site) AS k
                FROM p
            ) WHERE k = 1 AND elo = 1500 AND abs(diff) >= 100
        )
        SELECT p.player, p.elo, p.diff, p.opp, p.score, p.white,
               row_number() OVER (PARTITION BY p.player ORDER BY p.d, p.t, p.Site) AS k
        FROM p JOIN firsts USING (player)
        ORDER BY p.player, k
    """


def consistent_prefix(elo: np.ndarray, diff: np.ndarray) -> int:
    """How many games, from the first, have each rating equal to the last plus its change."""
    n = 1
    while n < len(elo) and elo[n] == elo[n - 1] + diff[n - 1]:
        n += 1
    return n


def implied_deviation(
    diff: float, elo: float, opp: float, score: float, white: bool
) -> float | None:
    """The deviation after a game, backed out of its rating change (None when not usable)."""
    if score == 0.5 or abs(diff) < 3:
        return None
    gj = g(ASSUMED_OPPONENT_RD / SCALE)
    adv = WHITE_ADVANTAGE if white else -WHITE_ADVANTAGE
    e = 1 / (1 + exp(-gj * (elo + adv - opp) / SCALE))
    change = diff / REGULATOR["blitz"] if diff > 0 else diff
    phi2 = change / (SCALE * gj * (score - e))
    return SCALE * sqrt(phi2) if phi2 > 0 else None


def summarize(rows: list[tuple], games: int = GAMES) -> dict:
    """Aggregates over new players from rows (player, elo, diff, opp, score, white, k)."""
    by_player: dict[str, list[tuple]] = {}
    for row in rows:
        by_player.setdefault(row[0], []).append(row)
    lengths, devs, first_established, finals, ratings = [], [], [], [], []
    abs_diff: list[list[float]] = [[] for _ in range(games)]
    for record in by_player.values():
        elo = np.array([r[1] for r in record], dtype=float)
        diff = np.array([r[2] for r in record], dtype=float)
        n = consistent_prefix(elo, diff)
        lengths.append(n)
        path = []
        for k in range(min(n, games)):
            _, e, d, o, s, w, _ = record[k]
            path.append(implied_deviation(d, e, o, s, bool(w)))
            if s != 0.5:
                abs_diff[k].append(abs(d))
        devs.append(path)
        below = [k + 1 for k, v in enumerate(path) if v is not None and v < PROVISIONAL]
        if below and n >= 30:
            first_established.append(below[0])
        if n > games:
            ratings.append(elo[: games + 1])
            finals.append(elo[games])
    deviation_by_game = []
    for k in range(games):
        vals = [p[k] for p in devs if len(p) > k and p[k] is not None]
        deviation_by_game.append(
            {"game": k + 1, "n": len(vals), "median": float(np.median(vals)) if vals else None}
        )
    out: dict = {
        "players": len(by_player),
        "games_in_record": {q: float(np.quantile(lengths, q / 100)) for q in (25, 50, 75, 90)},
        "players_with_30_games": int(sum(n >= 30 for n in lengths)),
        "players_with_more_than_60_games": len(finals),
        "median_abs_change_decisive": [
            {"game": k + 1, "n": len(v), "median": float(np.median(v)) if v else None}
            for k, v in enumerate(abs_diff)
        ],
        "implied_deviation": deviation_by_game,
        "first_game_below_110": {
            "players": len(first_established),
            "quartiles": [float(np.quantile(first_established, q)) for q in (0.25, 0.5, 0.75)]
            if first_established
            else None,
        },
    }
    if ratings:
        r = np.array(ratings)
        gap = np.abs(r - r[:, -1:])
        out["rating_after_60"] = {
            "histogram_50": {
                str(int(b)): int(c)
                for b, c in zip(*np.unique((r[:, -1] // 50) * 50, return_counts=True), strict=True)
            },
        }
        out["within_100_of_game_61"] = [
            {"game": k + 1, "share": float((gap[:, k] <= 100).mean())} for k in range(games)
        ]
    return out
