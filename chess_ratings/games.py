"""Lichess games as positions: parse movetext, track clocks, and choose a sample with DuckDB.

The Lichess open database (database.lichess.org, CC0) is read as Parquet, as the Hugging Face
dataset Lichess/standard-chess-games publishes it: one row per game, the PGN headers as
columns and the moves with their clock comments in `movetext`.
"""

import io
import re
from dataclasses import dataclass

import chess
import chess.pgn

from chess_ratings.maia import HISTORY

TIME_PRESSURE = 30.0  # seconds; Maia-3 drops positions after a player first falls below this


@dataclass(frozen=True)
class Ply:
    """One move of a game: the boards up to it (oldest first, current last), the move, and
    whether the mover was already in time pressure."""

    boards: tuple[chess.Board, ...]
    move: str
    ply: int
    white: bool
    pressed: bool
    clock: float = float("nan")  # the mover's seconds left before the move


def plies(movetext: str, start_seconds: float) -> list[Ply]:
    """Every move of a game with up to seven earlier boards and the time-pressure flag."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []
    out: list[Ply] = []
    board = game.board()
    past = [board.copy(stack=False)]
    clock = {chess.WHITE: start_seconds, chess.BLACK: start_seconds}
    pressed = {chess.WHITE: False, chess.BLACK: False}
    for i, node in enumerate(game.mainline()):
        mover = board.turn
        pressed[mover] = pressed[mover] or clock[mover] < TIME_PRESSURE
        out.append(
            Ply(
                tuple(past[-HISTORY:]),
                node.move.uci(),
                i,
                mover == chess.WHITE,
                pressed[mover],
                clock[mover],
            )
        )
        board.push(node.move)
        past.append(board.copy(stack=False))
        seconds = node.clock()
        if seconds is not None:
            clock[mover] = seconds
    return out


def base_seconds(time_control: str) -> float:
    """The starting clock of a time control such as '180+2'."""
    return float(time_control.split("+")[0]) if "+" in time_control else float("inf")


COMMENT = re.compile(r"\{[^}]*\}")
NUMBER = re.compile(r"^\d+\.+$")


def opening(movetext: str, plies: int) -> list[str]:
    """The first SAN moves of a movetext, without comments, move numbers or results."""
    words = COMMENT.sub(" ", movetext).split()
    sans = [w for w in words if not NUMBER.match(w) and w not in ("1-0", "0-1", "1/2-1/2", "*")]
    return [s.rstrip("?!") for s in sans[:plies]]


def blitz_sample_sql(files: list[str], per_band: int, salt: str, bands: tuple) -> str:
    """SQL for a stratified sample of established players' rated blitz games.

    A game qualifies if it is rated blitz, ended normally or on time, and moved neither rating
    by more than 15 points, which keeps players whose deviation is small. It falls in the band
    of the mean of the two ratings; within a band, games are ordered by md5(salt || Site), so the
    sample is fixed by the salt and the files. The moves are fetched afterwards, by
    `movetext_sql`, for the chosen games only.
    """
    listed = ", ".join(f"'{f}'" for f in files)
    return f"""
        WITH g AS (
            SELECT Site, WhiteElo, BlackElo, Result, TimeControl,
                   {band_case(bands)} AS band,
                   md5('{salt}' || Site) AS h, White, Black, filename
            FROM read_parquet([{listed}], filename = true)
            WHERE Event LIKE 'Rated Blitz%'
              AND Termination IN ('Normal', 'Time forfeit')
              AND abs(WhiteRatingDiff) <= 15 AND abs(BlackRatingDiff) <= 15
        )
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY band ORDER BY h) AS rank FROM g
            WHERE band IS NOT NULL
        ) WHERE rank <= {per_band} ORDER BY band, rank
    """


def movetext_sql(files: list[str], sites: list[str]) -> str:
    """SQL for the moves of the chosen games, by their Site."""
    listed = ", ".join(f"'{f}'" for f in files)
    wanted = ", ".join(f"'{s}'" for s in sites)
    return f"SELECT Site, movetext FROM read_parquet([{listed}]) WHERE Site IN ({wanted})"


def band_case(bands: tuple, otherwise: str = "NULL") -> str:
    """A SQL CASE that names the band of the mean of the two ratings."""
    cases = " ".join(
        f"WHEN (WhiteElo + BlackElo) / 2.0 >= {lo} AND (WhiteElo + BlackElo) / 2.0 < {hi} "
        f"THEN '{lo}-{hi}'"
        for lo, hi in bands
    )
    return f"CASE {cases} ELSE {otherwise} END"


def population_sample_sql(files: list[str], n: int, salt: str, bands: tuple) -> str:
    """SQL for qualifying games (as in `blitz_sample_sql`) in md5(salt || Site) order, whatever
    their ratings, so the sample follows Lichess's distribution of ratings. Games whose mean
    rating is outside the bands get the band 'other'."""
    listed = ", ".join(f"'{f}'" for f in files)
    return f"""
        WITH g AS (
            SELECT Site, WhiteElo, BlackElo, Result, TimeControl,
                   {band_case(bands, "'other'")} AS band,
                   md5('{salt}' || Site) AS h, White, Black, filename
            FROM read_parquet([{listed}], filename = true)
            WHERE Event LIKE 'Rated Blitz%'
              AND Termination IN ('Normal', 'Time forfeit')
              AND abs(WhiteRatingDiff) <= 15 AND abs(BlackRatingDiff) <= 15
        )
        SELECT * FROM (SELECT *, row_number() OVER (ORDER BY h) AS rank FROM g)
        WHERE rank <= {n} ORDER BY rank
    """


def rating_counts_sql(files: list[str], width: int = 25) -> str:
    """SQL for the number of qualifying players' ratings in bins of `width`, by time control:
    the population prior's data, as aggregates."""
    listed = ", ".join(f"'{f}'" for f in files)
    return f"""
        WITH g AS (
            SELECT TimeControl, WhiteElo, BlackElo FROM read_parquet([{listed}])
            WHERE Event LIKE 'Rated Blitz%'
              AND Termination IN ('Normal', 'Time forfeit')
              AND abs(WhiteRatingDiff) <= 15 AND abs(BlackRatingDiff) <= 15
        ), p AS (
            SELECT TimeControl, WhiteElo AS elo FROM g
            UNION ALL SELECT TimeControl, BlackElo FROM g
        )
        SELECT TimeControl, {width} * floor(elo / {width}) AS bin, count(*) AS n
        FROM p GROUP BY TimeControl, bin ORDER BY TimeControl, bin
    """


def month_sql(files: list[str], names: list[str]) -> str:
    """SQL for each named player's rated blitz games in the files: how many, and the rating
    before the last of them."""
    listed = ", ".join(f"'{f}'" for f in files)
    wanted = ", ".join("'" + n.replace("'", "''") + "'" for n in sorted(set(names)))
    return f"""
        WITH g AS (
            SELECT White, Black, WhiteElo, BlackElo, UTCDate + UTCTime AS played, Site
            FROM read_parquet([{listed}])
            WHERE Event LIKE 'Rated Blitz%' AND (White IN ({wanted}) OR Black IN ({wanted}))
        ), p AS (
            SELECT White AS name, WhiteElo AS elo, played, Site FROM g WHERE White IN ({wanted})
            UNION ALL
            SELECT Black, BlackElo, played, Site FROM g WHERE Black IN ({wanted})
        )
        SELECT name, count(*) AS games, arg_max(elo, (played, Site)) AS last FROM p GROUP BY name
    """
