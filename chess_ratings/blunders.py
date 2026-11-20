"""Blunders as Lichess defines them, from a light Stockfish evaluation of every legal move.

Lichess's analysis (lila modules/tree/src/main/Advice.scala, scalachess eval.scala) calls a
move a blunder when it lowers the mover's winning chances by 0.3 or more on the scale from -1
to 1, where the winning chances of an evaluation of cp centipawns are
2 / (1 + exp(-0.00368208 cp)) - 1. Moves into or out of a forced mate follow its separate
rules: allowing a mate is a blunder unless the mover was already worse than -700 centipawns,
and losing a forced mate is a blunder unless the mover is still better than +700.

Stockfish (GPL-3.0) runs as a separate program through python-chess's UCI client; it is not
part of this repository. Each move is scored by searching the position after it for a fixed
number of nodes, so the evaluations are light and somewhat noisy.
"""

from dataclasses import dataclass
from math import exp

import chess
import chess.engine

MULTIPLIER = -0.00368208
THRESHOLD = 0.3


def winning_chances(cp: float) -> float:
    """Lichess's winning chances in [-1, 1] for an evaluation in centipawns."""
    return max(-1.0, min(1.0, 2 / (1 + exp(MULTIPLIER * cp)) - 1))


@dataclass(frozen=True)
class Eval:
    """An evaluation from the mover's side: centipawns, or moves to mate (negative if mated)."""

    cp: int | None = None
    mate: int | None = None


def from_score(score: chess.engine.PovScore, mover: chess.Color) -> Eval:
    """Convert python-chess's score to the mover's point of view."""
    s = score.pov(mover)
    if s.is_mate():
        m = s.mate()
        return Eval(mate=m if m != 0 else -1)
    return Eval(cp=s.score())


def is_blunder(before: Eval, after: Eval) -> bool:
    """Lichess's blunder rule for the best line's evaluation `before` and the move's `after`."""
    if before.cp is not None and after.cp is not None:
        return winning_chances(before.cp) - winning_chances(after.cp) >= THRESHOLD
    if before.cp is not None and after.mate is not None and after.mate < 0:
        return before.cp >= -700  # a mate is now unavoidable
    if before.mate is not None and before.mate > 0:
        if after.cp is not None:
            return after.cp <= 700  # a forced mate was let go
        if after.mate is not None and after.mate < 0:
            return True
    return False


def evaluate_moves(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> dict[str, Eval]:
    """Each legal move's evaluation from the mover's side, by a search of the position after it."""
    mover = board.turn
    out = {}
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            out[move.uci()] = Eval(mate=1)
            continue
        if child.is_game_over():
            out[move.uci()] = Eval(cp=0)
            continue
        info = engine.analyse(child, chess.engine.Limit(nodes=nodes))
        out[move.uci()] = from_score(info["score"], mover)
    return out


def rank(e: Eval) -> float:
    """A total order on evaluations: mates beat any centipawn score, sooner mates first."""
    if e.mate is not None:
        return 100_000 - e.mate if e.mate > 0 else -100_000 - e.mate
    return float(e.cp or 0)


def blunder_moves(evals: dict[str, Eval]) -> set[str]:
    """The legal moves that are blunders against the best move's evaluation."""
    best = max(evals.values(), key=rank)
    return {m for m, e in evals.items() if is_blunder(best, e)}
