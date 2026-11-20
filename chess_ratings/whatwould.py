"""What would a player of a given rating play here? Maia-3's move distribution by rating.

    uv run python -m chess_ratings.whatwould "<FEN>" --checkpoint maia3-5m.pt
    uv run python -m chess_ratings.whatwould --moves e2e4 e7e5 d1h5 b8c6 f1c4 --checkpoint ...

With `--moves`, the game is replayed from the start and the model also sees up to seven earlier
positions; with a FEN it sees only the current one, as the authors' UCI engine does by default.
Both players are given the same rating. It needs torch and the `maia3` package (see maia.py).
"""

import argparse

import chess

from chess_ratings.maia import Maia, Query, history

RATINGS = (800, 1100, 1500, 1900, 2300)


def table(board: chess.Board, probs: dict[int, dict[str, float]], top: int) -> str:
    """Rows of moves (SAN), the most likely at any rating first; columns of ratings."""
    ratings = sorted(probs)
    chosen: set[str] = set()
    for r in ratings:
        chosen |= set(sorted(probs[r], key=lambda m: (-probs[r][m], m))[:top])
    middle = ratings[len(ratings) // 2]
    order = sorted(chosen, key=lambda m: (-probs[middle][m], m))
    head = "move    " + "".join(f"{r:>7}" for r in ratings)
    lines = [head]
    for m in order:
        san = board.san(chess.Move.from_uci(m))
        lines.append(f"{san:<8}" + "".join(f"{100 * probs[r][m]:>6.1f}%" for r in ratings))
    rest = [100 * (1 - sum(probs[r][m] for m in order)) for r in ratings]
    lines.append(f"{'other':<8}" + "".join(f"{x:>6.1f}%" for x in rest))
    return "\n".join(lines)


def distribution(maia: Maia, board: chess.Board, ratings, use_history: bool) -> dict:
    """Maia-3's probability of each legal move, for each rating (both players at that rating)."""
    boards = history(board, use_history)
    out = maia.probabilities([Query(boards, r, r) for r in ratings])
    return dict(zip(ratings, out, strict=True))


def run(maia: Maia, fen: str | None, moves: list[str], ratings, top: int) -> str:
    """The text the command prints."""
    board = chess.Board(fen) if fen else chess.Board()
    for m in moves:
        board.push_uci(m)
    probs = distribution(maia, board, ratings, use_history=bool(moves))
    side = "White" if board.turn == chess.WHITE else "Black"
    return f"{board.fen()}\n{side} to move; both players at the rating shown.\n" + table(
        board, probs, top
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fen", nargs="?", default=None)
    parser.add_argument("--moves", nargs="*", default=[], help="UCI moves from the start")
    parser.add_argument("--ratings", nargs="*", type=int, default=list(RATINGS))
    parser.add_argument("--top", type=int, default=3, help="moves shown per rating")
    parser.add_argument("--checkpoint", required=True, help="path to maia3-5m.pt")
    args = parser.parse_args(argv)
    maia = Maia(args.checkpoint)
    print(run(maia, args.fen, args.moves, args.ratings, args.top))


if __name__ == "__main__":
    main()
