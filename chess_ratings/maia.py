"""Maia-3's move probabilities for a position, by the ratings of the two players.

Maia-3 (Monroe, Eilender, Chalmers, Tang and Anderson, "Chessformer", ICLR 2026) is loaded with
its authors' inference package `maia3` (github.com/CSSLab/maia3, AGPL-3.0), which is not part of
this repository and is installed only where the model runs:

    uv pip install --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install "maia3 @ git+https://github.com/CSSLab/maia3@1e13597c42d4858b7cfd7cfdae01e297263364b2"

The checkpoint (`UofTCSSLab/Maia3-5M`, file `maia3-5m.pt`) is read from a local path. The model
sees the current position and up to seven earlier ones, both ratings, and no clock. This module
batches positions and ratings; `check_against_uci` compares it with the authors' UCI engine.
"""

import importlib
from collections import deque
from dataclasses import dataclass
from typing import Any

import chess

MAIA3_COMMIT = "1e13597c42d4858b7cfd7cfdae01e297263364b2"
HISTORY = 8  # the current position and seven before it


@dataclass(frozen=True)
class Query:
    """A position with the boards before it (oldest first, current last) and two ratings."""

    boards: tuple[chess.Board, ...]
    rating: int
    opponent: int

    @property
    def board(self) -> chess.Board:
        return self.boards[-1]


def history(board: chess.Board, use_history: bool = True) -> tuple[chess.Board, ...]:
    """The current position and up to seven before it, from the board's move stack."""
    if not use_history:
        return (board.copy(stack=False),)
    boards = [board.copy(stack=False)]
    replay = board.copy()
    while replay.move_stack and len(boards) < HISTORY:
        replay.pop()
        boards.append(replay.copy(stack=False))
    return tuple(reversed(boards))


class Maia:
    """A loaded Maia-3 checkpoint that returns move probabilities for batches of queries."""

    def __init__(self, checkpoint: str, model: str = "5m", threads: int | None = None):
        self.torch: Any = importlib.import_module("torch")
        if threads:
            self.torch.set_num_threads(threads)
        self.uci: Any = importlib.import_module("maia3.uci")
        self.data: Any = importlib.import_module("maia3.dataset")
        utils: Any = importlib.import_module("maia3.utils")
        self.mirror = utils.mirror_move
        self.cfg = self.uci.parse_args(
            ["--model", model, "--checkpoint", checkpoint, "--device", "cpu", "--no-use-amp"]
        )
        self.net = self.uci.load_model(self.cfg)
        self.moves = utils.get_all_possible_moves()
        self.index = {m: i for i, m in enumerate(self.moves)}

    def tokens(self, boards: tuple[chess.Board, ...]) -> Any:
        """The model's (64, features) input for a position and its history."""
        past = deque((self.data.tokenize_board(b) for b in boards), maxlen=self.cfg.history)
        return self.data.get_historical_tokens(past, self.cfg, 0.0, 0.0, 0.0, 0.0)

    def legal(self, board: chess.Board) -> list[tuple[chess.Move, int]]:
        """Each legal move with its index in the model's output (mirrored when Black moves)."""
        out = []
        for move in board.legal_moves:
            uci = move.uci() if board.turn == chess.WHITE else self.mirror(move.uci())
            out.append((move, self.index[uci]))
        return out

    def probabilities(self, queries: list[Query], batch: int = 512) -> list[dict[str, float]]:
        """For each query, the probability of every legal move (UCI notation), summing to one."""
        torch = self.torch
        out: list[dict[str, float]] = []
        cache: dict[int, Any] = {}
        moves: dict[int, list[tuple[chess.Move, int]]] = {}
        for start in range(0, len(queries), batch):
            chunk = queries[start : start + batch]
            rows = []
            for q in chunk:
                key = id(q.boards)
                if key not in cache:
                    cache = {key: self.tokens(q.boards)}
                    moves = {key: self.legal(q.board)}
                rows.append(cache[key])
            with torch.no_grad():
                logits, _, _ = self.net(
                    torch.stack(rows),
                    torch.tensor([q.rating for q in chunk], dtype=torch.long),
                    torch.tensor([q.opponent for q in chunk], dtype=torch.long),
                )
            for q, row in zip(chunk, logits.float(), strict=True):
                legal = moves.get(id(q.boards)) or self.legal(q.board)
                picked = torch.log_softmax(row[[i for _, i in legal]], dim=0)
                out.append(
                    {m.uci(): float(p.exp()) for (m, _), p in zip(legal, picked, strict=True)}
                )
        return out

    def check_against_uci(self, board: chess.Board, rating: int) -> list[tuple[str, float, float]]:
        """Top moves from the authors' UCI engine beside this module's, with no history."""
        cfg = self.uci.parse_args(
            ["--model", "5m", "--checkpoint", self.cfg.checkpoint_path, "--device", "cpu"]
            + ["--no-use-amp", "--multipv", "5", "--elo", str(rating), "--temperature", "0"]
        )
        engine = self.uci.Maia3UCIEngine(cfg)
        engine.model = self.net
        engine.board = board.copy(stack=False)
        engine._reset_history()
        _, top = engine.score_moves()
        ours = self.probabilities([Query(history(board, False), rating, rating)])[0]
        return [(t["move"].uci(), float(t["policy"]), ours[t["move"].uci()]) for t in top]
