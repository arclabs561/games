"""Boards, clues, the two roles, and a game loop.

The game is the one-team version used by the Codenames AI competition (Summerville et
al. 2019) and by Kim et al. (2019): 25 words, of which 9 are the team's, 8 the
opponent's, 7 neutral and 1 the assassin. Only the team plays. Each turn the giver
names a clue and a count n; the guesser picks up to n words, one at a time, and the
turn ends at the first pick that is not a team word. Picking the assassin, or the last
opponent word, loses; finding all 9 team words wins. The real game also allows one
pick beyond n; this model does not.

A **giver** is a function from a board to a clue. A **guesser** is a function from a
clue, a count and the unrevealed words to those words in the order it would pick them.
Any function with these signatures can play, including one that asks a language model.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM

SPLIT = {TEAM: 9, OPPONENT: 8, NEUTRAL: 7, ASSASSIN: 1}


@dataclass(frozen=True)
class Clue:
    word: str
    count: int


@dataclass
class Board:
    words: tuple[str, ...]
    kinds: tuple[str, ...]
    revealed: set[str] = field(default_factory=set)

    def unrevealed(self) -> tuple[str, ...]:
        return tuple(w for w in self.words if w not in self.revealed)

    def kind(self, word: str) -> str:
        return self.kinds[self.words.index(word)]

    def left(self, kind: str) -> int:
        return sum(
            1
            for w, k in zip(self.words, self.kinds, strict=True)
            if k == kind and w not in self.revealed
        )


Giver = Callable[[Board], Clue]
Guesser = Callable[[str, int, tuple[str, ...]], list[str]]


def deal(words: tuple[str, ...], rng: random.Random) -> Board:
    """25 distinct words with the 9/8/7/1 split, in random positions."""
    chosen = rng.sample(words, 25)
    kinds = [k for k, n in SPLIT.items() for _ in range(n)]
    rng.shuffle(kinds)
    return Board(tuple(chosen), tuple(kinds))


@dataclass
class Result:
    won: bool
    turns: int
    clues: list[tuple[Clue, list[str]]]


def play(board: Board, giver: Giver, guesser: Guesser, max_turns: int = 25) -> Result:
    """Play one game to the end, or until max_turns turns have passed (a loss)."""
    clues = []
    for turn in range(1, max_turns + 1):
        clue = giver(board)
        picks = guesser(clue.word, clue.count, board.unrevealed())[: clue.count]
        made = []
        for word in picks:
            board.revealed.add(word)
            made.append(word)
            kind = board.kind(word)
            if kind == ASSASSIN or (kind == OPPONENT and board.left(OPPONENT) == 0):
                clues.append((clue, made))
                return Result(False, turn, clues)
            if board.left(TEAM) == 0:
                clues.append((clue, made))
                return Result(True, turn, clues)
            if kind != TEAM:
                break
        clues.append((clue, made))
    return Result(False, max_turns, clues)
