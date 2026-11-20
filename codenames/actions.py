"""The legal actions an embedding giver proposes on a board: 30 clues times counts 1 to 4.

This is the action list of experiments C6 (a hosted language model picks one) and C7 (a
small language model is trained to pick one). Both use the same function, so both see
the same 120 actions on the same board.

- The **proposer** is EmbeddingGemma 2 with the bare template on both sides, and the
  temperature fitted on the train turns of C1's game split (`results/calibrate-game.json`).
- The **clues** are `rl.actions` over the whole clue list (`words.clue_words()`, 12,000
  words): the clues that `players.shortlist_rows` keeps for the margin of the first,
  second or third closest team word over the closest other word, sorted by their
  closest team word, the first K kept. Every clue is legal on the board, because
  `players.ClueTable.build` drops the others before the shortlist.
- The **actions** are each clue with each count 1 to 4, clue by clue: index 4 i + n - 1
  is clue i with count n.

Computing the list needs the proposer's vectors (`codenames/vectors/`, from the branch
`outputs/codenames-embed`). A job without them reads the list from a text file written
by `action_lines`: one line per board, `seed board clue_1 ... clue_K`.

    uv run python -m codenames.actions --seed 6004 --n 200 --out actions-6004.txt
"""

import argparse
import hashlib
from functools import cache
from pathlib import Path

import numpy as np

from codenames.game import Board
from codenames.players import Space, Weigher, space
from codenames.study import load
from codenames.words import clue_words, legal

PROPOSER = "google/embeddinggemma-2"
K = 30
COUNTS = (1, 2, 3, 4)
Action = tuple[str, int]


@cache
def proposer() -> tuple[Space, float]:
    """The proposer's space and Luce temperature, as calibrated on C1's game split."""
    best = load("calibrate-game")[PROPOSER]["best"]
    from codenames.evaluate import all_words

    return space(PROPOSER, best["clue"], best["word"], all_words()), best["tau"]


def shortlist(board: Board, sp: Space, tau: float, k: int = K, pool=None) -> list[str]:
    """The k clues of `rl.actions` from pool (default the whole clue list), in its order."""
    from codenames.rl import actions

    clues, _ = actions(sp, tau, board, pool or clue_words(), k)
    return [c for c, n in clues if n == COUNTS[0]]


def expand(clues: list[str]) -> list[Action]:
    """Each clue with each count, clue by clue."""
    return [(c, n) for c in clues for n in COUNTS]


def legal_actions(
    board: Board, sp: Space | None = None, tau: float | None = None, k: int = K, pool=None
) -> list[Action]:
    """The 4 k legal (clue, count) actions on this board, from the proposer when sp is None.
    C7 calls this with no arguments but the board."""
    if sp is None or tau is None:
        sp, tau = proposer()
    return expand(shortlist(board, sp, tau, k, pool))


def check(board: Board, actions: list[Action], spaces: list[Space]) -> None:
    """Every clue is legal on the board and has a vector in every space."""
    words = board.unrevealed()
    for clue, n in actions:
        assert legal(clue, words), (clue, words)
        assert n in COUNTS, n
        for sp in spaces:
            assert clue in sp.index, (clue, sp.name)


def values(weigh: Weigher, board: Board, actions: list[Action]) -> np.ndarray:
    """Exact expected turn value of every action under one Luce guesser (codenames.luce),
    with the costs of players.costs_for."""
    from codenames.rl import rewards

    return rewards(weigh, board, actions)


def values_by_loop(weigh: Weigher, board: Board, actions: list[Action]) -> np.ndarray:
    """The same values one action at a time, through players.expected_value."""
    from codenames.game import Clue
    from codenames.players import expected_value

    return np.array([expected_value(weigh, board, Clue(c, n)) for c, n in actions])


def action_lines(seed: int, lists: list[list[str]]) -> str:
    """One line per board: the seed, the board's index and its clues in order."""
    return "".join(f"{seed} {i} {' '.join(c)}\n" for i, c in enumerate(lists))


def read_action_lines(text: str) -> dict[int, list[list[str]]]:
    """The inverse of action_lines, for several seeds: {seed: clues of each board}."""
    out: dict[int, list[list[str]]] = {}
    for line in text.splitlines():
        seed, board, *clues = line.split()
        rows = out.setdefault(int(seed), [])
        assert int(board) == len(rows), line[:40]
        rows.append(clues)
    return out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    from codenames.c4 import boards

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = ""
    for seed, n in zip(args.seed, args.n, strict=True):
        sp, tau = proposer()
        text += action_lines(seed, [shortlist(b, sp, tau) for b in boards(seed, n)])
    args.out.write_text(text)
    print(args.out, sha256(text))


if __name__ == "__main__":
    main()
