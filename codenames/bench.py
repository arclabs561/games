"""The single-team game of the Codenames AI competition framework, and its word-vector bots.

The framework is the one of Summerville et al. (2019), CodenamesAICompetition/Game, as
revised by Stephenson et al. (2025, arXiv 2412.11373) in stepmat/Codenames_GPT. Its
single-team version is the game of codenames.game: only the red team plays, a turn
ends at the first pick that is not red, and the game is lost on the assassin or on the
last blue word. The **score** of a game is the number of turns (clues) it took to find
all 9 red words, or 25 if it was lost. Lower is better.

A board is dealt as the framework deals it: `random.seed(seed)`, then one shuffle of
the word pool (the first 25 words are the board) and one shuffle of the key.

The framework's word-vector bots, ported here so that they run on any board and with any
partner:

- **Codemaster** (players/codemaster_{w2v,glove,w2vglove}.py). For each count k from 1
  to 3, and each set of k red words, it looks for the clue in its list (cm_wordlist.txt)
  whose largest cosine distance to the k words is smallest, among clues closer to all k
  than to any other unrevealed word. It gives the largest k whose best clue is within
  distance 0.7 of all its words, and k = 1 otherwise. A clue may not be a board word, a
  part of one or contain one, and may not equal a board word's WordNet lemma or
  Lancaster stem (the clue's own, as NLTK computes them).
- **Guesser** (players/guesser_{w2v,glove,w2vglove}.py). It picks the unrevealed word
  at the smallest cosine distance to the clue, and keeps picking until it has made as
  many picks as the count. Ties go to the alphabetically first word. Words without a
  vector are never picked.

"w2v" uses the GoogleNews word2vec vectors (300 dimensions, float32), "glove" the
GloVe 6B 300d vectors (parsed to float64), and "w2vglove" both, concatenated.

    uv run python -m codenames.bench_play --help     # the runs
"""

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from codenames.game import Board, Clue, Result
from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM

KINDS = {"Red": TEAM, "Blue": OPPONENT, "Civilian": NEUTRAL, "Assassin": ASSASSIN}
KEY = ["Red"] * 9 + ["Blue"] * 8 + ["Civilian"] * 7 + ["Assassin"] * 1
LOSS_SCORE = 25
THRESHOLD = 0.7
MAX_COUNT = 3
BOTS = {"w2v": ("w2v",), "glove": ("glove",), "w2vglove": ("w2v", "glove")}


def framework_deal(pool: Sequence[str], seed: int) -> Board:
    """The board and key that the framework deals for this seed (game.py, Game.__init__).

    random.Random(seed) gives the same stream as the framework's random.seed(int(seed)).
    Words are lowercased; the framework shows them in capitals."""
    rng = random.Random(seed)
    words = list(pool)
    rng.shuffle(words)
    key = list(KEY)
    rng.shuffle(key)
    return Board(tuple(w.lower() for w in words[:25]), tuple(KINDS[k] for k in key))


# Vectors and the framework's cosine distance.


@dataclass
class Vectors:
    """Vectors of one or more tables, concatenated as the framework's concatenate() does;
    a word has a vector only if every table has it."""

    words: list[str]
    matrix: np.ndarray  # (len(words), dims), float32 or float64

    @cached_property
    def index(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def rows(self, words: Sequence[str]) -> np.ndarray:
        return self.matrix[[self.index[w] for w in words]]

    @staticmethod
    def concat(tables: Sequence["Vectors"]) -> "Vectors":
        if len(tables) == 1:
            return tables[0]
        common = [w for w in tables[0].words if all(w in t for t in tables[1:])]
        return Vectors(common, np.hstack([t.rows(common) for t in tables]))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance of every row of a to every row of b, shape (len(a), len(b)).

    scipy.spatial.distance.cosine computes 1 - u.v / sqrt((u.u)(v.v)) in the vectors'
    own precision, clipped to [0, 2]. For float32 vectors the result is a float32, as in
    the framework's logs; the order of the sums in the dot products may differ from
    scipy's in the last bit."""
    uv = a @ b.T
    uu = np.einsum("ij,ij->i", a, a)
    vv = np.einsum("ij,ij->i", b, b)
    denom = np.sqrt((uu[:, None] * vv[None, :]).astype(np.float64))
    ratio = (uv / denom).astype(a.dtype)
    return np.clip(a.dtype.type(1.0) - ratio, 0.0, 2.0)


# The legality rule of the framework's codemaster (arr_not_in_word).


def framework_legal(clue: str, forms: tuple[str, str], board: Sequence[str]) -> bool:
    """False if the clue is a board word, its lemma or stem is one, or it is a part of
    a board word or contains one. forms = (WordNet lemma, Lancaster stem) of the clue."""
    lemma, stem = forms
    if clue in board:
        return False
    return not any(w in (lemma, stem) or clue in w or w in clue for w in board)


# The bots.


@dataclass
class FrameworkCodemaster:
    """codemaster_{w2v,glove,w2vglove}.AICodemaster with threshold 0.7, vectorized.

    Within each count k, the framework's loops visit the sets of k red words in
    itertools.combinations order and the clues in list order, and keep the first pair
    with the smallest distance (strict comparisons); np.argmin over the flattened
    (set, clue) array keeps the same one."""

    name: str
    vectors: Vectors
    clues: list[str]
    forms: dict[str, tuple[str, str]]
    threshold: float = THRESHOLD
    _boards: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        missing = [c for c in self.clues if c not in self.vectors]
        if missing:
            raise KeyError(f"{self.name}: {len(missing)} clue words have no vector")
        self.clue_matrix = self.vectors.rows(self.clues)

    def board_tables(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        """Distances from every clue to every board word, and which pairs break the
        legality rule; computed once per board (the board's words do not change)."""
        key = board.words
        if key not in self._boards:
            missing = [w for w in board.words if w not in self.vectors]
            if missing:
                raise KeyError(f"{self.name}: board words without a vector: {missing}")
            dist = cosine(self.clue_matrix, self.vectors.rows(board.words))
            illegal = np.array(
                [
                    [not framework_legal(c, self.forms[c], (w,)) for w in board.words]
                    for c in self.clues
                ]
            )
            self._boards = {key: (dist, illegal)}  # one board at a time
        return self._boards[key]

    def plan(self, board: Board) -> tuple[str, int, list[tuple[tuple[str, ...], str, float]]]:
        dist, illegal = self.board_tables(board)
        live = [i for i, w in enumerate(board.words) if w not in board.revealed]
        red = [i for i in live if board.kinds[i] == TEAM]
        bad = [i for i in live if board.kinds[i] != TEAM]
        legal = ~illegal[:, live].any(axis=1)
        bad_min = dist[:, bad].min(axis=1)
        bests = []
        for k in range(1, MAX_COUNT + 1):
            combos = list(itertools.combinations(red, k))
            if not combos:
                bests.append(((), "", np.inf))
                continue
            worst = np.stack([dist[:, list(c)].max(axis=1) for c in combos])  # (sets, clues)
            value = np.where(legal[None, :] & (worst < bad_min[None, :]), worst, np.inf)
            flat = int(np.argmin(value))
            s, c = divmod(flat, value.shape[1])
            if value[s, c] == np.inf:
                bests.append(((), "", np.inf))
            else:
                bests.append(
                    (tuple(board.words[i] for i in combos[s]), self.clues[c], float(value[s, c]))
                )
        chosen, count = bests[0], 1
        for k, (targets, _, score) in enumerate(bests, start=1):
            # The framework recomputes the largest distance to the targets; it is score.
            if targets and score < self.threshold:
                chosen, count = bests[k - 1], k
        if chosen[2] == np.inf:
            raise RuntimeError(f"{self.name}: no clue for one red word (the framework fails here)")
        return chosen[1], count, bests

    def __call__(self, board: Board) -> Clue:
        clue, count, _ = self.plan(board)
        return Clue(clue, count)


@dataclass
class FrameworkGuesser:
    """guesser_{w2v,glove,w2vglove}.AIGuesser: unrevealed words by cosine distance to the
    clue, ties in alphabetical order. A clue without a vector gives no picks (the
    framework would stop with an error)."""

    name: str
    vectors: Vectors

    def __call__(self, clue: str, count: int, words: tuple[str, ...]) -> list[str]:
        if clue not in self.vectors:
            return []
        have = [w for w in words if w in self.vectors]
        if not have:
            return []
        d = cosine(self.vectors.rows([clue]), self.vectors.rows(have))[0]
        return [
            w
            for _, w in sorted(
                zip(d.tolist(), have, strict=True), key=lambda x: (x[0], x[1].upper())
            )
        ]


def framework_bots(tables: dict[str, Vectors], clues: list[str], forms: dict[str, tuple[str, str]]):
    """The three codemasters and three guessers, by the framework's names."""
    vecs = {name: Vectors.concat([tables[t] for t in parts]) for name, parts in BOTS.items()}
    givers = {n: FrameworkCodemaster(n, v, clues, forms) for n, v in vecs.items()}
    guessers = {n: FrameworkGuesser(n, v) for n, v in vecs.items()}
    return givers, guessers


# Scoring.


def loss_kind(result: Result, board: Board) -> str | None:
    """None for a win; else the kind of the pick that lost (assassin or opponent), or
    "turns" if the game ran out of turns."""
    if result.won:
        return None
    last = [w for _, made in result.clues for w in made][-1:]
    if last and board.kind(last[0]) in (ASSASSIN, OPPONENT):
        return board.kind(last[0])
    return "turns"


def score(result: Result, board: Board, rule: str = "framework") -> int:
    """The single-team score: turns taken, or 25 for a loss.

    rule "framework" is the rule of the framework and of Stephenson et al. (Sec. IV-B1):
    25 for picking the assassin or the last blue word. rule "table" scores only an
    assassin loss as 25 and a loss on the last blue word by its turns, which is what
    their Table I does (see bench_play.published_table)."""
    kind = loss_kind(result, board)
    if kind is None:
        return result.turns
    if rule == "table" and kind == OPPONENT:
        return result.turns
    return LOSS_SCORE


def transcript(result: Result) -> list[tuple[str, int, list[str]]]:
    """(clue, count, picks in capitals) for each turn, as the framework's move log has them."""
    return [(c.word, c.count, [w.upper() for w in made]) for c, made in result.clues]
