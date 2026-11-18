"""Givers and guessers built on word vectors or on association counts.

A **space** holds unit vectors for every word in the vocabulary under one model; the
similarity of a clue and a word is the dot product of their vectors. The clue side and
the word side may use different templates (codenames.embed.TEMPLATES).

Givers:
- threshold_giver: in the style of Kim et al. (2019). For each clue, rank the board by
  similarity; the targets are the team words ranked above every other word and at least
  t similar to the clue. Choose the clue with the most targets, then the widest gap
  between its weakest target and the closest other word.
- expected_giver: choose the clue and count with the highest expected value of the turn
  under the giver's own Luce guesser (codenames.luce), minus a penalty for rare clues:
  beta * max(0, z0 - Zipf frequency of the clue).

Guessers pick in order of similarity (embedding_guesser) or of association counts
(norm_guesser). For exact expected scores each guesser also has Luce weights.
"""

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from codenames.embed import vectors
from codenames.game import Board, Clue
from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM, best_count, luce_weights, turn_odds
from codenames.words import clue_words, legal, vocabulary, zipf

Weigher = Callable[[str, tuple[str, ...]], np.ndarray]


@dataclass
class Space:
    """Word vectors of one model: clue vectors and board-word vectors."""

    name: str
    clue_vecs: np.ndarray
    word_vecs: np.ndarray
    words: list[str]

    @cached_property
    def index(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}

    def sims(self, clue: str, words: tuple[str, ...]) -> np.ndarray:
        """Similarity of one clue to each word; 0 for a clue or word not in the vocabulary."""
        if clue not in self.index:
            return np.zeros(len(words))
        c = self.clue_vecs[self.index[clue]]
        out = np.zeros(len(words))
        for j, w in enumerate(words):
            if w in self.index:
                out[j] = float(self.word_vecs[self.index[w]] @ c)
        return out


def space(model: str, clue_template: str = "bare", word_template: str = "bare", words=None):
    words = list(words or vocabulary())
    cv = vectors(model, words, clue_template)
    wv = cv if word_template == clue_template else vectors(model, words, word_template)
    return Space(f"{model}:{clue_template}/{word_template}", cv, wv, words)


def costs_for(board: Board) -> dict[str, float]:
    """Costs of a wrong pick in the one-team game: an opponent word is a word for the other
    side (cost 1), a neutral word only ends the turn (0), and the assassin loses the game,
    so it costs every team word still to find."""
    return {OPPONENT: 1.0, NEUTRAL: 0.0, ASSASSIN: float(board.left(TEAM))}


@dataclass
class ClueTable:
    """Similarities of every candidate clue to the board words, under one space."""

    clues: list[str]
    sims: np.ndarray  # (len(clues), number of words)

    @staticmethod
    def build(sp: Space, words: tuple[str, ...], candidates: tuple[str, ...]) -> "ClueTable":
        ok = [c for c in candidates if legal(c, words) and c in sp.index]
        cmat = sp.clue_vecs[[sp.index[c] for c in ok]]
        wmat = sp.word_vecs[[sp.index[w] for w in words]]
        return ClueTable(ok, cmat @ wmat.T)


def threshold_giver(sp: Space, t: float = 0.5, candidates: tuple[str, ...] | None = None):
    pool = candidates or clue_words()

    def give(board: Board) -> Clue:
        words = board.unrevealed()
        kinds = tuple(board.kind(w) for w in words)
        tab = ClueTable.build(sp, words, pool)
        is_team = np.array([k == TEAM for k in kinds])
        order = np.argsort(-tab.sims, axis=1)
        ranked_team = is_team[order]
        lead = np.argmin(ranked_team, axis=1)  # team words before the first other word
        ranked_sims = np.take_along_axis(tab.sims, order, axis=1)
        above = (ranked_sims >= t) & ranked_team
        before_lead = np.arange(len(words))[None, :] < lead[:, None]
        n = (above & before_lead).sum(axis=1)
        best_bad = np.where(is_team, -np.inf, tab.sims).max(axis=1)
        # The n-th ranked word, or the first when n = 0.
        weakest = ranked_sims[np.arange(len(tab.clues)), np.maximum(n, 1) - 1]
        gap = weakest - best_bad
        i = int(np.lexsort((gap, n))[-1])
        return Clue(tab.clues[i], max(int(n[i]), 1))

    return give


class ExpectedGiver:
    """Maximize the expected value of the turn under the giver's own Luce guesser."""

    def __init__(
        self,
        sp: Space,
        tau: float,
        beta: float = 0.1,
        z0: float = 3.5,
        nmax: int = 4,
        shortlist: int = 300,
        candidates: tuple[str, ...] | None = None,
    ):
        self.sp, self.tau, self.beta, self.z0 = sp, tau, beta, z0
        self.nmax, self.shortlist = nmax, shortlist
        self.pool = candidates or clue_words()
        self.freq = {c: zipf(c) for c in self.pool}

    def __call__(self, board: Board) -> Clue:
        clue, n, _ = self.plan(board)
        return Clue(clue, n)

    def plan(self, board: Board) -> tuple[str, int, float]:
        """The chosen clue, its count, and its expected value under the giver's model."""
        words = board.unrevealed()
        kinds = tuple(board.kind(w) for w in words)
        tab = ClueTable.build(self.sp, words, self.pool)
        keep = shortlist_rows(tab.sims, kinds, self.shortlist)
        odds = turn_odds(luce_weights(tab.sims[keep], self.tau), kinds, self.nmax)
        n, value = best_count(odds, costs_for(board))
        rarity = np.array([max(0.0, self.z0 - self.freq[tab.clues[i]]) for i in keep])
        j = int(np.argmax(value - self.beta * rarity))
        return tab.clues[keep[j]], int(n[j]), float(value[j])


expected_giver = ExpectedGiver


class EnsembleGiver(ExpectedGiver):
    """Maximize the average expected value over several guesser models, each a space with
    its own temperature: a giver that does not assume its partner shares its vectors."""

    def __init__(self, members: list[tuple[Space, float]], **kw):
        super().__init__(members[0][0], members[0][1], **kw)
        self.members = members

    def plan(self, board: Board) -> tuple[str, int, float]:
        words = board.unrevealed()
        kinds = tuple(board.kind(w) for w in words)
        tabs = [ClueTable.build(sp, words, self.pool) for sp, _ in self.members]
        assert all(t.clues == tabs[0].clues for t in tabs)
        keep = sorted(
            set().union(*(shortlist_rows(t.sims, kinds, self.shortlist // 2) for t in tabs))
        )
        costs = costs_for(board)
        values = np.zeros((self.nmax, len(keep)))
        for tab, (_, tau) in zip(tabs, self.members, strict=True):
            odds = turn_odds(luce_weights(tab.sims[keep], tau), kinds, self.nmax)
            values += np.stack([odds.value(n, costs) for n in range(1, self.nmax + 1)])
        values /= len(self.members)
        n, value = values.argmax(axis=0) + 1, values.max(axis=0)
        clues = tabs[0].clues
        rarity = np.array([max(0.0, self.z0 - self.freq[clues[i]]) for i in keep])
        j = int(np.argmax(value - self.beta * rarity))
        return clues[keep[j]], int(n[j]), float(value[j])


def shortlist_rows(sims: np.ndarray, kinds: tuple[str, ...], k: int) -> list[int]:
    """Clues worth the exact computation: for each j up to 3, the k clues whose j-th
    closest team word beats the closest other word by the most."""
    team = np.array([x == TEAM for x in kinds])
    team_sorted = -np.sort(-sims[:, team], axis=1)
    best_bad = sims[:, ~team].max(axis=1)
    rows: set[int] = set()
    for j in range(min(3, team_sorted.shape[1])):
        margin = team_sorted[:, j] - best_bad
        rows |= set(np.argsort(-margin)[:k].tolist())
    return sorted(rows)


def embedding_guesser(sp: Space):
    def guess(clue: str, count: int, words: tuple[str, ...]) -> list[str]:
        s = sp.sims(clue, words)
        return [words[i] for i in np.argsort(-s, kind="stable")]

    return guess


def embedding_weigher(sp: Space, tau: float) -> Weigher:
    def weigh(clue: str, words: tuple[str, ...]) -> np.ndarray:
        return luce_weights(sp.sims(clue, words), tau)[0]

    return weigh


def tie_break(clue: str, word: str) -> int:
    """A fixed pseudo-random order for words with equal counts."""
    return zlib.crc32(f"{clue}|{word}".encode())


def norm_guesser(counts: dict[str, dict[str, int]]):
    """Pick board words in order of how many people gave them as associations of the clue."""

    def guess(clue: str, count: int, words: tuple[str, ...]) -> list[str]:
        row = counts.get(clue, {})
        return sorted(words, key=lambda w: (-row.get(w, 0), tie_break(clue, w)))

    return guess


def norm_weigher(counts: dict[str, dict[str, int]], eps: float) -> Weigher:
    """Luce weights count + eps: each person's choice, restricted to the board."""

    def weigh(clue: str, words: tuple[str, ...]) -> np.ndarray:
        row = counts.get(clue, {})
        return np.array([row.get(w, 0) + eps for w in words], dtype=float)

    return weigh


def expected_value(weigh: Weigher, board: Board, clue: Clue, nmax: int = 4) -> float:
    """Exact expected value of the turn when this guesser answers the clue."""
    words = board.unrevealed()
    kinds = tuple(board.kind(w) for w in words)
    odds = turn_odds(weigh(clue.word, words), kinds, min(clue.count, nmax))
    return float(odds.value(min(clue.count, nmax), costs_for(board))[0])


def expected_found(weigh: Weigher, board: Board, clue: Clue, nmax: int = 4) -> float:
    """Exact expected number of team words found in the turn."""
    words = board.unrevealed()
    kinds = tuple(board.kind(w) for w in words)
    n = min(clue.count, nmax)
    return float(turn_odds(weigh(clue.word, words), kinds, n).expected_team(n)[0])
