"""Exact outcome probabilities for a guesser that picks words with Luce's choice rule.

A **Luce guesser** with weights w (one per unrevealed word) picks its next word with
probability proportional to its weight among the words not yet picked. With weights
exp(similarity / tau) this is the softmax, or Plackett-Luce, guesser; tau is its
**temperature**, and tau -> 0 gives the greedy guesser that picks in order of similarity.

A turn: the guesser keeps picking while it picks team words, up to the count n the giver
named. It stops at the first word that is not a team word. With T the set of team
words, write p(S) for the chance that the first |S| picks are the team words in S, in
some order. Then p(empty set) = 1 and

    p(S) = sum over x in S of  p(S - x) * w_x / (W - W(S - x)),

where W is the total weight and W(A) the weight of the words in A. The chance that
the first j picks are all team words is the sum of p(S) over the sets S of j team
words, and the chance that pick j is the first non-team word, of kind k, is the sum
over sets S of j - 1 team words of p(S) * W_k / (W - W(S)). These sums are exact and
need only the sets of at most n team words: at most 256 sets for 9 team words and n = 4.
"""

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np

TEAM, OPPONENT, NEUTRAL, ASSASSIN = "T", "O", "N", "A"
WRONG = (OPPONENT, NEUTRAL, ASSASSIN)


@dataclass(frozen=True)
class TurnOdds:
    """Outcome chances of one turn, for each of C clues at once.

    all_team[j] has shape (C,): the chance that the first j picks are all team words.
    first_wrong[k][j] has shape (C,): the chance that pick j (from 1) is the first
    non-team pick and is of kind k. Index 0 of first_wrong[k] is 0.
    """

    all_team: np.ndarray  # (nmax + 1, C)
    first_wrong: dict[str, np.ndarray]  # kind -> (nmax + 1, C)

    def expected_team(self, n: int) -> np.ndarray:
        """Expected number of team words found when up to n picks are allowed."""
        return self.all_team[1 : n + 1].sum(axis=0)

    def stopped_by(self, kind: str, n: int) -> np.ndarray:
        """Chance that the turn ends on a word of this kind within n picks."""
        return self.first_wrong[kind][1 : n + 1].sum(axis=0)

    def value(self, n: int, costs: dict[str, float]) -> np.ndarray:
        """Expected team words found, minus the cost of each kind of wrong pick."""
        v = self.expected_team(n)
        for kind, cost in costs.items():
            v = v - cost * self.stopped_by(kind, n)
        return v


def luce_weights(sims: np.ndarray, tau: float) -> np.ndarray:
    """exp(sims / tau), shifted per row so the largest weight is 1."""
    s = np.atleast_2d(sims)
    return np.exp((s - s.max(axis=1, keepdims=True)) / tau)


def turn_odds(weights: np.ndarray, kinds: tuple[str, ...], nmax: int) -> TurnOdds:
    """Exact outcome chances of a turn of up to nmax picks, by the recursion above.

    weights has shape (C, m) or (m,): one row of word weights per clue.
    kinds has length m: the kind (T, O, N or A) of each unrevealed word.
    """
    # A floor keeps weights that underflowed at a low temperature from dividing 0 by 0.
    w = np.maximum(np.atleast_2d(np.asarray(weights, dtype=np.float64)), 1e-300)
    team = [i for i, k in enumerate(kinds) if k == TEAM]
    kind_weight = {k: w[:, [i for i, x in enumerate(kinds) if x == k]].sum(axis=1) for k in WRONG}
    nmax = min(nmax, len(team))
    all_team = np.zeros((nmax + 1, w.shape[0]))
    first_wrong = {k: np.zeros((nmax + 1, w.shape[0])) for k in WRONG}
    every = range(w.shape[1])

    def left_after(picked: tuple[int, ...]) -> np.ndarray:
        # Summed directly rather than as total - picked, which cancels to 0 when the
        # picked words hold almost all the weight (a low temperature).
        return w[:, [i for i in every if i not in picked]].sum(axis=1)

    p: dict[tuple[int, ...], np.ndarray] = {(): np.ones(w.shape[0])}
    for size in range(nmax + 1):
        if size > 0:
            new: dict[tuple[int, ...], np.ndarray] = {}
            for subset in combinations(team, size):
                acc = np.zeros(w.shape[0])
                for x in subset:
                    rest = tuple(i for i in subset if i != x)
                    acc += p[rest] * w[:, x] / left_after(rest)
                new[subset] = acc
            p = new
        for subset, prob in p.items():
            all_team[size] += prob
            if size < nmax:
                left = left_after(subset)
                for k in WRONG:
                    first_wrong[k][size + 1] += prob * kind_weight[k] / left
    return TurnOdds(all_team, first_wrong)


def turn_odds_brute(weights: np.ndarray, kinds: tuple[str, ...], nmax: int) -> TurnOdds:
    """The same chances by summing over every ordered sequence of picks (small cases only)."""
    w = np.asarray(weights, dtype=np.float64)
    m = len(w)
    nmax = min(nmax, kinds.count(TEAM))
    all_team = np.zeros((nmax + 1, 1))
    first_wrong = {k: np.zeros((nmax + 1, 1)) for k in WRONG}
    for length in range(nmax + 1):
        for seq in permutations(range(m), length):
            prob, left = 1.0, w.sum()
            for i in seq:
                prob *= w[i] / left
                left -= w[i]
            picked = [kinds[i] for i in seq]
            if all(k == TEAM for k in picked):
                all_team[length] += prob
            elif all(k == TEAM for k in picked[:-1]):
                first_wrong[picked[-1]][length] += prob
    return TurnOdds(all_team, first_wrong)


def best_count(odds: TurnOdds, costs: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """For each clue, the count n with the highest expected value, and that value."""
    nmax = odds.all_team.shape[0] - 1
    values = np.stack([odds.value(n, costs) for n in range(1, nmax + 1)])
    return values.argmax(axis=0) + 1, values.max(axis=0)
