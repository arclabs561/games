"""Deck edits as optimal transport, and the frequency-difference baseline.

A deck is a **measure** on cards: each copy carries mass 1/n for a deck of n cards. A
**target** is the average measure of the decks that won (or lost) games. Moving one
measure onto the other at least total cost, with the distance between card embeddings as
the cost of moving a unit of mass from one card to another, is an optimal transport
problem; its plan says which cards' mass goes to which. Mass that moves from card i to
card j is read as "cut i, add j".

**Lemma (kept mass).** If the cost is a metric (zero on the diagonal, symmetric, with the
triangle inequality), some optimal plan keeps min(mu_i, nu_i) on every card i. Proof:
in a plan that moves mass from i to j while card k sends mass to i, sending that mass
from k to j directly and keeping it at i costs no more, since d(k, j) <= d(k, i) + d(i, j).
Repeating this never increases cost and ends with the kept mass maximal. So the plan
only pairs the cards the deck has too many of with those it has too few of, and those
amounts are the positive and negative parts of mu - nu: the same lists of cuts and adds
as the frequency-difference baseline. What transport adds is the pairing.

The cost used is the Euclidean distance between unit vectors, sqrt(2 - 2 cos), which is a
metric; 1 - cos is not.
"""

import itertools

import numpy as np
import ot
from scipy.optimize import linprog

from mtg_decks.embed import cosine


def measure(counts: np.ndarray) -> np.ndarray:
    """A deck's card counts as a measure of total mass 1."""
    c = counts.astype(np.float64)
    return c / c.sum()


def ground_cost(x: np.ndarray) -> np.ndarray:
    """Euclidean distance between unit-normalized embedding rows: a metric."""
    sq = np.clip(2 - 2 * cosine(x), 0, None)
    d = np.sqrt(sq)
    np.fill_diagonal(d, 0.0)
    return d


def emd(mu: np.ndarray, nu: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact optimal plan and cost by POT's network simplex, on the supports of mu and nu."""
    a, b = np.flatnonzero(mu), np.flatnonzero(nu)
    sub = np.ascontiguousarray(cost[np.ix_(a, b)])
    p = ot.emd(mu[a], nu[b], sub, numItermax=1_000_000)
    plan = np.zeros((mu.size, nu.size))
    plan[np.ix_(a, b)] = p
    return plan, float((p * sub).sum())


def lp_cost(mu: np.ndarray, nu: np.ndarray, cost: np.ndarray) -> float:
    """The same optimum by a linear program (HiGHS): a second, independent solver."""
    a, b = np.flatnonzero(mu), np.flatnonzero(nu)
    sub = cost[np.ix_(a, b)]
    m, n = sub.shape
    rows = np.zeros((m + n, m * n))
    for i in range(m):
        rows[i, i * n : (i + 1) * n] = 1
    for j in range(n):
        rows[m + j, j::n] = 1
    res = linprog(sub.ravel(), A_eq=rows, b_eq=np.concatenate([mu[a], nu[b]]), method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def brute_cost(src: list[int], dst: list[int], cost: np.ndarray) -> float:
    """Transport between two decks of equal size by trying every matching of copies.

    With mass 1/n on each of n copies, an optimal plan can be taken to be a matching
    (Birkhoff), so the minimum over the n! matchings is the exact optimum.
    """
    n = len(src)
    best = min(
        sum(cost[src[i], dst[p[i]]] for i in range(n)) for p in itertools.permutations(range(n))
    )
    return best / n


def total_variation(mu: np.ndarray, nu: np.ndarray) -> float:
    """Transport cost with cost 1 between any two different cards: half the L1 distance."""
    return 0.5 * float(np.abs(mu - nu).sum())


Swap = tuple[int, int, float]  # (cut card, added card, mass)


def swaps(plan: np.ndarray, min_mass: float = 1e-12) -> list[Swap]:
    """The plan's off-diagonal entries as swaps, largest mass first, ties by index."""
    out = [
        (int(i), int(j), float(plan[i, j]))
        for i, j in zip(*np.nonzero(plan > min_mass), strict=True)
        if i != j
    ]
    return sorted(out, key=lambda t: (-t[2], t[0], t[1]))


def baseline_swaps(mu: np.ndarray, nu: np.ndarray) -> list[Swap]:
    """Frequency-difference swaps: cut cards by surplus, add by deficit, paired in rank order.

    The surplus and deficit are the positive and negative parts of mu - nu; the cards in
    each list are sorted by amount (ties by index) and matched greedily from the top, as
    a person reading two ranked lists side by side would.
    """
    d = mu - nu
    cuts = sorted(np.flatnonzero(d > 1e-15), key=lambda i: (-d[i], i))
    adds = sorted(np.flatnonzero(d < -1e-15), key=lambda j: (d[j], j))
    left = {int(i): float(d[i]) for i in cuts}
    need = {int(j): float(-d[j]) for j in adds}
    out: list[Swap] = []
    a = 0
    for i in cuts:
        i = int(i)
        while left[i] > 1e-15 and a < len(adds):
            j = int(adds[a])
            m = min(left[i], need[j])
            out.append((i, j, m))
            left[i] -= m
            need[j] -= m
            if need[j] <= 1e-15:
                a += 1
    return sorted(out, key=lambda t: (-t[2], t[0], t[1]))


def pairing_gap(sw: list[Swap], value: np.ndarray) -> float:
    """Mass-weighted mean |value[cut] - value[add]| over swaps (for example, mana value)."""
    mass = np.array([m for _, _, m in sw])
    gap = np.array([abs(value[i] - value[j]) for i, j, _ in sw])
    return float((mass * gap).sum() / mass.sum())


def pairing_match(sw: list[Swap], label: list[str]) -> float:
    """Mass-weighted share of swaps whose two cards have the same label (for example, type)."""
    mass = np.array([m for _, _, m in sw])
    same = np.array([label[i] == label[j] for i, j, _ in sw], dtype=float)
    return float((mass * same).sum() / mass.sum())
