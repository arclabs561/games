"""Diverse suggestion lists by maximal marginal relevance (Carbonell and Goldstein 1998).

From candidates with relevance r and pairwise similarity sim, MMR picks greedily

    argmax over unpicked j of  lam * r'(j) - (1 - lam) * max over picked s of sim(j, s),

with r' the relevance min-max scaled to [0, 1] over the candidates, as rankops'
`mmr_similarity` does; lam = 1 is the top-k list by relevance. `rankops` (PyPI) is the
second implementation, and `best_subset` is the exact optimum of a set objective that
MMR approximates greedily, for small cases.
"""

import itertools

import numpy as np


def scaled(r: np.ndarray) -> np.ndarray:
    """Relevance min-max scaled to [0, 1]; all ones when every value is equal."""
    lo, hi = r.min(), r.max()
    return np.ones_like(r) if hi == lo else (r - lo) / (hi - lo)


def mmr(r: np.ndarray, sim: np.ndarray, lam: float, k: int) -> list[int]:
    """Indices picked by MMR, in order; ties go to the lower index."""
    rel = scaled(r.astype(np.float64))
    picked: list[int] = []
    worst = np.full(r.size, -np.inf)  # max similarity to the picked set so far
    for _ in range(min(k, r.size)):
        pen = np.where(np.isinf(worst), 0.0, worst)
        score = lam * rel - (1 - lam) * pen
        score[picked] = -np.inf
        j = int(np.argmax(score))
        picked.append(j)
        worst = np.maximum(worst, sim[j])
    return picked


def mmr_rankops(r: np.ndarray, sim: np.ndarray, lam: float, k: int) -> list[int]:
    """The same selection by rankops' Rust implementation (single precision)."""
    import rankops

    cands = [(str(i), float(v)) for i, v in enumerate(r)]
    out = rankops.mmr_similarity(cands, sim.tolist(), lam, k)
    return [int(i) for i, _ in out]


def objective(r: np.ndarray, sim: np.ndarray, lam: float, s: list[int]) -> float:
    """A set objective of MMR's form: lam * mean scaled relevance - (1 - lam) * mean pair sim."""
    rel = scaled(r.astype(np.float64))
    pairs = list(itertools.combinations(s, 2))
    red = np.mean([sim[i, j] for i, j in pairs]) if pairs else 0.0
    return float(lam * rel[s].mean() - (1 - lam) * red)


def best_subset(r: np.ndarray, sim: np.ndarray, lam: float, k: int) -> list[int]:
    """The k-subset that maximizes `objective`, by trying every subset (small cases only)."""
    return list(
        max(itertools.combinations(range(r.size), k), key=lambda s: objective(r, sim, lam, list(s)))
    )


def relevance_kept(r: np.ndarray, picked: list[int]) -> float:
    """Relevance of the picked list as a share of the top-k list's."""
    top = np.sort(r)[::-1][: len(picked)]
    return float(r[picked].sum() / top.sum())


def redundancy(sim: np.ndarray, picked: list[int]) -> float:
    """Mean pairwise similarity within the picked list."""
    pairs = list(itertools.combinations(picked, 2))
    return float(np.mean([sim[i, j] for i, j in pairs]))


def distinct(labels: list[str], picked: list[int]) -> int:
    """Number of distinct labels (for example, card type and mana value) in the list."""
    return len({labels[i] for i in picked})
