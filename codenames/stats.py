"""The bootstrap over clusters (games) used for every interval on human turns.

Turns come in games, and turns of one game are not independent. A replicate draws as
many games as there are, with replacement, and pools their turns. The estimate is the
share of hits over the pooled turns (a ratio, since games differ in size). The interval
is the 2.5th and 97.5th percentiles over 10,000 replicates. A paired difference between
two guessers is the same computation on the per-turn differences, so both guessers see
the same games in every replicate.

The tests compare the draws with an exact enumeration of every resample on three or four
games.
"""

import itertools
from collections import Counter
from math import factorial

import numpy as np

DRAWS = 10_000


def cluster_draws(clusters: list[str], draws: int = DRAWS, seed: int = 0):
    """Cluster index of each unit, and a (draws, n_clusters) matrix of how often each
    cluster is drawn in each replicate (multinomial with equal chances)."""
    ids = sorted(set(clusters))
    pos = {g: i for i, g in enumerate(ids)}
    index = np.array([pos[g] for g in clusters])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    return index, counts


def replicates(values, index: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The pooled mean of the values in each replicate. values may be (units,) or
    (units, k) for k statistics at once; the result is (draws,) or (draws, k)."""
    v = np.asarray(values, dtype=float)
    n = counts.shape[1]
    size = np.bincount(index, minlength=n).astype(float)
    if v.ndim == 1:
        per = np.bincount(index, weights=v, minlength=n)
        return (counts @ per) / (counts @ size)
    per = np.stack([np.bincount(index, weights=v[:, j], minlength=n) for j in range(v.shape[1])])
    return (counts @ per.T) / (counts @ size)[:, None]


def clustered(values, index: np.ndarray, counts: np.ndarray) -> dict:
    """Mean of per-unit values, with a 95% percentile interval from resampling clusters."""
    v = np.asarray(values, dtype=float)
    lo, hi = np.percentile(replicates(v, index, counts), [2.5, 97.5])
    return {"mean": float(v.mean()), "ci": [float(lo), float(hi)], "n": len(v)}


def exact_distribution(values, clusters: list[str]) -> dict[float, float]:
    """Every resample of the clusters and its probability: the exact bootstrap
    distribution of the pooled mean, for a handful of clusters."""
    ids = sorted(set(clusters))
    n = len(ids)
    v = np.asarray(values, dtype=float)
    total = {g: float(v[[c == g for c in clusters]].sum()) for g in ids}
    size = Counter(clusters)
    out: dict[float, float] = {}
    for combo in itertools.combinations_with_replacement(ids, n):
        k = Counter(combo)
        ways = factorial(n)
        for c in k.values():
            ways //= factorial(c)
        mean = sum(k[g] * total[g] for g in k) / sum(k[g] * size[g] for g in k)
        key = round(mean, 12)
        out[key] = out.get(key, 0.0) + ways / n**n
    return out


def kendall_tau_b(x, y) -> float:
    """Kendall's tau-b between two score lists (ties counted as neither order)."""
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / denom) if denom else float("nan")


def clustered_kendall(hits: np.ndarray, reference, index, counts) -> dict:
    """Kendall's tau-b between each replicate's top-1 order of the guessers (columns of
    hits) and a fixed reference score list, with a percentile interval."""
    reps = replicates(hits, index, counts)
    taus = np.array([kendall_tau_b(r, reference) for r in reps])
    lo, hi = np.nanpercentile(taus, [2.5, 97.5])  # a replicate with all scores tied has none
    point = kendall_tau_b(np.asarray(hits, float).mean(axis=0), reference)
    return {"tau": point, "ci": [float(lo), float(hi)], "undefined": int(np.isnan(taus).sum())}
