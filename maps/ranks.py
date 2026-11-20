"""How well a map keeps neighbors: trustworthiness, continuity, precision and recall.

Venna and Kaski ("Neighborhood preservation in nonlinear projection methods", ICANN 2001;
"Local multidimensional scaling", Neural Networks 19, 2006) define two scores for a
neighborhood of size k. A map is **trustworthy** if the points shown near a point are
near it in the data, and **continuous** if the points near it in the data are shown near
it. With r(i, j) the rank of j among i's neighbors in the data, U_k(i) the points among
i's k nearest on the map but not in the data, and V_k(i) the reverse,

    T(k) = 1 - 2 / (N k (2N - 3k - 1)) * sum_i sum_{j in U_k(i)} (r(i, j) - k),
    C(k) = the same with V_k(i) and the ranks on the map.

Both are 1 for a map that keeps every k-neighborhood. A map with ties (a shelf, where a
book has a neighbor on each side, or a self-organizing map, where several items share a
cell) is scored as Venna and Kaski did: the best and worst orders of the tied points are
both computed and averaged.

Venna et al. ("Information retrieval perspective to nonlinear dimensionality reduction
for data visualization", JMLR 11, 2010) read the same two kinds of error as precision
(points retrieved from the map that are not true neighbors) and recall (true neighbors
not retrieved).
"""

import numpy as np

from maps.colors import distances


def rank_matrix(dist: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """ranks[i, j] = 1 + the number of points closer to i than j (self has rank 0).

    Ties in dist[i] are ordered by tiebreak[i] (smaller first), then by index.
    """
    n = len(dist)
    key2 = np.zeros_like(dist) if tiebreak is None else tiebreak
    ranks = np.empty((n, n), dtype=np.int64)
    idx = np.arange(n)
    for i in range(n):
        d = dist[i].copy()
        d[i] = -np.inf
        order = np.lexsort((idx, key2[i], d))
        ranks[i, order] = np.arange(n)
    return ranks


def _score(near_rank: np.ndarray, far_rank: np.ndarray, k: int) -> float:
    """1 - normalized sum of (far_rank - k) over pairs with far_rank > k >= near_rank."""
    n = len(near_rank)
    if not 1 <= k < n / 2:
        raise ValueError(f"k = {k} must be at least 1 and below N/2 = {n / 2}")
    mask = (near_rank <= k) & (near_rank > 0) & (far_rank > k)
    penalty = (far_rank[mask] - k).sum()
    return float(1 - 2 * penalty / (n * k * (2 * n - 3 * k - 1)))


def trust_continuity(
    data_dist: np.ndarray, map_dist: np.ndarray, ks: list[int]
) -> dict[str, list[float]]:
    """Trustworthiness and continuity at each k, averaged over the best and worst tie orders.

    Ties on the map are ordered by data rank: nearest-in-data first for the best case,
    last for the worst. Returns the averages and both cases.
    """
    r_data = rank_matrix(data_dist)
    best = rank_matrix(map_dist, tiebreak=r_data)
    worst = rank_matrix(map_dist, tiebreak=-r_data)
    out: dict[str, list[float]] = {}
    for name, r_map in (("best", best), ("worst", worst)):
        out[f"trust_{name}"] = [_score(r_map, r_data, k) for k in ks]
        out[f"cont_{name}"] = [_score(r_data, r_map, k) for k in ks]
    for s in ("trust", "cont"):
        out[s] = [(a + b) / 2 for a, b in zip(out[f"{s}_best"], out[f"{s}_worst"], strict=True)]
    return out


def trustworthiness(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """T(k) for data rows x shown at map rows y (the average over tie orders)."""
    return trust_continuity(distances(x), distances(y), [k])["trust"][0]


def continuity(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """C(k) for data rows x shown at map rows y (the average over tie orders)."""
    return trust_continuity(distances(x), distances(y), [k])["cont"][0]


def line_distances(order: list[int]) -> np.ndarray:
    """Distances between books on a shelf: how many places apart they stand."""
    pos = np.empty(len(order))
    pos[order] = np.arange(len(order))
    return np.abs(pos[:, None] - pos[None, :])


def precision_recall(
    data_dist: np.ndarray, map_dist: np.ndarray, relevant: int, retrieved: list[int]
) -> tuple[list[float], list[float]]:
    """Mean precision and recall when the `relevant` nearest points in the data are the
    true neighbors and the k nearest on the map are retrieved, for each k (Venna et al.
    2010, section 2.1). The map must have no ties."""
    r_data, r_map = rank_matrix(data_dist), rank_matrix(map_dist)
    true = (r_data >= 1) & (r_data <= relevant)
    precision, recall = [], []
    for k in retrieved:
        got = (r_map >= 1) & (r_map <= k)
        hits = (true & got).sum(axis=1)
        precision.append(float((hits / k).mean()))
        recall.append(float((hits / relevant).mean()))
    return precision, recall
