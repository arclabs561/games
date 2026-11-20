"""How often a random projection changes the word that an embedding guesser picks.

The **guesser** of codenames.players.embedding_guesser sees a clue and a board of 25 words
and ranks the board by cosine with the clue; its **pick** is the top word. Here the clue is
one of the 400 deck words and the **board** is any 25 of the other 399. A random map
R (256 x k, from jl.projection) moves every vector, and the **change rate** is the share
of clue-board pairs whose pick differs before and after the map.

The rate is exact over all 400 * C(399, 25) clue-board pairs, by counting. Fix a clue and
a word w. Both guessers pick w exactly when w is on the board and none of the m_w words
that beat w in either ranking is. Of the C(399, 25) boards, C(398 - m_w, 24) do that, so

    P(same pick | clue) = sum_w C(398 - m_w, 24) / C(399, 25),

with m_w = r_w + s_w - #{v : r_v < r_w and s_v < s_w}, where r and s are the ranks of
the other words (0 for the closest) by cosine before and after the map.

    uv run python -m maps.study guesses   # this section (about two minutes)
"""

from functools import cache
from itertools import combinations
from math import comb

import numpy as np

from maps import data, jl
from maps.ranktsne import summary

BOARD = 25
SEEDS = list(range(20))  # the same maps as the distortion study: default_rng(seed)
KS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536]


def cosines(x: np.ndarray) -> np.ndarray:
    """Cosine of every pair of rows."""
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    return u @ u.T


def clue_ranks(cos: np.ndarray) -> np.ndarray:
    """ranks[c, w]: how many other words are closer to clue c than w is (ties by index).

    The clue itself gets -1. Rows are clues and columns words, n x n.
    """
    n = len(cos)
    out = np.empty((n, n), dtype=np.int64)
    for c in range(n):
        key = -cos[c].copy()
        key[c] = -np.inf
        order = np.argsort(key, kind="stable")
        out[c, order] = np.arange(-1, n - 1)
    return out


@cache
def same_pick_weights(n: int, board: int) -> np.ndarray:
    """w[m] = C(n - 2 - m, board - 1) / C(n - 1, board): the chance that a random board
    holds a given word and none of m given others (n words, one of them the clue)."""
    total = comb(n - 1, board)
    return np.array([comb(n - 2 - m, board - 1) / total for m in range(n - 1)])


def same_pick(r: np.ndarray, s: np.ndarray, board: int = BOARD) -> np.ndarray:
    """For each clue, the exact chance over random boards that the two rankings pick the
    same word. r and s are clue_ranks before and after a map."""
    n = len(r)
    weights = same_pick_weights(n, board)
    out = np.empty(n)
    for c in range(n):
        keep = r[c] >= 0
        rc, sc = r[c, keep], s[c, keep]
        s_by_r = sc[np.argsort(rc)]  # projected ranks, in the order of the original ranks
        # both[i]: words ahead of the i-th word in both rankings
        both = np.tril(s_by_r[None, :] < s_by_r[:, None], -1).sum(axis=1)
        m = np.arange(n - 1) + s_by_r - both
        out[c] = weights[m].sum()
    return out


def change_rate(x: np.ndarray, y: np.ndarray, board: int = BOARD) -> float:
    """Exact share of clue-board pairs whose top pick differs between vectors x and y."""
    return float(1 - same_pick(clue_ranks(cosines(x)), clue_ranks(cosines(y)), board).mean())


def change_rate_brute(x: np.ndarray, y: np.ndarray, board: int) -> float:
    """The same share, by trying every clue and every board (small cases only)."""
    cx, cy = cosines(x), cosines(y)
    changed = total = 0
    for c in range(len(x)):
        others = [w for w in range(len(x)) if w != c]
        for b in combinations(others, board):
            b = list(b)
            changed += b[int(np.argmax(cx[c, b]))] != b[int(np.argmax(cy[c, b]))]
            total += 1
    return changed / total


def change_rate_sampled(
    x: np.ndarray, y: np.ndarray, pairs: int, seed: int, board: int = BOARD
) -> float:
    """The same share, estimated from random clue-board pairs."""
    rng = np.random.default_rng(seed)
    cx, cy = cosines(x), cosines(y)
    changed = 0
    for _ in range(pairs):
        c = int(rng.integers(len(x)))
        others = np.delete(np.arange(len(x)), c)
        b = rng.choice(others, board, replace=False)
        changed += b[np.argmax(cx[c, b])] != b[np.argmax(cy[c, b])]
    return changed / pairs


def pca_projection(x: np.ndarray, k: int) -> np.ndarray:
    """x on its k leading right singular vectors (uncentered): the best rank-k inner products."""
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:k].T


def margin_cdf(x: np.ndarray, t: float, board: int = BOARD) -> float:
    """Exact share of clue-board pairs whose pick wins by at most t in cosine over the
    runner-up. The closest two board words are the i-th and j-th closest of the n - 1
    (i < j) on C(n - 2 - j, board - 2) of the C(n - 1, board) boards."""
    cos = cosines(x)
    n = len(x)
    total = comb(n - 1, board)
    w = np.array([comb(n - 2 - j, board - 2) for j in range(n - 1)], float) / total
    upper = np.triu(np.ones((n - 1, n - 1), bool), 1)
    share = 0.0
    for c in range(n):
        a = np.sort(np.delete(cos[c], c))[::-1]
        close = upper & (a[:, None] - a[None, :] <= t)
        share += float((close * w[None, :]).sum())
    return share / n


def margin_quantile(x: np.ndarray, q: float, board: int = BOARD, tol: float = 1e-5) -> float:
    """The smallest margin t (to within tol) with margin_cdf(t) >= q, by bisection."""
    lo, hi = 0.0, 2.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if margin_cdf(x, mid, board) >= q:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


MARGIN_QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def wilson(hits: float, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a share hits / n."""
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


SAMPLED_PAIRS, SAMPLED_SEED, SAMPLED_K = 20_000, 1, 16


def section() -> dict:
    """Change rate and worst distortion for 20 maps of each kind at each k; PCA at each k."""
    _, x = data.codenames_words()
    d = x.shape[1]
    r = clue_ranks(cosines(x))
    out: dict = {"n": len(x), "board": BOARD, "k": KS, "kinds": {}}
    for kind in jl.KINDS:
        rows = []
        for k in KS:
            rates, worst = [], []
            for s in SEEDS:
                y = x @ jl.projection(d, k, kind, np.random.default_rng(s))
                rates.append(float(1 - same_pick(r, clue_ranks(cosines(y))).mean()))
                worst.append(jl.worst(jl.pair_sq_distances(y) / jl.pair_sq_distances(x)))
            rows.append(
                {"change": summary(rates), "rates": rates, "median_worst": float(np.median(worst))}
            )
        out["kinds"][kind] = rows
    out["pca"] = [change_rate(x, pca_projection(x, k)) for k in KS if k <= d]
    out["margin_quantiles"] = {str(q): margin_quantile(x, q) for q in MARGIN_QS}
    y = x @ jl.projection(d, SAMPLED_K, "gaussian", np.random.default_rng(0))
    sampled = change_rate_sampled(x, y, SAMPLED_PAIRS, SAMPLED_SEED)
    out["sampled_check"] = {
        "k": SAMPLED_K,
        "exact": out["kinds"]["gaussian"][KS.index(SAMPLED_K)]["rates"][0],
        "sampled": sampled,
        "wilson": wilson(sampled * SAMPLED_PAIRS, SAMPLED_PAIRS),
        "pairs": SAMPLED_PAIRS,
        "seed": SAMPLED_SEED,
    }
    return out
