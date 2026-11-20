"""Random projections and the Johnson-Lindenstrauss lemma.

Multiply n points in R^d by a random d x k matrix R, scaled by 1/sqrt(k). For one pair of
points, the **ratio** |f(u) - f(v)|^2 / |u - v|^2 has mean 1, and it is close to 1 with
high probability once k is large. The union bound over all n(n - 1)/2 pairs then keeps
every ratio near 1 at once. The lemma's k depends on n and on the tolerance, not on d.

Bounds used (each holds for squared distances, (1 - eps) <= ratio <= (1 + eps)):
- Dasgupta and Gupta ("An elementary proof of a theorem of Johnson and Lindenstrauss",
  Random Structures and Algorithms 22, 2003): k >= 4 ln n / (eps^2/2 - eps^3/3) gives
  success with probability at least 1/n.
- Achlioptas ("Database-friendly random projections: Johnson-Lindenstrauss with binary
  coins", JCSS 66, 2003): k >= (4 + 2 beta) ln n / (eps^2/2 - eps^3/3) gives success with
  probability at least 1 - n^-beta, for Gaussian, +-1 and sparse {+-sqrt 3, 0} entries.
"""

from math import comb, log

import numpy as np
from scipy.stats import chi2

KINDS = ("gaussian", "binary", "sparse")


def projection(d: int, k: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """A d x k matrix with independent entries of mean 0 and variance 1, divided by sqrt(k).

    gaussian: N(0, 1). binary: +1 or -1, each with chance 1/2. sparse: +sqrt(3) and
    -sqrt(3) with chance 1/6 each and 0 with chance 2/3 (Achlioptas's second matrix).
    """
    if kind == "gaussian":
        r = rng.normal(size=(d, k))
    elif kind == "binary":
        r = rng.choice([-1.0, 1.0], size=(d, k))
    elif kind == "sparse":
        r = np.sqrt(3) * rng.choice([-1.0, 0.0, 1.0], size=(d, k), p=[1 / 6, 2 / 3, 1 / 6])
    else:
        raise ValueError(kind)
    return r / np.sqrt(k)


def pair_sq_distances(x: np.ndarray) -> np.ndarray:
    """Squared distances of all pairs i < j, in the order of np.triu_indices.

    Computed as |a|^2 + |b|^2 - 2 a.b; the rounding error is far below the distortions
    measured here, since no two points coincide.
    """
    sq = (x**2).sum(axis=1)
    gram = x @ x.T
    i, j = np.triu_indices(len(x), 1)
    return sq[i] + sq[j] - 2 * gram[i, j]


def ratios(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """|f(u) - f(v)|^2 / |u - v|^2 for every pair, with f(x) = x R."""
    return pair_sq_distances(x @ r) / pair_sq_distances(x)


def worst(rat: np.ndarray) -> float:
    """The smallest eps with 1 - eps <= every ratio <= 1 + eps."""
    return float(max(1 - rat.min(), rat.max() - 1))


def k_needed(n: int, eps: float, beta: float = 0.0) -> float:
    """(4 + 2 beta) ln n / (eps^2/2 - eps^3/3): beta = 0 is Dasgupta and Gupta's bound."""
    return (4 + 2 * beta) * log(n) / (eps**2 / 2 - eps**3 / 3)


def eps_bound(n: int, k: int, beta: float = 0.0) -> float:
    """The eps that k_needed turns into k, found by bisection on (0, 1); nan if none."""
    lo, hi = 1e-9, 1.0
    if k_needed(n, hi, beta) > k:
        return float("nan")
    for _ in range(100):
        mid = (lo + hi) / 2
        if k_needed(n, mid, beta) > k:
            lo = mid
        else:
            hi = mid
    return hi


def gaussian_pair_tail(k: int, eps: float) -> float:
    """Exact chance that one pair's ratio leaves [1 - eps, 1 + eps] under a Gaussian map.

    For Gaussian entries, k times the ratio has the chi-squared distribution with k
    degrees of freedom, whatever the pair.
    """
    return float(chi2.cdf(k * (1 - eps), k) + chi2.sf(k * (1 + eps), k))


def eps_union(n: int, k: int, delta: float) -> float:
    """Smallest eps for which the union bound C(n, 2) * tail(k, eps) is at most delta."""
    pairs = comb(n, 2)
    lo, hi = 0.0, 1.0
    if pairs * gaussian_pair_tail(k, hi) > delta:
        return float("nan")
    for _ in range(100):
        mid = (lo + hi) / 2
        if pairs * gaussian_pair_tail(k, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def ratio_variance(a: np.ndarray, k: int, fourth_moment: float) -> float:
    """Exact variance of one pair's ratio for difference vector a and entries with E r^4 = m4.

    With z = sum_i r_i a_i, E z^2 = |a|^2 and E z^4 = 3|a|^4 + (m4 - 3) sum_i a_i^4, so the
    ratio, a mean of k independent z^2 / |a|^2, has variance (2 + (m4 - 3) sum a_i^4 / |a|^4) / k.
    m4 is 3 for Gaussian and sparse entries and 1 for +-1 entries.
    """
    s2 = float((a**2).sum())
    return (2 + (fourth_moment - 3) * float((a**4).sum()) / s2**2) / k


FOURTH_MOMENT = {"gaussian": 3.0, "binary": 1.0, "sparse": 3.0}
