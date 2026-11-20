"""A rating from one game: the likelihood of a player's moves under Maia-3 at each rating.

For a grid of ratings R, the **log-likelihood** of a player's game is the sum over their moves
of log P(move | position, the player rated R, the opponent rated R). Moves are treated as
independent given the positions, which they are not, so the likelihood interval is a nominal
one; its coverage on real games is measured, not assumed.

- The **estimate** is the grid maximum, refined by a parabola through it and its neighbors.
- The **likelihood interval** holds the ratings whose log-likelihood is within 1.92 of the
  maximum (a 95% interval if the moves were independent and the model were right).
- The **posterior** uses a flat prior over the grid; its mean and 2.5% and 97.5% points give a
  second estimate and interval.

Everything here is plain arithmetic on a (moves, ratings) array of log-probabilities.
"""

import numpy as np

GRID = tuple(range(600, 2601, 100))
DROP = 1.920729  # half the 95% point of chi-squared with one degree of freedom


def parabola_peak(xs: np.ndarray, ys: np.ndarray) -> float:
    """The x of the grid maximum, refined by the parabola through it and its two neighbors."""
    i = int(np.argmax(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i])
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom >= 0:
        return float(xs[i])
    step = xs[i + 1] - xs[i]
    return float(xs[i] + step * (y0 - y2) / (2 * denom))


def likelihood_interval(xs: np.ndarray, ys: np.ndarray, drop: float = DROP) -> tuple[float, float]:
    """The ratings within `drop` of the maximum log-likelihood, by linear interpolation on the
    grid; an end of the grid is returned when the interval reaches it."""
    top = ys.max()
    inside = np.nonzero(ys >= top - drop)[0]
    lo_i, hi_i = inside[0], inside[-1]

    def cross(i: int, j: int) -> float:
        # the point between grid points i (inside) and j (outside) where ys = top - drop
        t = (ys[i] - (top - drop)) / (ys[i] - ys[j])
        return float(xs[i] + t * (xs[j] - xs[i]))

    lo = float(xs[0]) if lo_i == 0 else cross(lo_i, lo_i - 1)
    hi = float(xs[-1]) if hi_i == len(xs) - 1 else cross(hi_i, hi_i + 1)
    return lo, hi


def posterior(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Mean and central 95% interval of the posterior with a flat prior on the grid points."""
    w = np.exp(ys - ys.max())
    w /= w.sum()
    cdf = np.cumsum(w)
    return (
        float(w @ xs),
        float(xs[np.searchsorted(cdf, 0.025)]),
        float(xs[np.searchsorted(cdf, 0.975)]),
    )


def estimate(logp: np.ndarray, grid: tuple[int, ...] = GRID) -> dict[str, float]:
    """Estimate, intervals and posterior mean from per-move log-probabilities (moves, ratings)."""
    xs = np.array(grid, dtype=float)
    ys = logp.sum(axis=0)
    lo, hi = likelihood_interval(xs, ys)
    mean, plo, phi = posterior(xs, ys)
    return {
        "estimate": parabola_peak(xs, ys),
        "lo": lo,
        "hi": hi,
        "posterior_mean": mean,
        "posterior_lo": plo,
        "posterior_hi": phi,
        "moves": int(logp.shape[0]),
    }


BANDS = ((600, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 2700))


def band(rating: float) -> str:
    """The rating band a true rating falls in, as 'lo-hi'."""
    for lo, hi in BANDS:
        if lo <= rating < hi:
            return f"{lo}-{hi}"
    return "other"


def bootstrap_mean(values: np.ndarray, seed: int, resamples: int = 10_000) -> tuple[float, float]:
    """95% percentile bootstrap interval of a mean."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
