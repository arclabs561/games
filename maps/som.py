"""Kohonen's self-organizing map ("Self-organized formation of topologically correct
feature maps", Biological Cybernetics 43, 1982).

A **map** is a grid of **units**, each holding a **weight**: a point in data space. Each
step draws one data point, finds its **best-matching unit** (the unit whose weight is
closest), and pulls that unit and its grid neighbors toward the point:

    w_u <- w_u + alpha(t) * h(u, bmu, t) * (x - w_u),

with h a Gaussian in grid distance whose width sigma(t) shrinks over training, so that
units near each other on the grid end up with weights near each other in data space.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SOM:
    """A trained map: unit grid coordinates and unit weights."""

    grid: np.ndarray  # (units, grid dims) integer coordinates
    weights: np.ndarray  # (units, data dims)
    history: dict[int, np.ndarray] = field(default_factory=dict)  # weights after some steps


def grid_coords(shape: tuple[int, ...]) -> np.ndarray:
    """Integer coordinates of every cell of a grid, in row-major order."""
    return np.array(np.unravel_index(np.arange(int(np.prod(shape))), shape)).T.astype(float)


def pca_init(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Weights on the plane (or line) of the leading principal components, spanning +-2 sd."""
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    vt = vt * np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])[:, None]
    sd = s / np.sqrt(len(x))
    span = grid.max(axis=0)
    span[span == 0] = 1
    unit = grid / span * 2 - 1  # each grid axis scaled to [-1, 1]
    dims = grid.shape[1]
    coef = unit * 2 * sd[:dims]
    return mean + (coef[:, :, None] * vt[None, :dims, :]).sum(axis=1)


def train(
    x: np.ndarray,
    shape: tuple[int, ...],
    steps: int,
    seed: int,
    alpha: tuple[float, float] = (0.5, 0.01),
    sigma: tuple[float, float] | None = None,
    init: str = "random",
    snapshots: tuple[int, ...] = (),
) -> SOM:
    """Online training; alpha and sigma fall geometrically from their first to last values.

    init "random" takes weights from randomly chosen data points; "pca" spreads them on
    the principal plane. sigma defaults to (half the longest grid side, 0.5). The weights
    after each number of steps in snapshots are kept in the map's history.
    """
    rng = np.random.default_rng(seed)
    grid = grid_coords(shape)
    if sigma is None:
        sigma = (max(shape) / 2, 0.5)
    if init == "pca":
        w = pca_init(x, grid)
    else:
        w = x[rng.integers(0, len(x), size=len(grid))].astype(float)
    grid_d2 = ((grid[:, None, :] - grid[None, :, :]) ** 2).sum(axis=-1)
    picks = rng.integers(0, len(x), size=steps)
    history = {0: w.copy()} if 0 in snapshots else {}
    for t, i in enumerate(picks):
        frac = t / max(steps - 1, 1)
        a = alpha[0] * (alpha[1] / alpha[0]) ** frac
        s = sigma[0] * (sigma[1] / sigma[0]) ** frac
        b = best_unit(w, x[i])
        h = np.exp(-grid_d2[b] / (2 * s * s))
        w += (a * h)[:, None] * (x[i] - w)
        if t + 1 in snapshots:
            history[t + 1] = w.copy()
    return SOM(grid, w, history)


def best_unit(weights: np.ndarray, point: np.ndarray) -> int:
    """Index of the unit whose weight is closest to the point (lowest index on ties)."""
    return int(np.argmin(((weights - point) ** 2).sum(axis=1)))


def best_units(som: SOM, x: np.ndarray) -> np.ndarray:
    """Best-matching unit of every data row."""
    return np.array([best_unit(som.weights, p) for p in x])


def quantization_error(som: SOM, x: np.ndarray) -> float:
    """Mean distance from each data point to its best-matching unit's weight."""
    b = best_units(som, x)
    return float(np.sqrt(((x - som.weights[b]) ** 2).sum(axis=1)).mean())


def topographic_error(som: SOM, x: np.ndarray) -> float:
    """Share of data points whose two closest units are not grid neighbors (Kiviluoto 1996).

    Neighbors are cells at grid distance 1 (sides, not corners).
    """
    bad = 0
    for p in x:
        first, second = np.argsort(((som.weights - p) ** 2).sum(axis=1), kind="stable")[:2]
        if np.abs(som.grid[first] - som.grid[second]).sum() != 1:
            bad += 1
    return bad / len(x)


def u_matrix(som: SOM, shape: tuple[int, int]) -> np.ndarray:
    """Mean weight distance from each unit to its side neighbors, laid out on the grid."""
    w = som.weights.reshape(*shape, -1)
    out = np.zeros(shape)
    for r in range(shape[0]):
        for c in range(shape[1]):
            near = [
                w[r + dr, c + dc]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]
            ]
            out[r, c] = np.mean([np.linalg.norm(w[r, c] - v) for v in near])
    return out


def positions(som: SOM, x: np.ndarray) -> np.ndarray:
    """Each data point's place on the map: the grid coordinates of its best unit."""
    return som.grid[best_units(som, x)]


def line_order(som: SOM, x: np.ndarray) -> list[int]:
    """Order of data rows along a one-dimensional map.

    Rows are sorted by best unit. Rows sharing a unit are sorted by where they fall
    between the units on either side: by d(previous) - d(next), nearer the previous first.
    """
    b = best_units(som, x)
    m = len(som.weights)

    def lean(i: int) -> float:
        prev = som.weights[max(b[i] - 1, 0)]
        nxt = som.weights[min(b[i] + 1, m - 1)]
        return float(np.linalg.norm(x[i] - prev) - np.linalg.norm(x[i] - nxt))

    return sorted(range(len(x)), key=lambda i: (b[i], lean(i), i))
