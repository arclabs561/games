"""Stochastic neighbor embedding (Hinton and Roweis 2002) and t-SNE (van der Maaten and Hinton 2008).

Each data point i spreads one unit of probability over the others, p(j | i), with a
Gaussian whose width is chosen so that the **perplexity** 2^H(P_i) equals a target: the
effective number of neighbors. A map does the same with its own points, q(j | i), and
the map is moved to make the two sets of distributions agree, measured by the
Kullback-Leibler divergence. SNE uses a Gaussian on the map; t-SNE symmetrizes the p's
into one joint distribution and uses a Student t with one degree of freedom, whose
heavy tail lets moderately distant data points sit far apart on the map.

The loops use only elementwise numpy operations, no matrix products, and the exp and log
of maps.portable, so that a run gives the same bits on any machine with the same numpy.
"""

from dataclasses import dataclass, field

import numpy as np

from maps.portable import exp, log, log2


def sq_distances(y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows, one coordinate at a time (no BLAS)."""
    out = np.zeros((len(y), len(y)))
    for c in range(y.shape[1]):
        out += (y[:, c, None] - y[None, :, c]) ** 2
    return out


def row_distribution(d2: np.ndarray, beta: float, i: int) -> np.ndarray:
    """p(. | i) for precision beta = 1 / (2 sigma^2), with p(i | i) = 0."""
    d = d2.copy()
    d[i] = np.inf
    p = exp(-beta * (d - d.min()))  # shifted for stability; the shift cancels
    return p / p.sum()


def entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits, with 0 log 0 = 0."""
    nz = p[p > 0]
    return float(-(nz * log2(nz)).sum())


def conditional_p(
    d2: np.ndarray, perplexity: float, tol: float = 1e-10, max_steps: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Rows p(. | i) of the given perplexity, by bisection on each row's precision beta.

    Entropy falls as beta rises, so bisection on log beta finds the unique root. Returns
    the matrix and the betas.
    """
    n = len(d2)
    target = float(log2(perplexity))
    p = np.zeros((n, n))
    betas = np.empty(n)
    for i in range(n):
        lo, hi = -50.0, 50.0  # bounds on log beta, relative to the scale of the row
        scale = np.median(np.delete(d2[i], i))
        for _ in range(max_steps):
            mid = (lo + hi) / 2
            beta = float(exp(mid)) / scale
            row = row_distribution(d2[i], beta, i)
            h = entropy_bits(row)
            if abs(h - target) < tol:
                break
            if h > target:
                lo = mid
            else:
                hi = mid
        p[i], betas[i] = row, beta
    return p, betas


def joint_p(p_cond: np.ndarray) -> np.ndarray:
    """Symmetrized joint probabilities p_ij = (p(j | i) + p(i | j)) / 2n."""
    return (p_cond + p_cond.T) / (2 * len(p_cond))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) summed over all entries, with 0 log 0 = 0.

    A q that underflowed to 0 where p > 0 (early in SNE) is read as 1e-300.
    """
    m = p > 0
    return float((p[m] * log(p[m] / np.maximum(q[m], 1e-300))).sum())


def _pull(force: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sum_j force[i, j] (y_i - y_j) for each i, one coordinate at a time (no BLAS)."""
    total = force.sum(axis=1)
    return np.column_stack(
        [y[:, c] * total - (force * y[None, :, c]).sum(axis=1) for c in range(y.shape[1])]
    )


def sne_q_grad(y: np.ndarray, p_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SNE: Gaussian q(j | i) of variance 1/2, and the gradient of sum_i KL(P_i || Q_i),

    dC/dy_i = 2 sum_j (p(j|i) - q(j|i) + p(i|j) - q(i|j)) (y_i - y_j).
    """
    d2 = sq_distances(y)
    np.fill_diagonal(d2, np.inf)
    w = exp(-(d2 - d2.min(axis=1, keepdims=True)))  # shifted per row; the shift cancels
    q = w / w.sum(axis=1, keepdims=True)
    m = p_cond - q
    return q, 2 * _pull(m + m.T, y)


def ssne_q_grad(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric SNE: joint Gaussian q_ij; gradient of KL(P || Q) is 4 sum_j (p_ij - q_ij)(y_i - y_j)."""
    d2 = sq_distances(y)
    np.fill_diagonal(d2, np.inf)
    w = exp(-(d2 - d2.min()))  # shifted; the shift cancels
    q = w / w.sum()
    return q, 4 * _pull(p - q, y)


def tsne_q_grad(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """t-SNE: q_ij proportional to 1 / (1 + |y_i - y_j|^2); the gradient of KL(P || Q) is

    dC/dy_i = 4 sum_j (p_ij - q_ij) (y_i - y_j) / (1 + |y_i - y_j|^2).
    """
    w = 1.0 / (1.0 + sq_distances(y))
    np.fill_diagonal(w, 0.0)
    q = w / w.sum()
    return q, 4 * _pull((p - q) * w, y)


Q_GRAD = {"sne": sne_q_grad, "ssne": ssne_q_grad, "tsne": tsne_q_grad}


def cost_grad(kind: str, y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """The cost KL(P || Q) (summed over rows for SNE) and its gradient."""
    q, grad = Q_GRAD[kind](y, p)
    return kl(p, q), grad


@dataclass
class Run:
    """An embedding and the cost after each recorded iteration."""

    y: np.ndarray
    costs: list[float] = field(default_factory=list)


def embed(
    x_d2: np.ndarray,
    kind: str = "tsne",
    dims: int = 2,
    perplexity: float = 30.0,
    seed: int = 0,
    iterations: int = 1000,
    learning_rate: float = 100.0,
    exaggeration: float = 4.0,
    exaggerate_until: int = 50,
    record_every: int = 50,
    p_cond: np.ndarray | None = None,
) -> Run:
    """Minimize the chosen cost with van der Maaten and Hinton's optimizer (section 3.4).

    Start from N(0, 1e-4 I); multiply P by 4 for the first 50 iterations; momentum 0.5
    before iteration 250 and 0.8 after; learning rate 100 with Jacobs's adaptive gains.
    SNE gets the same optimizer, without the annealing noise of Hinton and Roweis, and
    with the learning rate divided by 2n: its n conditional distributions carry 2n times
    the mass of the joint one, so its gradient is about 2n times as large.
    x_d2 holds squared distances in the data; p_cond, if given, is
    conditional_p(x_d2, perplexity)[0], computed once for several runs.
    """
    rng = np.random.default_rng(seed)
    n = len(x_d2)
    if p_cond is None:
        p_cond, _ = conditional_p(x_d2, perplexity)
    target = p_cond if kind == "sne" else joint_p(p_cond)
    q_grad = Q_GRAD[kind]
    if kind == "sne":
        learning_rate /= 2 * n
    y = rng.normal(0.0, 1e-2, size=(n, dims))
    step = np.zeros_like(y)
    gains = np.ones_like(y)
    run = Run(y)
    for t in range(iterations):
        boost = exaggeration if t < exaggerate_until else 1.0
        _, grad = q_grad(y, target * boost)
        momentum = 0.5 if t < 250 else 0.8
        same = (grad > 0) == (step > 0)
        gains = np.maximum(np.where(same, gains * 0.8, gains + 0.2), 0.01)
        step = momentum * step - learning_rate * gains * grad
        y = y + step
        y = y - y.mean(axis=0)
        if t % record_every == 0 or t == iterations - 1:
            run.costs.append(kl(target, q_grad(y, target)[0]))
    run.y = y
    return run
