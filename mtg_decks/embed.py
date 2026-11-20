"""Card embeddings from deck co-occurrence, and tests of what their neighbors share.

Two methods:

- **PPMI + SVD.** Positive pointwise mutual information over decks,
  PPMI[i, j] = max(0, log(C[i, j] * N / (n_i * n_j))), with C the number of decks holding
  both cards, n_i the number holding card i and N the number of decks. The embedding is
  U_k * sqrt(L_k) for the k largest eigenvalues of this symmetric matrix, the symmetric
  form of the truncated SVD that Levy, Goldberg and Dagan (2015) use. Exact and
  deterministic: no seed.

Logarithms, exponentials, the eigenvectors and dot products come from `maps.portable`,
which computes them from IEEE-exact operations, so every machine gets the same bits and
the committed results can be compared byte for byte.
- **Skip-gram with negative sampling** (word2vec), with each deck a sentence and the
  whole deck as the window. Its positive pairs are then the pairs of cards in a deck, so
  they can be drawn straight from C, which is what `sgns` does. Levy and Goldberg (2014)
  showed its optimum satisfies w_i . c_j = PMI(i, j) - log k on the pair distribution;
  `tests/test_mtg_decks.py` checks that on a small case. Learned: seeded.
"""

import numpy as np

from maps.portable import dot_rows, exp, log, symmetric_eigen

# ---- PPMI and SVD ----------------------------------------------------------------


def ppmi(s: np.ndarray, held: np.ndarray, n_decks: int) -> np.ndarray:
    """Positive PMI over decks from co-occurrence counts s, per-card counts and N."""
    s = s.astype(np.float64)
    expected = np.outer(held, held) / n_decks
    ratio = np.divide(s, expected, out=np.zeros_like(s), where=s > 0)
    pmi = np.zeros_like(s)
    pmi[ratio > 0] = log(ratio[ratio > 0])
    out = np.maximum(pmi, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def eigen_embedding(m: np.ndarray, dim: int) -> np.ndarray:
    """Rows of U_k * sqrt(max(L_k, 0)) for the dim largest eigenvalues, with signs fixed.

    The sign of each column is set so that its largest entry in absolute value is positive.
    """
    values, vectors = symmetric_eigen(m)
    u = vectors[:, :dim] * np.sqrt(np.maximum(values[:dim], 0.0))
    for k in range(u.shape[1]):
        if u[np.argmax(np.abs(u[:, k])), k] < 0:
            u[:, k] = -u[:, k]
    return u


# ---- Skip-gram with negative sampling ---------------------------------------------


def sgns(
    s: np.ndarray,
    dim: int,
    seed: int,
    *,
    negatives: int = 5,
    steps: int = 4000,
    batch: int = 1024,
    lr: float = 0.2,
    power: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Train skip-gram with negative sampling on pairs drawn with probability ∝ s[i, j].

    Negatives are drawn ∝ (pair-count marginal)^power, as word2vec does with unigram
    counts. Plain minibatch SGD with a learning rate that falls linearly to zero.
    Returns the word and context vectors.
    """
    rng = np.random.default_rng(seed)
    n = s.shape[0]
    flat = np.triu(s, 1).ravel().astype(np.float64)
    pair_p = flat / flat.sum()
    marginal = s.sum(axis=1).astype(np.float64)
    neg_p = marginal**power
    neg_p /= neg_p.sum()
    w = (rng.random((n, dim)) - 0.5) / dim
    c = np.zeros((n, dim))
    for step in range(steps):
        rate = lr * (1 - step / steps)
        idx = rng.choice(flat.size, size=batch, p=pair_p)
        i, j = np.divmod(idx, n)
        flip = rng.random(batch) < 0.5  # each unordered pair is seen in both directions
        i, j = np.where(flip, j, i), np.where(flip, i, j)
        neg = rng.choice(n, size=(batch, negatives), p=neg_p)
        wi = w[i]
        # positive pair: gradient of log sigmoid(w_i . c_j)
        g_pos = 1 - _sigmoid((wi * c[j]).sum(axis=1))
        # negatives: gradient of log sigmoid(-w_i . c_n)
        cn = c[neg]
        g_neg = -_sigmoid((wi[:, None, :] * cn).sum(axis=2))
        grad_w = g_pos[:, None] * c[j] + (g_neg[:, :, None] * cn).sum(axis=1)
        gc = np.zeros_like(c)
        np.add.at(gc, j, g_pos[:, None] * wi)
        np.add.at(gc, neg, g_neg[:, :, None] * wi[:, None, :])
        gw = np.zeros_like(w)
        np.add.at(gw, i, grad_w)
        # each row moves by its mean gradient over the batch, so a card drawn often in
        # one batch does not take a step many times the learning rate
        nc = np.bincount(np.concatenate([j, neg.ravel()]), minlength=n)
        nw = np.bincount(i, minlength=n)
        c += rate * gc / np.maximum(nc, 1)[:, None]
        w += rate * gw / np.maximum(nw, 1)[:, None]
    return w, c


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + exp(-np.clip(x, -30, 30)))


def pair_pmi(s: np.ndarray) -> np.ndarray:
    """PMI on the pair distribution that `sgns` samples, with marginals Σ_j s[i, j]."""
    s = s.astype(np.float64)
    total = s.sum()
    row = s.sum(axis=1)
    ratio = s * total / np.outer(row, row)
    out = np.full_like(ratio, -np.inf)
    out[ratio > 0] = log(ratio[ratio > 0])
    return out


# ---- The shuffled-deck null --------------------------------------------------------


def null_counts(held: np.ndarray, sizes: np.ndarray, seed: int, trades: int = 10) -> np.ndarray:
    """Random 0/1 decks with exactly the real deck sizes and the real per-card deck counts.

    Card i is in held[i] decks and deck d holds sizes[d] distinct cards, as in the data;
    everything else is random. A first matrix with these margins is built by Ryser's
    construction (each deck, largest first, takes the cards with most copies left), then
    mixed by the curveball algorithm (Strona et al. 2014): two decks pool the cards that
    only one of them holds and deal them back at random, keeping both sizes. `trades`
    such trades per deck are made. Returns a decks x cards 0/1 matrix.
    """
    rng = np.random.default_rng(seed)
    n, m = sizes.size, held.size
    assert held.sum() == sizes.sum(), "card totals and deck sizes must agree"
    left = held.astype(np.int64).copy()
    rows: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n
    for d in np.argsort(-sizes, kind="stable"):
        pick = np.argsort(-left, kind="stable")[: sizes[d]]
        assert np.all(left[pick] > 0), "no 0/1 matrix has these margins"
        left[pick] -= 1
        rows[d] = pick
    sets = [set(r.tolist()) for r in rows]
    for _ in range(trades * n):
        a, b = rng.choice(n, size=2, replace=False)
        only_a, only_b = sets[a] - sets[b], sets[b] - sets[a]
        if not only_a or not only_b:
            continue
        pool = np.array(sorted(only_a | only_b))
        rng.shuffle(pool)
        k = len(only_a)
        shared = sets[a] & sets[b]
        sets[a] = shared | set(pool[:k].tolist())
        sets[b] = shared | set(pool[k:].tolist())
    out = np.zeros((n, m), dtype=np.int8)
    for d, s in enumerate(sets):
        out[d, sorted(s)] = 1
    return out


# ---- Neighbors and what they share -------------------------------------------------


def cosine(x: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows; a zero row has similarity 0 with every row."""
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True))
    u = x / np.where(norm == 0, 1.0, norm)
    return dot_rows(u, u)


def neighbors(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest rows by cosine, self excluded, ties by index."""
    sim = cosine(x)
    np.fill_diagonal(sim, -np.inf)
    # a stable sort on -sim breaks ties toward the lower index
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def agreement(nbrs: np.ndarray, labels: list[str | None]) -> float:
    """Mean share of a card's neighbors with the card's label, over labeled cards.

    Neighbors without a label count as disagreeing; cards without one are skipped.
    """
    shares = [
        np.mean([labels[j] == labels[i] for j in nbrs[i]])
        for i in range(len(labels))
        if labels[i] is not None
    ]
    return float(np.mean(shares))


def chance_agreement(labels: list[str | None]) -> float:
    """Expected `agreement` if each card's neighbors were drawn at random from the rest."""
    n = len(labels)
    shares = []
    for i, li in enumerate(labels):
        if li is None:
            continue
        same = sum(1 for j, lj in enumerate(labels) if j != i and lj == li)
        shares.append(same / (n - 1))
    return float(np.mean(shares))
