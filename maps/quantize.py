"""How many bits per dimension keep the guesser's pick: quantized word vectors.

The question of guesses.py, with the stored word table quantized instead of projected.
The guesser keeps the clue's own vector and ranks the 399 other words by their
approximate squared distance to it, which for unit vectors is 2 - 2 cos. Three ways to
store a word in b bits per dimension:

- **RaBitQ**, from qntz (github.com/arclabs561/qntz, the Python binding, version 0.1.3):
  a seeded random rotation, then b-bit codes with per-word correction factors, and an
  estimate of the distance from a full-precision query to the original vector.
- **Rounding**: each dimension split into 2^b equal cells between its smallest and largest
  value over the 400 words, and stored as the midpoint of its cell.
- **Rounding after a rotation**: the same, after a seeded random orthogonal rotation of
  all vectors, which spreads each vector's weight over the dimensions as RaBitQ's does.

The share of picks changed is exact over all clues and boards, by guesses.same_pick.

    uv run python -m maps.study quantize   # this section (about 20 minutes)
"""

from functools import cache

import numpy as np
import qntz

from maps import data, guesses
from maps.ranktsne import summary

BITS = [1, 2, 3, 4, 5, 6, 8]
SEEDS = list(range(10))
METHODS = ("RaBitQ", "rounding", "rotated rounding")


@cache
def unit_words() -> np.ndarray:
    """The 400 word vectors scaled to length 1 (they are stored as float16)."""
    _, x = data.codenames_words()
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sq_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distance from each row of a to each row of b."""
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def round_to_cells(x: np.ndarray, bits: int) -> np.ndarray:
    """Each column cut into 2^bits equal cells between its min and max; values become the
    midpoints of their cells."""
    lo, hi = x.min(axis=0), x.max(axis=0)
    cells = 2**bits
    step = (hi - lo) / cells
    code = np.clip(np.floor((x - lo) / step), 0, cells - 1)
    return lo + (code + 0.5) * step


def rotation(d: int, seed: int) -> np.ndarray:
    """A random orthogonal d x d matrix (QR of a Gaussian matrix, signs fixed)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def rabitq_distances(x: np.ndarray, bits: int, seed: int) -> np.ndarray:
    """RaBitQ's estimate of |clue - word|^2 for every clue (full precision) and word (coded).

    qntz's fit needs a 2-D float32 array; a list of lists raises TypeError in 0.1.3.
    """
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    q = qntz.RaBitQQuantizer(x.shape[1], seed=seed, total_bits=bits)
    q.fit(x32)
    coded = [q.quantize(v) for v in x32]
    return np.array([[q.approximate_l2_sqr(c, w) for w in coded] for c in x32])


def distances(method: str, bits: int, seed: int = 0) -> np.ndarray:
    """Squared distances from full-precision clues to stored words, by one method."""
    x = unit_words()
    if method == "RaBitQ":
        return rabitq_distances(x, bits, seed)
    if method == "rounding":
        return sq_distances(x, round_to_cells(x, bits))
    if method == "rotated rounding":
        rot = rotation(x.shape[1], seed)
        return sq_distances(x @ rot, round_to_cells(x @ rot, bits))
    raise ValueError(method)


@cache
def original_ranks() -> np.ndarray:
    return guesses.clue_ranks(guesses.cosines(unit_words()))


def change_rate(d2: np.ndarray) -> float:
    """Exact share of clue-board pairs whose pick by smallest distance differs from the
    pick by the original cosines (ties go to the lower index)."""
    return float(1 - guesses.same_pick(original_ranks(), guesses.clue_ranks(-d2)).mean())


def section() -> dict:
    """Change rate for each method and bit count: 10 seeds where the method has one."""
    x = unit_words()
    out: dict = {"bits": BITS, "seeds": SEEDS, "methods": {}}
    for method in METHODS:
        rows = []
        for b in BITS:
            if method == "rounding":
                rows.append({"rate": change_rate(distances(method, b))})
            else:
                rates = [change_rate(distances(method, b, s)) for s in SEEDS]
                rows.append({"change": summary(rates), "rates": rates})
        out["methods"][method] = rows
    out["float32"] = change_rate(sq_distances(x.astype(np.float32), x.astype(np.float32)))
    return out
