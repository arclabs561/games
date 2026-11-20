"""Elementary functions and principal axes that give the same bits on every machine.

numpy's exp, log, log2, cbrt and power pick a different implementation for each CPU
(AVX-512 kernels on some, the C library on others), and the results can differ in the
last bit. LAPACK's SVD and BLAS's dot products differ the same way, through the kernel
that OpenBLAS picks for the CPU. SNE and t-SNE amplify such differences until whole
results move, so the maps package computes these functions here instead, from addition,
subtraction, multiplication, division, square roots and exact scaling by powers of two.
IEEE 754 fixes the result of each of those operations, so every machine computes the
same bits. Accuracy is within a few units in the last place.
"""

from math import factorial

import numpy as np

# ln 2 split into a high part with 21 trailing zero bits, so that k * LN2_HI is exact for
# |k| < 2^11, and the rest (Cody and Waite's reduction, as in fdlibm).
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
LN2 = 0.6931471805599453
INV_LN2 = 1.4426950408889634
SQRT_HALF = 0.7071067811865476
PI = 3.141592653589793
MAX_EXP = 709.782712893384  # the largest x with e^x below the largest float

EXP_TERMS = [1.0 / factorial(n) for n in range(14)]  # Taylor series to r^13 / 13!
ATAN_TERMS = [1.0 / (2 * j + 1) for j in range(10)]  # atan(z) / z in powers of -z^2
ATANH_TERMS = [1.0 / (2 * j + 1) for j in range(11)]  # atanh(f) / f in powers of f^2


def _horner(coefficients: list[float], r: np.ndarray) -> np.ndarray:
    """sum_n coefficients[n] r^n."""
    out = np.full_like(r, coefficients[-1])
    for c in reversed(coefficients[:-1]):
        out *= r
        out += c
    return out


def exp(x) -> np.ndarray:
    """e^x: x = k ln 2 + r with |r| <= ln 2 / 2, a Taylor series for e^r, then times 2^k."""
    x = np.asarray(x, dtype=float)
    # e^-746 rounds to 0, and e^x overflows for x above MAX_EXP.
    clipped = np.clip(np.where(np.isnan(x), 0.0, x), -746.0, MAX_EXP)
    k = np.rint(clipped * INV_LN2)
    r = (clipped - k * LN2_HI) - k * LN2_LO
    out = np.ldexp(_horner(EXP_TERMS, r), k.astype(np.int32))
    return np.where(np.isnan(x), np.nan, np.where(x > MAX_EXP, np.inf, out))


def _log_parts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = 2^e m with sqrt(1/2) <= m < sqrt(2): returns e and ln m = 2 atanh((m - 1) / (m + 1))."""
    m, e = np.frexp(x)
    low = m < SQRT_HALF
    m = np.where(low, 2 * m, m)
    e = (e - low).astype(float)
    f = (m - 1) / (m + 1)
    return e, 2 * f * _horner(ATANH_TERMS, f * f)


def _special_log(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    """log(0) = -inf, log(inf) = inf, and nan for negative x."""
    out = np.where(x == 0, -np.inf, out)
    out = np.where(x == np.inf, np.inf, out)
    return np.where((x < 0) | np.isnan(x), np.nan, out)


def log(x) -> np.ndarray:
    """Natural logarithm."""
    x = np.asarray(x, dtype=float)
    safe = np.where((x > 0) & (x < np.inf), x, 1.0)
    e, ln_m = _log_parts(safe)
    return _special_log(x, e * LN2_HI + (ln_m + e * LN2_LO))


def log2(x) -> np.ndarray:
    """Base-2 logarithm."""
    x = np.asarray(x, dtype=float)
    safe = np.where((x > 0) & (x < np.inf), x, 1.0)
    e, ln_m = _log_parts(safe)
    return _special_log(x, e + ln_m * INV_LN2)


def power(x, y: float) -> np.ndarray:
    """x^y for x >= 0 and y >= 0: x^n by multiplication for the whole part n of y, times
    e^(f ln x) for the fraction f, which keeps the exponent small and the error low."""
    x = np.asarray(x, dtype=float)
    whole, frac = divmod(y, 1.0)
    out = exp(frac * log(np.where(x == 0, 1.0, x)))
    for _ in range(int(whole)):
        out = out * x
    return np.where(x == 0, 0.0 if y > 0 else 1.0, out)


def cbrt(x) -> np.ndarray:
    """Real cube root: e^(ln|x| / 3), then one Newton step, with the sign of x."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    safe = np.where(a > 0, a, 1.0)
    y = exp(log(safe) / 3)
    y = y - (y * y * y - safe) / (3 * y * y)
    return np.where(a > 0, np.copysign(y, x), x)


def atan2(y, x) -> np.ndarray:
    """The angle of the point (x, y) in radians, in [-pi, pi] (x = -0 counts as +0).

    atan of z = min / max in [0, 1], after halving the angle twice by
    atan z = 2 atan(z / (1 + sqrt(1 + z^2))), so that the series in z converges fast.
    """
    y, x = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
    ay, ax = np.abs(y), np.abs(x)
    big = np.maximum(ay, ax)
    z = np.minimum(ay, ax) / np.where(big > 0, big, 1.0)
    for _ in range(2):
        z = z / (1 + np.sqrt(1 + z * z))
    angle = 4 * z * _horner(ATAN_TERMS, -(z * z))  # atan of min / max, in [0, pi / 4]
    angle = np.where(ay > ax, PI / 2 - angle, angle)
    angle = np.where(x < 0, PI - angle, angle)
    return np.copysign(angle, y)


def dot_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b.T for 2D arrays, as elementwise products summed in a fixed order (no BLAS)."""
    return np.stack([(a * row).sum(axis=1) for row in b], axis=1)


def symmetric_eigen(a: np.ndarray, sweeps: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and unit eigenvectors (columns) of a symmetric matrix.

    Jacobi's method: each rotation zeroes one off-diagonal pair, and in each round of a
    sweep the pairs form a round-robin tournament, so that n / 2 disjoint rotations are
    applied at once. Sweeps stop when every off-diagonal entry is below 1e-15 times the
    largest diagonal one.
    """
    a = np.array(a, dtype=float)
    n = len(a)
    v = np.eye(n)
    players = list(range(n)) + ([-1] if n % 2 else [])  # -1 sits out a round
    m = len(players)
    for _ in range(sweeps):
        off = np.abs(a - np.diag(np.diag(a))).max(initial=0.0)
        if off <= 1e-15 * np.abs(np.diag(a)).max(initial=0.0):
            break
        for _ in range(m - 1):
            pairs = [(players[i], players[m - 1 - i]) for i in range(m // 2)]
            pairs = [(min(p, q), max(p, q)) for p, q in pairs if p >= 0 and q >= 0]
            p = np.array([p for p, _ in pairs], dtype=int)
            q = np.array([q for _, q in pairs], dtype=int)
            apq = a[p, q]
            nonzero = apq != 0
            theta = (a[q, q] - a[p, p]) / np.where(nonzero, 2 * apq, 1.0)
            t = np.where(theta >= 0, 1.0, -1.0) / (np.abs(theta) + np.sqrt(theta * theta + 1))
            t = np.where(nonzero, t, 0.0)
            c = 1 / np.sqrt(t * t + 1)
            s = t * c
            ap, aq = a[p, :], a[q, :]
            a[p, :], a[q, :] = c[:, None] * ap - s[:, None] * aq, s[:, None] * ap + c[:, None] * aq
            ap, aq = a[:, p], a[:, q]
            a[:, p], a[:, q] = ap * c - aq * s, ap * s + aq * c
            vp, vq = v[:, p], v[:, q]
            v[:, p], v[:, q] = vp * c - vq * s, vp * s + vq * c
            players = [players[0], players[-1], *players[1:-1]]
    values = np.diag(a).copy()
    order = np.argsort(-values, kind="stable")
    return values[order], v[:, order]


def principal_axes(x: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviations along the leading principal axes of the rows of x, and the axes
    as rows, each signed so that its largest loading is positive."""
    xc = x - x.mean(axis=0)
    cov = np.stack([(xc * xc[:, j, None]).sum(axis=0) for j in range(x.shape[1])]) / len(x)
    values, vectors = symmetric_eigen(cov)
    axes = vectors[:, :dims].T
    axes = (
        axes * np.where(axes[np.arange(dims), np.abs(axes).argmax(axis=1)] < 0, -1.0, 1.0)[:, None]
    )
    return np.sqrt(np.maximum(values[:dims], 0.0)), axes
