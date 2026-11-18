"""Who is the better player, after k wins in n games?

With a uniform prior on a player's chance p of winning a game, the chance that p > 1/2 after k wins
in n games equals the chance of at most k heads in n + 1 fair coin flips. This script checks
that identity against the Beta integral, prints how many wins it takes to be 95% sure, measures
two logistic approximations, and writes peggle/better.csv for the figure in peggle.typ.

    uv run python -m peggle.peggle
"""

from fractions import Fraction
from functools import cache
from itertools import accumulate
from math import ceil, comb, exp, log, sqrt
from pathlib import Path

CSV = Path(__file__).resolve().parent / "better.csv"


def better(k: int, n: int) -> Fraction:
    """P(p > 1/2 | k wins in n games), uniform prior: P(at most k heads in n + 1 flips)."""
    return Fraction(heads_at_most(n + 1)[k], 2 ** (n + 1))


@cache
def heads_at_most(flips: int) -> list[int]:
    """Entry k: how many of the 2^flips sequences of flips have at most k heads."""
    return list(accumulate(comb(flips, j) for j in range(flips + 1)))


def beta_integral(
    k: int, n: int, lo: Fraction = Fraction(0), hi: Fraction = Fraction(1)
) -> Fraction:
    """The exact integral of q^k (1 - q)^(n - k) from lo to hi, by expanding (1 - q)^(n - k)."""
    total = Fraction(0)
    for i in range(n - k + 1):
        power = k + i + 1
        total += comb(n - k, i) * (-1) ** i * (hi**power - lo**power) / power
    return total


def better_by_integral(k: int, n: int) -> Fraction:
    """The same probability, from the posterior Beta(k + 1, n - k + 1) density."""
    return beta_integral(k, n, Fraction(1, 2)) / beta_integral(k, n)


def fitted_slope(n: int) -> float:
    """The slope in the earlier LyX note, 3 n^-0.48, from a curve fit."""
    return 3 * n**-0.48


def posterior_slope(n: int) -> float:
    """The slope implied by the posterior's spread.

    Near k = n/2 the posterior Beta(k + 1, n - k + 1) has mean (k + 1)/(n + 2) and standard
    deviation 1/(2 sqrt(n + 3)), so P(p > 1/2) is about Phi(z) with
    z = 2 (k - n/2) sqrt(n + 3)/(n + 2); and 1/(1 + exp(-1.702 z)) is within 0.01 of Phi(z).
    """
    return 2 * 1.702 * sqrt(n + 3) / (n + 2)


def logistic(k: int, n: int, slope=posterior_slope) -> float:
    """The approximation 1 / (1 + exp(-slope(n) (k - n/2)))."""
    return 1 / (1 + exp(-slope(n) * (k - n / 2)))


def largest_error(slope, lowest: int, highest: int = 300) -> tuple[float, int, int]:
    """The largest |exact - logistic| over lowest <= n <= highest, with the n and k where it occurs."""
    return max(
        (abs(float(better(k, n)) - logistic(k, n, slope)), n, k)
        for n in range(lowest, highest + 1)
        for k in range(n + 1)
    )


def wins_needed(n: int, sure: Fraction = Fraction(95, 100)) -> int:
    """The fewest wins in n games that make P(p > 1/2) at least `sure`."""
    return next(k for k in range(n + 1) if better(k, n) >= sure)


assert better(6, 10) == Fraction(1486, 2048)
assert all(better(k, n) == better_by_integral(k, n) for n in range(1, 13) for k in range(n + 1))
assert all(better(k, n) + better(n - k, n) == 1 for n in range(1, 30) for k in range(n + 1))


def main() -> None:
    print("Six wins in ten games:", better(6, 10), f"= {float(better(6, 10)):.4f}")
    print("\nGames needed so that |p - k/n| < delta with probability at least 95%:")
    for delta in (Fraction(1, 10), Fraction(1, 20)):
        chernoff = ceil((delta + 2) / delta**2 * log(2 / 0.05))
        hoeffding = ceil(log(2 / 0.05) / (2 * delta**2))
        print(
            f"  delta = {float(delta)}: {chernoff} by the Chernoff bound, {hoeffding} by Hoeffding's"
        )
    print("\nWins needed to be 95% sure of being the better player:")
    for n in (10, 20, 50, 100, 200, 1000):
        k = wins_needed(n)
        print(f"  {n:5} games: {k:4} wins ({k / n:.1%}), P = {float(better(k, n)):.4f}")
    print("\nLargest error of each logistic approximation, against the exact values:")
    for name, slope in (("3 n^-0.48", fitted_slope), ("posterior", posterior_slope)):
        for lowest in (1, 10, 30):
            err, n, k = largest_error(slope, lowest)
            print(f"  {name:9} {lowest:2} <= n <= 300: {err:.4f} (k = {k}, n = {n})")

    rows = ["n,k,share,exact,logistic"]
    for n in (10, 50, 250):
        rows += [
            f"{n},{k},{k / n:.4f},{float(better(k, n)):.6f},{logistic(k, n):.6f}"
            for k in range(n + 1)
        ]
    CSV.write_text("\n".join(rows) + "\n")
    print(f"\nwrote {CSV}")


if __name__ == "__main__":
    main()
