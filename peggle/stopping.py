"""Play until someone is 95% sure: optional stopping, exactly.

Two friends play until the posterior chance that the first player is better, P(p > 1/2 | k, n)
from peggle.py, reaches 95% (the first player is declared better) or falls to 5% (the second
player is 95% sure and is declared better), or until a cap of N games (no one is declared).

The stopping rule looks only at (wins, games), so the number of game sequences that reach each
(k, n) without having stopped is an integer that does not depend on p. One dynamic program over
those counts gives every exact answer: a sequence with k wins in n games has chance
p^k (1 - p)^(n - k) when p is fixed, and 1 / ((n + 1) C(n, k)) when p is drawn from the
uniform prior.

    uv run python -m peggle.stopping
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb, sqrt

from peggle.peggle import beta_integral, better, heads_at_most

SURE = Fraction(95, 100)


@dataclass(frozen=True)
class Stop:
    """A place where play stops: k wins in n games, and who is declared better (+1, -1 or 0)."""

    k: int
    n: int
    verdict: int
    paths: int  # how many win/loss sequences first stop here


def verdict(k: int, n: int, sure: Fraction = SURE) -> int:
    """+1 if the first player is at least `sure` to be better, -1 if the second is, else 0."""
    # better(k, n) = below / 2^(n + 1); compare whole numbers rather than build a Fraction.
    below, total = heads_at_most(n + 1)[k], 2 ** (n + 1)
    if below * sure.denominator >= sure.numerator * total:
        return 1
    return -1 if (total - below) * sure.denominator >= sure.numerator * total else 0


Rule = Callable[[int, int], int]  # (k, n) -> verdict, as `verdict` returns it


@cache
def binomial_test(alpha: Fraction) -> Rule:
    """The rule of Armitage, McPherson and Rowe (1969): after each game, a two-sided test of
    p = 1/2 at level 2 alpha, as if n had been fixed. A verdict when k >= b_n or k <= n - b_n,
    where b_n is the fewest wins with P(at least b_n heads in n fair flips) <= alpha."""

    @cache
    def fewest(n: int) -> int:
        tail = [2**n - c for c in [0, *heads_at_most(n)[:-1]]]  # tail[b]: sequences with >= b
        return next(b for b in range(n + 2) if b > n or tail[b] <= alpha * 2**n)

    def rule(k: int, n: int) -> int:
        b = fewest(n)
        return 1 if k >= b else -1 if k <= n - b else 0

    return rule


@cache
def stops(cap: int, rule: Rule = verdict) -> tuple[Stop, ...]:
    """Every place where play stops, with the number of sequences that first stop there."""
    alive = {0: 1}  # wins -> number of sequences that reach (wins, n) without stopping
    found = []
    for n in range(1, cap + 1):
        step: dict[int, int] = {}
        for k, paths in alive.items():
            step[k + 1] = step.get(k + 1, 0) + paths
            step[k] = step.get(k, 0) + paths
        alive = {}
        for k, paths in sorted(step.items()):
            v = rule(k, n)
            if v or n == cap:
                found.append(Stop(k, n, v, paths))
            else:
                alive[k] = paths
    return tuple(found)


def path_chance(k: int, n: int, p: Fraction) -> Fraction:
    """The chance of one particular sequence with k wins in n games, when p is fixed."""
    return p**k * (1 - p) ** (n - k)


def prior_path_chance(k: int, n: int) -> Fraction:
    """The same chance averaged over the uniform prior: the integral of p^k (1 - p)^(n - k)."""
    return Fraction(1, (n + 1) * comb(n, k))


@dataclass(frozen=True)
class Outcome:
    """Chances that the first player, the second, or no one is declared better; mean games."""

    first: Fraction
    second: Fraction
    neither: Fraction
    games: Fraction


def outcome(p: Fraction, cap: int, rule: Rule = verdict) -> Outcome:
    """Exact chances for a fixed win chance p of the first player."""
    totals = {1: Fraction(0), -1: Fraction(0), 0: Fraction(0)}
    games = Fraction(0)
    for s in stops(cap, rule):
        chance = s.paths * path_chance(s.k, s.n, p)
        totals[s.verdict] += chance
        games += s.n * chance
    return Outcome(totals[1], totals[-1], totals[0], games)


def fixed_length(p: Fraction, n: int, rule: Rule = verdict) -> Outcome:
    """The same chances when the friends play exactly n games and look once."""
    totals = {1: Fraction(0), -1: Fraction(0), 0: Fraction(0)}
    for k in range(n + 1):
        totals[rule(k, n)] += comb(n, k) * path_chance(k, n, p)
    return Outcome(totals[1], totals[-1], totals[0], Fraction(n))


@dataclass(frozen=True)
class Bayes:
    """With p drawn from the uniform prior: chance of a verdict, and of a wrong one."""

    declared: Fraction
    wrong: Fraction

    @property
    def wrong_given_declared(self) -> Fraction:
        return self.wrong / self.declared


def bayes(cap: int, by_integral: bool = False) -> Bayes:
    """Exact chances under the uniform prior. A verdict at (k, n) is wrong with chance
    1 - better(k, n) or better(k, n); with by_integral, that comes from the Beta integral."""
    declared = wrong = Fraction(0)
    for s in stops(cap):
        if not s.verdict:
            continue
        reach = s.paths * prior_path_chance(s.k, s.n)
        if by_integral:
            above = beta_integral(s.k, s.n, Fraction(1, 2)) / beta_integral(s.k, s.n)
        else:
            above = better(s.k, s.n)
        declared += reach
        wrong += reach * (1 - above if s.verdict > 0 else above)
    return Bayes(declared, wrong)


def laplace_path_chance(wins: list[bool]) -> Fraction:
    """The chance of a sequence under the uniform prior, game by game, by Laplace's rule:
    after k wins in n games the next is won with chance (k + 1) / (n + 2)."""
    chance, k = Fraction(1), 0
    for n, won in enumerate(wins):
        next_win = Fraction(k + 1, n + 2)
        chance *= next_win if won else 1 - next_win
        k += won
    return chance


def thresholds(cap: int, rule: Rule = verdict) -> list[tuple[int, int]]:
    """For each n <= cap, the (lo, hi) with a verdict exactly when k <= lo or k >= hi."""
    rows = [(-1, 1)]  # n = 0: no verdict
    for n in range(1, cap + 1):
        verdicts = [rule(k, n) for k in range(n + 1)]
        lo = max((k for k, v in enumerate(verdicts) if v < 0), default=-1)
        hi = min((k for k, v in enumerate(verdicts) if v > 0), default=n + 1)
        rows.append((lo, hi))
    return rows


def simulate(p: float | None, cap: int, trials: int, seed: int) -> dict[str, float]:
    """Seeded simulation. With p None, each trial draws p from the uniform prior and also
    records whether the verdict was wrong."""
    rng = random.Random(seed)
    bounds = thresholds(cap)
    first = second = wrong = 0
    games = 0
    for _ in range(trials):
        q = rng.random() if p is None else p
        k = n = 0
        while n < cap:
            n += 1
            k += rng.random() < q
            lo, hi = bounds[n]
            if k >= hi:
                first += 1
                wrong += q < 0.5
                break
            if k <= lo:
                second += 1
                wrong += q > 0.5
                break
        games += n
    return {
        "first": first / trials,
        "second": second / trials,
        "wrong": wrong / trials,
        "games": games / trials,
    }


assert [verdict(k, 4) for k in range(5)] == [-1, 0, 0, 0, 1]  # better(4, 4) = 31/32
assert all(
    verdict(k, n) == (1 if better(k, n) >= SURE else -1 if better(k, n) <= 1 - SURE else 0)
    for n in range(1, 40)
    for k in range(n + 1)
)
assert sum(s.paths for s in stops(1)) == 2
assert all(sum(s.paths * 2 ** (10 - s.n) for s in stops(m)) == 2**10 for m in (1, 5, 10))


def main() -> None:
    caps = (100, 1000)
    print("Play until P(better) >= 0.95 for either player, or a cap of N games. Exact.\n")
    print("Equal players, p = 1/2: chance that someone is declared better")
    for cap in caps:
        o, f = outcome(Fraction(1, 2), cap), fixed_length(Fraction(1, 2), cap)
        print(
            f"  N = {cap:4}: stopping at 95%: {float(o.first + o.second):.4f}"
            f" (mean {float(o.games):.1f} games);"
            f" one look after N games: {float(f.first + f.second):.4f}"
        )
    print("\nFirst player wins each game with chance p (the second player is the worse one):")
    print("     p      N  better declared  worse declared  undecided  mean games")
    for p in (Fraction(55, 100), Fraction(6, 10), Fraction(7, 10)):
        for cap in caps:
            o = outcome(p, cap)
            print(
                f"  {float(p):4}  {cap:5}  {float(o.first):15.4f}  {float(o.second):14.4f}"
                f"  {float(o.neither):9.4f}  {float(o.games):10.1f}"
            )
    print("\nWith p drawn from the uniform prior (the Bayesian reading):")
    for cap in caps:
        b = bayes(cap)
        print(
            f"  N = {cap:4}: verdict {float(b.declared):.4f},"
            f" wrong verdicts {float(b.wrong):.4f},"
            f" wrong given a verdict {float(b.wrong_given_declared):.4f}"
        )

    print("\nCheck against Armitage, McPherson and Rowe (1969), Table 1: repeated two-sided")
    print("binomial tests at level 2 alpha after each game, p = 1/2, chance of a verdict by n:")
    for two_alpha, n, published in (
        (Fraction(1, 100), 100, 0.05586),
        (Fraction(5, 100), 100, 0.22731),
        (Fraction(5, 100), 150, 0.26108),
    ):
        o = outcome(Fraction(1, 2), n, binomial_test(two_alpha / 2))
        print(
            f"  2 alpha = {float(two_alpha)}, n = {n}: {float(o.first + o.second):.5f} (published {published})"
        )

    cap, trials, seed = 100, 200_000, 1
    print(f"\nSimulation, N = {cap}, {trials:,} trials each, seed {seed}, 95% intervals:")
    for p in (Fraction(1, 2), Fraction(55, 100), Fraction(7, 10), None):
        sim = simulate(None if p is None else float(p), cap, trials, seed)
        if p is None:
            b = bayes(cap)
            exact = {"declared": b.declared, "wrong": b.wrong}
            got = {"declared": sim["first"] + sim["second"], "wrong": sim["wrong"]}
        else:
            o = outcome(p, cap)
            exact = {"first": o.first, "second": o.second}
            got = {"first": sim["first"], "second": sim["second"]}
        label = "uniform" if p is None else f"p = {float(p)}"
        cells = ", ".join(
            f"{key} {got[key]:.4f} ± {1.96 * sqrt(got[key] * (1 - got[key]) / trials):.4f}"
            f" (exact {float(exact[key]):.4f})"
            for key in exact
        )
        print(f"  {label:8}: {cells}")


if __name__ == "__main__":
    main()
