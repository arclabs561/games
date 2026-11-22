"""Variations on "who is the better player?": other priors, match formats, ratings, and games
whose chances are not all the same.

Every function returns an exact Fraction unless its name or docstring says floating point.

- **Priors.** A Beta(a, a) prior with a whole number a puts 2a - 2 extra points on Bayes's
  billiard table, so the chance of being better is again a count of coin flips. The Jeffreys
  prior Beta(1/2, 1/2) gives 1/2 + R/pi with R a fraction.
- **Wald's test as a random walk.** The lead stops at +c or -c; an absorbing Markov chain solved
  by exact elimination checks the gambler's-ruin formulas of stopping.py.
- **Match formats.** Best of n and win by c, with their chance of crowning the better player
  and their mean length.
- **Ratings.** An Elo gap D gives p = 1 / (1 + 10^(-D/400)).
- **Chances that change.** Who shoots first, and players who improve: a dynamic program over
  (wins, games, last winner) with integer weights.

    uv run python -m peggle.variations
"""

import random
from collections.abc import Callable
from fractions import Fraction
from functools import cache
from math import ceil, comb, cos, exp, factorial, lgamma, log, log10, pi, sin
from pathlib import Path

from peggle.peggle import beta_integral, better, heads_at_most
from peggle.stopping import (
    SURE,
    Outcome,
    Rule,
    bounds_hi,
    fixed_length,
    lead_exact,
    outcome,
    stops,
    verdict,
)

HERE = Path(__file__).resolve().parent
HALF = Fraction(1, 2)
P55 = Fraction(55, 100)

# ---------- priors ----------


def better_beta(k: int, n: int, a: int) -> Fraction:
    """P(p > 1/2 | k wins in n games) with the prior Beta(a, a), for a whole number a >= 1:
    the chance of at most k + a - 1 heads in n + 2a - 1 fair flips."""
    flips = n + 2 * a - 1
    return Fraction(heads_at_most(flips)[k + a - 1], 2**flips)


def better_beta_by_integral(k: int, n: int, a: int) -> Fraction:
    """The same chance from the posterior density, proportional to q^(k+a-1) (1-q)^(n-k+a-1)."""
    j, m = k + a - 1, n + 2 * a - 2
    return beta_integral(j, m, HALF) / beta_integral(j, m)


def beta_path_chance(k: int, n: int, a: int) -> Fraction:
    """The chance of one sequence with k wins in n games when p is drawn from Beta(a, a):
    B(k + a, n - k + a) / B(a, a), written with factorials."""

    def beta(x: int, y: int) -> Fraction:
        return Fraction(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))

    return beta(k + a, n - k + a) / beta(a, a)


def jeffreys_part(k: int, n: int) -> Fraction:
    """R with P(p > 1/2 | k wins in n games) = 1/2 + R / pi under the prior Beta(1/2, 1/2).

    With a = i + 1/2 and b = j + 1/2, the regularized incomplete beta I(a, b) at 1/2 starts
    from I(1/2, 1/2) = 1/2 and moves by I(a, b + 1) = I(a, b) + 2^-(a+b) / (b B(a, b)) and
    I(a + 1, b) = I(a, b) - 2^-(a+b) / (a B(a, b)). Each step is a fraction over pi, since
    1 / B(i + 1/2, j + 1/2) = 4^(i+j) i! j! (i+j)! / ((2i)! (2j)! pi). The answer is 1 - I.
    """

    def inv_beta_times_pi(i: int, j: int) -> Fraction:
        return Fraction(
            4 ** (i + j) * factorial(i) * factorial(j) * factorial(i + j),
            factorial(2 * i) * factorial(2 * j),
        )

    def step(i: int, j: int) -> Fraction:  # 2^-(a+b) / B(a, b), times pi
        return inv_beta_times_pi(i, j) / 2 ** (i + j + 1)

    i, j = k, n - k
    part = Fraction(0)  # I(1/2, 1/2) = 1/2 + part / pi
    for t in range(j):  # raise b from 1/2 to j + 1/2, with a = 1/2
        part += step(0, t) / Fraction(2 * t + 1, 2)
    for s in range(i):  # raise a from 1/2 to i + 1/2, with b = j + 1/2
        part -= step(s, j) / Fraction(2 * s + 1, 2)
    return -part  # P = 1 - I = 1/2 - part / pi


# pi to 40 places, as a bracket of fractions, for exact comparisons with 1/2 + R/pi.
PI_LO = Fraction("3.1415926535897932384626433832795028841971")
PI_HI = PI_LO + Fraction(1, 10**40)


def better_jeffreys(k: int, n: int) -> float:
    """1/2 + R / pi, in floating point."""
    return 0.5 + float(jeffreys_part(k, n)) / pi


def jeffreys_float(k: int, n: int) -> float:
    """1/2 + R / pi summed in floating point with logarithms, for speed; the same steps as
    jeffreys_part."""

    def step(i: int, j: int, half: float) -> float:  # 2^-(a+b) / (half B(a, b))
        a, b = i + 0.5, j + 0.5
        return exp(lgamma(a + b) - lgamma(a) - lgamma(b) - (a + b) * log(2)) / half

    part = sum(step(0, t, t + 0.5) for t in range(n - k))
    part -= sum(step(s, n - k, s + 0.5) for s in range(k))
    return 0.5 - part


def jeffreys_at_least(k: int, n: int, sure: Fraction) -> bool:
    """Whether 1/2 + R/pi >= sure: in floating point when far from `sure`, otherwise exactly
    with the bracket on pi."""
    approx = jeffreys_float(k, n)
    if abs(approx - float(sure)) > 1e-9:
        return approx > float(sure)
    r = jeffreys_part(k, n)
    need = sure - HALF  # compare r / pi with need
    lo, hi = (r / PI_HI, r / PI_LO) if r >= 0 else (r / PI_LO, r / PI_HI)
    assert lo >= need or hi < need, "pi bracket too wide"
    return lo >= need


def better_jeffreys_by_quadrature(k: int, n: int, steps: int = 4000) -> float:
    """The same chance by Simpson's rule, in floating point. With p = sin^2 t the density of
    Beta(k + 1/2, n - k + 1/2) becomes proportional to sin^2k t cos^2(n-k) t on [0, pi/2]."""

    def simpson(lo: float, hi: float) -> float:
        h = (hi - lo) / steps
        total = 0.0
        for i in range(steps + 1):
            t = lo + i * h
            w = 1 if i in (0, steps) else 4 if i % 2 else 2
            total += w * sin(t) ** (2 * k) * cos(t) ** (2 * (n - k))
        return total * h / 3

    return simpson(pi / 4, pi / 2) / simpson(0.0, pi / 2)


def fewest_wins_beta(n: int, a: int, sure: Fraction = SURE) -> int:
    """The fewest wins in n games that make P(p > 1/2) at least `sure` under Beta(a, a);
    n + 1 if no record of n games is that sure."""
    counts, flips = heads_at_most(n + 2 * a - 1), n + 2 * a - 1
    lo, hi = n // 2, n + 1  # the chance rises with k
    while lo < hi:
        mid = (lo + hi) // 2
        if counts[mid + a - 1] * sure.denominator >= sure.numerator * 2**flips:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fewest_wins_jeffreys(n: int, sure: Fraction = SURE) -> int:
    """The same under the Jeffreys prior; n + 1 if no record of n games is that sure."""
    lo, hi = n // 2, n + 1  # the chance rises with k, and is below 1/2 at k < n/2
    while lo < hi:
        mid = (lo + hi) // 2
        if jeffreys_at_least(mid, n, sure):
            hi = mid
        else:
            lo = mid + 1
    return lo


@cache
def rule_from_fewest(fewest: Callable[[int], int]) -> Rule:
    """The 95% rule for a symmetric prior, given the fewest wins that make it sure at each n."""
    table: dict[int, int] = {}

    def rule(k: int, n: int) -> int:
        if n not in table:
            table[n] = fewest(n)
        hi = table[n]
        return 1 if k >= hi else -1 if k <= n - hi else 0

    return rule


@cache
def beta_rule(a: int) -> Rule:
    """Play until the Beta(a, a) posterior is 95% sure for either player."""
    return rule_from_fewest(lambda n: fewest_wins_beta(n, a))


JEFFREYS_RULE = rule_from_fewest(fewest_wins_jeffreys)


def prior_check(cap: int, rule: Rule, a: int) -> tuple[Fraction, Fraction]:
    """With p drawn from Beta(a, a): the chance of a verdict, and of a wrong one."""
    declared = wrong = Fraction(0)
    for s in stops(cap, rule):
        if not s.verdict:
            continue
        reach = s.paths * beta_path_chance(s.k, s.n, a)
        above = better_beta(s.k, s.n, a)
        declared += reach
        wrong += reach * (1 - above if s.verdict > 0 else above)
    return declared, wrong


def calibration(cap: int, edges: tuple[int, ...]) -> list[tuple[int, int, Fraction, Fraction]]:
    """Uniform prior, 95% rule: for verdicts reached at lo < n <= hi, their chance and the
    share of them that are wrong."""
    rows = []
    for lo, hi in zip((0, *edges), edges, strict=False):
        declared = wrong = Fraction(0)
        for s in stops(cap):
            if s.verdict and lo < s.n <= hi:
                reach = s.paths * Fraction(1, (s.n + 1) * comb(s.n, s.k))
                declared += reach
                b = better(s.k, s.n)
                wrong += reach * (1 - b if s.verdict > 0 else b)
        rows.append((lo, hi, declared, wrong / declared))
    return rows


# ---------- the lead as a random walk ----------


def lead_chain(p: Fraction, c: int) -> tuple[Fraction, Fraction]:
    """Wald's test with no cap, by solving the absorbing chain on leads -c + 1, ..., c - 1.

    h(x) is the chance of reaching +c first from lead x, and m(x) the mean number of games
    still to play: h(x) = p h(x+1) + q h(x-1), m(x) = 1 + p m(x+1) + q m(x-1), with h(c) = 1,
    h(-c) = 0, m(+-c) = 0. Gaussian elimination in fractions; this shares nothing with the
    closed forms of stopping.lead_exact. Returns (h(0), m(0))."""
    q = 1 - p
    size = 2 * c - 1  # unknowns at x = -c + 1, ..., c - 1
    rows = []
    for i in range(size):
        x = i - c + 1
        h_row = [Fraction(0)] * size
        m_row = [Fraction(0)] * size
        h_row[i] = m_row[i] = Fraction(1)
        h_rhs, m_rhs = Fraction(0), Fraction(1)
        if x + 1 < c:
            h_row[i + 1] -= p
            m_row[i + 1] -= p
        else:
            h_rhs += p
        if x - 1 > -c:
            h_row[i - 1] -= q
            m_row[i - 1] -= q
        rows.append((h_row, m_row, h_rhs, m_rhs))
    h = solve([r[0] for r in rows], [r[2] for r in rows])
    m = solve([r[1] for r in rows], [r[3] for r in rows])
    return h[c - 1], m[c - 1]


def solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Gauss-Jordan elimination with exact fractions."""
    a = [row[:] + [b] for row, b in zip(matrix, rhs, strict=True)]
    size = len(a)
    for col in range(size):
        pivot = next(r for r in range(col, size) if a[r][col] != 0)
        a[col], a[pivot] = a[pivot], a[col]
        lead = a[col][col]
        a[col] = [v / lead for v in a[col]]
        for r in range(size):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [v - factor * w for v, w in zip(a[r], a[col], strict=True)]
    return [row[-1] for row in a]


def wald_bounds(cap: int) -> list[tuple[int, int]]:
    """For n = 0..cap, the lead 2 hi - n that the 95% rule needs after n games (-1 if none)."""
    rows = []
    for n in range(cap + 1):
        hi = next((k for k in range(n + 1) if verdict(k, n) > 0), None) if n else None
        rows.append((n, 2 * hi - n if hi is not None else -1))
    return rows


def sample_walk(p: float, lead: int, seed: int, length: int) -> list[int]:
    """One seeded path of the lead, until it reaches +-lead or `length` games."""
    rng = random.Random(seed)
    walk = [0]
    while abs(walk[-1]) < lead and len(walk) <= length:
        walk.append(walk[-1] + (1 if rng.random() < p else -1))
    return walk


# ---------- match formats ----------


def best_of(p: Fraction, n: int) -> Fraction:
    """The chance that the player who wins each game with chance p wins a best-of-n match
    (n odd): at least (n + 1)/2 wins in n games."""
    assert n % 2 == 1
    q = 1 - p
    return sum((comb(n, j) * p**j * q ** (n - j) for j in range((n + 1) // 2, n + 1)), Fraction(0))


def first_to(m: int) -> Rule:
    """Stop when either player has m wins: a best-of-(2m - 1) match."""

    @cache
    def rule(k: int, n: int) -> int:
        return 1 if k >= m else -1 if n - k >= m else 0

    return rule


def best_of_games(p: Fraction, n: int) -> Fraction:
    """The mean length of a best-of-n match that stops as soon as it is decided."""
    m = (n + 1) // 2
    q = 1 - p
    # The match ends at game g when the winner takes their m-th win there.
    return sum(
        (
            g * comb(g - 1, m - 1) * (p**m * q ** (g - m) + q**m * p ** (g - m))
            for g in range(m, n + 1)
        ),
        Fraction(0),
    )


def _best_of_one(p: Fraction | float, alpha: Fraction | float) -> bool:
    """Check feasibility and whether one game already meets the target."""
    if not 0 <= p <= 1 or not 0 <= alpha <= 1:
        raise ValueError("p and alpha must be probabilities")
    if p >= 1 - alpha:
        return True
    if p <= 0.5 or alpha == 0:
        raise ValueError("no finite best-of match reaches the requested win chance")
    return False


def fewest_best_of(p: Fraction, alpha: Fraction = Fraction(5, 100)) -> int:
    """The shortest best-of-n match (n odd) that the better player wins with chance >= 1 - alpha."""
    if _best_of_one(p, alpha):
        return 1
    n = 3
    while best_of(p, n) < 1 - alpha:
        n += 2
    return n


def slope_at_half_best_of(n: int) -> Fraction:
    """d/dp of best_of(p, n) at p = 1/2: n C(n - 1, (n - 1)/2) / 2^(n - 1)."""
    return Fraction(n * comb(n - 1, (n - 1) // 2), 2 ** (n - 1))


# ---------- ratings ----------


def elo_chance(gap: float) -> float:
    """The Elo (Bradley-Terry) chance of winning for a rating gap of `gap` points."""
    return 1 / (1 + 10 ** (-gap / 400))


def elo_gap(p: float) -> float:
    """The rating gap that gives winning chance p."""
    return 400 * log10(p / (1 - p))


def lead_for_gap(gap: float, alpha: float = 0.05) -> int:
    """The smallest lead whose test errs with chance <= alpha at the Elo chance for `gap`:
    r^c / (1 + r^c) <= alpha with r = 10^(-gap/400), or c >= 400 log10((1 - alpha)/alpha) / gap."""
    return ceil(400 * log10((1 - alpha) / alpha) / gap - 1e-12)


def lead_float(p: float, c: int) -> tuple[float, float]:
    """lead_exact in floating point: (chance the worse player wins, mean games)."""
    r = ((1 - p) / p) ** c
    return r / (1 + r), c * (1 - r) / ((1 + r) * (2 * p - 1))


def fewest_best_of_float(p: float, alpha: float = 0.05) -> int:
    """fewest_best_of in floating point, summing the binomial tail with logs."""
    if _best_of_one(p, alpha):
        return 1
    n = 3
    while True:
        tail = sum(
            exp(
                lgamma(n + 1)
                - lgamma(j + 1)
                - lgamma(n - j + 1)
                + j * log(p)
                + (n - j) * log(1 - p)
            )
            for j in range((n + 1) // 2, n + 1)
        )
        if tail >= 1 - alpha:
            return n
        n += 2


# ---------- chances that change from game to game ----------

# chance(n, last) -> the chance that the first player wins game n (1, 2, ...), given who won
# game n - 1 (+1 first player, -1 second, 0 before the first game).
Chance = Callable[[int, int], Fraction]


def changing_outcome(cap: int, rule: Rule, chance: Chance, denominator: int) -> Outcome:
    """Exact chances of each verdict when the chance of each game may depend on its number and
    on who won the game before. Every chance must be a multiple of 1 / denominator; weights are
    kept as whole numbers over denominator^n."""
    alive: dict[tuple[int, int], int] = {(0, 0): 1}  # (wins, last winner) -> weight
    totals = {1: Fraction(0), -1: Fraction(0), 0: Fraction(0)}
    games = Fraction(0)
    for n in range(1, cap + 1):
        step: dict[tuple[int, int], int] = {}
        for (k, last), w in alive.items():
            c = chance(n, last) * denominator
            assert c.denominator == 1
            win = int(c)
            step[k + 1, 1] = step.get((k + 1, 1), 0) + w * win
            step[k, -1] = step.get((k, -1), 0) + w * (denominator - win)
        alive = {}
        scale = denominator**n
        for (k, last), w in step.items():
            v = rule(k, n)
            if v or n == cap:
                share = Fraction(w, scale)
                totals[v] += share
                games += n * share
            elif w:
                alive[k, last] = w
    return Outcome(totals[1], totals[-1], totals[0], games)


def at_end(rule: Rule, cap: int) -> Rule:
    """The same verdicts, but only after exactly cap games: one look."""

    def once(k: int, n: int) -> int:
        return rule(k, n) if n == cap else 0

    return once


def constant(p: Fraction) -> Chance:
    return lambda n, last: p


def alternate(edge: Fraction) -> Chance:
    """The players take turns to shoot first, the first player in odd games; the one who shoots
    first wins with chance edge."""
    return lambda n, last: edge if n % 2 else 1 - edge


def winner_starts(edge: Fraction) -> Chance:
    """The first player starts game 1; after that the last winner shoots first."""
    return lambda n, last: edge if last >= 0 else 1 - edge


def loser_starts(edge: Fraction) -> Chance:
    """The first player starts game 1; after that the last loser shoots first."""
    return lambda n, last: edge if last <= 0 else 1 - edge


def step_change(before: Fraction, after: Fraction, switch: int) -> Chance:
    """The first player wins with chance `before` in games 1..switch and `after` later."""
    return lambda n, last: before if n <= switch else after


# ---------- figures ----------


def beta_density(x: float, a: float, b: float) -> float:
    """The Beta(a, b) density at x, in floating point."""
    return exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * log(x) + (b - 1) * log(1 - x))


def write_csvs() -> None:
    """priors.csv, walk.csv and formats.csv, read by the figures in peggle.typ."""
    k, n = 6, 10
    rows = ["x,uniform,jeffreys,skeptical"]
    for i in range(1, 200):
        x = i / 200
        rows.append(
            f"{x:.3f},{beta_density(x, k + 1, n - k + 1):.5f},"
            f"{beta_density(x, k + 0.5, n - k + 0.5):.5f},"
            f"{beta_density(x, k + SKEPTIC, n - k + SKEPTIC):.5f}"
        )
    (HERE / "priors.csv").write_text("\n".join(rows) + "\n")

    walks = [sample_walk(0.55, 15, seed, WALK_LENGTH) for seed in WALK_SEEDS]
    length = max(len(w) for w in walks) + 10
    posterior = dict(wald_bounds(length))
    rows = ["n," + ",".join(f"seed{seed}" for seed in WALK_SEEDS) + ",posterior,bayes"]
    for g in range(length):
        cells = [str(w[g]) if g < len(w) else "" for w in walks]
        cells.append(str(posterior[g]) if posterior[g] >= 0 else "")
        hi = bounds_hi(g) if g else 0
        cells.append(str(2 * hi - g) if g and hi <= g else "")
        rows.append(f"{g}," + ",".join(cells))
    (HERE / "walk.csv").write_text("\n".join(rows) + "\n")

    rows = ["p,single,best3,best7,lead2,lead15"]
    for i in range(0, 101):
        p = Fraction(1, 2) + Fraction(i, 400)  # 0.5 to 0.75
        cells = [
            p,
            best_of(p, 3),
            best_of(p, 7),
            1 - lead_exact(p, 2)[0] if p != HALF else HALF,
            1 - lead_exact(p, 15)[0] if p != HALF else HALF,
        ]
        rows.append(f"{float(p):.4f}," + ",".join(f"{float(c):.5f}" for c in cells))
    (HERE / "formats.csv").write_text("\n".join(rows) + "\n")


SKEPTIC = 12  # Beta(12, 12): mean 1/2, standard deviation 1/10
WALK_SEEDS = (1, 2, 3, 4, 5)  # the paths drawn in the random-walk figure
WALK_LENGTH = 400


def show(x: Fraction, places: int = 4) -> str:
    return f"{float(x):.{places}f}"


def main() -> None:
    k, n = 6, 10
    print("Other priors, six wins in ten games. Exact unless marked.")
    print(f"  uniform Beta(1, 1):     {better(k, n)} = {show(better(k, n))}")
    r = jeffreys_part(k, n)
    print(f"  Jeffreys Beta(1/2, 1/2): 1/2 + ({r})/pi = {better_jeffreys(k, n):.4f}")
    b = better_beta(k, n, SKEPTIC)
    print(f"  skeptical Beta({SKEPTIC}, {SKEPTIC}): {b} = {show(b)}")
    m = SKEPTIC - 1  # the prior's own chance above 0.6, from its density q^m (1 - q)^m
    above = beta_integral(m, 2 * m, Fraction(3, 5)) / beta_integral(m, 2 * m)
    print(f"  Beta({SKEPTIC}, {SKEPTIC}) puts chance {show(above, 3)} above 0.6")
    print("  wins needed to be 95% sure (uniform, Jeffreys, skeptical):")
    for m in (10, 20, 50, 100, 200, 1000):
        print(
            f"    {m:4} games: {fewest_wins_beta(m, 1):3} {fewest_wins_jeffreys(m):3}"
            f" {fewest_wins_beta(m, SKEPTIC):3}"
        )
    print(
        "  shortest verdict: uniform",
        next(m for m in range(1, 20) if fewest_wins_beta(m, 1) <= m),
        "games, Jeffreys",
        next(m for m in range(1, 20) if fewest_wins_jeffreys(m) <= m),
        "games, skeptical",
        next(m for m in range(1, 40) if fewest_wins_beta(m, SKEPTIC) <= m),
    )
    print("  play until 95% sure under each prior (equal players; p = 0.55):")
    rules = (("uniform", verdict), ("Jeffreys", JEFFREYS_RULE), ("skeptical", beta_rule(SKEPTIC)))
    for name, rule in rules:
        for cap in (100, 1000):
            e, g = outcome(HALF, cap, rule), outcome(P55, cap, rule)
            print(
                f"    {name:9} N = {cap:4}: verdict {show(e.first + e.second, 3)},"
                f" mean {float(e.games):.1f}; better {show(g.first, 3)},"
                f" worse {show(g.second, 3)}, mean {float(g.games):.1f}"
            )
    print(f"  p drawn from Beta({SKEPTIC}, {SKEPTIC}): share of verdicts that are wrong")
    for name, rule in (rules[0], rules[2]):
        for cap in (100, 1000):
            declared, wrong = prior_check(cap, rule, SKEPTIC)
            print(f"    {name:9} rule, N = {cap:4}: {float(wrong / declared):.1%}")

    print("\nUniform prior, 95% rule, N = 1000: verdicts by when they come, and the share wrong")
    for lo, hi, declared, wrong in calibration(1000, (10, 100, 1000)):
        print(f"  {lo + 1:4} <= n <= {hi:4}: chance {show(declared, 3)}, wrong {float(wrong):.1%}")

    print("\nWald's test as an absorbing chain, solved exactly (agrees with the closed forms):")
    for p in (HALF, P55):
        h, m = lead_chain(p, 15)
        print(
            f"  p = {float(p)}: first player reaches +15 first {show(h)}, mean games {float(m):.1f}"
        )

    print("\nMatch formats at p = 0.55: chance the better player wins, mean games, slope at 1/2")
    for m in (1, 3, 5, 7):
        print(
            f"  best of {m}: {show(best_of(P55, m))}, {float(best_of_games(P55, m)):.2f},"
            f" {slope_at_half_best_of(m)}"
        )
    for c in (2, 15):
        worse, games = lead_exact(P55, c)
        print(f"  win by {c:2}: {show(1 - worse)}, {float(games):.2f}, {c}")
    nb = fewest_best_of(P55)
    print(
        f"  shortest best-of-n with 95%: n = {nb} ({show(best_of(P55, nb))};"
        f" best of {nb - 2}: {show(best_of(P55, nb - 2))}), mean {float(best_of_games(P55, nb)):.2f},"
        f" slope {float(slope_at_half_best_of(nb)):.1f}"
    )

    print(
        f"\nLead needed for 5% error at an Elo gap D: 400 log10(19) / D = {400 * log10(19):.1f} / D"
    )
    print("Elo gaps (floating point): chance, lead needed for 5% error, its error, mean games,")
    print("and the shortest best-of-n with 95%")
    for gap in (elo_gap(0.55), 50, 100, 200, 400):
        p = elo_chance(gap)
        c = lead_for_gap(gap)
        worse, games = lead_float(p, c)
        print(
            f"  {gap:6.1f} points: p = {p:.4f}, lead {c:2}, error {worse:.4f},"
            f" mean {games:6.1f}, best of {fewest_best_of_float(p)}"
        )
    worse, games = lead_exact(Fraction(10, 11), 2)
    print(
        f"  400 points exactly: p = 10/11, win by 2 errs {worse}, mean {games} = {float(games):.2f}"
    )
    p = elo_chance(400 * 400 / 561)
    worse, games = lead_float(p, 2)
    print(
        f"  400 points on the 561 scale: p = {p:.4f}, win by 2 errs {worse:.4f}, mean {games:.2f},"
        f" best of {fewest_best_of_float(p)}"
    )

    print("\nEqual players, but whoever shoots first wins with chance e. Exact.")
    print("  rule                 e    one look 100   stop N=100   stop N=1000")
    iid100 = fixed_length(HALF, 100)
    print(
        f"  independent (iid)   -     {show(iid100.first + iid100.second)}"
        f"       {show(outcome(HALF, 100).first * 2)}       {show(outcome(HALF, 1000).first * 2)}"
    )
    for edge in (Fraction(3, 5), Fraction(7, 10)):
        for name, chance in (
            ("alternate", alternate(edge)),
            ("winner shoots first", winner_starts(edge)),
            ("loser shoots first", loser_starts(edge)),
        ):
            look = changing_outcome(100, at_end(verdict, 100), chance, 10)
            cells = [show(look.first + look.second)]
            for cap in (100, 1000):
                o = changing_outcome(cap, verdict, chance, 10)
                cells.append(show(o.first + o.second))
            print(f"  {name:19} {float(edge)}   " + "       ".join(cells))

    print("\nA player who improves: p = before in games 1-500, after in games 501-1000. Exact.")
    iid = fixed_length(HALF, 1000)
    print(f"  constant 1/2, one look after 1000: {show(iid.first + iid.second)}")
    for before, after in ((Fraction(45, 100), P55), (Fraction(2, 5), Fraction(3, 5))):
        chance = step_change(before, after, 500)
        look = changing_outcome(1000, at_end(verdict, 1000), chance, 20)
        o = changing_outcome(1000, verdict, chance, 20)
        print(
            f"  {float(before)} then {float(after)}: one look {show(look.first + look.second)};"
            f" stop at 95%: improver declared better {show(o.first, 3)}, the other"
            f" {show(o.second, 3)}, mean {float(o.games):.1f} games"
        )

    write_csvs()
    print(f"\nwrote {HERE / 'priors.csv'}, walk.csv and formats.csv")


if __name__ == "__main__":
    main()
