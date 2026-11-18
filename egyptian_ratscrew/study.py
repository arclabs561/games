"""Seeded batches of Egyptian Ratscrew games, and models of who wins when slap shares differ.

The large runs behind the notebook's last sections take under an hour on two cores. They
are recorded in `runs.json`, next to this file, and the tests replay a slice of several of them:

    uv run python -m egyptian_ratscrew.study
"""

import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from math import exp, log
from multiprocessing import Pool
from pathlib import Path
from statistics import fmean

from egyptian_ratscrew.game import Game, play_game

RUNS = Path(__file__).with_name("runs.json")
CHUNK = 5_000  # games per seeded chunk; chunk k of a run with seed s uses Random(s * 1000 + k)


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% Wilson score interval for a binomial proportion."""
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return center - half, center + half


assert wilson(50, 100) == (wilson(50, 100)[0], 1 - wilson(50, 100)[0])


def rotated(values: Sequence[float], seat: int) -> list[float]:
    """`values` by player, rearranged by seat: player p sits in seat (p + seat) mod n."""
    n = len(values)
    return [values[(i - seat) % n] for i in range(n)]


assert rotated([0.7, 0.2, 0.1], 1) == [0.1, 0.7, 0.2]


def games(
    shares: Sequence[float],
    n: int,
    seed: int,
    slaps: bool = True,
    misslaps: Sequence[float] | None = None,
    rotate: bool = False,
) -> list[Game]:
    """`n` seeded games. With rotate=True, game g seats player p in seat (p + g) mod n, and the
    winner is reported as a player, not a seat, so that no player always leads."""
    rng = random.Random(seed)
    players = len(shares)
    results = []
    for g in range(n):
        seat = g % players if rotate else 0
        game = play_game(
            rotated(shares, seat),
            rng,
            slaps=slaps,
            misslaps=None if misslaps is None else rotated(misslaps, seat),
        )
        if game.winner is not None:
            game = game._replace(winner=(game.winner - seat) % players)
        results.append(game)
    return results


def chunk_summary(args: tuple) -> dict:
    """Play one chunk of a run and summarize it: wins by player, unfinished games, and the
    counts of game lengths in cards and in tricks."""
    shares, seed, chunk, kwargs = args
    results = games(shares, CHUNK, seed * 1000 + chunk, **kwargs)
    return {
        "wins": [sum(g.winner == i for g in results) for i in range(len(shares))],
        "unfinished": sum(g.winner is None for g in results),
        "cards": Counter(g.cards for g in results if g.winner is not None),
        "tricks": Counter(g.tricks for g in results if g.winner is not None),
        "paid": sum(g.paid for g in results),
    }


def run(
    shares: Sequence[float], n: int, seed: int, pool=None, histograms: bool = False, **kwargs
) -> dict:
    """`n` games (a multiple of CHUNK) in seeded chunks, merged; the wins of every chunk are
    kept so that any chunk can be replayed and checked. With histograms=True the number of
    games of each length in tricks is kept too."""
    jobs = [(list(shares), seed, k, kwargs) for k in range(n // CHUNK)]
    parts = list(pool.map(chunk_summary, jobs) if pool else map(chunk_summary, jobs))
    cards, tricks = Counter(), Counter()
    for part in parts:
        cards += part["cards"]
        tricks += part["tricks"]
    summary = {
        "shares": list(shares),
        "seed": seed,
        "options": kwargs,
        "games": n,
        "wins": [sum(p["wins"][i] for p in parts) for i in range(len(shares))],
        "chunk_wins": [p["wins"] for p in parts],
        "unfinished": sum(p["unfinished"] for p in parts),
        "paid": sum(p["paid"] for p in parts),
        "cards": length_summary(cards),
        "tricks": length_summary(tricks),
        "tricks_6_to_28": sum(c for t, c in tricks.items() if 6 <= t <= 28),
        "tricks_mode": max(tricks, key=lambda t: (tricks[t], -t)),
        "tricks_at_mode": max(tricks.values()),
    }
    if histograms:
        summary["tricks_counts"] = {str(x): tricks[x] for x in sorted(tricks)}
    return summary


def length_summary(counts: Counter) -> dict:
    """Mean, standard deviation and quantiles of lengths given as {length: games}."""
    n = sum(counts.values())
    mean = sum(x * c for x, c in counts.items()) / n
    var = sum(c * (x - mean) ** 2 for x, c in counts.items()) / (n - 1)
    ordered = sorted(counts)

    def quantile(q: float) -> int:
        seen = 0
        for x in ordered:
            seen += counts[x]
            if seen > q * n:
                return x
        return ordered[-1]

    return {
        "mean": mean,
        "sd": var**0.5,
        "median": quantile(0.5),
        "q90": quantile(0.9),
        "q99": quantile(0.99),
        "max": ordered[-1],
    }


assert length_summary(Counter({1: 1, 2: 1, 3: 1}))["median"] == 2


# Models of the win chance. Each maps a list of slap shares to a list of win chances.

type Model = Callable[[Sequence[float]], list[float]]


def logistic(x: float) -> float:
    return 1 / (1 + exp(-x))


def normalized(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    return [w / total for w in weights]


def proportional(shares: Sequence[float]) -> list[float]:
    """Win chance equal to slap share."""
    return normalized(shares)


def power(k: float) -> Model:
    """Bradley-Terry (Luce) strengths s**k: player i wins with chance s_i**k / sum s_j**k."""
    return lambda shares: normalized([s**k for s in shares])


def softmax(c: float) -> Model:
    """Strengths exp(c * s): in two-player games, logit(win) = c * (s0 - s1)."""
    return lambda shares: normalized([exp(c * s) for s in shares])


def two_player_against_field(c: float) -> Model:
    """The two-player curve logistic(c * (2x - 1)), at the effective share
    x = s_i / (s_i + mean share of the others), normalized to sum to 1."""

    def model(shares: Sequence[float]) -> list[float]:
        n = len(shares)
        raw = []
        for s in shares:
            field = (sum(shares) - s) / (n - 1)
            raw.append(logistic(c * (2 * s / (s + field) - 1)))
        return normalized(raw)

    return model


assert power(1)([0.5, 0.3, 0.2]) == proportional([0.5, 0.3, 0.2])
assert abs(softmax(4)([0.7, 0.3])[0] - logistic(4 * 0.4)) < 1e-12
assert abs(two_player_against_field(4)([0.7, 0.3])[0] - logistic(4 * 0.4)) < 1e-12


def fit(
    make: Callable[[float], Model],
    data: Sequence[tuple[Sequence[float], Sequence[int]]],
    lo: float,
    hi: float,
) -> float:
    """The parameter in [lo, hi] that maximizes the multinomial likelihood of the observed
    wins, found by golden-section search. `data` pairs share vectors with wins by player."""

    def loglik(theta: float) -> float:
        model = make(theta)
        return sum(
            w * log(p) for shares, wins in data for w, p in zip(wins, model(shares), strict=True)
        )

    g = (5**0.5 - 1) / 2
    a, b = lo, hi
    for _ in range(80):
        x1, x2 = b - g * (b - a), a + g * (b - a)
        if loglik(x1) < loglik(x2):
            a = x1
        else:
            b = x2
    return (a + b) / 2


def random_shares(players: int, rng: random.Random, alpha: float = 2.0) -> list[float]:
    """Slap shares drawn from a symmetric Dirichlet distribution, rounded to 3 places."""
    draws = [rng.gammavariate(alpha, 1) for _ in range(players)]
    shares = [round(d / sum(draws), 3) for d in draws]
    shares[-1] = round(1 - sum(shares[:-1]), 3)
    return shares


def entropy(shares: Sequence[float]) -> float:
    """Shannon entropy of the shares, in nats."""
    return -sum(s * log(s) for s in shares if s > 0)


def rmse(predicted: Sequence[float], observed: Sequence[float]) -> float:
    return fmean((p - o) ** 2 for p, o in zip(predicted, observed, strict=True)) ** 0.5


def ranks(xs: Sequence[float]) -> list[float]:
    """Ranks from 0, with ties given their average rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    result = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman's rank correlation."""
    rx, ry = ranks(xs), ranks(ys)
    mx, my = fmean(rx), fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


assert ranks([3, 1, 3, 2]) == [2.5, 0, 2.5, 1]
assert abs(spearman([1, 2, 3], [10, 20, 25]) - 1) < 1e-12


TWO_PLAYER_SHARES = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
FAST_SHARES = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
MISSLAP_RATES = (0.0, 0.01, 0.02, 0.05, 0.1)
VECTORS, VECTOR_GAMES = 60, 4_000


def vector_wins(args: tuple) -> dict:
    """Wins by player in VECTOR_GAMES seeded games with rotated seats."""
    shares, seed = args
    results = games(shares, VECTOR_GAMES, seed, rotate=True)
    return {
        "shares": shares,
        "seed": seed,
        "wins": [sum(g.winner == i for g in results) for i in range(len(shares))],
    }


def share_vectors(players: int, seed: int) -> list[list[float]]:
    """VECTORS random share vectors for `players` players, seeded."""
    rng = random.Random(seed)
    return [random_shares(players, rng) for _ in range(VECTORS)]


def large_runs(pool) -> dict[str, Callable[[], object]]:
    """The recorded runs, by name, each as a function that plays it."""

    def vectors(players: int) -> list[dict]:
        jobs = [(v, 1000 * players + k) for k, v in enumerate(share_vectors(players, players))]
        return pool.map(vector_wins, jobs)

    return {
        "first_player_2": lambda: run([0.5, 0.5], 1_000_000, 1, pool, True),
        "first_player_4": lambda: run([0.25] * 4, 200_000, 2, pool, True),
        "beggar_my_neighbour": lambda: run([1, 1], 1_000_000, 3, pool, True, slaps=False),
        "two_player_curve": lambda: [
            run([s, round(1 - s, 2)], 20_000, 10 + i, pool, rotate=True)
            for i, s in enumerate(TWO_PLAYER_SHARES)
        ],
        "misslaps": lambda: [
            run(
                [s, round(1 - s, 2)], 20_000, 100 + 10 * i + j, pool, rotate=True, misslaps=[m, 0.0]
            )
            for i, s in enumerate(FAST_SHARES)
            for j, m in enumerate(MISSLAP_RATES)
        ],
        "vectors_3": lambda: vectors(3),
        "vectors_4": lambda: vectors(4),
    }


def main() -> None:
    """Play every recorded run that `runs.json` lacks, saving after each."""
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    with Pool() as pool:
        for name, play_run in large_runs(pool).items():
            if name not in runs:
                runs[name] = play_run()
                RUNS.write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n")
                print(f"{name}: saved to {RUNS.name}", flush=True)


if __name__ == "__main__":
    main()
