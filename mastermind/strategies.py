"""Mastermind guessing strategies, scored on every secret code.

A **code** is four pegs, each one of six colors, so there are 6^4 = 1,296 codes. After each
**guess** the codemaker gives a **reply**: how many pegs are the right color in the right
place, and how many more are a right color in the wrong place. The **consistent codes** are
those that would have given every reply so far. A guess **partitions** the consistent codes by
the reply each one would give.

Each strategy below picks its next guess from the partitions. Deterministic strategies are
compiled once into a guess tree and then played against all 1,296 codes, so the averages are
exact. The random strategy plays every code once with a fixed seed. The script prints each
strategy's average, worst case and distribution, and which of these strategies are on the
Pareto frontier of (average, worst case), both minimized, among themselves. The same dominance
test is `pare::pareto_front` in the Rust crate pare.

The four published strategies and the first-consistent one reproduce, guess for guess, the
distributions in Table 7 of B. Kooi, "Yet another Mastermind strategy", ICGA Journal 28(1),
2005. Of these, only minimax never needs a sixth guess. None is on the frontier over all
strategies: exhaustive search finds a total of 5,625 guesses (average 4.340) with worst case
6, and 5,626 (4.341) with worst case 5 (Koyama and Lai, 1993; reproduced by optimal.py).

    uv run python -m mastermind.strategies          # about half a minute
    uv run python -m mastermind.strategies --plot   # also writes mastermind_strategies.png
"""

import itertools
import math
import random
import sys
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from mastermind.mastermind import score

COLORS = 6
HOLES = 4
SEED = 1

type Code = tuple[int, ...]
type Reply = int  # 10 * (right place) + (right color, wrong place)
type Strategy = Callable[[list[int]], int]  # consistent code indices -> guess index

CODES: list[Code] = list(itertools.product(range(COLORS), repeat=HOLES))
WIN: Reply = 10 * HOLES


def reply(code: Code, guess: Code) -> Reply:
    """The codemaker's reply to `guess` when the secret is `code`, as one number."""
    exact, near = score(code, guess)
    return 10 * exact + near


assert reply((0, 0, 1, 1), (0, 0, 1, 1)) == WIN
assert reply((0, 1, 2, 3), (3, 2, 1, 0)) == 4  # all four colors right, none in place
assert reply((0, 0, 0, 1), (0, 1, 1, 1)) == 20  # extra 1s in the guess earn nothing
assert reply((1, 1, 2, 2), (2, 2, 1, 1)) == reply((2, 2, 1, 1), (1, 1, 2, 2)) == 4

# REPLIES[g][s] is the reply to guess g when the secret is s, for code indices g and s.
REPLIES: list[list[Reply]] = [[reply(s, g) for s in CODES] for g in CODES]


def partition_sizes(guess: int, consistent: list[int]) -> list[int]:
    """How many consistent codes give each possible reply to `guess`."""
    row = REPLIES[guess]
    return list(Counter(row[s] for s in consistent).values())


def best_guess(consistent: list[int], cost: Callable[[list[int]], float]) -> int:
    """The guess whose partition sizes have the lowest cost, preferring a consistent code,
    then the lowest index.

    This is Knuth's tie-break. Any of the 1,296 codes may be guessed, consistent or not.
    """
    if len(consistent) <= 2:
        return consistent[0]
    preferred = set(consistent)
    return min(
        range(len(CODES)),
        key=lambda g: (cost(partition_sizes(g, consistent)), g not in preferred, g),
    )


# The cost of a partition, given only its part sizes; each strategy picks the guess of
# lowest cost. `fallout3.py` uses the same four costs on terminal word lists.


def largest_part(sizes: list[int]) -> float:
    """Knuth (1977): the size of the largest part."""
    return max(sizes)


def fewer_parts(sizes: list[int]) -> float:
    """Kooi (2005): minus the number of parts."""
    return -len(sizes)


def sum_of_squares(sizes: list[int]) -> float:
    """Irving (1979): the sum of squared part sizes, which is n times the expected size of
    the part the secret falls in."""
    return sum(k * k for k in sizes)


def negative_entropy(sizes: list[int]) -> float:
    """Neuwirth (1982): minus the entropy of the reply, in bits."""
    n = sum(sizes)
    return sum(k / n * math.log2(k / n) for k in sizes)


def minimax(consistent: list[int]) -> int:
    """Knuth (1977): make the largest partition as small as possible."""
    return best_guess(consistent, largest_part)


def most_parts(consistent: list[int]) -> int:
    """Kooi (2005): make as many partitions as possible."""
    return best_guess(consistent, fewer_parts)


def expected_size(consistent: list[int]) -> int:
    """Irving (1979): make the expected size of the remaining partition as small as possible."""
    return best_guess(consistent, sum_of_squares)


def entropy(consistent: list[int]) -> int:
    """Neuwirth (1982): make the reply as informative as possible (maximum entropy)."""
    return best_guess(consistent, negative_entropy)


def first_consistent(consistent: list[int]) -> int:
    """Guess the lowest-numbered code that is still possible."""
    return consistent[0]


def guess_counts(strategy: Strategy) -> list[int]:
    """Guesses needed for each of the 1,296 secret codes, by compiling a guess tree."""
    counts = [0] * len(CODES)

    def play(consistent: list[int], depth: int) -> None:
        guess = strategy(consistent)
        branches: dict[Reply, list[int]] = {}
        for s in consistent:
            branches.setdefault(REPLIES[guess][s], []).append(s)
        for r, codes in branches.items():
            if r == WIN:
                counts[guess] = depth
            else:
                play(codes, depth + 1)

    play(list(range(len(CODES))), 1)
    return counts


def random_consistent_counts(seed: int = SEED) -> list[int]:
    """Guesses needed for each secret code when every guess is a random consistent code."""
    rng = random.Random(seed)
    counts = []
    for secret in range(len(CODES)):
        consistent, depth = list(range(len(CODES))), 1
        while (guess := rng.choice(consistent)) != secret:
            consistent = [s for s in consistent if REPLIES[guess][s] == REPLIES[guess][secret]]
            depth += 1
        counts.append(depth)
    return counts


def pareto_front(points: dict[str, tuple[float, int]]) -> set[str]:
    """Names whose (average, worst) no other point beats in one and matches or beats in both."""

    def dominates(p: tuple[float, int], q: tuple[float, int]) -> bool:
        return p[0] <= q[0] and p[1] <= q[1] and p != q

    return {n for n, q in points.items() if not any(dominates(p, q) for p in points.values())}


def main() -> None:
    strategies: dict[str, Strategy] = {
        "most parts": most_parts,
        "expected size": expected_size,
        "entropy": entropy,
        "minimax": minimax,
        "first consistent": first_consistent,
    }
    results = {name: guess_counts(fn) for name, fn in strategies.items()}
    results[f"random consistent (seed {SEED})"] = random_consistent_counts()

    worst_overall = max(max(c) for c in results.values())
    print(f"Guesses needed over all {len(CODES):,} codes ({COLORS} colors, {HOLES} pegs)\n")
    header = "".join(f"{k:>6}" for k in range(1, worst_overall + 1))
    print(f"{'strategy':<28}{'total':>7}{'average':>9}{'worst':>7}{header}")
    points = {}
    for name, counts in results.items():
        dist = Counter(counts)
        average = Fraction(sum(counts), len(counts))
        points[name] = (float(average), max(counts))
        row = "".join(f"{dist[k]:>6}" for k in range(1, worst_overall + 1))
        print(f"{name:<28}{sum(counts):>7}{float(average):>9.3f}{max(counts):>7}{row}")

    front = pareto_front(points)
    print("\nPareto frontier among these strategies (average and worst case both minimized):")
    for name in sorted(front, key=lambda n: points[n]):
        print(f"  {name}: average {points[name][0]:.3f}, worst {points[name][1]}")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        stacked = Counter()  # labels of points with the same worst case go one above another
        for name, (average, worst) in sorted(points.items(), key=lambda item: item[1]):
            color = "#0072B2" if name in front else "#999999"
            ax.scatter(average, worst, s=60, color=color, zorder=3)
            ax.annotate(
                name,
                (average, worst),
                textcoords="offset points",
                xytext=(6, 4 + 13 * stacked[worst]),
                fontsize=9,
            )
            stacked[worst] += 1
        ax.set_yticks(range(5, worst_overall + 2))
        ax.set_ylim(4.6, worst_overall + 1)
        ax.set_xlabel("Average guesses over all codes")
        ax.set_ylabel("Most guesses for any code")
        ax.set_title("Mastermind strategies (blue: Pareto frontier)")
        fig.tight_layout()
        png = Path(__file__).with_name("mastermind_strategies.png")
        fig.savefig(png, dpi=150)
        print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
