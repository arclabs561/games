# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
# ]
# ///
"""Pareto frontier of Mastermind guessing strategies.

Implements three strategies with different tradeoffs between average guesses and
worst-case guesses, evaluates each over many random codes, then finds the Pareto
frontier of (avg_guesses, worst_case_guesses).

pare equivalents:
  - Pareto frontier: pare::pareto_front (non-dominated sort on N objectives)
  - Dominance check: pare::dominates (point-wise comparison)
  - Pareto ranking: pare::pareto_rank (assign rank 0..k to each point)
  - Multi-objective optimization uses the same domination logic to select
    strategy configurations that are not strictly worse than any other.
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter

COLORS = 6
HOLES = 4


def grade(code: list[int], guess: list[int]) -> tuple[int, int]:
    """Return (exact_matches, color_matches) for guess against code."""
    exact = sum(c == g for c, g in zip(code, guess))
    code_counts = Counter(code)
    guess_counts = Counter(guess)
    total_color = sum(min(code_counts[c], guess_counts[c]) for c in code_counts)
    return exact, total_color - exact


def all_codes() -> list[list[int]]:
    return [list(c) for c in itertools.product(range(COLORS), repeat=HOLES)]


# --- Strategy 1: Random consistent ---
# Each guess is a random code consistent with all previous feedback.
# Fast per-guess but can get unlucky.
def strategy_random_consistent(code: list[int]) -> int:
    candidates = all_codes()
    guess = random.choice(candidates)
    guesses = 1
    while guess != code:
        response = grade(code, guess)
        candidates = [c for c in candidates if grade(c, guess) == response]
        guess = random.choice(candidates)
        guesses += 1
    return guesses


# --- Strategy 2: Minimax size ---
# Pick the guess that minimizes the maximum remaining partition size.
# Slower per-guess, better worst-case.
def strategy_minimax(code: list[int]) -> int:
    candidates = all_codes()
    guess = [0, 0, 1, 1]  # known good first guess for 6-color 4-hole
    guesses = 1
    while guess != code:
        response = grade(code, guess)
        candidates = [c for c in candidates if grade(c, guess) == response]
        if len(candidates) == 1:
            guess = candidates[0]
        else:
            # Among candidates, pick the one that minimizes max partition.
            best_guess = None
            best_max_part = len(candidates) + 1
            # Only search over candidates (not full universe) for speed.
            sample = candidates if len(candidates) <= 50 else random.sample(candidates, 50)
            for g in sample:
                partitions: dict[tuple[int, int], int] = {}
                for c in candidates:
                    key = grade(c, g)
                    partitions[key] = partitions.get(key, 0) + 1
                max_part = max(partitions.values())
                if max_part < best_max_part:
                    best_max_part = max_part
                    best_guess = g
            guess = best_guess  # type: ignore[assignment]
        guesses += 1
    return guesses


# --- Strategy 3: Entropy maximizer ---
# Pick the guess that maximizes Shannon entropy of the partition distribution.
# Good average case.
def strategy_entropy(code: list[int]) -> int:
    import math

    candidates = all_codes()
    guess = [0, 0, 1, 1]
    guesses = 1
    while guess != code:
        response = grade(code, guess)
        candidates = [c for c in candidates if grade(c, guess) == response]
        if len(candidates) == 1:
            guess = candidates[0]
        else:
            best_guess = None
            best_entropy = -1.0
            sample = candidates if len(candidates) <= 50 else random.sample(candidates, 50)
            for g in sample:
                partitions: dict[tuple[int, int], int] = {}
                for c in candidates:
                    key = grade(c, g)
                    partitions[key] = partitions.get(key, 0) + 1
                n = len(candidates)
                entropy = -sum((cnt / n) * math.log2(cnt / n) for cnt in partitions.values())
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_guess = g
            guess = best_guess  # type: ignore[assignment]
        guesses += 1
    return guesses


def evaluate_strategy(
    name: str,
    strategy_fn,
    n_games: int,
) -> tuple[str, float, int]:
    """Run strategy over n_games random codes, return (name, avg, worst)."""
    results = []
    for _ in range(n_games):
        code = [random.randint(0, COLORS - 1) for _ in range(HOLES)]
        results.append(strategy_fn(code))
    avg = sum(results) / len(results)
    worst = max(results)
    return name, avg, worst


def pareto_front(
    points: list[tuple[str, float, int]],
) -> list[tuple[str, float, int]]:
    """Find non-dominated points. Minimizing both avg and worst.

    A point P dominates Q if P.avg <= Q.avg AND P.worst <= Q.worst,
    with at least one strict inequality.
    """
    front = []
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            # q dominates p?
            if q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def main() -> None:
    n_games = 30
    if "--fast" in sys.argv:
        n_games = 10

    print(f"Evaluating 3 strategies over {n_games} random codes each...")
    print(f"(Code space: {COLORS} colors, {HOLES} holes)\n")

    strategies = [
        ("random_consistent", strategy_random_consistent),
        ("minimax", strategy_minimax),
        ("entropy", strategy_entropy),
    ]

    results = []
    for name, fn in strategies:
        name, avg, worst = evaluate_strategy(name, fn, n_games)
        results.append((name, avg, worst))
        print(f"  {name:<20s}  avg={avg:.2f}  worst={worst}")

    front = pareto_front(results)
    print(f"\n=== Pareto frontier ({len(front)}/{len(results)} strategies) ===")
    print("(Non-dominated on (avg_guesses, worst_case_guesses), both minimized)")
    for name, avg, worst in sorted(front, key=lambda x: x[1]):
        print(f"  {name:<20s}  avg={avg:.2f}  worst={worst}")

    dominated = [r for r in results if r not in front]
    if dominated:
        print("\nDominated strategies:")
        for name, avg, worst in dominated:
            print(f"  {name:<20s}  avg={avg:.2f}  worst={worst}")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, avg, worst in results:
            color = "tab:blue" if (name, avg, worst) in front else "tab:gray"
            ax.scatter(avg, worst, s=100, color=color, zorder=3)
            ax.annotate(name, (avg, worst), textcoords="offset points", xytext=(5, 5))

        front_sorted = sorted(front, key=lambda x: x[1])
        ax.plot(
            [p[1] for p in front_sorted],
            [p[2] for p in front_sorted],
            "b--",
            alpha=0.5,
            label="Pareto front",
        )
        ax.set_xlabel("Average guesses (lower is better)")
        ax.set_ylabel("Worst-case guesses (lower is better)")
        ax.set_title("Mastermind strategy Pareto frontier")
        ax.legend()
        fig.tight_layout()
        plt.savefig("mastermind_pareto.png", dpi=150)
        print("\nSaved mastermind_pareto.png")
        plt.close()


if __name__ == "__main__":
    main()
