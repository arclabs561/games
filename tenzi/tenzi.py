"""How many turns does a game of Tenzi take?

Each player rolls ten dice, keeps the dice showing one face, and rerolls the rest until all
ten match. Under the **fixed-face** strategy the player picks the most common face after the
first roll and never changes it. The **switching** strategy changes face whenever the rerolled
dice show more of another face than are kept. Either way the number of matching dice is a
Markov chain, so the distribution of turns can be computed exactly; a seeded simulation of
each strategy checks the exact numbers.

    uv run python -m tenzi.tenzi           # exact summary, simulations, and tenzi/turns.csv
    uv run python -m tenzi.tenzi --plot    # also a histogram with a fitted gamma density

Only --plot needs the packages in the header; the rest uses the standard library.
"""

import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb, factorial
from pathlib import Path
from statistics import mean, stdev

DICE = 10
FACES = 6
P = Fraction(1, FACES)  # chance that one rerolled die shows the chosen face
CSV = Path(__file__).resolve().parent / "turns.csv"

type Dist = dict[int, Fraction]


def compositions(n: int, k: int):
    """Every way to write n as an ordered sum of k nonnegative counts."""
    if k == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in compositions(n - first, k - 1):
            yield (first, *rest)


def first_roll() -> Dist:
    """Distribution of the largest number of dice showing one face after rolling all ten."""
    dist: Counter[int] = Counter()
    for counts in compositions(DICE, FACES):
        ways = factorial(DICE)
        for c in counts:
            ways //= factorial(c)
        dist[max(counts)] += ways
    return {m: Fraction(w, FACES**DICE) for m, w in sorted(dist.items())}


def reroll(m: int) -> Dist:
    """With m dice matching, the distribution of matching dice after rerolling the rest."""
    r = DICE - m
    return {m + j: comb(r, j) * P**j * (1 - P) ** (r - j) for j in range(r + 1)}


def reroll_switching(m: int) -> Dist:
    """Like `reroll`, but a larger group of another face among the rerolled dice is kept instead."""
    r = DICE - m
    dist: defaultdict[int, Fraction] = defaultdict(Fraction)
    for counts in compositions(r, FACES):  # counts[0] is the kept face
        ways = factorial(r)
        for c in counts:
            ways //= factorial(c)
        dist[max(m + counts[0], *counts[1:])] += Fraction(ways, FACES**r)
    return dict(dist)


def turn_distribution(max_turns: int = 200, step=reroll) -> Dist:
    """P(the strategy finishes on turn t), for t = 1 .. max_turns; `step` is its reroll."""
    dist, state = {}, first_roll()
    for t in range(1, max_turns + 1):
        dist[t] = state.get(DICE, Fraction(0))
        nxt: defaultdict[int, Fraction] = defaultdict(Fraction)
        for m, pr in state.items():
            if m < DICE:
                for m2, q in step(m).items():
                    nxt[m2] += pr * q
        state = nxt
    return dist


def turns_left(step=reroll) -> Dist:
    """Expected further turns to finish from m matching dice (0 when m = 10)."""
    e: Dist = {DICE: Fraction(0)}
    for m in range(DICE - 1, 0, -1):
        moves = step(m)
        gain = sum((q * e[m2] for m2, q in moves.items() if m2 != m), Fraction(0))
        e[m] = (1 + gain) / (1 - moves.get(m, Fraction(0)))
    return e


def expected_turns(step=reroll) -> Fraction:
    """Expected turns for the strategy, counting the first roll."""
    e = turns_left(step)
    return 1 + sum((pr * e[m] for m, pr in first_roll().items()), Fraction(0))


assert sum(first_roll().values()) == 1
assert turn_distribution(2)[1] == Fraction(1, FACES ** (DICE - 1))  # six ways, one per face
assert turns_left()[DICE - 1] == FACES  # the last die needs six turns on average
assert all(sum(reroll_switching(m).values()) == 1 for m in range(1, DICE))
assert reroll_switching(9) == reroll(9)  # one rerolled die cannot outnumber nine


def play_fixed(rng: random.Random) -> int:
    """Turns to finish, keeping the face that was most common after the first roll."""
    first = Counter(rng.randint(1, FACES) for _ in range(DICE))
    face, matched = first.most_common(1)[0]
    turns = 1
    while matched < DICE:
        matched += sum(rng.randint(1, FACES) == face for _ in range(DICE - matched))
        turns += 1
    return turns


def play_switching(rng: random.Random) -> int:
    """Turns to finish, switching to whichever face is most common after each roll."""
    dice = [rng.randint(1, FACES) for _ in range(DICE)]
    face = Counter(dice).most_common(1)[0][0]
    turns = 1
    while dice.count(face) < DICE:
        dice = [d if d == face else rng.randint(1, FACES) for d in dice]
        counts = Counter(dice)
        best = max(counts.values())
        if counts[face] < best:
            face = min(f for f, c in counts.items() if c == best)
        turns += 1
    return turns


def summary(samples: list[int]) -> str:
    """Mean with a 95% interval, and the largest sample."""
    half = 1.96 * stdev(samples) / len(samples) ** 0.5
    return f"mean {mean(samples):.3f} ± {half:.3f}, longest {max(samples)}"


def write_csv(upto: int = 60) -> None:
    """Write P(finish on turn t) for both strategies, for t = 1 .. upto."""
    fixed = turn_distribution(upto)
    switching = turn_distribution(upto, step=reroll_switching)
    rows = ["turns,fixed,switching"]
    rows += [f"{t},{float(fixed[t]):.6e},{float(switching[t]):.6e}" for t in range(1, upto + 1)]
    CSV.write_text("\n".join(rows) + "\n")


def main() -> None:
    print("Largest count after the first roll:")
    for m, pr in first_roll().items():
        print(f"  {m:2}: {float(pr):.4%}")
    dist = turn_distribution()
    print(f"P(done on the first roll) = 1/6^9 = {float(dist[1]):.3e}")
    left = turns_left()
    print("\nFixed face, expected turns still to come with m dice matching:")
    print("  " + ", ".join(f"{m}: {float(left[m]):.2f}" for m in range(2, DICE)))
    tail = 1 - sum(p for t, p in dist.items() if t <= 30)
    print(f"Fixed face, P(more than 30 turns) = {float(tail):.4f}")
    print("\nExact, by strategy:")
    for name, step in [("fixed face", reroll), ("switching", reroll_switching)]:
        d = turn_distribution(step=step)
        cum, median = Fraction(0), None
        for t in sorted(d):
            cum += d[t]
            if median is None and cum >= Fraction(1, 2):
                median = t
        mode = max(d, key=lambda t: d[t])
        mean_t = float(expected_turns(step))
        print(f"  {name:11}  mean {mean_t:.4f} turns, median {median}, most likely {mode}")

    n = 100_000
    fixed = [play_fixed(random.Random(i)) for i in range(n)]
    switching = [play_switching(random.Random(i)) for i in range(n)]
    print(f"\nSimulated, {n:,} games each (game i seeded with i):")
    print(f"  fixed face:  {summary(fixed)}")
    print(f"  switching:   {summary(switching)}")

    write_csv()
    print(f"\nwrote {CSV}")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        shape, loc, scale = stats.gamma.fit(fixed, floc=1)
        print(f"gamma fit: shape {shape:.3f}, scale {scale:.3f} (location fixed at 1)")
        x = np.arange(1, 61)
        plt.bar(x, [float(dist[int(t)]) for t in x], color="#bbbbbb", label="exact")
        plt.plot(x, stats.gamma(shape, loc, scale).pdf(x), color="#0072B2", label="gamma fit")
        plt.xlabel("turns")
        plt.ylabel("probability")
        plt.legend()
        plt.savefig("tenzi.png", dpi=150)
        print("wrote tenzi.png")


if __name__ == "__main__":
    main()
