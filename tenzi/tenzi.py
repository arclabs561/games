"""How many turns does a game of Tenzi take, and who wins the race?

Each player rolls ten dice, keeps the dice showing one face, and rerolls the rest until all
ten match. Three strategies are compared. The **target-first** strategy names a face before the
first roll. The **fixed-face** strategy picks the most common face after the first roll and
never changes it. The **switching** strategy changes face whenever the rerolled dice show more
of another face than are kept. In each the number of matching dice is a Markov chain, so the
distribution of turns can be computed exactly; so can a race between n players who roll in
lockstep. Seeded simulations check the exact numbers. `tenzi.optimal` shows that no strategy
beats switching.

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


def target_first() -> Dist:
    """Matching dice after the first roll when the face is named before it: Bin(10, 1/6)."""
    return reroll(0)


def state_distributions(max_turns: int = 200, step=reroll, first: Dist | None = None) -> list[Dist]:
    """Element t - 1 gives P(m dice match after turn t, and the game did not end earlier).

    `first` is the count after the first roll (default `first_roll()`), `step` the reroll.
    """
    states, state = [], first_roll() if first is None else first
    for _ in range(max_turns):
        states.append(state)
        nxt: defaultdict[int, Fraction] = defaultdict(Fraction)
        for m, pr in state.items():
            if m < DICE:
                for m2, q in step(m).items():
                    nxt[m2] += pr * q
        state = nxt
    return states


def turn_distribution(max_turns: int = 200, step=reroll, first: Dist | None = None) -> Dist:
    """P(the strategy finishes on turn t), for t = 1 .. max_turns; `step` is its reroll."""
    states = state_distributions(max_turns, step, first)
    return {t: s.get(DICE, Fraction(0)) for t, s in enumerate(states, 1)}


def turns_left(step=reroll) -> Dist:
    """Expected further turns to finish from m matching dice (0 when m = 10)."""
    e: Dist = {DICE: Fraction(0)}
    for m in range(DICE - 1, -1, -1):
        moves = step(m)
        gain = sum((q * e[m2] for m2, q in moves.items() if m2 != m), Fraction(0))
        e[m] = (1 + gain) / (1 - moves.get(m, Fraction(0)))
    return e


def expected_turns(step=reroll, first: Dist | None = None) -> Fraction:
    """Expected turns for the strategy, counting the first roll."""
    e, first = turns_left(step), first_roll() if first is None else first
    return 1 + sum((pr * e[m] for m, pr in first.items()), Fraction(0))


STRATEGIES = {  # name: (count after the first roll, reroll step)
    "target first": (target_first(), reroll),
    "fixed face": (first_roll(), reroll),
    "switching": (first_roll(), reroll_switching),
}


def target_first_moments() -> tuple[Fraction, Fraction]:
    """Mean and variance of the turns when the face is named first, in closed form.

    Each die needs a Geometric(1/6) number of turns, and the game lasts the longest of the ten:
    P(T > k) = 1 - (1 - q^k)^10 = sum_j (-1)^(j+1) C(10, j) q^(jk) with q = 5/6. Summing over
    k gives E[T] = sum P(T > k) and E[T^2] = sum (2k + 1) P(T > k).
    """
    q = 1 - P
    terms = [((-1) ** (j + 1) * comb(DICE, j), q**j) for j in range(1, DICE + 1)]
    mean_t = sum(c / (1 - x) for c, x in terms)
    square = sum(c * (1 + x) / (1 - x) ** 2 for c, x in terms)
    return mean_t, square - mean_t**2


def moments(dist: Dist) -> tuple[Fraction, Fraction]:
    """Mean and variance of a distribution over turns."""
    m = sum((t * p for t, p in dist.items()), Fraction(0))
    return m, sum(((t - m) ** 2 * p for t, p in dist.items()), Fraction(0))


def race(n: int, states: list) -> tuple[list, float | Fraction, float | Fraction]:
    """Race of n players who roll together, from one player's `state_distributions`.

    Returns P(the first finish is on turn t) for t = 1, 2, ..., P(two or more players finish
    on that turn), and the expected number of players with nine dice matched when it comes.
    The numbers can be Fractions or floats; the sums stop where `states` does.
    """
    done = [s.get(DICE, 0) for s in states]
    survive = [1]  # survive[t] = P(one player is not done after turn t)
    for p in done:
        survive.append(survive[-1] - p)
    win = [survive[t - 1] ** n - survive[t] ** n for t in range(1, len(states) + 1)]
    tie = sum(
        w - n * p * survive[t] ** (n - 1) for t, (w, p) in enumerate(zip(win, done, strict=True), 1)
    )
    one_left = n * sum(
        s.get(DICE - 1, 0) * (survive[t - 1] ** (n - 1) - survive[t] ** (n - 1))
        for t, s in enumerate(states, 1)
    )
    return win, tie, one_left


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


def play(rng: random.Random, strategy: str) -> list[int]:
    """Dice matching the chosen face after each turn, for one of the names in STRATEGIES."""
    dice = [rng.randint(1, FACES) for _ in range(DICE)]
    face = 1 if strategy == "target first" else Counter(dice).most_common(1)[0][0]
    matched = [dice.count(face)]
    while matched[-1] < DICE:
        dice = [d if d == face else rng.randint(1, FACES) for d in dice]
        if strategy == "switching":
            counts = Counter(dice)
            if counts[face] < max(counts.values()):
                face = counts.most_common(1)[0][0]
        matched.append(dice.count(face))
    return matched


def play_race(rng: random.Random, n: int, strategy: str) -> tuple[int, int, int]:
    """One simulated race: the winning turn, how many finish on it, how many have nine matched."""
    games = [play(rng, strategy) for _ in range(n)]
    turn = min(len(g) for g in games)
    at_end = [g[turn - 1] for g in games]
    return turn, at_end.count(DICE), at_end.count(DICE - 1)


# Each simulation draws all its games from its own generator, so no two simulations share
# random numbers and each is an independent check of the exact values.
SEEDS = {
    "target first": 1,
    "fixed face": 2,
    "switching": 3,
    "race, target first": 4,
    "race, switching": 5,
}
assert len(set(SEEDS.values())) == len(SEEDS)


def play_target_first(rng: random.Random) -> int:
    """Turns to finish, with the face named before the first roll."""
    return len(play(rng, "target first"))


GAMES = {"target first": play_target_first, "fixed face": play_fixed, "switching": play_switching}


def simulate(name: str, n: int) -> list[int]:
    """Turns in n games of one strategy, all drawn from that strategy's own generator."""
    rng = random.Random(SEEDS[name])
    return [GAMES[name](rng) for _ in range(n)]


def interval(samples: list[int] | list[float]) -> str:
    """Mean with a 95% interval."""
    half = 1.96 * stdev(samples) / len(samples) ** 0.5
    return f"{mean(samples):.3f} ± {half:.3f}"


def summary(samples: list[int]) -> str:
    """Mean with a 95% interval, and the largest sample."""
    return f"mean {interval(samples)}, longest {max(samples)}"


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
    print("\nExact, by strategy (turns count the first roll):")
    for name, (first, step) in STRATEGIES.items():
        d = turn_distribution(step=step, first=first)
        cum, median = Fraction(0), None
        for t in sorted(d):
            cum += d[t]
            if median is None and cum >= Fraction(1, 2):
                median = t
        mode = max(d, key=lambda t: d[t])
        mean_t, var = moments(d)
        assert abs(mean_t - expected_turns(step, first)) < Fraction(1, 10**9)
        print(
            f"  {name:12}  mean {float(mean_t):.4f}, sd {float(var) ** 0.5:.4f},"
            f" median {median}, most likely {mode}"
        )
    mean_t, var = target_first_moments()
    print(f"  target first, closed form: mean {float(mean_t):.6f}, sd {float(var) ** 0.5:.6f}")

    n = 100_000
    target, fixed, switching = (
        simulate(name, n) for name in ("target first", "fixed face", "switching")
    )
    print(f"\nSimulated, {n:,} games each, one generator per strategy:")
    print(f"  target first (seed {SEEDS['target first']}): {summary(target)}")
    print(f"  fixed face (seed {SEEDS['fixed face']}):   {summary(fixed)}")
    print(f"  switching (seed {SEEDS['switching']}):    {summary(switching)}")

    print("\nRace of n players rolling together, exact (sums stop at 200 turns):")
    for name in ("target first", "switching"):
        first, step = STRATEGIES[name]
        states = [
            {m: float(p) for m, p in s.items()} for s in state_distributions(200, step, first)
        ]
        print(f"  {name}:  n, mean winning turn, P(tie), players with nine dice matched")
        for players in (2, 3, 4, 5, 6, 10, 20, 30):
            win, tie, one_left = race(players, states)
            mean_win = sum(t * w for t, w in enumerate(win, 1))
            print(f"    {players:2}  {mean_win:7.3f}  {tie:.4f}  {one_left:.3f}")

    races, players = 20_000, 4
    for name in ("target first", "switching"):
        seed = SEEDS[f"race, {name}"]
        rng = random.Random(seed)
        sims = [play_race(rng, players, name) for _ in range(races)]
        turns = [t for t, _, _ in sims]
        ties = [float(k > 1) for _, k, _ in sims]
        nines = [c for _, _, c in sims]
        print(f"\nSimulated, {races:,} races of {players} players, {name} (seed {seed}):")
        print(f"  winning turn {interval(turns)}, P(tie) {interval(ties)},", end=" ")
        print(f"players with nine dice matched {interval(nines)}")

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
        png = Path(__file__).with_name("tenzi.png")
        plt.savefig(png, dpi=150)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
