"""The best way to play Tenzi, as a Markov decision process solved exactly.

After each roll the player sees how many dice show each face and decides which dice to keep;
the rest are rerolled, and every roll costs one turn. Faces are interchangeable, so all that
matters about a roll is its **counts**: the sorted multiset of how many dice show each face,
such as (4, 3, 2, 1). A **keep** is the counts of the dice set aside. Two rule sets are solved:

- *one face*: keep some or all of the dice of one face, or none (the rules of the game);
- *any dice*: keep any dice at all, of several faces, which the rules do not allow.

Policy iteration with exact fractions starts from the switching strategy (keep every die of
the most common face) and stops when no keep improves on it. For ten dice and six faces it
stops at once under both rule sets, so switching is optimal; `brute_force` confirms the method
on small games by value iteration over every ordered roll and every subset of dice.

`by_turn` asks a stronger question: does switching also give the largest chance of being done
by turn t, for every t at once? A finite-horizon dynamic program per t, in whole numbers,
says yes for ten dice up to the turn where that chance is within 10^-12 of 1. A strategy that
is best for every deadline is best for any race (see tenzi.typ). `brute_force_by_turn` checks
the program on small games over every ordered roll and every subset of dice.

    uv run python -m tenzi.optimal
"""

from fractions import Fraction
from functools import cache
from itertools import product
from math import factorial

from tenzi.tenzi import compositions

type Counts = tuple[int, ...]  # dice per face, largest first, zeros dropped


def counts(values) -> Counts:
    """The sorted counts of a sequence of per-face counts."""
    return tuple(sorted((v for v in values if v), reverse=True))


def partitions(n: int, parts: int, largest: int | None = None):
    """Every Counts of n dice on at most `parts` faces, each count at most `largest`."""
    largest = n if largest is None else largest
    if n == 0:
        yield ()
        return
    for first in range(min(n, largest), 0, -1):
        if parts > 0:
            for rest in partitions(n - first, parts - 1, first):
                yield (first, *rest)


assert list(partitions(4, 2)) == [(4,), (3, 1), (2, 2)]


@cache
def outcomes(keep: Counts, dice: int, faces: int) -> dict[Counts, Fraction]:
    """Distribution of the counts after keeping `keep` and rerolling the other dice."""
    r = dice - sum(keep)
    kept = keep + (0,) * (faces - len(keep))
    dist: dict[Counts, Fraction] = {}
    for rolled in compositions(r, faces):
        ways = factorial(r)
        for c in rolled:
            ways //= factorial(c)
        after = counts(k + c for k, c in zip(kept, rolled, strict=True))
        dist[after] = dist.get(after, Fraction(0)) + Fraction(ways, faces**r)
    return dist


assert outcomes((1,), 2, 2) == {(2,): Fraction(1, 2), (1, 1): Fraction(1, 2)}


def keeps(roll: Counts, any_dice: bool) -> list[Counts]:
    """The keeps allowed after `roll`; at least one die is always rerolled."""
    if not any_dice:
        return [(), *((k,) for k in range(1, roll[0] + 1))]
    options = {counts(k) for k in product(*(range(c + 1) for c in roll))}
    options.discard(roll)
    return sorted(options)


assert keeps((2, 1), any_dice=False) == [(), (1,), (2,)]
assert keeps((2, 1), any_dice=True) == [(), (1,), (1, 1), (2,)]


def solve_linear(rows: list[list[Fraction]]) -> list[Fraction]:
    """Solve a square system given as augmented rows, by Gauss-Jordan elimination."""
    n = len(rows)
    for c in range(n):
        pivot = next(r for r in range(c, n) if rows[r][c] != 0)
        rows[c], rows[pivot] = rows[pivot], rows[c]
        rows[c] = [x / rows[c][c] for x in rows[c]]
        for r in range(n):
            if r != c and rows[r][c] != 0:
                f = rows[r][c]
                rows[r] = [x - f * y for x, y in zip(rows[r], rows[c], strict=True)]
    return [row[n] for row in rows]


def evaluate(policy: dict[Counts, Counts], dice: int, faces: int) -> dict[Counts, Fraction]:
    """Expected further turns from each roll when the player follows `policy` (roll -> keep)."""
    rolls = list(policy)
    index = {roll: i for i, roll in enumerate(rolls)}
    rows = []
    for roll in rolls:  # V(roll) - sum P(roll -> next) V(next) = 1
        row = [Fraction(0)] * len(rolls) + [Fraction(1)]
        row[index[roll]] += 1
        for nxt, p in outcomes(policy[roll], dice, faces).items():
            if nxt != (dice,):
                row[index[nxt]] -= p
        rows.append(row)
    values = dict(zip(rolls, solve_linear(rows), strict=True))
    values[(dice,)] = Fraction(0)
    return values


def solve(dice: int = 10, faces: int = 6, any_dice: bool = False):
    """Optimal policy, its expected number of turns, and the value of every keep it compared.

    The expected number of turns counts the first roll: it is the value of keeping nothing.
    Ties keep the current choice, so a policy that is already optimal is returned unchanged.
    """
    rolls = [roll for roll in partitions(dice, faces) if roll != (dice,)]
    policy: dict[Counts, Counts] = {roll: (roll[0],) for roll in rolls}  # switching
    while True:
        values = evaluate(policy, dice, faces)

        @cache
        def keep_value(keep: Counts, values=values) -> Fraction:
            dist = outcomes(keep, dice, faces)
            return 1 + sum((p * values[nxt] for nxt, p in dist.items()), Fraction(0))

        better = {
            roll: min(keeps(roll, any_dice), key=lambda k: (keep_value(k), k != policy[roll]))
            for roll in rolls
        }
        if better == policy:
            return (
                policy,
                keep_value(()),
                {k: keep_value(k) for r in rolls for k in keeps(r, any_dice)},
            )
        policy = better


def brute_force(dice: int, faces: int, tol: float = 1e-13) -> float:
    """Optimal expected turns by value iteration over ordered rolls and subsets of dice to keep.

    Feasible only for small games: faces**dice rolls, each with 2**dice - 1 choices.
    """
    rolls = list(product(range(faces), repeat=dice))
    value = dict.fromkeys(rolls, 0.0)
    while True:
        change = 0.0
        for roll in rolls:
            if len(set(roll)) == 1:
                continue
            best = float("inf")
            for mask in range(2**dice - 1):  # bit i set: keep die i
                free = [i for i in range(dice) if not mask >> i & 1]
                total = 0.0
                for new in product(range(faces), repeat=len(free)):
                    after = list(roll)
                    for i, face in zip(free, new, strict=True):
                        after[i] = face
                    total += value[tuple(after)]
                best = min(best, 1 + total / faces ** len(free))
            change = max(change, abs(best - value[roll]))
            value[roll] = best
        if change < tol:
            return 1 + sum(value.values()) / len(rolls)


def by_turn(
    dice: int = 10, faces: int = 6, any_dice: bool = False, within: Fraction = Fraction(1, 10**12)
) -> tuple[list[Fraction], list[tuple[int, Counts, list[Counts]]]]:
    """The best chance of being done by turn t, for t = 1, 2, ... until it is within `within` of 1.

    V_t(roll) is the best chance of finishing within t more rolls after seeing `roll`; it is the
    largest, over keeps, of sum P(keep -> next) V_(t-1)(next). Values are kept as whole numbers
    times faces^(-dice t), so the arithmetic is exact and fast. Returns the best chances from the
    start, and every (t, roll, best keeps) where keeping every die of the most common face is
    not among the best keeps; switching is best for every deadline when that list is empty.
    """
    scale, done = faces**dice, (dice,)
    weights = {  # outcomes(keep) times faces^dice, as whole numbers
        keep: {nxt: int(p * scale) for nxt, p in outcomes(keep, dice, faces).items()}
        for roll in partitions(dice, faces)
        for keep in keeps(roll, any_dice)
    }
    rolls = [roll for roll in partitions(dice, faces) if roll != done]
    value = {roll: int(roll == done) for roll in partitions(dice, faces)}  # V_0
    best_chances, worse = [], []
    for t in range(1, 10**6):
        start = sum(w * value[nxt] for nxt, w in weights[()].items())
        best_chances.append(Fraction(start, scale**t))  # first roll, then t - 1 more
        if 1 - best_chances[-1] < within:
            return best_chances, worse
        new = {done: scale * value[done]}
        for roll in rolls:
            options = {
                keep: sum(w * value[nxt] for nxt, w in weights[keep].items())
                for keep in keeps(roll, any_dice)
            }
            new[roll] = max(options.values())
            if options[(roll[0],)] != new[roll]:
                worse.append((t, roll, [k for k, v in options.items() if v == new[roll]]))
        value = new
    raise AssertionError("unreachable")


def brute_force_by_turn(dice: int, faces: int, turns: int) -> list[float]:
    """The best chance of being done by turn t, for t = 1 .. turns, over ordered rolls and every
    subset of dice to keep. Feasible only for small games."""
    rolls = list(product(range(faces), repeat=dice))
    value = {roll: float(len(set(roll)) == 1) for roll in rolls}  # done within 0 more rolls
    chances = []
    for _ in range(turns):
        chances.append(sum(value.values()) / len(rolls))
        new = {}
        for roll in rolls:
            if len(set(roll)) == 1:
                new[roll] = 1.0
                continue
            best = 0.0
            for mask in range(2**dice - 1):  # bit i set: keep die i
                free = [i for i in range(dice) if not mask >> i & 1]
                total = 0.0
                for faces_rolled in product(range(faces), repeat=len(free)):
                    after = list(roll)
                    for i, face in zip(free, faces_rolled, strict=True):
                        after[i] = face
                    total += value[tuple(after)]
                best = max(best, total / faces ** len(free))
            new[roll] = best
        value = new
    return chances


def main() -> None:
    print("Small games: exact optimum (one face, any dice) and value iteration over subsets")
    for dice, faces in [(3, 2), (4, 2), (5, 2), (3, 3), (4, 3)]:
        one = solve(dice, faces)[1]
        anyd = solve(dice, faces, any_dice=True)[1]
        print(
            f"  {dice} dice, {faces} faces: {one} = {anyd} = {float(one):.6f}, "
            f"value iteration {brute_force(dice, faces):.6f}"
        )

    for any_dice in (False, True):
        policy, best, values = solve(any_dice=any_dice)
        rule = "any dice" if any_dice else "one face"
        switching = all(keep == (roll[0],) for roll, keep in policy.items())
        print(f"\nTen dice, six faces, {rule}: optimal mean {float(best):.10f} turns")
        print(f"  the optimal policy keeps every die of the most common face: {switching}")
        margin = min(
            min(v for k, v in values.items() if k != keep and k in keeps(roll, any_dice))
            - values[keep]
            for roll, keep in policy.items()
        )
        print(f"  smallest margin over the next best keep: {float(margin):.4f} turns")
    print("\nExpected turns still to come after keeping k dice of one face (exact):")
    print("  " + ", ".join(f"{k}: {float(values[(k,) if k else ()]):.3f}" for k in range(10)))
    print(f"  as a fraction, the optimal mean is {best}")

    print("\nBest chance of being done by turn t, one finite-horizon program per t:")
    for any_dice in (False, True):
        chances, worse = by_turn(any_dice=any_dice)
        rule = "any dice" if any_dice else "one face"
        print(
            f"  {rule}: up to t = {len(chances)}, where it is {float(chances[-1]):.13f},"
            f" switching is among the best keeps at every roll and every t: {not worse}"
        )
    small = [
        (dice, faces, any_dice)
        for dice in range(2, 7)
        for faces in range(2, 7)
        for any_dice in (False, True)
        if by_turn(dice, faces, any_dice)[1]
    ]
    print(f"  two to six dice on two to six faces, games where switching is not best: {small}")
    print("  t, best chance by turn t:", end="")
    for t in (5, 10, 14, 20, 30, 50):
        print(f"  {t}: {float(chances[t - 1]):.4f}", end="")
    print()


if __name__ == "__main__":
    main()
