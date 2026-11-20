"""Reproduce the simulated figures in the paper (a few minutes). Seeds are fixed.

For each deck it deals 30,000 random deals and reports how often a deal cannot be
cleared with addition only and with + and -, how many of the + and - failures are
the parity obstruction, and how many leave exactly one card under best play.
For the 0 to 10 deck it also plays "grab the pairs first" on 30,000 deals that
can be cleared with +, - and x. Last, for each deck, it asks how often a deal can
be cleared with small groups only, and with every group worked as a running total
(no brackets).
"""

import random
import sys
from collections import Counter
from functools import cache

from erase import make, solve

N = 30000


def groups(hand, center, minus):
    """Bitmasks of hand cards that form a group, with or without one center card.

    With + only, a group is a set of cards that sums to 10. With + and -, brackets
    reach every choice of signs with at least one plus, so a set makes 10 exactly
    when some of its cards sum to (total + 10) / 2.
    """
    n = len(hand)
    total = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        total[mask] = total[mask ^ low] + hand[low.bit_length() - 1]
    out = []
    for mask in range(1, 1 << n):
        size = bin(mask).count("1")
        if not minus:
            ok = (size > 1 and total[mask] == 10) or any(total[mask] + c == 10 for c in center)
        else:
            parts = set()
            sub = mask
            while True:
                parts.add(total[sub])
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            ok = False
            for c in [None] + sorted(set(center)):
                if size + (c is not None) < 2:
                    continue
                t = total[mask] + (c or 0) + 10
                if t % 2 == 0 and (t // 2 in parts or (c is not None and t // 2 - c in parts)):
                    ok = True
                    break
        if ok:
            out.append(mask)
    return out


def most_erased(n, gs):
    """The largest number of hand cards that disjoint groups can erase."""
    best = [0] * (1 << n)
    size = {g: bin(g).count("1") for g in gs}
    for mask in range(1, 1 << n):
        low = mask & -mask
        b = best[mask ^ low]
        for g in gs:
            if g & low and g & mask == g:
                b = max(b, size[g] + best[mask ^ g])
        best[mask] = b
    return best[-1]


def share(k, n):
    p = k / n
    return f"{p:.1%} +- {1.96 * (p * (1 - p) / n) ** 0.5:.1%}"


def readings(cards, seed, n=N):
    """Deals stuck with addition only and with + and -, and how the + and - failures look."""
    rng = random.Random(seed)
    add = pm = parity = one = 0
    for _ in range(n):
        rng.shuffle(cards)
        hand, center = sorted(cards[:7]), sorted(cards[7:11])
        if most_erased(7, groups(hand, center, False)) < 7:
            add += 1
        left = 7 - most_erased(7, groups(hand, center, True))
        if left:
            pm += 1
            one += left == 1
            parity += sum(v % 2 for v in hand) % 2 == 1 and all(c % 2 == 0 for c in center)
    return {"deals": n, "addition": add, "plus_minus": pm, "parity": parity, "one_left": one}


def pairs_first(cards, seed, n=N):
    """Of n deals that clear with +, - and x, how many grabbing the pairs first strands."""
    rng = random.Random(seed)
    done = bad = 0
    while done < n:
        rng.shuffle(cards)
        hand, center = tuple(sorted(cards[:7])), tuple(sorted(cards[7:11]))
        if solve(hand, center) is None:
            continue
        done += 1
        count = Counter(hand)
        rest = []
        for a in range(5):
            x, y = count[a], count[10 - a]
            if 10 - a not in center:
                rest += [a] * (x - min(x, y))
            if a not in center:
                rest += [10 - a] * (y - min(x, y))
        if count[5] % 2 and 5 not in center:
            rest.append(5)
        if rest and solve(tuple(sorted(rest)), center) is None:
            bad += 1
    return {"deals": n, "stranded": bad}


@cache
def chain(ms):
    """Values of running totals: start with one card, then +c, -c or xc, one card at a time."""
    if len(ms) == 1:
        return frozenset(ms)
    out = set()
    for i in range(len(ms)):
        if i and ms[i] == ms[i - 1]:
            continue
        c = ms[i]
        for v in chain(ms[:i] + ms[i + 1 :]):
            out |= {v + c, v - c, v * c}
    return frozenset(out)


def full_groups(hand, center, makes10, largest=8):
    """Bitmasks of hand cards that form a group of at most `largest` cards under +, - and x."""
    out = []
    for mask in range(1, 1 << len(hand)):
        cards = [hand[i] for i in range(len(hand)) if mask >> i & 1]
        for c in [None] + sorted(set(center)):
            group = tuple(sorted(cards + ([c] if c is not None else [])))
            if 2 <= len(group) <= largest and makes10(group):
                out.append(mask)
                break
    return out


def restricted(cards, seed, n=N):
    """Deals that clear with groups of at most three or four cards, or with running totals."""
    rng = random.Random(seed)
    three = four = running = 0
    for _ in range(n):
        rng.shuffle(cards)
        hand, center = sorted(cards[:7]), sorted(cards[7:11])
        three += most_erased(7, full_groups(hand, center, lambda g: make(g, 10), 3)) == 7
        four += most_erased(7, full_groups(hand, center, lambda g: make(g, 10), 4)) == 7
        running += most_erased(7, full_groups(hand, center, lambda g: 10 in chain(g))) == 7
    return {"deals": n, "three": three, "four": four, "running": running}


ZERO = [v for v in range(11) for _ in range(4)]
NO_ZERO = [v for v in range(1, 11) for _ in range(4)]


def run(n=N):
    """Every simulated figure, with the seeds the paper used."""
    return {
        "zero": readings(ZERO[:], 1, n),
        "no_zero": readings(NO_ZERO[:], 2, n),
        "pairs_first": pairs_first(ZERO[:], 3, n),
        "small_zero": restricted(ZERO[:], 4, n),
        "small_no_zero": restricted(NO_ZERO[:], 5, n),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N
    r = run(n)
    for name, key in (("cards 0 to 10", "zero"), ("cards 1 to 10", "no_zero")):
        x = r[key]
        print(
            f"{name}: addition only stuck {share(x['addition'], n)}; "
            f"+ and - stuck {share(x['plus_minus'], n)}"
        )
        print(
            f"{name}: of the {x['plus_minus']:,} + and - failures, {x['parity']:,} are the "
            f"parity obstruction and {x['one_left']:,} leave exactly one card"
        )
    print(
        f"cards 0 to 10: grabbing pairs first strands a card {share(r['pairs_first']['stranded'], n)}"
    )
    for name, key in (("cards 0 to 10", "small_zero"), ("cards 1 to 10", "small_no_zero")):
        x = r[key]
        print(
            f"{name}: clears with groups of at most three cards {share(x['three'], n)}; "
            f"at most four {share(x['four'], n)} ({x['four']:,} of {n:,})"
        )
        print(
            f"{name}: clears with every group a running total (no brackets) in {x['running']:,} of {n:,} deals"
        )


if __name__ == "__main__":
    main()
