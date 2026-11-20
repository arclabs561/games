"""Count the deals that cannot be cleared with +, - and x (about 35 seconds).

Prints 12362536 2530881913560 1 in 204,722.
With --no-zero (a deck of 1 to 10, four of each, 40 cards) it prints the count for that deck.
With --addition it counts the deals that cannot be cleared with addition alone (about 2 min
for each deck), with --pairs those that the printed rules, pairs that sum to 10, cannot clear
(a second), and with --alone the seven-card hands that cannot be cleared with no center
cards, as in the original version of the game (about 10 s).
"""

import sys
from functools import cache
from itertools import combinations
from math import comb

from erase import multisets, solve


def ways(cards, avail):
    """How many ways to draw this multiset of values from the cards available."""
    n = 1
    for v in set(cards):
        n *= comb(avail[v], cards.count(v))
    return n


def stuck(zero=True):
    """(stuck deals, all deals) for the deck of 0 to 10, or of 1 to 10 without zero."""
    deck = [4] * 11
    if not zero:
        deck[0] = 0
    dead = 0
    for hand in multisets(7, deck[:]):
        if solve(hand, ()) is not None:  # most hands clear without help
            continue
        left = [deck[v] - hand.count(v) for v in range(11)]
        for center in multisets(4, left[:]):
            if solve(hand, center) is None:
                dead += ways(hand, deck) * ways(center, left)
    n = sum(deck)
    return dead, comb(n, 7) * comb(n - 7, 4)


def minimal(masks):
    """The masks that contain no other mask of the list."""
    out = []
    for m in sorted(set(masks), key=int.bit_count):
        if not any(o & m == o for o in out):
            out.append(m)
    return tuple(out)


@cache
def piles(hand):
    """With addition alone: the smallest sets of center values that let the hand clear.

    Each set is a bitmask with bit c for the center value c. The hand must split into piles
    that each sum to 10, from two or more hand cards, or to 10 - c with the center card c.
    A center holds four cards, so sets of more than four values are dropped.
    """
    if not hand:
        return (0,)
    first, rest = hand[0], hand[1:]
    out, seen = [], set()
    for k in range(len(rest) + 1):
        for pick in combinations(range(len(rest)), k):
            pile = tuple(rest[i] for i in pick)
            total = first + sum(pile)
            if pile in seen or total > 10:
                continue
            seen.add(pile)
            need = 0 if k and total == 10 else 1 << (10 - total)
            left = tuple(rest[i] for i in range(len(rest)) if i not in pick)
            out += [m | need for m in piles(left) if (m | need).bit_count() <= 4]
    return minimal(out)


def clears_by_addition(hand, center):
    have = sum(1 << c for c in set(center))
    return any(m & have == m for m in piles(tuple(sorted(hand))))


def stuck_addition(zero=True):
    """(deals that addition alone cannot clear, all deals); the deal counted as in stuck()."""
    deck = [4] * 11
    if not zero:
        deck[0] = 0
    dead = 0
    for hand in multisets(7, deck[:]):
        if 0 in piles(hand):  # clears with no center card
            continue
        left = [deck[v] - hand.count(v) for v in range(11)]
        dead += ways(hand, deck) * sum(
            ways(center, left)
            for center in multisets(4, left[:])
            if not clears_by_addition(hand, center)
        )
    n = sum(deck)
    return dead, comb(n, 7) * comb(n - 7, 4)


def pairs_needs(hand, lone_ten=False):
    """Under the printed rules, the center values a hand needs (the pairs-only proposition).

    Within the hand each a pairs with a 10 - a, and 5s pair with each other. A card left
    over needs its complement in the center. With lone_ten, a 10 may be erased by itself.
    """
    need = set()
    for a in range(5):
        x, y = hand.count(a), hand.count(10 - a)
        if x > y:
            need.add(10 - a)
        if y > x and not (a == 0 and lone_ten):
            need.add(a)
    if hand.count(5) % 2:
        need.add(5)
    return need


def stuck_pairs(zero=True, lone_ten=False):
    """(deals that pairs alone cannot clear, all deals), counting the centers that hold every
    needed value by inclusion and exclusion."""
    deck = [4] * 11
    if not zero:
        deck[0] = 0
    clear = 0
    for hand in multisets(7, deck[:]):
        need = sorted(pairs_needs(hand, lone_ten))
        left = [deck[v] - hand.count(v) for v in range(11)]
        centers = sum(
            (-1) ** k * comb(sum(left) - sum(left[v] for v in drop), 4)
            for k in range(len(need) + 1)
            for drop in combinations(need, k)
        )
        clear += ways(hand, deck) * centers
    n = sum(deck)
    total = comb(n, 7) * comb(n - 7, 4)
    return total - clear, total


def stuck_alone():
    """(seven-card hands that cannot be cleared with no center cards, all hands)."""
    deck = [4] * 11
    dead = sum(ways(h, deck) for h in multisets(7, deck[:]) if solve(h, ()) is None)
    return dead, comb(44, 7)


def main():
    if "--alone" in sys.argv:
        dead, total = stuck_alone()
    elif "--pairs" in sys.argv:
        dead, total = stuck_pairs(zero="--no-zero" not in sys.argv)
    elif "--addition" in sys.argv:
        dead, total = stuck_addition(zero="--no-zero" not in sys.argv)
    else:
        dead, total = stuck(zero="--no-zero" not in sys.argv)
    share = f"{100 * dead / total:.2f}%" if dead > total / 100 else f"1 in {total / dead:,.0f}"
    print(dead, total, share)


if __name__ == "__main__":
    main()
