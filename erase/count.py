"""Count the deals that cannot be cleared with +, - and x (about 35 seconds).

Prints 12362536 2530881913560 1 in 204,722.
With --no-zero (a deck of 1 to 10, four of each, 40 cards) it prints the count for that deck.
"""

import sys
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


def main():
    dead, total = stuck(zero="--no-zero" not in sys.argv)
    print(dead, total, f"1 in {total / dead:,.0f}")


if __name__ == "__main__":
    main()
