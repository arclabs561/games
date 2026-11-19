"""Spot It decks from projective planes, and the one-shared-symbol property checked pair by pair.

A **deck** is a list of **cards**; a card is a frozenset of **symbols**, numbered 0 to v - 1.
A deck has the **Spot It property** if every two cards share exactly one symbol. A
**projective plane of order n** is a deck of n² + n + 1 cards with n + 1 symbols each, on
n² + n + 1 symbols, in which every two symbols also lie on exactly one card. The cards are the
plane's lines and the symbols its points.

Two constructions are compared:

- `projective_plane(q)`: points and lines of GF(q)³ in homogeneous coordinates. A symbol is a
  nonzero vector up to scaling, a card is one too, and symbol s is on card c when s · c = 0.
- `slope_deck(n, add, mul)`: the construction most Spot It generators use. The symbols are the
  n² points (x, y), one symbol per slope 0..n-1, and one for vertical lines. The cards are the
  lines y = ax + b with their slope symbol, the vertical lines x = c with the vertical symbol,
  and one card of all the slope symbols. With arithmetic mod n it is a plane only for prime n;
  with the tables of GF(q) it is a plane for every prime power q.

    uv run python -m spot_it.plane     # checks both constructions for every order up to 13
"""

from collections import Counter
from itertools import combinations, product

from spot_it.field import gf, integers_mod, prime_power

type Card = frozenset[int]
type Deck = list[Card]


def projective_plane(q: int) -> Deck:
    """PG(2, q): cards and symbols are the nonzero vectors of GF(q)³ whose first nonzero entry is 1."""
    f = gf(q)
    vectors = [
        v
        for v in product(f.elements, repeat=3)
        if any(v) and v[next(i for i in range(3) if v[i])] == 1
    ]
    index = {v: i for i, v in enumerate(vectors)}

    def dot(a, b):
        s = 0
        for x, y in zip(a, b, strict=True):
            s = f.add[s][f.mul[x][y]]
        return s

    deck = [frozenset(index[v] for v in vectors if dot(v, c) == 0) for c in vectors]
    return deck


def slope_deck(n: int, add=None, mul=None) -> Deck:
    """Lines y = ax + b and x = c over the given arithmetic (default: integers mod n), plus slopes."""
    if add is None or mul is None:
        add, mul = integers_mod(n)
    point = {(x, y): x * n + y for x in range(n) for y in range(n)}
    slope = {a: n * n + a for a in range(n)}
    vertical = n * n + n
    deck = [
        frozenset({point[x, add[mul[a][x]][b]] for x in range(n)} | {slope[a]})
        for a in range(n)
        for b in range(n)
    ]
    deck += [frozenset({point[c, y] for y in range(n)} | {vertical}) for c in range(n)]
    deck.append(frozenset(slope.values()) | {vertical})
    return deck


def field_slope_deck(q: int) -> Deck:
    f = gf(q)
    return slope_deck(q, f.add, f.mul)


def shared_counts(deck: Deck) -> Counter[int]:
    """How many pairs of cards share 0, 1, 2, ... symbols."""
    return Counter(len(a & b) for a, b in combinations(deck, 2))


def has_spot_it_property(deck: Deck) -> bool:
    return set(shared_counts(deck)) <= {1}


def symbol_counts(deck: Deck, v: int) -> list[int]:
    """Number of cards showing each symbol 0..v-1."""
    counts = [0] * v
    for card in deck:
        for s in card:
            counts[s] += 1
    return counts


def is_projective_plane(deck: Deck, n: int) -> bool:
    """Checks the definition directly: sizes, every pair of cards, every pair of symbols."""
    v = n * n + n + 1
    if len(deck) != v or any(len(c) != n + 1 for c in deck):
        return False
    if set().union(*deck) != set(range(v)):
        return False
    pairs = Counter(pair for card in deck for pair in combinations(sorted(card), 2))
    return (
        has_spot_it_property(deck) and len(pairs) == v * (v - 1) // 2 and set(pairs.values()) == {1}
    )


def main() -> None:
    print("order  cards  PG(2,q)  slopes over GF(q)  slopes mod n: pairs sharing 0, 1, 2+ symbols")
    for n in range(2, 14):
        mod = shared_counts(slope_deck(n))
        zero, one, more = mod[0], mod[1], sum(c for k, c in mod.items() if k >= 2)
        if prime_power(n):
            pg = is_projective_plane(projective_plane(n), n)
            sl = is_projective_plane(field_slope_deck(n), n)
            assert pg and sl
            fields = "plane    plane            "
        else:
            fields = "no field                  "
        v = n * n + n + 1
        print(f"{n:5}  {v:5}  {fields}{zero:5} {one:6} {more:5}")


if __name__ == "__main__":
    main()
