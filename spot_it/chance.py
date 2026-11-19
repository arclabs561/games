"""Chances in the 55-card deck: which symbol two cards share, and the Triplet mini-game.

Every number here is exact: it comes from counting cards of PG(2, 7), as fractions.

- **Symbol counts.** In the 57-card plane every symbol is on 8 cards. With two cards removed,
  one symbol is on 6, fourteen are on 7 and the other 42 are on 8.
- **The shared symbol.** Two cards drawn from the deck share exactly one symbol; symbol s is
  that symbol with chance C(c_s, 2) / C(m, 2), where c_s cards of the m show it.
- **Three cards.** The chance that three cards drawn from the deck all show one symbol.
- **Triplet.** Nine cards lie face up and players look for three with a common symbol. A
  **triplet-free** set of cards is one in which no symbol is on three of them. Nine cards are
  never triplet-free, and the counts for fewer cards come from an exhaustive search.

    uv run python -m spot_it.chance    # about 5 seconds
"""

from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import comb

from spot_it.field import gf
from spot_it.plane import Card, Deck, projective_plane, symbol_counts

ORDER = 7
PLANE = projective_plane(ORDER)
V = len(PLANE)  # 57


def deck_55(i: int = 0, j: int = 1) -> Deck:
    """The plane of order 7 without cards i and j."""
    return [c for k, c in enumerate(PLANE) if k not in (i, j)]


def count_profile(deck: Deck) -> dict[int, int]:
    """How many symbols are on 0, 1, 2, ... cards of the deck."""
    return dict(sorted(Counter(symbol_counts(deck, V)).items()))


def shared_symbol(deck: Deck) -> dict[int, Fraction]:
    """Chance that two cards drawn from the deck share symbol s, for each symbol s."""
    pairs = comb(len(deck), 2)
    return {s: Fraction(comb(c, 2), pairs) for s, c in enumerate(symbol_counts(deck, V))}


def shared_symbol_by_pairs(deck: Deck) -> dict[int, Fraction]:
    """The same chances by listing every pair of cards (the slow check)."""
    tally: Counter[int] = Counter()
    for a, b in combinations(deck, 2):
        (s,) = a & b
        tally[s] += 1
    pairs = comb(len(deck), 2)
    return {s: Fraction(tally[s], pairs) for s in range(V)}


def three_share(deck: Deck) -> Fraction:
    """Chance that three cards drawn from the deck show a common symbol."""
    return Fraction(sum(comb(c, 3) for c in symbol_counts(deck, V)), comb(len(deck), 3))


def fixed_symbol_count() -> dict[int, Fraction]:
    """For a fixed symbol and two cards removed at random, the chance it is left on 8, 7 or 6."""
    tally: Counter[int] = Counter()
    for i, j in combinations(range(V), 2):
        tally[8 - (0 in PLANE[i]) - (0 in PLANE[j])] += 1
    total = comb(V, 2)
    return {c: Fraction(t, total) for c, t in sorted(tally.items())}


def triplet_free_counts(deck: Deck, kmax: int = 10) -> list[int]:
    """Number of k-card subsets of the deck with no symbol on three cards, for k = 0..kmax.

    Depth-first over cards in index order. `union` holds the symbols on some chosen card and
    `covered` the symbols on two; a card can join when it shows no covered symbol.
    """
    masks = [sum(1 << s for s in card) for card in deck]
    counts = [0] * (kmax + 1)

    def dfs(k: int, cands: list[int], union: int, covered: int) -> None:
        counts[k] += 1
        if k == kmax:
            return
        for idx, i in enumerate(cands):
            new_covered = covered | (masks[i] & union)
            rest = [j for j in cands[idx + 1 :] if not masks[j] & new_covered]
            dfs(k + 1, rest, union | masks[i], new_covered)

    dfs(0, list(range(len(deck))), 0, 0)
    return counts


def triplet_free_sets(deck: Deck, k: int) -> set[frozenset[int]]:
    """The k-card triplet-free subsets themselves, as sets of card positions in the deck."""
    masks = [sum(1 << s for s in card) for card in deck]
    found: set[frozenset[int]] = set()

    def dfs(chosen: list[int], cands: list[int], union: int, covered: int) -> None:
        if len(chosen) == k:
            found.add(frozenset(chosen))
            return
        for idx, i in enumerate(cands):
            new_covered = covered | (masks[i] & union)
            rest = [j for j in cands[idx + 1 :] if not masks[j] & new_covered]
            dfs([*chosen, i], rest, union | masks[i], new_covered)

    dfs([], list(range(len(deck))), 0, 0)
    return found


def conics(q: int) -> set[Card]:
    """Zero sets of the nondegenerate quadratic forms on GF(q)³, as sets of point positions.

    Points are numbered as in `projective_plane`, where point i and card i are the same vector.
    Cards a, b, c share a symbol exactly when the vectors a, b, c are linearly dependent, that
    is, when points a, b, c lie on a line; so a triplet-free set of cards is the same index set
    as a set of points with no three on a line. A form ax² + by² + cz² + dxy + exz + fyz is
    nondegenerate when its zero set is not a line, a point, two lines or empty; for odd q those
    zero sets have q + 1 points with no three on a line, and those are the ones kept here.
    """
    f = gf(q)
    plane = projective_plane(q)
    points = [
        v
        for v in product(f.elements, repeat=3)
        if any(v) and v[next(i for i in range(3) if v[i])] == 1
    ]
    add, mul = f.add, f.mul

    def value(coef, x, y, z):
        a, b, c, d, e, g = coef
        terms = (
            mul[a][mul[x][x]],
            mul[b][mul[y][y]],
            mul[c][mul[z][z]],
            mul[d][mul[x][y]],
            mul[e][mul[x][z]],
            mul[g][mul[y][z]],
        )
        t = 0
        for term in terms:
            t = add[t][term]
        return t

    out = set()
    for coef in product(f.elements, repeat=6):
        if not any(coef):
            continue
        zeros = frozenset(i for i, (x, y, z) in enumerate(points) if value(coef, x, y, z) == 0)
        if len(zeros) == q + 1 and all(len(zeros & line) <= 2 for line in plane):
            out.add(zeros)
    return out


def triplet_free_formula(k: int) -> int:
    """k-card triplet-free sets of the full plane for k <= 5, counted by hand.

    Choose cards one at a time. The first is any of 57; the second any of the other 56. The
    third avoids the point the first two share: 57 - 8 = 49 choices. The fourth avoids the three
    points the first three share in pairs, 21 cards in all: 36 choices. The fifth avoids the six
    points of four cards in general position, which lie on 37 cards: 20 choices. Divide by k!.
    """
    choices = [57, 56, 49, 36, 20]
    total = 1
    for c in choices[:k]:
        total *= c
    for i in range(2, k + 1):
        total //= i
    return total


def show(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator} = {float(x):.6f}"


def main() -> None:
    deck = deck_55()
    print("symbols on 6, 7, 8 cards:", count_profile(deck))
    profiles = Counter(
        tuple(count_profile(deck_55(i, j)).items()) for i, j in combinations(range(V), 2)
    )
    print(f"the same for all {comb(V, 2)} pairs of removed cards:", len(profiles) == 1)
    print(
        "a fixed symbol, two cards removed at random:",
        {c: show(p) for c, p in fixed_symbol_count().items()},
    )

    print("\nTwo cards drawn, chance that they share a given symbol:")
    full = shared_symbol(PLANE)
    print("  57 cards, every symbol:", show(full[0]))
    chances = shared_symbol(deck)
    assert chances == shared_symbol_by_pairs(deck)
    counts = symbol_counts(deck, V)
    for c in (8, 7, 6):
        s = counts.index(c)
        print(f"  55 cards, a symbol on {c} cards:", show(chances[s]))

    print("\nThree cards drawn, chance that they show a common symbol:")
    print("  57 cards:", show(three_share(PLANE)))
    print("  56 cards:", show(three_share(PLANE[1:])))
    print("  55 cards:", show(three_share(deck)))
    point = next(iter(PLANE[0] & PLANE[1]))
    through = [k for k, c in enumerate(PLANE) if point in c][:3]
    triangle = [
        0,
        1,
        next(k for k, c in enumerate(PLANE) if not c & (PLANE[0] & PLANE[1]) and k > 1),
    ]
    for name, removed in (
        ("three cards with a common symbol", through),
        ("three cards without", triangle),
    ):
        rest = [c for k, c in enumerate(PLANE) if k not in removed]
        print(f"  54 cards, removing {name}:", show(three_share(rest)))

    print("\nTriplet-free sets of k cards (k = 0..10):")
    full_counts = triplet_free_counts(PLANE)
    print("  57 cards:", full_counts)
    assert all(full_counts[k] == triplet_free_formula(k) for k in range(6))
    ovals = triplet_free_sets(PLANE, 8)
    print(
        f"  the {len(ovals)} triplet-free 8-card sets are exactly the conics:",
        ovals == conics(ORDER),
    )
    sevens = {frozenset(c) for oval in ovals for c in combinations(oval, 7)}
    print(
        "  their 7-card subsets are all different and are all the triplet-free sets of seven:",
        len(sevens) == 8 * len(ovals) == full_counts[7],
    )
    counts55 = triplet_free_counts(deck)
    print("  55 cards:", counts55)
    other = triplet_free_counts(deck_55(5, 40))
    print("  55 cards, another pair removed, same counts:", other == counts55)
    for k in (7, 8, 9):
        print(
            f"  chance that {k} cards drawn from 55 are triplet-free:",
            show(Fraction(counts55[k], comb(55, k))),
        )
    print("  expected triplets among 9 cards:", show(comb(9, 3) * three_share(deck)))


if __name__ == "__main__":
    main()
