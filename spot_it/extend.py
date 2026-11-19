"""Which decks with the Spot It property extend to a projective plane?

Fix n and v = n² + n + 1 symbols. A **family** is a set of m cards of n + 1 symbols each, drawn
from those v symbols, every two sharing exactly one symbol. Its **deficit** is d = v - m, and
the deficit of a symbol is n + 1 minus the number of cards showing it. A family **extends** if
some projective plane of order n on the same symbols contains it, and is **maximal** if no
card can be added. The README proves:

- every symbol's deficit is at least 0, the deficits add up to d(n + 1), the products
  deficit × (deficit - 1) add up to d(d - 1), and the deficits on each card add up to d;
- a symbol x of deficit 1 forces a card: x with the n symbols that never appear with x;
- so every family with m ≥ n² cards extends, in exactly one way, and for odd n so does every
  family with m = n² - 1.

This module checks those statements by computer: the identities on many families, the forced
cards, and, for n = 2 and 3, every maximal family that contains two fixed cards. It also counts
the completions of a fixed pencil to test that the planes of orders 2, 3 and 4 are unique, and
runs seeded random searches for large maximal families that are not planes.

    uv run python -m spot_it.extend      # about 35 seconds
"""

import random
from collections import Counter
from itertools import combinations
from math import factorial

from spot_it.missing import cards_meeting_once
from spot_it.plane import (
    Card,
    Deck,
    has_spot_it_property,
    is_projective_plane,
    projective_plane,
    symbol_counts,
)


def deficits(deck: Deck, n: int) -> list[int]:
    return [n + 1 - c for c in symbol_counts(deck, n * n + n + 1)]


def identities_hold(deck: Deck, n: int) -> bool:
    """The three counting identities and the card-sum identity of the README."""
    v = n * n + n + 1
    d = v - len(deck)
    delta = deficits(deck, n)
    return (
        min(delta) >= 0
        and sum(delta) == d * (n + 1)
        and sum(x * (x - 1) for x in delta) == d * (d - 1)
        and all(sum(delta[s] for s in card) == d for card in deck)
    )


def forced_card(deck: Deck, n: int, x: int) -> Card:
    """Symbol x together with every symbol that never appears on a card with x."""
    v = n * n + n + 1
    beside = {s for card in deck if x in card for s in card}
    return frozenset({x} | (set(range(v)) - beside))


def complete(deck: Deck, n: int) -> Deck:
    """Add forced cards while some symbol has deficit 1; return the family reached."""
    deck = list(deck)
    while True:
        delta = deficits(deck, n)
        ones = [s for s, x in enumerate(delta) if x == 1]
        if not ones:
            return deck
        deck.append(forced_card(deck, n, ones[0]))


def random_subfamilies(n: int, trials: int, seed: int):
    """Random subsets of PG(2, n), every size from 0 to v, `trials` of each size."""
    rng = random.Random(seed)
    plane = projective_plane(n)
    for m in range(len(plane) + 1):
        for _ in range(trials):
            yield rng.sample(plane, m)


# Maximal families by exhaustive search, for small n.


def two_fixed_cards(n: int) -> tuple[Card, Card]:
    """{0, 1, ..., n} and {0, n + 1, ..., 2n}. Any two cards sharing one symbol look like these."""
    return frozenset(range(n + 1)), frozenset({0} | set(range(n + 1, 2 * n + 1)))


def maximal_families(n: int) -> list[Deck]:
    """Every maximal family that contains the two fixed cards (Bron–Kerbosch with pivoting)."""
    v = n * n + n + 1
    a, b = two_fixed_cards(n)
    cand = [
        frozenset(c)
        for c in combinations(range(v), n + 1)
        if len(a & set(c)) == 1 and len(b & set(c)) == 1
    ]
    nbr = [0] * len(cand)
    for i, j in combinations(range(len(cand)), 2):
        if len(cand[i] & cand[j]) == 1:
            nbr[i] |= 1 << j
            nbr[j] |= 1 << i
    out: list[Deck] = []

    def bits(x: int):
        while x:
            low = x & -x
            yield low.bit_length() - 1
            x ^= low

    def bk(r: list[int], p: int, x: int) -> None:
        if not p and not x:
            out.append([a, b, *(cand[i] for i in r)])
            return
        pivot = max(bits(p | x), key=lambda u: (nbr[u] & p).bit_count())
        for u in bits(p & ~nbr[pivot]):
            bk([*r, u], p & nbr[u], x & nbr[u])
            p &= ~(1 << u)
            x |= 1 << u

    bk([], (1 << len(cand)) - 1, 0)
    return out


def summarize_maximal(n: int) -> dict:
    """Sizes of the maximal families through the two fixed cards, planes and others."""
    fams = maximal_families(n)
    planes = [f for f in fams if is_projective_plane(f, n)]
    others = [f for f in fams if len(f) < n * n + n + 1]
    assert len(planes) + len(others) == len(fams)
    for f in others:  # maximal, so no card can be added: confirm with the exact-cover search
        assert not cards_meeting_once(f, n * n + n + 1, n + 1, limit=1)
        assert n * n + n + 1 - len(f) >= n + 2 + n % 2
    return {
        "maximal": len(fams),
        "planes": len(planes),
        "other sizes": dict(sorted(Counter(len(f) for f in others).items())),
        "largest other": max(others, key=len) if others else None,
        "other profiles": Counter(
            tuple(sorted(Counter(symbol_counts(f, n * n + n + 1)).items())) for f in others
        ),
    }


def complete_graph_family(n: int) -> Deck:
    """n + 2 cards, one per vertex of the complete graph on n + 2 vertices; the symbols are its
    edges, and a card shows the n + 1 edges at its vertex. Two cards share their one edge."""
    edges = {e: i for i, e in enumerate(combinations(range(n + 2), 2))}
    deck = [frozenset(i for e, i in edges.items() if u in e) for u in range(n + 2)]
    assert has_spot_it_property(deck)
    return deck


def five_card_decks_of_order_two() -> Counter[str]:
    """Every deck of five 3-symbol cards with the Spot It property, symbols from 0 to 10 (enough
    for any such deck), that contains {0, 1, 2}: is it a sunflower, or on how many symbols?"""
    triples = [frozenset(c) for c in combinations(range(11), 3)]
    tally: Counter[str] = Counter()

    def grow(deck: list[Card], start: int) -> None:
        if len(deck) == 5:
            common = frozenset.intersection(*deck)
            tally["sunflower" if common else f"{len(frozenset().union(*deck))} symbols"] += 1
            return
        for i in range(start, len(triples)):
            if all(len(triples[i] & c) == 1 for c in deck):
                grow([*deck, triples[i]], i + 1)

    grow([triples[0]], 1)
    return tally


# Uniqueness of the planes of order 2, 3 and 4, by counting completions.


def automorphisms(deck: Deck) -> int:
    """Number of permutations of the symbols that map cards to cards (backtracking)."""
    v = len(set().union(*deck))
    line = {}
    for i, card in enumerate(deck):
        for s, t in combinations(card, 2):
            line[s, t] = line[t, s] = i
    order = list(range(v))
    count = 0

    def extend(image: dict[int, int], line_map: dict[int, int], used: set[int]) -> None:
        nonlocal count
        if len(image) == v:
            count += 1
            return
        x = order[len(image)]
        for y in range(v):
            if y in used:
                continue
            new = dict(line_map)
            ok = True
            for a, a2 in image.items():
                src, dst = line[a, x], line[a2, y]
                if new.get(src, dst) != dst or (src not in new and dst in new.values()):
                    ok = False
                    break
                new[src] = dst
            if ok:
                extend({**image, x: y}, new, used | {y})

    extend({}, {}, set())
    return count


def pencil_and_line(n: int) -> Deck:
    """n + 1 cards through symbol 0 that cover every symbol, and one card that avoids 0."""
    pencil = [frozenset({0} | set(range(1 + n * i, 1 + n * (i + 1)))) for i in range(n + 1)]
    line = frozenset(1 + n * i for i in range(n + 1))
    return [*pencil, line]


def count_completions(deck: Deck, n: int) -> int:
    """Number of projective planes of order n on 0..v-1 that contain every card of the deck."""
    v = n * n + n + 1
    covered = {pair for card in deck for pair in combinations(sorted(card), 2)}
    total = 0

    def search(cards: list[Card], pairs: set) -> None:
        nonlocal total
        if len(cards) == v:
            total += 1
            return
        # The first symbol pair on no card yet must go on exactly one new card.
        s, t = next((s, t) for s, t in combinations(range(v), 2) if (s, t) not in pairs)
        for card in cards_meeting_once(cards, v, n + 1):
            if (
                s in card
                and t in card
                and not any(p in pairs for p in combinations(sorted(card), 2))
            ):
                search([*cards, card], pairs | set(combinations(sorted(card), 2)))

    search(list(deck), covered)
    return total


def expected_completions(n: int, aut: int) -> int:
    """Completions of `pencil_and_line(n)` if every plane of order n has `aut` automorphisms.

    A plane has v pencils (one per point) and v - n - 1 cards avoiding each pencil's point. On v
    labeled symbols there are v · (v - 1)! / ((n!)^(n + 1) · (n + 1)!) pencils and n^(n + 1)
    cards avoiding the point, one symbol from each card of the pencil. The plane has v! / aut
    labelings.
    """
    v = n * n + n + 1
    pencils = v * factorial(v - 1) // (factorial(n) ** (n + 1) * factorial(n + 1))
    configs = pencils * n ** (n + 1)
    num = factorial(v) * v * (v - n - 1)
    assert num % (aut * configs) == 0
    return num // (aut * configs)


# Random searches for maximal families that are not planes.


def random_card(deck: Deck, n: int, rng: random.Random) -> Card | None:
    """One card that meets every card of the deck once, by a randomized search; None if none."""
    v = n * n + n + 1
    covers = [0] * v
    for i, card in enumerate(deck):
        for s in card:
            covers[s] |= 1 << i
    unused = [s for s in range(v) if not covers[s]]
    full = (1 << len(deck)) - 1

    def search(chosen: list[int], hit: int) -> Card | None:
        if len(chosen) > n + 1:
            return None
        if hit == full:
            pad = n + 1 - len(chosen)
            return (
                frozenset(chosen) | frozenset(rng.sample(unused, pad))
                if pad <= len(unused)
                else None
            )
        first = (~hit & full) & -(~hit & full)
        symbols = sorted(deck[first.bit_length() - 1])
        rng.shuffle(symbols)
        for s in symbols:
            if not covers[s] & hit:
                found = search([*chosen, s], hit | covers[s])
                if found is not None:
                    return found
        return None

    return search([], 0)


def search_maximal(n: int, removed: range, trials: int, seed: int) -> set[int]:
    """Remove k random cards from PG(2, n), add random fitting cards until none fits, and keep
    the sizes of the maximal families reached that are not planes."""
    rng = random.Random(seed)
    plane = projective_plane(n)
    sizes = set()
    for k in removed:
        for _ in range(trials):
            deck = rng.sample(plane, len(plane) - k)
            while (card := random_card(deck, n, rng)) is not None:
                deck.append(card)
            if not is_projective_plane(deck, n):
                assert not cards_meeting_once(
                    deck, len(plane), n + 1, limit=1
                )  # maximal, checked again
                assert len(plane) - len(deck) >= n + 2 + n % 2  # as the README proves
                sizes.add(len(deck))
    return sizes


def main() -> None:
    # 1. Identities and forced completion on random subfamilies of the planes.
    for n in (2, 3, 4, 5, 7, 8):
        checked = 0
        for fam in random_subfamilies(n, trials=20, seed=n):
            assert identities_hold(fam, n)
            d = n * n + n + 1 - len(fam)
            if d <= n + 1 or (n % 2 == 1 and d == n + 2):
                full = complete(fam, n)
                assert is_projective_plane(full, n)
            checked += 1
        print(f"order {n}: identities hold on {checked} random subfamilies of PG(2,{n})")

    print("five-card decks of order 2 containing {0, 1, 2}:", dict(five_card_decks_of_order_two()))

    # 2. Every maximal family through two fixed cards, n = 2 and 3.
    for n in (2, 3):
        s = summarize_maximal(n)
        print(
            f"order {n}: {s['maximal']} maximal families, {s['planes']} planes,"
            f" other sizes {s['other sizes']}"
        )
        if s["largest other"]:
            print("   largest other:", [sorted(c) for c in s["largest other"]])
            print(
                "   symbols on 0, 1, 2, ... cards, for each other family:",
                dict(s["other profiles"]),
            )
    for n in range(2, 10):
        fam = complete_graph_family(n)
        v = n * n + n + 1
        maximal = not cards_meeting_once(fam, v, n + 1, limit=1)
        print(f"order {n}: the complete-graph family of {n + 2} cards is maximal: {maximal}")
        assert maximal == (n % 2 == 1)

    # 3. Uniqueness: completions of a pencil and one more card.
    for n in (2, 3, 4):
        aut = automorphisms(projective_plane(n))
        exp = expected_completions(n, aut)
        got = count_completions(pencil_and_line(n), n)
        print(f"order {n}: |Aut PG(2,{n})| = {aut}, completions {got}, expected if unique {exp}")
        assert got == exp

    # 4. Random searches for large maximal families that are not planes.
    for n, removed, trials in (
        (3, range(5, 11), 30),
        (4, range(6, 19), 30),
        (5, range(7, 29), 30),
        (7, range(30, 46), 20),
    ):
        found = search_maximal(n, removed, trials, seed=n)
        print(
            f"order {n}: largest maximal family found that is not a plane: {max(found)} cards,"
            f" sizes found {sorted(found)}"
        )


if __name__ == "__main__":
    main()
