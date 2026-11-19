"""The puzzle of the two missing cards: given 55 cards of the plane of order 7, find the other two.

**Counting method.** Count how often each symbol appears. In the full plane every symbol is on
n + 1 cards. The symbol both missing cards share is on n - 1 of the cards left, the other 2n
symbols of the missing cards are on n, and every other symbol is on n + 1. Two of the 2n
symbols belong to the same missing card exactly when no card left shows both of them, so
"never together" splits them into the two cards.

**Search method.** Shares no code with the first: list every set of n + 1 symbols that meets
every card left in exactly one symbol, by an exact-cover search over the cards. The answer is
unique when the search returns exactly the two removed cards.

    uv run python -m spot_it.missing   # both methods on every pair of removed cards, orders 2 to 9
"""

from collections import Counter
from itertools import combinations

from spot_it.plane import Card, Deck, projective_plane, symbol_counts


def together(deck: Deck) -> set[tuple[int, int]]:
    """Pairs of symbols (smaller first) that appear on a common card."""
    return {pair for card in deck for pair in combinations(sorted(card), 2)}


def missing_by_counting(deck: Deck, n: int) -> list[Card]:
    """The two cards that complete a plane of order n with two cards removed."""
    v = n * n + n + 1
    counts = symbol_counts(deck, v)
    (shared,) = [s for s in range(v) if counts[s] == n - 1]
    rest = sorted(s for s in range(v) if counts[s] == n)
    assert len(rest) == 2 * n and all(c in (n - 1, n, n + 1) for c in counts)
    seen = together(deck)
    first = rest[0]
    group = {first} | {s for s in rest[1:] if (first, s) not in seen}
    other = set(rest) - group
    return sorted([frozenset(group | {shared}), frozenset(other | {shared})], key=sorted)


def cards_meeting_once(deck: Deck, v: int, size: int, limit: int | None = None) -> list[Card]:
    """Every set of `size` symbols from 0..v-1 that meets each card of the deck exactly once.

    Each symbol "covers" the cards that show it; a valid set is an exact cover of the deck by
    the cover sets of its symbols, padded with symbols that appear on no card. With a limit,
    stop once that many have been found.
    """
    covers = [frozenset(i for i, card in enumerate(deck) if s in card) for s in range(v)]
    unused = [s for s in range(v) if not covers[s]]
    found: set[Card] = set()

    def search(chosen: list[int], hit: frozenset[int]) -> None:
        if len(chosen) > size or (limit is not None and len(found) >= limit):
            return
        if len(hit) == len(deck):
            for pad in combinations(unused, size - len(chosen)):
                found.add(frozenset(chosen) | frozenset(pad))
                if limit is not None and len(found) >= limit:
                    return
            return
        target = min(set(range(len(deck))) - hit)  # the first card not yet met
        for s in deck[target]:
            if not covers[s] & hit:
                search([*chosen, s], hit | covers[s])

    search([], frozenset())
    return sorted(found, key=sorted)


def check_order(n: int) -> Counter[str]:
    """Remove every pair of cards from PG(2, n) and solve the puzzle both ways."""
    plane = projective_plane(n)
    v = len(plane)
    outcome: Counter[str] = Counter()
    for i, j in combinations(range(v), 2):
        rest = [c for k, c in enumerate(plane) if k not in (i, j)]
        removed = sorted([plane[i], plane[j]], key=sorted)
        by_count = missing_by_counting(rest, n)
        by_search = cards_meeting_once(rest, v, n + 1)
        outcome["pairs"] += 1
        outcome["counting finds them"] += by_count == removed
        outcome["search finds exactly them"] += by_search == removed
    return outcome


def main() -> None:
    for n in (2, 3, 4, 5, 7, 8, 9):
        result = check_order(n)
        print(f"order {n}: {dict(result)}")
        assert len(set(result.values())) == 1


if __name__ == "__main__":
    main()
