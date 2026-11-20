"""Which small decks have a trapped position? Every deck of a given size, up to renaming.

`termination.decks_with_traps` stands for the number cards by the ranks 2 to 5, which covers
decks with at most four different number ranks. This census covers every deck of n cards:

- **Deck types.** Slaps look only at which cards are equal, so the number cards of a deck
  matter only through how many copies each rank has: a partition of their count into parts
  of at most 4. The royals J, Q, K and A are not interchangeable, since each asks for a
  different number of cards. A deck type is a multiset of royals (at most 4 of each, at
  least one royal) and a partition of the rest.
- **Trapped.** A deck has a trap when some position reachable from one of its deals cannot
  reach the end of the game under any run of slap winners (`termination.can_end`).
- **Checks.** For every trapped position found, play from it is checked to loop without a
  slappable pile (`termination.slap_free_cycle`), the kind of trap that Theorem 2 of the
  README describes. A trap of another kind would be reported, not hidden.

    uv run python -m egyptian_ratscrew.traps 7            # every 7-card deck, 2 and 3 players
    uv run python -m egyptian_ratscrew.traps 8 --players 2 --slice 0 --slices 8 --out DIR
"""

import argparse
import json
import time
from collections.abc import Iterator
from itertools import combinations_with_replacement
from pathlib import Path

from egyptian_ratscrew.game import CHANCES, Card
from egyptian_ratscrew.termination import can_end, chain, slap_free_cycle

HERE = Path(__file__).parent
RESULTS = HERE / "traps.json"
NUMBERS = (2, 3, 4, 5, 6, 7, 8, 9, 10)


def partitions(n: int, largest: int = 4) -> Iterator[tuple[int, ...]]:
    """Partitions of n into parts of at most `largest`, largest part first."""
    if n == 0:
        yield ()
        return
    for first in range(min(n, largest), 0, -1):
        for rest in partitions(n - first, first):
            yield (first, *rest)


def deck_types(cards: int) -> list[tuple[Card, ...]]:
    """One deck of each type with `cards` cards, numbers first, then royals in J Q K A order."""
    out = []
    for royals in range(1, cards + 1):
        for chosen in combinations_with_replacement(tuple(CHANCES), royals):
            if any(chosen.count(r) > 4 for r in chosen):
                continue
            for parts in partitions(cards - royals):
                numbers = [NUMBERS[i] for i, k in enumerate(parts) for _ in range(k)]
                out.append((*numbers, *chosen))
    return out


def examine(deck: tuple[Card, ...], players: int) -> dict:
    """The deck's positions, its trapped positions and the orders that deal into one."""
    c = chain(deck, players)
    good = can_end(c)
    trapped = [k for k, ok in enumerate(good) if not ok]
    slap_free = all(slap_free_cycle(c.states[k]) is not None for k in trapped)
    return {
        "deck": " ".join(map(str, deck)),
        "positions": len(c.states),
        "trapped_positions": len(trapped),
        "trapped_orders": sum(w for k, w in c.deals.items() if not good[k]),
        "all_traps_slap_free": slap_free,
    }


def census(cards: int, players: int, part: int = 0, parts: int = 1) -> dict:
    """Every deck type of `cards` cards (every `parts`-th from `part`), and which have traps."""
    decks = deck_types(cards)[part::parts]
    t0 = time.process_time()
    rows = [examine(d, players) for d in decks]
    return {
        "cards": cards,
        "players": players,
        "slice": [part, parts],
        "deck_types": len(decks),
        "seconds": round(time.process_time() - t0),
        "with_traps": [r for r in rows if r["trapped_positions"]],
        "positions": sum(r["positions"] for r in rows),
        "all_traps_slap_free": all(r["all_traps_slap_free"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cards", type=int, nargs="+")
    ap.add_argument("--players", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--slices", type=int, default=1)
    ap.add_argument("--out", type=Path, help="write census.json in this folder")
    ap.add_argument("--write", action="store_true", help=f"write {RESULTS.name}")
    args = ap.parse_args()
    results = []
    for cards in args.cards:
        for players in args.players:
            r = census(cards, players, args.slice, args.slices)
            traps = [(t["deck"], t["trapped_orders"]) for t in r["with_traps"]]
            print(f"{cards} cards, {players} players: {r['deck_types']} deck types, "
                  f"{len(traps)} with a trap, all slap-free: {r['all_traps_slap_free']}, "
                  f"{r['seconds']} s", traps, flush=True)  # fmt: skip
            results.append(r)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "census.json").write_text(json.dumps(results, indent=1) + "\n")
    if args.write:
        RESULTS.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
