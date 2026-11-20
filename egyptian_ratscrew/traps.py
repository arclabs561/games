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
- **Kinds of trap.** For every trapped position, the census records whether play from it
  loops without ever offering a slap (`termination.slap_free_cycle`), the kind of trap that
  Theorem 2 of the README describes, and whether a slap can happen there. It also finds the
  closed classes of trapped positions, which play can enter and never leave, and says
  whether any of them holds a slap: that would be a trap of another kind.

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
from egyptian_ratscrew.termination import can_end, chain, ended, moves, slap_free_cycle, start

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


def components(nodes: list[int], edges: list[list[tuple[int | None, int]]]) -> list[list[int]]:
    """Strongly connected components of the graph on `nodes` (iterative Tarjan)."""
    inside = set(nodes)
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    on: set[int] = set()
    out: list[list[int]] = []
    for root in nodes:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = len(index)
                stack.append(v)
                on.add(v)
            succ = [t for _, t in edges[v] if t in inside]
            if i < len(succ):
                work.append((v, i + 1))
                w = succ[i]
                if w not in index:
                    work.append((w, 0))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return out


def examine(deck: tuple[Card, ...], players: int) -> dict:
    """The deck's positions, its trapped positions and the orders that deal into one.

    Trapped positions only lead to trapped positions. Among them, a **closed class** is a set
    that play can enter and never leave, and inside which every position can reach every
    other: a bottom strongly connected component. A closed class with a slap in it would be a
    trap of a second kind, in which slaps happen forever and the game never ends.
    """
    c = chain(deck, players)
    good = can_end(c)
    trapped = [k for k, ok in enumerate(good) if not ok]
    comps = components(trapped, c.edges)
    where = {k: i for i, comp in enumerate(comps) for k in comp}
    closed = [
        comp
        for i, comp in enumerate(comps)
        if all(where[t] == i for k in comp for _, t in c.edges[k])
    ]
    return {
        "deck": " ".join(map(str, deck)),
        "positions": len(c.states),
        "trapped_positions": len(trapped),
        "trapped_orders": sum(w for k, w in c.deals.items() if not good[k]),
        "trapped_with_slap": sum(len(c.edges[k]) > 1 for k in trapped),
        "closed_classes": len(closed),
        "closed_classes_with_slap": sum(any(len(c.edges[k]) > 1 for k in comp) for comp in closed),
        "all_traps_slap_free": all(slap_free_cycle(c.states[k]) is not None for k in trapped),
    }


def is_closed(states: set) -> bool:
    """Whether no game from these positions can end: none has ended, and every move from each,
    whoever wins a slap, stays among them. Uses only the rules (`moves`, `ended`), not the
    chain or the backward search."""
    return all(not ended(s) and all(t in states for _, t in moves(s)) for s in states)


# A deal of 2 2 3 4 5 6 7 J J, player 0 first, that never ends although slaps happen in it
# (found by `second_kind`; see the README).
NEVER_ENDS = (("J", 2, 4, 6, 7), (2, 3, 5, "J"))


def never_ends(hands) -> tuple[int, int] | None:
    """If no run of slap winners ends the game from this deal: the number of positions it can
    reach and how many of them offer a slap. None if some reachable position has ended.
    Uses only the rules (`moves`, `ended`)."""
    first = start(hands)
    seen, todo, slaps = {first}, [first], 0
    while todo:
        s = todo.pop()
        if ended(s):
            return None
        options = moves(s)
        slaps += len(options) > 1
        for _, t in options:
            if t not in seen:
                seen.add(t)
                todo.append(t)
    return len(seen), slaps


def second_kind(deck: tuple[Card, ...], players: int = 2) -> dict | None:
    """A deal of `deck` from which every run of slap winners ends in a closed class of trapped
    positions with a slap in it, with the smallest such class; None if there is none."""
    c = chain(deck, players)
    good = can_end(c)
    trapped = [k for k, ok in enumerate(good) if not ok]
    comps = components(trapped, c.edges)
    where = {k: i for i, comp in enumerate(comps) for k in comp}
    closed = {
        i for i, comp in enumerate(comps) if all(where[t] == i for k in comp for _, t in c.edges[k])
    }
    with_slap = {i for i in closed if any(len(c.edges[k]) > 1 for k in comps[i])}
    for first in sorted(k for k in c.deals if not good[k]):
        seen, todo = {first}, [first]
        while todo:
            k = todo.pop()
            for _, t in c.edges[k]:
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        reached = {where[k] for k in seen if where[k] in closed}
        if reached and reached <= with_slap:
            smallest = min(reached, key=lambda i: len(comps[i]))
            states = [c.states[k] for k in comps[smallest]]
            return {
                "hands": [list(h) for h in c.states[first].hands],
                "classes_reached": len(reached),
                "class_size": len(states),
                "slap_positions": sum(len(c.edges[k]) > 1 for k in comps[smallest]),
                "states": states,
            }
    return None


def census(cards: int, players: int, part: int = 0, parts: int = 1) -> dict:
    """Every deck type of `cards` cards (every `parts`-th from `part`), and which have traps."""
    decks = deck_types(cards)[part::parts]
    rows = [examine(d, players) for d in decks]
    return {
        "cards": cards,
        "players": players,
        "slice": [part, parts],
        "deck_types": len(decks),
        "with_traps": [r for r in rows if r["trapped_positions"]],
        "positions": sum(r["positions"] for r in rows),
        "all_traps_slap_free": all(r["all_traps_slap_free"] for r in rows),
        "closed_classes_with_slap": sum(r["closed_classes_with_slap"] for r in rows),
    }


def merge(slices: list[list[dict]]) -> list[dict]:
    """One census per number of cards and players from the slices run apart."""
    groups: dict[tuple[int, int], list[dict]] = {}
    for rows in slices:
        for r in rows:
            groups.setdefault((r["cards"], r["players"]), []).append(r)
    out = []
    for (cards, players), rs in sorted(groups.items()):
        parts = rs[0]["slice"][1]
        assert sorted(r["slice"][0] for r in rs) == list(range(parts)), (cards, players)
        order = {" ".join(map(str, d)): i for i, d in enumerate(deck_types(cards))}
        traps = sorted((t for r in rs for t in r["with_traps"]), key=lambda t: order[t["deck"]])
        out.append(
            {
                "cards": cards,
                "players": players,
                "slice": [0, 1],
                "deck_types": sum(r["deck_types"] for r in rs),
                "with_traps": traps,
                "positions": sum(r["positions"] for r in rs),
                "all_traps_slap_free": all(r["all_traps_slap_free"] for r in rs),
                "closed_classes_with_slap": sum(r["closed_classes_with_slap"] for r in rs),
            }
        )
        assert out[-1]["deck_types"] == len(deck_types(cards))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cards", type=int, nargs="+")
    ap.add_argument("--players", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--slices", type=int, default=1)
    ap.add_argument("--out", type=Path, help="write census.json in this folder")
    ap.add_argument("--write", action="store_true", help=f"write {RESULTS.name}")
    ap.add_argument("--merge", type=Path, nargs="*", help="census.json files of slices to add")
    args = ap.parse_args()
    results = []
    for cards in args.cards:
        for players in args.players:
            t0 = time.process_time()
            r = census(cards, players, args.slice, args.slices)
            traps = [(t["deck"], t["trapped_orders"]) for t in r["with_traps"]]
            print(f"{cards} cards, {players} players: {r['deck_types']} deck types, "
                  f"{len(traps)} with a trap, all slap-free: {r['all_traps_slap_free']}, "
                  f"closed classes with a slap: {r['closed_classes_with_slap']}, "
                  f"{time.process_time() - t0:.0f} s", traps, flush=True)  # fmt: skip
            results.append(r)
    if args.merge:
        results += merge([json.loads(p.read_text()) for p in args.merge])
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "census.json").write_text(json.dumps(results, indent=1) + "\n")
    if args.write:
        RESULTS.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
