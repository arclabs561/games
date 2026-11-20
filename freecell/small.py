"""Exact counts on small decks, where brute force can check both solvers on every deal.

A small game keeps the rules and shrinks the deck (fewer ranks or suits) and the tableau.
A **deal** is an order of the deck, dealt in rows across the columns as in Microsoft
FreeCell, so every order counts once, as in a shuffle. For each game and number of free
cells, `count` runs the brute force (`freecell.brute`), Solver A and Solver B on every deal,
or on a seeded sample when the deck is too large, and requires the three verdicts to agree.

    uv run python -m freecell.small            # every game below, about 10 minutes
    uv run python -m freecell.small --quick    # every 50th deal
"""

import argparse
import json
import math
import random
import time
from itertools import islice, permutations
from pathlib import Path

from freecell import native
from freecell.brute import brute_winnable
from freecell.certificates import blocked_at_start, static_stuck
from freecell.deals import deal_columns
from freecell.rules import Rules, replay
from freecell.solver import solve

# (ranks, suits, columns, levels of free cells, sample size or None for every deal)
GAMES = [
    (3, (0, 1), 2, (0, 1), None),
    (4, (0, 1), 2, (0, 1, 2), None),
    (2, (0, 1, 2, 3), 2, (0, 1, 2), None),
    (4, (0, 1), 3, (0, 1), None),
    (5, (0, 1), 3, (0, 1, 2), 3000),
    (3, (0, 1, 2, 3), 3, (0, 1, 2), 3000),
]


def orders(rules: Rules, sample: int | None, step: int = 1, seed: int = 0):
    """Every order of the deck (every `step`-th), or `sample` seeded random orders."""
    deck = rules.deck()
    if sample is None:
        yield from islice(permutations(deck), 0, None, step)
        return
    rng = random.Random(seed)
    for _ in range(0, sample, step):
        order = deck[:]
        rng.shuffle(order)
        yield tuple(order)


def count(ranks, suits, columns, cells, sample, step=1) -> dict:
    """Deals, lost deals, and disagreements between the three engines."""
    rules = Rules(cells=cells, columns=columns, ranks=ranks, suits=suits)
    layouts = [deal_columns(list(o), columns) for o in orders(rules, sample, step)]
    answers = native.run([native.deck_problem(rules, lay) for lay in layouts])
    lost = disagree = bad_lines = blocked = stuck = frozen = unsound = 0
    for lay, b in zip(layouts, answers, strict=True):
        a = solve(rules, lay)
        brute = brute_winnable(lay, cells)
        if not (brute == (a.verdict == "won") == (b.verdict == "won")):
            disagree += 1
        if a.verdict == "lost" and b.verdict == "lost":
            if a.frozen != b.frozen:
                disagree += 1
            frozen += bool(a.frozen)
        if cells == 0:
            st = static_stuck(lay)
            bl = blocked_at_start(lay, 0)
            stuck += bool(st)
            blocked += bl
            if (st or bl) and brute or not st <= a.frozen:
                unsound += 1
        if a.verdict == "won" and not replay(rules, lay, a.line):
            bad_lines += 1
        if b.verdict == "won" and not replay(rules, lay, b.line):
            bad_lines += 1
        lost += not brute
    return {
        "ranks": ranks,
        "suits": len(suits),
        "columns": columns,
        "cells": cells,
        "deals": len(layouts),
        "all deals": sample is None and step == 1,
        "orders of the deck": math.factorial(rules.size()),
        "lost": lost,
        "disagreements": disagree,
        "bad lines": bad_lines,
        "lost with a frozen card": frozen,
        "blocked at the start": blocked,
        "with stuck cards": stuck,
        "unsound certificates": unsound,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="every 50th deal")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = []
    for ranks, suits, columns, levels, sample in GAMES:
        for cells in levels:
            t = time.time()
            row = count(ranks, suits, columns, cells, sample, step=50 if args.quick else 1)
            rows.append(row)
            print(
                f"{ranks} ranks x {row['suits']} suits, {columns} columns, {cells} cells: "
                f"{row['lost']:,} of {row['deals']:,} deals lost; "
                f"{row['lost with a frozen card']:,} with a frozen card, "
                f"{row['blocked at the start']:,} blocked, {row['with stuck cards']:,} stuck; "
                f"{row['disagreements']} disagreements, {row['bad lines']} bad lines, "
                f"{row['unsound certificates']} unsound "
                f"({time.time() - t:.0f} s)"
            )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
