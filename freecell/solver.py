"""Solver A: does a deal have a winning line?

`solve` searches the positions reachable from the deal, best first. Every position is reduced
by safe autoplay (see `freecell.rules`) and stored by its key, so a position is expanded at
most once whatever the order of columns and cells. The search stops at the first won
position and returns its line, which `rules.replay` checks move by move.

If the queue runs dry, every position the search can reach has been seen and none is won:
the deal is **lost**. The set of keys seen is then a **fence**: it holds the deal, every
single-card move from a position in it leads, after autoplay, back into it, and it holds no
won position. `freecell.fence` checks fences without this module.

Besides single-card moves, the search may try **supermoves**: a run of cards in sequence moved
at once, written out as the single-card moves through free cells and empty columns that do
it. A run of k cards moves onto a card when k ≤ (f + 1) · 2ᵉ, with f empty free cells and e
empty columns, and into an empty column when k ≤ (f + 1) · 2ᵉ⁻¹. Supermoves reach no
position that single moves cannot, so they change no verdict. They make a win much quicker
to find. The fence of a lost deal stays a fence: every single-card move from each position
was still tried.

The order of the search changes only how fast a win is found. `priority` counts cards not
home, cards in free cells, cards lying above a lower card or out of sequence, and cards
above the next card of each suit. A nonzero `seed` adds a random amount to each priority, so
that a restart explores in another order; `decide` restarts with a few seeds before it runs
one search without a budget.
"""

import heapq
import random
from dataclasses import dataclass, field
from itertools import count

from freecell.deals import Card, Layout, rank, suit
from freecell.rules import (
    Move,
    Position,
    Rules,
    apply,
    autoplay,
    can_stack,
    key,
    moves,
    start,
    unkey,
    won,
)


@dataclass
class Result:
    """A verdict: 'won', 'lost', or 'unknown' when the budget ran out."""

    verdict: str
    states: int
    line: list[Move] = field(default_factory=list)
    fence: set[bytes] | None = None
    most_home: int = 0
    frozen: frozenset[Card] = frozenset()  # for a lost deal: cards no reachable position moves


def priority(rules: Rules, pos: Position) -> int:
    """Smaller looks closer to a win."""
    cols, cells, found = pos
    score = 3 * (rules.size() - sum(found)) + 2 * len(cells)
    for col in cols:
        lowest = 99
        in_sequence = True
        for i, c in enumerate(col):
            if i and not can_stack(c, col[i - 1]):
                in_sequence = False
            r = rank(c)
            if r > lowest:
                score += 2
            elif not in_sequence:
                score += 1
            lowest = min(lowest, r)
        for depth, c in enumerate(reversed(col)):
            if found[suit(c)] == rank(c) - 1:
                score += 2 * depth
    return score - 2 * sum(1 for col in cols if not col)


def run_moves(run: list[Card], dest: int | str, free: int, temps: int) -> list[Move]:
    """Single-card moves that carry `run` (bottom card first) onto `dest`.

    `free` empty free cells and `temps` empty columns other than the destination may hold
    cards on the way. A run that fits through the free cells goes one card at a time; a
    longer one moves its top part to an empty column, the rest to `dest`, and the top part
    back on top.
    """
    if len(run) == 1:
        return [(run[0], dest)]
    if len(run) <= free + 1:
        out: list[Move] = [(c, "cell") for c in reversed(run[1:])]
        out.append((run[0], dest))
        out += [(run[i], run[i - 1]) for i in range(1, len(run))]
        return out
    half = (free + 1) * 2 ** (temps - 1)
    if len(run) <= half:
        return run_moves(run, dest, free, temps - 1)
    top, rest = run[-half:], run[:-half]
    return (
        run_moves(top, "empty", free, temps - 1)
        + run_moves(rest, dest, free, temps - 1)
        + run_moves(top, rest[-1], free, temps - 1)
    )


def supermoves(rules: Rules, pos: Position) -> list[list[Move]]:
    """Moves of runs of two or more cards, each as its list of single-card moves."""
    cols, cells, _ = pos
    free = rules.cells - len(cells)
    empties = sum(1 for col in cols if not col)
    tops = [(j, col[-1]) for j, col in enumerate(cols) if col]
    out = []
    for i, col in enumerate(cols):
        k = 1
        while k < len(col) and can_stack(col[-k], col[-k - 1]):
            k += 1
        if k < 2:
            continue
        for j, t in tops:
            if j == i:
                continue
            for length in range(2, k + 1):
                if can_stack(col[-length], t):
                    if length <= (free + 1) * 2**empties:
                        out.append(run_moves(list(col[-length:]), t, free, empties))
                    break
        if empties:
            length = min(k, (free + 1) * 2 ** (empties - 1))
            if 2 <= length < len(col):
                out.append(run_moves(list(col[-length:]), "empty", free, empties - 1))
    return out


def solve(
    rules: Rules,
    layout: Layout,
    budget: int | None = None,
    seed: int = 0,
    use_supermoves: bool = True,
    keep_fence: bool = False,
) -> Result:
    """Search the deal; `budget` caps the number of positions stored."""
    first, auto = autoplay(rules, start(rules, layout))
    k0 = key(first)
    parent: dict[bytes, tuple[bytes, list[Move]] | None] = {k0: None}
    if won(rules, first):
        return Result("won", 1, auto, most_home=sum(first[2]))
    rng = random.Random(seed) if seed else None
    tie = count()
    queue = [(priority(rules, first), 0, k0)]
    most_home = sum(first[2])
    moved = {c for c, _ in auto}
    while queue:
        if budget is not None and len(parent) >= budget:
            return Result("unknown", len(parent), most_home=most_home)
        _, _, k = heapq.heappop(queue)
        pos = unkey(rules, k)
        children = [([m], nxt) for m, nxt in moves(rules, pos)]
        if use_supermoves:
            for line in supermoves(rules, pos):
                nxt = pos
                for m in line:
                    nxt = apply(rules, nxt, m)
                children.append((line, nxt))
        for line, nxt in children:
            nxt, made = autoplay(rules, nxt)
            nk = key(nxt)
            if nk in parent:
                continue
            parent[nk] = (k, line + made)
            moved.update(c for c, _ in line + made)
            most_home = max(most_home, sum(nxt[2]))
            if won(rules, nxt):
                return Result("won", len(parent), auto + _line(parent, nk), most_home=most_home)
            p = priority(rules, nxt) + (rng.randrange(8) if rng else 0)
            heapq.heappush(queue, (p, -next(tie), nk))
    fence = set(parent) if keep_fence else None
    frozen = frozenset(rules.deck()) - moved
    return Result("lost", len(parent), fence=fence, most_home=most_home, frozen=frozen)


def _line(parent: dict[bytes, tuple[bytes, list[Move]] | None], k: bytes) -> list[Move]:
    out: list[list[Move]] = []
    step = parent[k]
    while step is not None:
        k, made = step
        out.append(made)
        step = parent[k]
    return [m for made in reversed(out) for m in made]


def decide(
    rules: Rules,
    layout: Layout,
    first: int = 50_000,
    restarts: int = 8,
    each: int = 100_000,
    cap: int | None = None,
) -> Result:
    """A verdict with restarts: one budgeted search, `restarts` reseeded ones, then one with
    budget `cap` (None for none). The states reported are those of the deciding search."""
    res = solve(rules, layout, budget=first)
    for seed in range(1, restarts + 1):
        if res.verdict != "unknown":
            return res
        res = solve(rules, layout, budget=each, seed=seed)
    if res.verdict != "unknown":
        return res
    return solve(rules, layout, budget=cap)
