"""Trade routes as cycles in a graph, with exact methods and brute-force checks.

Vocabulary, used the same way in the README and the tests:

- A **system** is a node of the jump graph; a **jump** moves a ship to a linked system.
- A **stop** is a planet or station with a commodity exchange. Its **prices** map each commodity
  it trades to one price, paid when buying and received when selling (neither game has a spread).
- A **leg** goes from one stop to the next by the fewest jumps, carrying the one commodity with
  the largest price gain between them, or nothing if no commodity gains.
- A **route** is a cycle of stops, flown over and over. Its **rate** is the profit of one pass
  divided by its **cost**: jumps plus `landing` times the number of stops.

The best route is a maximum-ratio cycle. `best_route` finds it with Howard's policy iteration,
first in floating point and then exactly with fractions, and returns potentials that prove no
cycle does better (`check_certificate`). `karp` finds a maximum-mean cycle when every edge costs
one unit, and `brute_force_route` lists every simple cycle; the tests compare all three.
"""

import heapq
import itertools
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

type System = str
type Stop = str
type Commodity = str
type Prices = Mapping[Commodity, int]
type Number = Fraction | int


@dataclass(frozen=True)
class Galaxy:
    """Systems with their outgoing jumps, and the stops with their systems and prices."""

    links: Mapping[System, tuple[System, ...]]
    stop_system: Mapping[Stop, System]
    prices: Mapping[Stop, Prices]
    refuel_systems: frozenset[System] = field(default_factory=frozenset)

    def stops(self) -> list[Stop]:
        """Every stop, in sorted order."""
        return sorted(self.stop_system)


# Jumps


def jumps_from(links: Mapping[System, Iterable[System]], start: System) -> dict[System, int]:
    """The fewest jumps from `start` to each system it can reach (breadth-first search)."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        for there in links.get(here, ()):
            if there not in dist:
                dist[there] = dist[here] + 1
                queue.append(there)
    return dist


def jump_table(links: Mapping[System, Iterable[System]]) -> dict[System, dict[System, int]]:
    """The fewest jumps between every pair of systems, where one is reachable."""
    return {s: jumps_from(links, s) for s in links}


def return_jumps(links: Mapping[System, Iterable[System]], start: System) -> int | None:
    """The fewest jumps (at least one) that leave `start` and come back to it."""
    best = None
    for there in links.get(start, ()):
        back = jumps_from(links, there).get(start)
        if back is not None and (best is None or back + 1 < best):
            best = back + 1
    return best


def fuel_jumps(
    links: Mapping[System, Iterable[System]],
    refuel_systems: frozenset[System],
    tank: int,
    landing: Fraction,
    start: System,
) -> dict[System, tuple[Fraction, int]]:
    """Cheapest cost (jumps plus `landing` per refueling stop) from `start`, with `tank` jumps
    of fuel, refueling only in `refuel_systems`; also the number of refueling stops.

    The ship leaves `start` with a full tank. Dijkstra's algorithm on (system, fuel left).
    """
    best: dict[System, tuple[Fraction, int]] = {}
    seen: set[tuple[System, int]] = set()
    heap: list[tuple[Fraction, int, System, int]] = [(Fraction(0), 0, start, tank)]
    while heap:
        cost, stops, here, fuel = heapq.heappop(heap)
        if (here, fuel) in seen:
            continue
        seen.add((here, fuel))
        if here not in best:
            best[here] = (cost, stops)
        if here in refuel_systems and fuel < tank:
            heapq.heappush(heap, (cost + landing, stops + 1, here, tank))
        if fuel > 0:
            for there in links.get(here, ()):
                heapq.heappush(heap, (cost + 1, stops, there, fuel - 1))
    return best


# Legs and routes


def best_trade(here: Prices, there: Prices) -> tuple[int, Commodity | None]:
    """The largest gain per ton from buying here and selling there, and the commodity.

    Ties go to the commodity first in alphabetical order; no gain means carrying nothing.
    """
    gain, best = 0, None
    for c in sorted(here.keys() & there.keys()):
        if there[c] - here[c] > gain:
            gain, best = there[c] - here[c], c
    return gain, best


def capital_trade(here: Prices, there: Prices, cargo: int, credits: int) -> tuple[int, str | None]:
    """The largest profit from one load bought with at most `credits`, and the commodity.

    With enough credits this is `cargo` times `best_trade`; with fewer, a cheap commodity that
    fills the hold can beat a dear one that cannot.
    """
    gain, best = 0, None
    for c in sorted(here.keys() & there.keys()):
        tons = min(cargo, credits // here[c])
        if tons * (there[c] - here[c]) > gain:
            gain, best = tons * (there[c] - here[c]), c
    return gain, best


type Edges = dict[tuple[Stop, Stop], tuple[int, Fraction]]
"""A trade graph: for each ordered pair of stops, the leg's gain and its cost."""


def trade_graph(
    galaxy: Galaxy,
    landing: Number = 0,
    *,
    same_system: bool = True,
    gain: Callable[[Prices, Prices], int] | None = None,
    cost: Callable[[Stop, Stop], Fraction | None] | None = None,
) -> Edges:
    """Every leg between two different stops, with its gain per ton and its cost.

    By default a leg costs its fewest jumps plus `landing`. A leg between two stops in the same
    system costs nothing in jumps, so with `landing` = 0 such legs get the cost of leaving the
    system and coming back (`return_jumps`) instead; `same_system=False` drops them.
    """
    landing = Fraction(landing)
    table = jump_table(galaxy.links)
    returns = {s: return_jumps(galaxy.links, s) for s in galaxy.links}
    edges: Edges = {}
    for a, b in itertools.permutations(galaxy.stops(), 2):
        sa, sb = galaxy.stop_system[a], galaxy.stop_system[b]
        if cost is not None:
            t = cost(a, b)
        elif sa != sb:
            d = table[sa].get(sb)
            t = None if d is None else d + landing
        elif not same_system:
            t = None
        elif landing > 0:
            t = landing
        else:
            r = returns[sa]
            t = None if r is None else Fraction(r)
        if t is None:
            continue
        g = (gain or (lambda x, y: best_trade(x, y)[0]))(galaxy.prices[a], galaxy.prices[b])
        edges[a, b] = (g, Fraction(t))
    return edges


def route_rate(edges: Edges, route: tuple[Stop, ...]) -> Fraction:
    """Gain divided by cost for one pass of a route."""
    legs = [edges[a, b] for a, b in zip(route, route[1:] + route[:1], strict=True)]
    return Fraction(sum(g for g, _ in legs)) / sum(t for _, t in legs)


# Exact best route: Howard's policy iteration, with a certificate


@dataclass(frozen=True)
class Best:
    """The best route, its rate, and potentials that prove nothing beats it.

    `potentials` maps each stop to its potential and `rates` to the best rate of a route through
    the stop's strongly connected part of the graph. For every leg a -> b inside one part,
    potential[a] >= gain - rate * cost + potential[b]; adding this around any cycle of that part
    shows its rate is at most the part's rate.
    """

    rate: Fraction
    route: tuple[Stop, ...]
    rates: dict[Stop, Fraction]
    potentials: dict[Stop, Fraction]


def strongly_connected(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Strongly connected parts of a directed graph (Kosaraju's algorithm, iterative)."""
    nodes = list(nodes)
    out: dict[str, list[str]] = {v: [] for v in nodes}
    into: dict[str, list[str]] = {v: [] for v in nodes}
    for a, b in edges:
        out[a].append(b)
        into[b].append(a)
    order, seen = [], set()
    for root in nodes:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(out[root]))]
        while stack:
            v, it = stack[-1]
            nxt = next((u for u in it if u not in seen), None)
            if nxt is None:
                order.append(v)
                stack.pop()
            else:
                seen.add(nxt)
                stack.append((nxt, iter(out[nxt])))
    parts, placed = [], set()
    for root in reversed(order):
        if root in placed:
            continue
        part, stack = [], [root]
        placed.add(root)
        while stack:
            v = stack.pop()
            part.append(v)
            for u in into[v]:
                if u not in placed:
                    placed.add(u)
                    stack.append(u)
        parts.append(sorted(part))
    return parts


def _evaluate(policy, out, nodes, num):
    """Rate and potential of each node under a policy (one chosen leg per node)."""
    rate, pot = {}, {}
    for start in nodes:
        if start in rate:
            continue
        path, index = [], {}
        v = start
        while v not in rate and v not in index:
            index[v] = len(path)
            path.append(v)
            v = out[v][policy[v]][0]
        if v in index:  # a new cycle: path[index[v]:]
            cycle = path[index[v] :]
            g = sum(out[u][policy[u]][1] for u in cycle)
            t = sum(out[u][policy[u]][2] for u in cycle)
            r = num(g) / num(t)
            root = min(cycle)
            k = cycle.index(root)
            ordered = cycle[k:] + cycle[:k]
            rate[root], pot[root] = r, num(0)
            for u in reversed(ordered[1:]):
                w, g_u, t_u = out[u][policy[u]]
                rate[u], pot[u] = r, g_u - r * t_u + pot[w]
            path = path[: index[v]]
        for u in reversed(path):
            w, g_u, t_u = out[u][policy[u]]
            rate[u], pot[u] = rate[w], g_u - rate[w] * t_u + pot[w]
    return rate, pot


def _howard(nodes, out, num, policy=None, max_rounds=10_000, settle=True):
    """Howard's policy iteration for the maximum-ratio cycle of one strongly connected part.

    Returns rates, potentials and the final policy. With `settle` False, running out of rounds
    returns the current policy instead of failing (used for the floating-point pass).
    """
    if policy is None:
        policy = {v: max(range(len(out[v])), key=lambda i: num(out[v][i][1])) for v in nodes}
    for _ in range(max_rounds):
        rate, pot = _evaluate(policy, out, nodes, num)
        changed = False
        for v in nodes:  # first, move to legs that reach a better cycle
            i = max(range(len(out[v])), key=lambda i: rate[out[v][i][0]])
            if rate[out[v][i][0]] > rate[v]:
                policy[v], changed = i, True
        if not changed:  # then, among legs that keep the rate, raise the potential
            for v in nodes:
                best, best_i = pot[v], None
                for i, (u, g, t) in enumerate(out[v]):
                    if rate[u] == rate[v]:
                        val = g - rate[v] * t + pot[u]
                        if val > best:
                            best, best_i = val, i
                if best_i is not None:
                    policy[v], changed = best_i, True
        if not changed:
            return rate, pot, policy
    if not settle:
        return rate, pot, policy
    raise RuntimeError("Howard's iteration did not settle")


def best_route(edges: Edges) -> Best | None:
    """The route with the highest rate, exactly, or None if there is no cycle.

    Every cycle must have a positive cost. Floating point finds a good policy fast; exact
    iteration from that policy then settles the answer in fractions.
    """
    nodes = sorted({a for a, _ in edges} | {b for _, b in edges})
    rates: dict[Stop, Fraction] = {}
    pots: dict[Stop, Fraction] = {}
    best: tuple[Fraction, tuple[Stop, ...]] | None = None
    every: dict[Stop, list[tuple[Stop, int, Fraction]]] = {v: [] for v in nodes}
    for (a, b), (g, t) in sorted(edges.items()):
        every[a].append((b, g, t))
    for part in strongly_connected(nodes, edges):
        inside = set(part)
        out = {v: [leg for leg in every[v] if leg[0] in inside] for v in part}
        if any(not legs for legs in out.values()):
            continue  # a single stop with no cycle through it
        quick = {v: [(b, float(g), float(t)) for b, g, t in legs] for v, legs in out.items()}
        _, _, policy = _howard(part, quick, float, max_rounds=1000, settle=False)
        rate, pot, policy = _howard(part, out, Fraction, policy)
        rates.update(rate)
        pots.update(pot)
        top = max(part, key=lambda v: (rate[v], v))
        cycle, v = [], top
        while v not in cycle:
            cycle.append(v)
            v = out[v][policy[v]][0]
        cycle = cycle[cycle.index(v) :]
        k = cycle.index(min(cycle))
        route = tuple(cycle[k:] + cycle[:k])
        if best is None or rate[top] > best[0]:
            best = (rate[top], route)
    if best is None:
        return None
    return Best(best[0], best[1], rates, pots)


def check_certificate(edges: Edges, best: Best) -> bool:
    """Whether the potentials prove that no cycle beats `best.rate`, and the route attains it.

    Independent of how the answer was found: it checks every leg once, in exact arithmetic.
    """
    if route_rate(edges, best.route) != best.rate:
        return False
    for (a, b), (g, t) in edges.items():
        if a in best.rates and best.rates.get(b) == best.rates[a]:
            if best.potentials[a] < g - best.rates[a] * t + best.potentials[b]:
                return False
    nodes = {a for a, _ in edges} | {b for _, b in edges}
    for part in strongly_connected(sorted(nodes), edges):
        if len(part) == 1:
            continue
        if any(v not in best.rates for v in part):
            return False
        rates = {best.rates[v] for v in part}
        if len(rates) != 1 or max(rates) > best.rate:
            return False
    return True


def brute_force_route(edges: Edges) -> tuple[Fraction, tuple[Stop, ...]] | None:
    """The best simple cycle by listing them all (each once, from its smallest stop).

    The best route is always a simple cycle: a cycle that visits a stop twice splits into two
    shorter cycles, and its rate lies between theirs.
    """
    out: dict[Stop, list[Stop]] = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)
    best = None
    for start in sorted(out):
        stack: list[tuple[Stop, tuple[Stop, ...]]] = [(start, (start,))]
        while stack:
            v, path = stack.pop()
            for u in out.get(v, ()):
                if u == start:
                    r = route_rate(edges, path)
                    if best is None or r > best[0]:
                        best = (r, path)
                elif u > start and u not in path:
                    stack.append((u, (*path, u)))
    return best


# Karp's algorithm for unit costs


def karp(n: int, src: np.ndarray, dst: np.ndarray, gain: np.ndarray) -> Fraction | None:
    """The maximum mean gain of a cycle in a graph whose every edge costs one unit.

    Karp (1978): with D[k][v] the best gain of a walk of exactly k edges ending at v (from
    anywhere), the answer is max over v of min over k < n of (D[n][v] - D[k][v]) / (n - k).
    Gains must be integers. Floating point picks the extreme pairs; the answer is exact.
    """
    neg = np.iinfo(np.int64).min // 4
    order = np.argsort(dst, kind="stable")
    src, dst, gain = src[order], dst[order], gain[order].astype(np.int64)
    heads = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
    targets = dst[heads]
    d = np.full((n + 1, n), neg, dtype=np.int64)
    d[0] = 0
    for k in range(n):
        vals = d[k][src] + gain
        vals[d[k][src] == neg] = neg
        d[k + 1, targets] = np.maximum.reduceat(vals, heads)
    final = d[n]
    live = np.flatnonzero(final > neg)
    if live.size == 0:
        return None
    best = None
    with np.errstate(invalid="ignore"):
        for v in live:
            col = d[:n, v]
            ks = np.flatnonzero(col > neg)
            ratios = (final[v] - col[ks]) / (n - ks)
            low = ratios.min()
            close = ks[ratios <= low + 1e-9 * max(1.0, abs(low))]
            exact = min(Fraction(int(final[v] - d[k, v]), int(n - k)) for k in close)
            if best is None or exact > best:
                best = exact
    return best


def brute_force_mean(edges: Iterable[tuple[int, int, int]]) -> Fraction | None:
    """The maximum mean gain of a simple cycle, by listing them all; for checking `karp`."""
    as_trade: Edges = {(str(a), str(b)): (g, Fraction(1)) for a, b, g in edges}
    found = brute_force_route(as_trade)
    return None if found is None else found[0]


# The same question on systems: one node per (system, cargo held)


def system_state_graph(
    galaxy: Galaxy, *, landing_step: bool
) -> tuple[list[tuple[System, Commodity | None]], list[tuple[int, int, int]]]:
    """A unit-cost graph whose nodes are (system, commodity held, or None for empty).

    With `landing_step` False, each edge is one jump, and before it the ship may land at one stop
    in the system it leaves, sell what it holds and buy something else: profit per jump when
    landing is free but each visit to a system lands once. With `landing_step` True, a jump and a
    landing each cost one unit, and a ship may land many times in one system.
    """
    commodities = sorted({c for p in galaxy.prices.values() for c in p})
    holds: list[Commodity | None] = [None, *commodities]
    nodes = [(s, c) for s in sorted(galaxy.links) for c in holds]
    index = {v: i for i, v in enumerate(nodes)}
    at: dict[System, list[Stop]] = {}
    for stop, sys in galaxy.stop_system.items():
        at.setdefault(sys, []).append(stop)

    def trade(stop: Stop, c: Commodity | None, c2: Commodity | None) -> int | None:
        p = galaxy.prices[stop]
        if (c is not None and c not in p) or (c2 is not None and c2 not in p):
            return None
        return (p[c] if c else 0) - (p[c2] if c2 else 0)

    edges: list[tuple[int, int, int]] = []
    for s in sorted(galaxy.links):
        moves: dict[tuple[Commodity | None, Commodity | None], int] = {(c, c): 0 for c in holds}
        for stop in at.get(s, []):
            for c, c2 in itertools.product(holds, holds):
                g = trade(stop, c, c2)
                if g is not None and g > moves.get((c, c2), g - 1):
                    moves[c, c2] = g
        if landing_step:
            for stop in at.get(s, []):
                for c, c2 in itertools.product(holds, holds):
                    g = trade(stop, c, c2)
                    if g is not None:
                        edges.append((index[s, c], index[s, c2], g))
            for t in galaxy.links[s]:
                for c in holds:
                    edges.append((index[s, c], index[t, c], 0))
        else:
            for t in galaxy.links[s]:
                for (c, c2), g in moves.items():
                    edges.append((index[s, c], index[t, c2], g))
    best: dict[tuple[int, int], int] = {}
    for a, b, g in edges:
        if (a, b) not in best or g > best[a, b]:
            best[a, b] = g
    return nodes, [(a, b, g) for (a, b), g in sorted(best.items())]


def karp_rate(galaxy: Galaxy, *, landing_step: bool) -> Fraction | None:
    """The best rate per ton on the system state graph, by Karp's algorithm."""
    nodes, edges = system_state_graph(galaxy, landing_step=landing_step)
    if not edges:
        return None
    arr = np.array(edges, dtype=np.int64)
    return karp(len(nodes), arr[:, 0], arr[:, 1], arr[:, 2])


# How the best route changes with the cost of a landing


def route_parts(
    galaxy: Galaxy, route: tuple[Stop, ...], table: dict[System, dict[System, int]] | None = None
) -> tuple[int, int, int]:
    """Gain per ton, jumps and stops of one pass; its rate is gain / (jumps + landing * stops)."""
    table = jump_table(galaxy.links) if table is None else table
    legs = list(zip(route, route[1:] + route[:1], strict=True))
    gain = sum(best_trade(galaxy.prices[a], galaxy.prices[b])[0] for a, b in legs)
    jumps = sum(table[galaxy.stop_system[a]][galaxy.stop_system[b]] for a, b in legs)
    return gain, jumps, len(route)


def landing_envelope(
    galaxy: Galaxy, low: Fraction, high: Fraction
) -> list[tuple[Fraction, tuple[Stop, ...]]]:
    """The best route for every cost of a landing from `low` to `high`, exactly.

    Returns (cost where a route starts being best, route) pairs in order. Whether route E beats
    route C at landing cost x, G_E (J_C + x S_C) > G_C (J_E + x S_E), is linear in x. So a route
    that is best at both ends of an interval is best on all of it, and two routes best at the
    two ends of an interval, if no third route beats them where they tie, split the interval
    there. Otherwise the third route goes between them and each half is solved the same way.
    """

    table = jump_table(galaxy.links)

    def best_at(x: Fraction) -> tuple[Stop, ...]:
        found = best_route(trade_graph(galaxy, x))
        assert found is not None
        return found.route

    def tie(c: tuple[Stop, ...], d: tuple[Stop, ...]) -> Fraction | None:
        (g1, j1, s1), (g2, j2, s2) = route_parts(galaxy, c, table), route_parts(galaxy, d, table)
        den = g1 * s2 - g2 * s1
        return None if den == 0 else Fraction(g2 * j1 - g1 * j2, den)

    def rate(route: tuple[Stop, ...], x: Fraction) -> Fraction:
        g, j, s = route_parts(galaxy, route, table)
        return Fraction(g) / (j + x * s)

    def solve(a, ca, b, cb):
        if ca == cb or rate(ca, b) == rate(cb, b) or rate(ca, a) == rate(cb, a):
            return []
        x = tie(ca, cb)
        assert x is not None and a < x < b
        cx = best_at(x)
        if rate(cx, x) == rate(ca, x):
            return [(x, cb)]
        return solve(a, ca, x, cx) + solve(x, cx, b, cb)

    first, last = best_at(low), best_at(high)
    return [(low, first), *solve(low, first, high, last)]


# Round trips


def best_round_trips(galaxy: Galaxy, top: int = 10) -> list[tuple[Fraction, Stop, Stop]]:
    """Two-stop routes between different systems, ranked by gain per jump, exactly."""
    table = jump_table(galaxy.links)
    found = []
    for a, b in itertools.combinations(galaxy.stops(), 2):
        sa, sb = galaxy.stop_system[a], galaxy.stop_system[b]
        if sa == sb or sb not in table[sa] or sa not in table[sb]:
            continue
        g = best_trade(galaxy.prices[a], galaxy.prices[b])[0]
        g += best_trade(galaxy.prices[b], galaxy.prices[a])[0]
        found.append((Fraction(g, table[sa][sb] + table[sb][sa]), a, b))
    found.sort(key=lambda x: (-x[0], x[1], x[2]))
    return found[:top]


# Structure of the jump graph


def undirected(links: Mapping[System, Iterable[System]]) -> dict[System, set[System]]:
    """The jump graph with each jump usable both ways."""
    adj: dict[System, set[System]] = {s: set() for s in links}
    for s, outs in links.items():
        for t in outs:
            adj[s].add(t)
            adj.setdefault(t, set()).add(s)
    return adj


def components(adj: Mapping[System, Iterable[System]]) -> list[list[System]]:
    """Connected parts of an undirected graph, largest first."""
    seen: set[System] = set()
    parts = []
    for s in sorted(adj):
        if s not in seen:
            part = sorted(jumps_from(adj, s))
            seen.update(part)
            parts.append(part)
    return sorted(parts, key=lambda p: (-len(p), p[0]))


def eccentricities(adj: Mapping[System, Iterable[System]]) -> dict[System, int]:
    """For each system, the most jumps to any system it can reach."""
    return {s: max(jumps_from(adj, s).values()) for s in adj}


def articulation_points(adj: Mapping[System, Iterable[System]]) -> set[System]:
    """Systems whose removal disconnects the graph (Hopcroft and Tarjan, iterative)."""
    order: dict[System, int] = {}
    low: dict[System, int] = {}
    found: set[System] = set()
    for root in sorted(adj):
        if root in order:
            continue
        order[root] = low[root] = len(order)
        children = 0
        stack = [(root, None, iter(sorted(adj[root])))]
        while stack:
            v, parent, it = stack[-1]
            u = next(it, None)
            if u is None:
                stack.pop()
                if parent is not None:
                    low[parent] = min(low[parent], low[v])
                    if parent != root and low[v] >= order[parent]:
                        found.add(parent)
            elif u not in order:
                order[u] = low[u] = len(order)
                if v == root:
                    children += 1
                stack.append((u, v, iter(sorted(adj[u]))))
            elif u != parent:
                low[v] = min(low[v], order[u])
        if children > 1:
            found.add(root)
    return found


def betweenness(adj: Mapping[System, Iterable[System]]) -> dict[System, Fraction]:
    """For each system, the sum over pairs {s, t} of the share of shortest s-t paths through it.

    Brandes (2001), with exact fractions; each unordered pair is counted once.
    """
    score = {v: Fraction(0) for v in adj}
    for s in sorted(adj):
        stack, preds = [], {v: [] for v in adj}
        sigma = dict.fromkeys(adj, 0)
        dist = dict.fromkeys(adj, -1)
        sigma[s], dist[s] = 1, 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = {v: Fraction(0) for v in adj}
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += Fraction(sigma[v], sigma[w]) * (1 + delta[w])
            if w != s:
                score[w] += delta[w]
    return {v: x / 2 for v, x in score.items()}


def brute_force_betweenness(adj: Mapping[System, Iterable[System]]) -> dict[System, Fraction]:
    """Betweenness by listing every shortest path; for checking `betweenness` on small graphs."""

    def paths(s: System, t: System) -> list[tuple[System, ...]]:
        dist = jumps_from(adj, s)
        if t not in dist:
            return []
        found, frontier = [], [(s,)]
        while frontier:
            nxt = []
            for p in frontier:
                if p[-1] == t:
                    found.append(p)
                    continue
                nxt += [(*p, w) for w in adj[p[-1]] if dist.get(w) == len(p)]
            frontier = nxt
        return [p for p in found if len(p) - 1 == dist[t]]

    score = {v: Fraction(0) for v in adj}
    for s, t in itertools.combinations(sorted(adj), 2):
        ps = paths(s, t)
        for p in ps:
            for v in p[1:-1]:
                score[v] += Fraction(1, len(ps))
    return score
