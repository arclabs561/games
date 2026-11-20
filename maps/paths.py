"""Putting points in a line so that neighbors are close: shortest Hamiltonian paths.

An **ordering** of n books is a permutation of range(n). Its **total jump** is the sum of
the distances between books standing next to each other. The best ordering is a shortest
Hamiltonian path with free ends, a traveling-salesman problem. It is solved here exactly
in two ways (Held and Karp's dynamic program for small shelves, an integer program with
subtour cuts for large ones) and checked against brute force on the smallest.
"""

from collections.abc import Sequence
from itertools import permutations

import numpy as np

Order = list[int]


def total_jump(order: Sequence[int], dist: np.ndarray) -> float:
    """Sum of distances between consecutive items of the ordering."""
    return float(sum(dist[a, b] for a, b in zip(order, order[1:], strict=False)))


def brute_force_path(dist: np.ndarray) -> tuple[float, Order]:
    """Shortest open path by trying every ordering whose first item is below its last."""
    n = len(dist)
    if n == 1:
        return 0.0, [0]
    best: tuple[float, Order] = (float("inf"), [])
    for perm in permutations(range(n)):
        if perm[0] < perm[-1]:
            length = total_jump(perm, dist)
            if length < best[0]:
                best = (length, list(perm))
    return best


def held_karp_path(dist: np.ndarray) -> tuple[float, Order]:
    """Shortest open path with free ends, by Held and Karp's dynamic program over subsets.

    best[S, j] is the shortest path that visits exactly the set S (a bit mask) and ends
    at j. A path may start anywhere, so best[{j}, j] = 0. Time O(2^n n^2).
    """
    n = len(dist)
    if n == 1:
        return 0.0, [0]
    full = 1 << n
    best = np.full((full, n), np.inf)
    back = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        best[1 << j, j] = 0.0
    masks = np.arange(full)
    sizes = np.array([m.bit_count() for m in range(full)])
    for size in range(2, n + 1):
        for mask in masks[sizes == size]:
            for j in range(n):
                if not mask >> j & 1:
                    continue
                prev = mask ^ (1 << j)
                cand = best[prev] + dist[:, j]
                i = int(np.argmin(cand))
                best[mask, j], back[mask, j] = cand[i], i
    mask, j = full - 1, int(np.argmin(best[full - 1]))
    length, order = float(best[mask, j]), [j]
    while back[mask, j] >= 0:
        mask, j = mask ^ (1 << j), int(back[mask, j])
        order.append(j)
    return length, order[::-1]


def exact_path(dist: np.ndarray, time_limit: float = 120.0) -> tuple[float, Order]:
    """Shortest open path for larger n: an integer program solved by HiGHS, with subtour cuts.

    A dummy node at distance 0 from every item turns the open path into a closed tour on
    n + 1 nodes. Each node gets degree 2; whenever the solution splits into several
    cycles, the cut "at most |S| - 1 edges inside S" is added for each cycle and the
    program is solved again. The final solution is one cycle, so it is optimal.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n = len(dist) + 1  # node n - 1 is the dummy
    d = np.zeros((n, n))
    d[:-1, :-1] = dist
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cost = np.array([d[i, j] for i, j in edges])
    degree = lil_matrix((n, len(edges)))
    for e, (i, j) in enumerate(edges):
        degree[i, e] = degree[j, e] = 1
    constraints = [LinearConstraint(degree.tocsr(), 2, 2)]
    while True:
        res = milp(
            cost,
            constraints=constraints,
            integrality=np.ones(len(edges)),
            bounds=Bounds(0, 1),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0},
        )
        if res.x is None or res.status != 0:
            raise RuntimeError(f"HiGHS did not finish: {res.message}")
        chosen = [edges[e] for e in np.flatnonzero(res.x > 0.5)]
        cycles = components(n, chosen)
        if len(cycles) == 1:
            break
        for cycle in cycles:
            inside = set(cycle)
            row = np.array([1.0 if i in inside and j in inside else 0.0 for i, j in edges])
            constraints.append(LinearConstraint(row[None, :], -np.inf, len(cycle) - 1))
    tour = walk(n, chosen, start=n - 1)[1:]  # drop the dummy
    return total_jump(tour, dist), tour


def components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components of a graph on range(n), each as a sorted list."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values())


def walk(n: int, edges: list[tuple[int, int]], start: int) -> Order:
    """The nodes of a single cycle in order, starting at start."""
    nbrs: dict[int, list[int]] = {a: [] for a in range(n)}
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    order = [start, min(nbrs[start])]
    while len(order) < n:
        a, b = nbrs[order[-1]]
        order.append(b if a == order[-2] else a)
    return order


def nearest_neighbor(dist: np.ndarray, start: int) -> Order:
    """Greedy path: from start, always step to the closest book not yet placed."""
    n = len(dist)
    order, left = [start], set(range(n)) - {start}
    while left:
        here = order[-1]
        order.append(min(left, key=lambda j: (dist[here, j], j)))
        left.remove(order[-1])
    return order


def two_opt(order: Sequence[int], dist: np.ndarray) -> Order:
    """Reverse segments of an open path while that shortens it (first improvement).

    Reversing order[i:j] changes only the two links at its ends; a segment that reaches
    an end of the shelf changes one link.
    """
    order = list(order)
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                if i == 0 and j == n:
                    continue
                a, b = order[i - 1] if i > 0 else None, order[i]
                c, e = order[j - 1], order[j] if j < n else None
                before = (dist[a, b] if a is not None else 0) + (dist[c, e] if e is not None else 0)
                after = (dist[a, c] if a is not None else 0) + (dist[b, e] if e is not None else 0)
                if after < before - 1e-12:
                    order[i:j] = order[i:j][::-1]
                    improved = True
    return order


def nn_two_opt(dist: np.ndarray) -> Order:
    """Nearest neighbor from every start, each improved by 2-opt; the shortest result."""
    best = min(
        (two_opt(nearest_neighbor(dist, s), dist) for s in range(len(dist))),
        key=lambda o: (total_jump(o, dist), o),
    )
    return canonical(best)


def canonical(order: Sequence[int]) -> Order:
    """The ordering read in whichever direction puts the smaller end first."""
    order = list(order)
    return order if order[0] <= order[-1] else order[::-1]


def sort_by(keys: Sequence[Sequence[float]]) -> Order:
    """Indices sorted by a tuple of keys per item, ties broken by index."""
    return sorted(range(len(keys)), key=lambda i: (*keys[i], i))


def hilbert_index(point: Sequence[int], bits: int) -> int:
    """Position along the Hilbert curve that fills the cube [0, 2^bits)^d.

    Skilling's transpose algorithm ("Programming the Hilbert curve", AIP Conference
    Proceedings 707, 2004): undo the excess work, Gray-decode, then interleave the bits.
    """
    x = list(point)
    d = len(x)
    m = 1 << (bits - 1)
    q = m
    while q > 1:  # inverse undo
        p = q - 1
        for i in range(d):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q >>= 1
    for i in range(1, d):  # Gray encode
        x[i] ^= x[i - 1]
    t, q = 0, m
    while q > 1:
        if x[d - 1] & q:
            t ^= q - 1
        q >>= 1
    x = [xi ^ t for xi in x]
    index = 0
    for b in range(bits - 1, -1, -1):
        for i in range(d):
            index = index << 1 | (x[i] >> b & 1)
    return index
