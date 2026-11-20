"""What a fixed strategy does: which states it reaches, how often, and its score distribution.

A **strategy** here is the one that a value array E defines: in each turn it makes the choices
that maximize the points of the turn plus E of the state the turn leads to. With the values from
`solver.solve` that is optimal play; with E = 0 it is Glenn's greedy strategy.

`graph` follows the strategy forward from the empty score card, one turn at a time. For each
state it reaches it records the **edges**: every way the turn can end, as the next state, the
points earned and the chance, with the final rolls that lead to the same next state and the same
points merged into one edge. The **occupancy** of a state is the chance that the game passes
through it. Expected values of anything earned turn by turn (points per box, Yahtzee bonuses,
Yahtzees rolled) are sums of occupancy times the expected earnings of the turn, so `graph`
collects them on the way.

`distribution` then runs backwards over the edges to give the exact distribution of the final
score (or of another count earned turn by turn), state by state: a state's distribution is the
mixture, over its edges, of the next state's distribution shifted by the edge's points.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from yahtzee.solver import Rule, Tables, final_distribution, turn


@dataclass
class Layer:
    """The states reached with a given number of boxes filled, and the edges out of them."""

    states: np.ndarray  # (n,) sorted state indices
    occupancy: np.ndarray  # (n,) the chance of passing through each state
    start: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )  # first edge of each state
    next: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )  # (e,) state index
    points: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )  # (e,) box + bonus
    yahtzee: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )  # final roll is one
    prob: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (e,) chance given the state


@dataclass
class Graph:
    tables: Tables
    layers: list[Layer]
    stats: dict[str, np.ndarray]  # expected values, accumulated over the turns


def graph(t: Tables, values: np.ndarray, batch: int = 2048, rule: Rule | None = None) -> Graph:
    """Follow the strategy that E defines from the empty score card to the end.

    With `rule`, the choices it fixes are followed instead (see `solver.turn`).
    """
    cats = t.rules.categories
    n_boxes = len(cats)
    stats = {k: np.zeros(n_boxes) for k in ("points", "squares", "zeros")} | {
        k: np.zeros(1) for k in ("bonus", "yahtzees", "jokers", "bonuses")
    }
    layers = [Layer(np.array([t.index(0, 0, 0)]), np.array([1.0]))]
    for _ in range(n_boxes):
        layer = layers[-1]
        starts, nexts, points, yahtzees, probs, mass = [], [], [], [], [], []
        offset = 0
        for i in range(0, len(layer.states), batch):
            s, occ = layer.states[i : i + batch], layer.occupancy[i : i + batch]
            tr = turn(t, values, s, choices=True, rule=rule)
            pi = final_distribution(t, tr)
            w = occ[:, None] * pi
            box = tr.box.ravel()
            stats["points"] += np.bincount(box, (w * tr.points).ravel(), n_boxes)
            stats["squares"] += np.bincount(box, (w * tr.points**2).ravel(), n_boxes)
            stats["zeros"] += np.bincount(box, (w * (tr.points == 0)).ravel(), n_boxes)
            stats["bonus"] += (w * tr.bonus).sum()
            stats["bonuses"] += (w * (tr.bonus > 0)).sum()
            stats["yahtzees"] += (w * t.yahtzee[None, :]).sum()
            stats["jokers"] += (w * tr.joker).sum()
            # Merge final rolls with the same next state, points and Yahtzee flag.
            reward = tr.points + tr.bonus
            assert reward.max() < 256
            y = np.broadcast_to(t.yahtzee, reward.shape)
            row = np.arange(len(s))[:, None]
            key = ((row * t.n_states + tr.next) * 256 + reward) * 2 + y
            keep = pi > 0
            uniq, inv = np.unique(key[keep], return_inverse=True)
            p = np.bincount(inv, pi[keep], len(uniq))
            src = uniq // (2 * 256 * t.n_states)
            starts.append(offset + np.searchsorted(src, np.arange(len(s))))
            offset += len(uniq)
            nexts.append(uniq // (2 * 256) % t.n_states)
            points.append(uniq // 2 % 256)
            yahtzees.append(uniq % 2 == 1)
            probs.append(p)
            mass.append(occ[src] * p)
        layer.start, layer.next = np.concatenate(starts), np.concatenate(nexts)
        layer.points, layer.yahtzee = np.concatenate(points), np.concatenate(yahtzees)
        layer.prob = np.concatenate(probs)
        states, inv = np.unique(layer.next, return_inverse=True)
        layers.append(Layer(states, np.bincount(inv, np.concatenate(mass), len(states))))
    last = layers[-1]
    total = (last.states // 2) % t.width
    stats["upper_bonus"] = np.array([last.occupancy[total == t.cap].sum() if t.cap else 0.0])
    return Graph(t, layers, stats)


def terminal_points(g: Graph) -> np.ndarray:
    """The upper bonus earned at each final state."""
    t = g.tables
    total = (g.layers[-1].states // 2) % t.width
    return np.where((total == t.cap) & (t.cap > 0), t.rules.upper_bonus, 0)


def distribution(g: Graph, count: str = "points", chunk: int = 256) -> np.ndarray:
    """The exact distribution, from the start, of the final score (or of the Yahtzees rolled).

    Entry s of the result is the chance that the total is s. `count="yahtzees"` counts the final
    rolls that are Yahtzees instead of points, and `count="bonuses"` the Yahtzee bonuses.
    """
    last = g.layers[-1]
    if count == "points":
        end = terminal_points(g)
    else:
        end = np.zeros(len(last.states), dtype=np.int64)
    dist = np.zeros((len(last.states), end.max() + 1))
    dist[np.arange(len(dist)), end] = 1.0
    for layer, after in zip(g.layers[-2::-1], g.layers[:0:-1], strict=True):
        shift = {
            "points": layer.points,
            "yahtzees": layer.yahtzee.astype(np.int64),
            "bonuses": (layer.points >= 100).astype(np.int64),
        }[count]
        top = int(shift.max())
        width = dist.shape[1] + top
        padded = np.zeros((len(dist), top + width))
        padded[:, top : top + dist.shape[1]] = dist
        windows = sliding_window_view(
            padded, width, axis=1
        )  # windows[j, top - s] is D[j] shifted by s
        dst = np.searchsorted(after.states, layer.next)
        new = np.zeros((len(layer.states), width))
        bounds = np.append(layer.start, len(layer.next))
        for a in range(0, len(layer.states), chunk):
            lo, hi = bounds[a], bounds[min(a + chunk, len(layer.states))]
            part = windows[dst[lo:hi], top - shift[lo:hi]] * layer.prob[lo:hi, None]
            new[a : a + chunk] = np.add.reduceat(part, layer.start[a : a + chunk] - lo, axis=0)
        dist = new
    return dist[0]


def summary(dist: np.ndarray) -> dict[str, float]:
    """Mean, standard deviation, median and extremes of a distribution over 0, 1, 2, ..."""
    s = np.arange(len(dist))
    mean = float(s @ dist)
    cdf = np.cumsum(dist)
    support = np.nonzero(dist > 0)[0]
    return {
        "mean": mean,
        "sd": float(np.sqrt(((s - mean) ** 2) @ dist)),
        "median": int(np.searchsorted(cdf, 0.5)),
        "min": int(support[0]),
        "max": int(support[-1]),
        "mode": int(dist.argmax()),
    }
