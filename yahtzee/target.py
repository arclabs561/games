"""Playing to reach a score, or to beat an opponent, instead of playing for the highest mean.

Optimal play for the mean (`solver.py`) stores one number per state between turns. A player
who wants the largest chance of scoring at least T needs more: the chance depends on how many
points are still **needed**, so each state stores a vector, one entry per number of points
still needed (Pawlewicz 2010 stores the same vector). For a state S and n points needed,

    F(S, n) = the largest chance, over strategies, that the rest of the game scores n or more.

F(S, n) = 1 for n <= 0. Scoring p points in a turn moves from (S, n) to (S', n - p). The turn is
solved as in `solver.turn`, entry by entry: the best box, keep and box again for each n, and the
sums over rolls are the same matrix products with one row per (state, n). Since every n is a
separate maximum, the vector at the empty score card gives, for each T, the best chance of
scoring T or more, and in general each T has its own strategy.

The same recursion with another value at the end of the game gives the **best response** to an
opponent whose final score has a known distribution and who does not react to the game (as
when both play from separate score cards and compare at the end). With a points the player
has scored so far, the value at the end is

    U(a) = P(opponent scores less than a) + w P(opponent scores exactly a),

with w = 0 to count only wins and w = 1/2 to count a tie as half a win. Writing n = A - a for a
number A above the opponent's highest score makes it the same recursion: U is 1 from A up, so
F(S, n) = 1 for n <= 0 again, and only the end of the game changes.

Layers are solved from the full score card back to the empty one, and only the next layer is
kept in memory. The full game needs about 3 GB and some hours, so it runs on GitHub Actions:

    uv run python -m yahtzee.target reach --workers 4 --out DIR         # the full game
    uv run python -m yahtzee.target respond-win --workers 4 --out DIR   # against optimal play
    uv run python -m yahtzee.target respond-half --workers 4 --out DIR  # a tie counts half

Each takes hours on one core and about 3 GB, so they run on GitHub Actions
(experiment.yml, one job each); `report` then writes results/target.json from their outputs.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from yahtzee.rules import JOKER_SCORES, LOWER, OFFICIAL, UPPER, Rules
from yahtzee.solver import Tables

HERE = Path(__file__).parent
# Columns n = 0 .. 1576 for the full game: its highest score is 1575, so the last is 0.
OFFICIAL_WIDTH = 1577
# Against optimal play for the mean (count only wins, or a tie as half a win), and against
# the cheat sheet (only wins): (opponent, weight of a tie).
RESPOND = {
    "respond-win": ("optimal", 0.0),
    "respond-half": ("optimal", 0.5),
    "respond-sheet-win": ("sheet", 0.0),
}


def box_moves(t: Tables, states: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each box: (allowed, points earned, next state), each (B, R), for every final roll.

    The points include the Yahtzee bonus and the Joker. This is `solver.final_rolls` without
    the values, so a test can check that the two agree.
    """
    rules = t.rules
    cats = rules.categories
    flag = states % 2
    total = (states // 2) % t.width
    mask = states // (2 * t.width)
    nb, nr = len(states), len(t.rolls)
    y = t.bit.get("yahtzee", 0)
    y_filled = (mask & y) > 0 if y else np.zeros(nb, dtype=bool)
    yahtzee = t.yahtzee[None, :] & y_filled[:, None]
    face_bit = np.array([t.bit.get(UPPER[f - 1], 0) if f else 0 for f in t.face])
    face_filled = ((mask[:, None] & face_bit[None, :]) > 0) | (face_bit[None, :] == 0)
    is_joker = yahtzee & face_filled & (rules.joker != "none")
    bonus = np.where(t.yahtzee[None, :] & (flag[:, None] == 1), rules.yahtzee_bonus, 0)
    if rules.joker == "forced":
        upper_open = ~face_filled
        lower_bits = sum(t.bit[c] for c in cats if c in LOWER)
        lower_open = (mask & lower_bits) != lower_bits
    out = []
    for i, c in enumerate(cats):
        bit = 1 << i
        open_ = (mask & bit) == 0
        points = np.broadcast_to(t.score[:, i], (nb, nr))
        if c in JOKER_SCORES:
            points = np.where(is_joker, JOKER_SCORES[c], points)
        allowed = np.broadcast_to(open_[:, None], (nb, nr))
        if rules.joker == "forced":
            if c in UPPER:
                rule = np.where(upper_open, face_bit[None, :] == bit, ~lower_open[:, None])
            else:
                rule = ~upper_open
            allowed = allowed & (~yahtzee | rule)
        up = points if c in UPPER else 0
        new_total = np.minimum(total[:, None] + up, t.cap)
        new_flag = flag[:, None] | ((c == "yahtzee") & (points == 50) & (rules.yahtzee_bonus > 0))
        nxt = t.index((mask | bit)[:, None], new_total, new_flag)
        out.append((allowed, np.asarray(points + bonus, dtype=np.int64), nxt))
    return out


def most_points(rules: Rules) -> int:
    """The most points one turn can earn: 50 in the Yahtzee box or a bonus with 30 or 40."""
    best = max(rules.dice * rules.faces, 50 if "yahtzee" in rules.categories else 0, 40, 35)
    return best + rules.yahtzee_bonus


class Keeps:
    """The averages and maxima over keeps, with one row per keep and one column per (state, n).

    The value of keeping k is the average, over one more die, of keeping k and that die:
    rerolling dice one at a time gives the same chances as rerolling them together. That is
    `faces` additions per keep in place of a product with the 462 x 252 matrix of chances, which
    matters here because every state has up to some 1,600 columns.
    """

    def __init__(self, t: Tables):
        n, faces = t.rules.dice, t.rules.faces
        self.t = t
        self.sizes = []
        for size in range(n - 1, -1, -1):
            idx = [i for i, k in enumerate(t.keeps) if len(k) == size]
            up = [
                [t.keep_index[tuple(sorted(t.keeps[i] + (f,)))] for f in range(1, faces + 1)]
                for i in idx
            ]
            self.sizes.append((np.array(idx), np.array(up)))

    def average(self, roll_value: np.ndarray) -> np.ndarray:
        """(K, N): the value of each keep before a reroll, from (R, N) values of the rolls."""
        t = self.t
        e = np.empty((len(t.keeps), roll_value.shape[1]))
        e[t.roll_keep] = roll_value
        for idx, up in self.sizes:
            acc = e[up[:, 0]]
            for j in range(1, up.shape[1]):
                acc += e[up[:, j]]
            acc /= up.shape[1]
            e[idx] = acc
        return e

    def best(self, keep_value: np.ndarray) -> np.ndarray:
        """(R, N): the value of each roll, the best keep inside it (as `solver.best_keep`).

        Overwrites `keep_value`.
        """
        for idx, kids in self.t.children:
            m = keep_value[kids[:, 0]]
            for j in range(1, kids.shape[1]):
                np.maximum(m, keep_value[kids[:, j]], out=m)
            np.maximum(m, keep_value[idx], out=m)
            keep_value[idx] = m
        return keep_value[self.t.roll_keep]


def box_most(t: Tables) -> np.ndarray:
    """The most points each box can hold, Joker included, Yahtzee bonus not."""
    most = t.score.max(axis=0)
    for i, c in enumerate(t.rules.categories):
        if c in JOKER_SCORES and t.rules.joker != "none":
            most[i] = max(most[i], JOKER_SCORES[c])
    return most


def bounds(t: Tables, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each state, the most points the rest of the game can earn and the most so far.

    The rest of the game: the open boxes at their most, the upper bonus, and a Yahtzee bonus in
    every turn that can earn one (every turn if the Yahtzee box holds 50; every turn but one if
    it is open; none if it holds 0). So far: the filled boxes at their most, and a Yahtzee
    bonus in every turn but one if the Yahtzee box holds 50. A turn that earns p points moves
    to a state whose bounds are at least p closer, which a test checks on small games.
    """
    rules = t.rules
    most = box_most(t)
    flag = states % 2
    mask = states // (2 * t.width)
    n = len(rules.categories)
    filled_bits = (mask[:, None] >> np.arange(n)[None, :]) & 1
    filled = filled_bits.sum(axis=1)
    y = t.bit.get("yahtzee", 0)
    y_open = (mask & y) == 0 if y else np.zeros(len(states), dtype=bool)
    turns = np.where(flag == 1, n - filled, np.where(y_open & (y > 0), n - filled - 1, 0))
    yb = rules.yahtzee_bonus
    upper = rules.upper_bonus if t.cap else 0
    rest = ((1 - filled_bits) * most).sum(axis=1) + yb * np.maximum(turns, 0) + upper
    so_far = (filled_bits * most).sum(axis=1) + np.where(flag == 1, yb * (filled - 1), 0)
    return rest, so_far


_SHARED: dict = {}


def _batch(lo: int, hi: int) -> np.ndarray:
    """F(S, n) for states lo .. hi - 1 of the layer in `_SHARED`, over the batch's columns.

    Columns past the most the rest of the game can earn are 0 (when reaching a score); columns
    for more points than the game can have earned so far are NaN (against an opponent), since
    no game gets there and a NaN read by mistake would show in the result.
    """
    t, keeps, states, cols = _SHARED["t"], _SHARED["keeps"], _SHARED["states"], _SHARED["cols"]
    padded, nxt_states, pad = _SHARED["padded"], _SHARED["nxt_states"], _SHARED["pad"]
    width = padded.shape[1] - pad
    s = states[lo:hi]
    c0, c1 = int(cols[lo:hi, 0].min()), int(cols[lo:hi, 1].max())
    out = np.full((len(s), width), _SHARED["fill"])
    if c1 <= c0:
        return out
    # windows[j, pad - p + c0] holds F(S', n - p) of row j for n = c0 .. c1 - 1.
    windows = sliding_window_view(padded[:, c0:], c1 - c0, axis=1)
    zero = len(nxt_states)  # an extra row of zeros, for boxes that cannot be chosen
    nr = len(t.rolls)
    best = np.zeros((nr, len(s), c1 - c0))
    for allowed, points, nxt in box_moves(t, s):
        if not allowed.any():
            continue
        rows = np.minimum(np.searchsorted(nxt_states, nxt), zero - 1)
        assert (nxt_states[rows[allowed]] == nxt[allowed]).all(), "unsolved state reached"
        rows = np.where(allowed, rows, zero)
        shift = pad - np.where(allowed, points, 0)
        np.maximum(best, windows[rows.T, shift.T], out=best)
    roll_value = best.reshape(nr, len(s) * (c1 - c0))
    for _ in range(1, t.rules.rolls):
        roll_value = keeps.best(keeps.average(roll_value))
    out[:, c0:c1] = (t.first @ roll_value).reshape(len(s), c1 - c0)
    return out


def layer_turn(
    t: Tables,
    states: np.ndarray,
    nxt_states: np.ndarray,
    nxt_values: np.ndarray,
    cols: np.ndarray,
    fill: float,
    batch: int = 8,
    workers: int = 1,
) -> np.ndarray:
    """F(S, n) for each state S of a layer, from the vectors F(S', n) of the next layer.

    `nxt_states` is sorted, and `nxt_values` holds one row per next state, with columns
    n = 0, 1, ... `cols` gives for each state the columns [lo, hi) that matter; the others
    are `fill`, or 0 past hi. States are batched in the order given, so the caller orders them
    to share filled boxes and Yahtzee flag, which lets a batch skip filled boxes and columns.
    """
    width = nxt_values.shape[1]
    pad = most_points(t.rules)
    padded = np.ones((len(nxt_values) + 1, pad + width))
    padded[:-1, pad:] = nxt_values
    padded[-1] = 0.0
    _SHARED.update(
        t=t,
        keeps=Keeps(t),
        states=states,
        cols=cols,
        fill=fill,
        padded=padded,
        nxt_states=nxt_states,
        pad=pad,
    )
    spans = [(lo, min(lo + batch, len(states))) for lo in range(0, len(states), batch)]
    out = np.empty((len(states), width))
    if workers > 1:
        import multiprocessing

        # Forked workers share the next layer's values without copying them.
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for (lo, hi), part in zip(spans, pool.imap(_span, spans, chunksize=8), strict=True):
                out[lo:hi] = part
    else:
        for lo, hi in spans:
            out[lo:hi] = _batch(lo, hi)
    _SHARED.clear()
    return out


def _span(span: tuple[int, int]) -> np.ndarray:
    return _batch(*span)


def solve(
    rules: Rules,
    end,
    width: int,
    mode: str,
    verbose: bool = False,
    workers: int = 1,
    window: bool = True,
) -> np.ndarray:
    """The vector F(empty card, n), n = 0 .. width - 1, for the value `end` at the end.

    `end(bonus)` gives the vector at a full score card that earned `bonus` upper-bonus points.
    `mode` is "reach" (columns past the most the game can still earn are 0) or "respond"
    (columns for more points than the game can have earned are never reached). With `window`
    off, every column of every state is computed, which a test compares.
    """
    t = Tables(rules)
    full = (1 << len(rules.categories)) - 1
    states = np.sort(t.states(len(rules.categories)))
    on_card = (states // 2) % t.width == t.cap
    values = np.array([end(rules.upper_bonus if t.cap and b else 0) for b in on_card])
    assert values.shape == (len(states), width)
    assert (states // (2 * t.width) == full).all()
    for filled in range(len(rules.categories) - 1, -1, -1):
        start = time.time()
        layer = np.sort(t.states(filled))
        rest, so_far = bounds(t, layer)
        cols = np.zeros((len(layer), 2), dtype=np.int64)
        cols[:, 1] = width
        if window and mode == "reach":
            cols[:, 1] = np.minimum(rest + 1, width)
        elif window:
            cols[:, 0] = np.maximum(width - 1 - so_far, 0)
        # Batch states that share filled boxes and Yahtzee flag; put the results back in order.
        order = np.lexsort((layer, layer % 2, layer // (2 * t.width)))
        fill = 0.0 if mode == "reach" else np.nan
        part = layer_turn(t, layer[order], states, values, cols[order], fill, workers=workers)
        del values
        values = np.empty_like(part)
        values[order] = part
        del part
        states = layer
        if verbose:
            seconds = time.time() - start
            print(
                f"{filled:2d} boxes filled: {len(layer):7,d} states, {seconds:6.0f} s", flush=True
            )
    return values[0]


def reach_end(width: int):
    """At the end of the game: needing n, the upper bonus is enough when n <= bonus."""
    return lambda bonus: (np.arange(width) <= bonus).astype(float)


def respond_end(opponent: np.ndarray, tie: float):
    """At the end of the game against an opponent's score distribution, with n = A - a."""
    top = len(opponent)  # A: the opponent scores at most top - 1
    below = np.concatenate([[0.0], np.cumsum(opponent)])  # below[a] = P(opponent < a)
    a = top - np.arange(top + 1)  # n = 0 .. top

    def end(bonus):
        final = np.minimum(a + bonus, top)
        return below[final] + tie * np.where(final < top, opponent[np.minimum(final, top - 1)], 0)

    return end


def reach(
    rules: Rules = OFFICIAL, width: int | None = None, verbose: bool = False, workers: int = 1
) -> np.ndarray:
    """For each T = 0, 1, ..., the largest chance of scoring T or more."""
    width = width or highest(rules) + 2
    return solve(rules, reach_end(width), width, "reach", verbose, workers)


def respond(
    opponent: np.ndarray,
    tie: float,
    rules: Rules = OFFICIAL,
    verbose: bool = False,
    workers: int = 1,
) -> float:
    """The best chance of beating a player whose score has the distribution `opponent`."""
    top = len(opponent)
    f = solve(rules, respond_end(opponent, tie), top + 1, "respond", verbose, workers)
    return float(f[top])


def highest(rules: Rules) -> int:
    """An upper bound on the final score: every turn at its most, plus the upper bonus."""
    return len(rules.categories) * most_points(rules) + rules.upper_bonus


def optimal_distribution() -> np.ndarray:
    """The score distribution of optimal play for the mean, from `distribution.csv`."""
    rows = [line.split(",") for line in (HERE / "distribution.csv").read_text().splitlines()[1:]]
    dist = np.zeros(int(rows[-1][0]) + 1)
    for s, p in rows:
        dist[int(s)] = float(p)
    return dist


def sheet_distribution() -> np.ndarray:
    """The cheat sheet's exact score distribution, as `versus.py` wrote it."""
    return np.array(
        json.loads((HERE / "results" / "versus.json").read_text())["sheet_distribution"]
    )


def main(argv: list[str]) -> None:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 1
    start = time.time()
    if argv[0] == "reach":
        f = reach(OFFICIAL, OFFICIAL_WIDTH, verbose=True, workers=workers)
        data = {"reach": f.tolist()}
        print(f"mean of the distribution of best chances: {f[1:].sum():.6f}", flush=True)
    elif argv[0] in RESPOND:
        who, tie = RESPOND[argv[0]]
        opponent = optimal_distribution() if who == "optimal" else sheet_distribution()
        data = {"opponent": who, "tie": tie}
        data["value"] = respond(opponent, tie, OFFICIAL, True, workers)
        print(f"{argv[0]}: {data['value']:.12f}", flush=True)
    else:
        raise SystemExit(__doc__)
    data["seconds"] = round(time.time() - start)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{argv[0]}.json").write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
