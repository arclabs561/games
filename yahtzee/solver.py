"""The optimal solitaire strategy, by dynamic programming over states between turns.

Between turns, all that matters for the rest of the game is the **state** (Glenn 2006):

- the set of boxes already filled (2^13 sets);
- the upper total so far, capped at 63, since more than 63 earns no more bonus (64 values);
- whether the Yahtzee box holds 50, so that a further Yahtzee earns the bonus (2 values).

The **value** E(S) of a state is the expected score still to come under optimal play. A state
with every box filled is worth 35 if its upper total reached 63 and 0 otherwise. A turn from
state S is a small game of its own (Glenn's **widget**), solved from the end backwards:

    final roll r:        V2(r) = max over open boxes c of  points(S, r, c) + E(next(S, r, c))
    keep k, one reroll:  K1(k) = sum over rolls r of P(k -> r) V2(r)
    second roll r:       V1(r) = max over keeps k inside r of K1(k)
    keep k, two rerolls: K0(k) = sum over r of P(k -> r) V1(r)
    first roll r:        V0(r) = max over keeps k inside r of K0(k)
    start of turn:       E(S)  = sum over r of P(nothing kept -> r) V0(r)

Keeping all five dice is one of the keeps, so standing pat needs no special case. Filling a box
moves to a state with one more box filled, so the states are solved in order of how many boxes
are filled, most first. Each step is done for thousands of states at once with numpy: the sums
are a matrix product with the 462 x 252 matrix P, and the maximum over keeps inside a roll is
built one die at a time, from keeps of four dice down to keeps of none.

Only reachable states are solved. A state is unreachable when no rolls give its upper total
with its filled upper boxes, or when the Yahtzee box holds 50 while it is still open. Of the
4,096 pairs of filled upper boxes and capped totals, 1,302 are unreachable (Glenn reports 1,260),
which leaves 536,448 states, the number Verhoeff reports.

    uv run python -m yahtzee.solver     # the official rules; about a minute
"""

import sys
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np

from yahtzee.rules import JOKER_SCORES, LOWER, OFFICIAL, UPPER, Rules, multisets, probability, score


class Tables:
    """Rolls, keeps, transition probabilities and base scores for one version of the game."""

    def __init__(self, rules: Rules):
        self.rules = rules
        n, faces = rules.dice, rules.faces
        self.rolls = multisets(faces, n)
        self.keeps = [k for size in range(n + 1) for k in multisets(faces, size)]
        self.keep_index = {k: i for i, k in enumerate(self.keeps)}
        self.roll_keep = np.array([self.keep_index[r] for r in self.rolls])
        roll_index = {r: i for i, r in enumerate(self.rolls)}
        # P[k, r]: the chance that keeping k and rerolling the other dice gives roll r.
        self.P = np.zeros((len(self.keeps), len(self.rolls)))
        for k, keep in enumerate(self.keeps):
            for rest in multisets(faces, n - len(keep)):
                r = roll_index[tuple(sorted(keep + rest))]
                self.P[k, r] += float(probability(rest, faces))
        self.first = self.P[self.keep_index[()]]
        # For the max over keeps inside a roll: each keep's keeps with one die fewer.
        self.children = []
        for size in range(1, n + 1):
            idx = [i for i, k in enumerate(self.keeps) if len(k) == size]
            kids = [
                sorted({self.keep_index[drop(self.keeps[i], v)] for v in self.keeps[i]})
                for i in idx
            ]
            width = max(map(len, kids))
            self.children.append(
                (np.array(idx), np.array([c + c[:1] * (width - len(c)) for c in kids]))
            )
        # Every keep inside each roll, the largest first, so that ties keep more dice.
        subs = [
            sorted(
                {sub for size in range(n + 1) for sub in combinations(r, size)},
                key=lambda k: -len(k),
            )
            for r in self.rolls
        ]
        width = max(map(len, subs))
        self.subkeeps = np.array(
            [
                [self.keep_index[k] for k in s] + [self.keep_index[s[0]]] * (width - len(s))
                for s in subs
            ]
        )
        # Scores and Yahtzee facts per roll, for the boxes of this version.
        cats = rules.categories
        self.score = np.array([[score(r, c) for c in cats] for r in self.rolls])
        self.yahtzee = np.array([len(set(r)) == 1 for r in self.rolls])
        self.face = np.array([r[0] if len(set(r)) == 1 else 0 for r in self.rolls])
        self.bit = {c: 1 << i for i, c in enumerate(cats)}
        # States: (filled boxes, capped upper total, Yahtzee box holds 50).
        has_upper = any(c in UPPER for c in cats)
        self.cap = rules.threshold if rules.upper_bonus and has_upper else 0
        self.width = self.cap + 1
        self.n_states = (1 << len(cats)) * self.width * 2
        self.reach = upper_totals(rules, self.cap)

    def index(self, mask, total, flag):
        """The position of states in the value array."""
        return (mask * self.width + total) * 2 + flag

    def states(self, filled: int) -> np.ndarray:
        """The reachable states with `filled` boxes filled, as indices."""
        rules, out = self.rules, []
        cats = rules.categories
        upper_bits = sum(self.bit[c] for c in cats if c in UPPER)
        y = self.bit.get("yahtzee", 0)
        flags = (0, 1) if rules.yahtzee_bonus and y else (0,)
        for chosen in combinations(range(len(cats)), filled):
            mask = sum(1 << i for i in chosen)
            for total in self.reach[mask & upper_bits]:
                for flag in flags:
                    if flag and not mask & y:
                        continue
                    out.append(self.index(mask, total, flag))
        return np.array(out, dtype=np.int64)


def drop(keep: tuple[int, ...], v: int) -> tuple[int, ...]:
    """The keep with one die of face v set back."""
    i = keep.index(v)
    return keep[:i] + keep[i + 1 :]


def upper_totals(rules: Rules, cap: int) -> dict[int, list[int]]:
    """For each set of filled upper boxes (as a bit mask), the capped totals rolls can give."""
    upper = [(1 << i, c) for i, c in enumerate(rules.categories) if c in UPPER]
    out = {}
    for size in range(len(upper) + 1):
        for chosen in combinations(upper, size):
            totals = {0}
            for _, c in chosen:
                v = UPPER.index(c) + 1
                totals = {t + v * k for t in totals for k in range(rules.dice + 1)}
            out[sum(b for b, _ in chosen)] = sorted({min(t, cap) for t in totals})
    return out


@dataclass
class Turn:
    """One widget solved for a batch of states, with the optimal choices that go with it."""

    value: np.ndarray  # (B,) expected score still to come at the start of the turn
    box: np.ndarray  # (B, R) the box chosen for each final roll (index into categories)
    points: np.ndarray  # (B, R) the points written in that box, Joker included
    bonus: np.ndarray  # (B, R) the Yahtzee bonus earned by that roll
    next: np.ndarray  # (B, R) the state after filling that box
    joker: np.ndarray  # (B, R) whether the box was filled as a Joker
    keeps: list[np.ndarray]  # per reroll, (B, R): the keep chosen after each roll
    levels: list[np.ndarray]  # per roll, (B, R): the value of each roll at that point
    keep_values: list[np.ndarray]  # per reroll, (B, K): the value of each keep


def final_rolls(t: Tables, values: np.ndarray, states: np.ndarray, allow=None, every=False):
    """For each state and final roll: the best box, its points and the value of choosing it.

    With `allow`, a (B, R, boxes) array of booleans, only the boxes it marks are considered.
    With `every`, the value of each box is returned too, as (B, R, boxes), -inf where closed.
    """
    rules = t.rules
    cats = rules.categories
    flag = states % 2
    total = (states // 2) % t.width
    mask = states // (2 * t.width)
    nb, nr = len(states), len(t.rolls)
    best = np.full((nb, nr), -np.inf)
    out = {
        k: np.zeros((nb, nr), dtype=np.int64) for k in ("box", "points", "bonus", "next", "joker")
    }
    each = []
    y = t.bit.get("yahtzee", 0)
    y_filled = (mask & y) > 0 if y else np.zeros(nb, dtype=bool)
    yahtzee = t.yahtzee[None, :] & y_filled[:, None]  # a Yahtzee roll after the box is filled
    # Upper box of the Yahtzee's face filled (or not in this version), roll by roll.
    face_bit = np.array([t.bit.get(UPPER[f - 1], 0) if f else 0 for f in t.face])
    face_filled = ((mask[:, None] & face_bit[None, :]) > 0) | (face_bit[None, :] == 0)
    is_joker = yahtzee & face_filled & (rules.joker != "none")
    bonus = np.where(t.yahtzee[None, :] & (flag[:, None] == 1), rules.yahtzee_bonus, 0)
    if rules.joker == "forced":
        upper_open = ~face_filled
        lower_bits = sum(t.bit[c] for c in cats if c in LOWER)
        lower_open = (mask & lower_bits) != lower_bits
    for i, c in enumerate(cats):
        bit = 1 << i
        open_ = (mask & bit) == 0
        points = np.broadcast_to(t.score[:, i], (nb, nr))
        if c in JOKER_SCORES:
            points = np.where(is_joker, JOKER_SCORES[c], points)
        allowed = np.broadcast_to(open_[:, None], (nb, nr))
        if allow is not None:
            allowed = allowed & allow[:, :, i]
        if rules.joker == "forced":
            if c in UPPER:
                rule = np.where(upper_open, face_bit[None, :] == bit, ~lower_open[:, None])
            else:
                rule = ~upper_open
            allowed = allowed & (~yahtzee | rule)
        up = points if c in UPPER else 0
        new_total = np.minimum(total[:, None] + up, t.cap)
        new_flag = flag[:, None] | ((c == "yahtzee") & (points == 50) & (rules.yahtzee_bonus > 0))
        raw = t.index((mask | bit)[:, None], new_total, new_flag)
        nxt = np.where(allowed, raw, 0)
        value = np.where(allowed, points + bonus + values[nxt], -np.inf)
        assert not np.isnan(value).any(), f"unsolved state reached by filling {c}"
        if every:
            each.append(
                np.where(
                    open_[:, None],
                    points + bonus + values[np.where(open_[:, None], raw, 0)],
                    -np.inf,
                )
            )
        better = value > best
        best = np.where(better, value, best)
        for k, v in (
            ("box", i),
            ("points", points),
            ("next", nxt),
            ("bonus", bonus),
            ("joker", is_joker),
        ):
            out[k] = np.where(better, v, out[k])
    out["joker"] = out["joker"] & np.isin(
        out["box"], [cats.index(c) for c in JOKER_SCORES if c in cats]
    )
    if every:
        return best, out, np.stack(each, axis=2)
    return best, out


def best_keep(t: Tables, keep_value: np.ndarray) -> np.ndarray:
    """The value of each roll: the best keep inside it, built up one die at a time."""
    best = keep_value.copy()
    for idx, kids in t.children:
        best[:, idx] = np.maximum(best[:, idx], best[:, kids].max(axis=2))
    return best[:, t.roll_keep]


class Rule(Protocol):
    """Fixed choices for some or all decisions of a turn; the solver makes the rest optimally."""

    def boxes(self, t: Tables, states: np.ndarray) -> np.ndarray | None:
        """(B, R, boxes): the boxes allowed for each final roll, or None for all of them."""
        ...

    def keeps(self, t: Tables, states: np.ndarray, left: int) -> np.ndarray | None:
        """The keeps allowed after each roll with `left` rolls to come, or None for all.

        Either (B, R), one keep per roll with -1 for any keep, or (B, R, m), a menu of keeps
        per roll from which the best is taken.
        """
        ...


def turn(
    t: Tables,
    values: np.ndarray,
    states: np.ndarray,
    choices: bool = False,
    rule: Rule | None = None,
):
    """Solve the widget of each state; with `choices`, also return the chosen moves.

    With `rule`, the choices it fixes are made as it says and the others optimally. When it
    fixes every choice, the values are the exact expected scores of the strategy it describes.
    """
    allow = rule.boxes(t, states) if rule else None
    roll_value, out = final_rolls(t, values, states, allow)
    levels, keeps, keep_values = [roll_value], [], []
    rows = np.arange(len(states))[:, None]
    for left in range(1, t.rules.rolls):
        keep_value = roll_value @ t.P.T
        forced = rule.keeps(t, states, left) if rule else None
        if forced is None:
            roll_value = best_keep(t, keep_value)
        elif forced.ndim == 3:  # a menu of keeps for each roll: the best of them
            menu = keep_value[rows[:, :, None], forced]
            pick = menu.argmax(axis=2)
            forced = np.take_along_axis(forced, pick[:, :, None], axis=2)[:, :, 0]
            roll_value = np.take_along_axis(menu, pick[:, :, None], axis=2)[:, :, 0]
            fixed = np.ones(forced.shape, bool)
        else:
            fixed = forced >= 0
            roll_value = keep_value[rows, np.maximum(forced, 0)]
            if not fixed.all():
                roll_value = np.where(fixed, roll_value, best_keep(t, keep_value))
        if choices:
            pick = keep_value[:, t.subkeeps].argmax(axis=2)  # first maximum: the largest keep
            chosen = t.subkeeps[np.arange(len(t.rolls))[None, :], pick]
            if forced is not None:
                chosen = np.where(fixed, forced, chosen)
            keeps.insert(0, chosen)
            keep_values.insert(0, keep_value)
        levels.insert(0, roll_value)
    value = roll_value @ t.first
    if not choices:
        return value
    return Turn(
        value,
        out["box"],
        out["points"],
        out["bonus"],
        out["next"],
        out["joker"].astype(bool),
        keeps,
        levels,
        keep_values,
    )


def terminal(t: Tables) -> np.ndarray:
    """The value array with only the full score cards filled in, the rest NaN."""
    values = np.full(t.n_states, np.nan)
    full = (1 << len(t.rules.categories)) - 1
    for total in range(t.width):
        for flag in (0, 1):
            values[t.index(full, total, flag)] = (
                t.rules.upper_bonus if t.cap and total == t.cap else 0
            )
    return values


def solve(
    rules: Rules = OFFICIAL,
    batch: int = 2048,
    anchor=None,
    stop: int = 0,
    verbose: bool = False,
    rule: Rule | None = None,
):
    """The value of every reachable state under optimal play.

    Only states with at least `stop` boxes filled are solved; the rest stay NaN.

    With `anchor`, a function from state indices to estimated values, each turn is played to
    maximize its own points plus `anchor` of the state it leads to, instead of the true value.
    Glenn's greedy strategy is `anchor = lambda s: 0`. The values returned are then the
    estimates the strategy acts on, not its true expected scores (see `policy.graph`).

    With `rule`, the choices it fixes are followed (see `turn`).
    """
    t = Tables(rules)
    values = terminal(t)
    if anchor is not None:
        values = np.where(np.isnan(values), np.nan, 0.0)
    for filled in range(len(rules.categories) - 1, stop - 1, -1):
        states = t.states(filled)
        if anchor is not None:  # the next layer's values, as the strategy estimates them
            nxt = t.states(filled + 1)
            values[nxt] = anchor(nxt)
        start = time.time()
        for i in range(0, len(states), batch):
            s = states[i : i + batch]
            values[s] = turn(t, values, s, rule=rule)
        if verbose:
            print(
                f"{filled:2d} boxes filled: {len(states):7,d} states, {time.time() - start:5.1f} s",
                flush=True,
            )
    return t, values


def final_distribution(t: Tables, tr: Turn) -> np.ndarray:
    """(B, R): the chance of each final roll under the turn's choices."""
    nb, nr, nk = tr.box.shape[0], len(t.rolls), len(t.keeps)
    q = np.broadcast_to(t.first, (nb, nr))
    rows = np.arange(nb)[:, None] * nk
    for keep in tr.keeps:
        z = np.bincount((rows + keep).ravel(), weights=q.ravel(), minlength=nb * nk)
        q = z.reshape(nb, nk) @ t.P
    return q


def reachable_count(rules: Rules = OFFICIAL) -> int:
    """How many states between turns can occur, terminal ones included."""
    t = Tables(rules)
    return sum(len(t.states(k)) for k in range(len(rules.categories) + 1))


def main(argv: list[str]) -> None:
    rules = OFFICIAL
    start = time.time()
    t, values = solve(rules, verbose=True)
    print(f"expected score {values[t.index(0, 0, 0)]:.10f} in {time.time() - start:.0f} s")
    print(f"{reachable_count(rules):,d} reachable states of {t.n_states:,d}")


if __name__ == "__main__":
    main(sys.argv[1:])
