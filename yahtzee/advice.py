"""Common Yahtzee advice, each piece tested against the optimal strategy.

A piece of **advice** applies to some decisions (a keep after the first or second roll, or the
box for the final roll) and restricts what may be done there. Its **cost** is the optimal score
minus the best score of a player who follows the advice wherever it applies and plays optimally
everywhere else. Both are exact: the second comes from `solver.solve` with the advice as a rule.

How often optimal play itself follows the advice is measured too: the expected number of times
per game the advice applies, and the share of those times the optimal move obeys it anyway.

    uv run python -m yahtzee.advice        # every piece; about 15 minutes
"""

import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from yahtzee import policy
from yahtzee.human import BOX, Card, Dice, Strategy
from yahtzee.rules import OFFICIAL
from yahtzee.solver import Tables, final_distribution, solve, turn

CARD = Strategy()  # for its helpers: dice facts and keeps


@dataclass(frozen=True)
class Advice:
    """Where the advice applies, and what it allows there.

    `level` is 2 or 1 for the keep with that many rolls to come, 0 for the box. `where` maps
    (dice, card, rolls to come) to a (B, R) boolean array. For a keep, `keep` names the dice to
    keep (a key of `Dice.keep`) or `avoid` names dice not to keep. For a box, `boxes` maps
    (dice, card, points) to the (B, R, 13) boxes allowed; if it allows none, any box is allowed.
    """

    name: str
    source: str
    quote: str
    levels: tuple[int, ...]
    where: Callable
    keep: str | None = None
    avoid: str | None = None
    boxes: Callable | None = None

    def rule(self) -> "Follow":
        return Follow(self)


class Follow:
    """The advice as a rule for the solver: obey it where it applies, else play optimally."""

    def __init__(self, advice: Advice):
        self.a = advice

    def keeps(self, t: Tables, states: np.ndarray, left: int):
        a = self.a
        if left not in a.levels or a.boxes is not None:
            return None
        d, c = CARD.dice(t), Card(t, states)
        hit = np.broadcast_to(a.where(d, c, left), (len(states), len(t.rolls)))
        if a.keep is not None:
            return np.where(hit, CARD.keep_of(a.keep, d, c), -1)
        # Every keep inside the roll except the one to avoid (the whole roll stands in for it).
        menu = np.broadcast_to(t.subkeeps[None], (len(states), *t.subkeeps.shape))
        bad = CARD.keep_of(a.avoid, d, c)[:, :, None] if a.avoid else None
        return np.where(hit[:, :, None] & (menu == bad), t.subkeeps[None, :, :1], menu)

    def boxes(self, t: Tables, states: np.ndarray):
        a = self.a
        if a.boxes is None:
            return None
        d, c = CARD.dice(t), Card(t, states)
        points = CARD.points(t, d, c)
        hit = np.broadcast_to(a.where(d, c, 0), (len(states), len(t.rolls)))
        allowed = a.boxes(d, c, points) & c.open[:, None, :]
        allowed = allowed | ~allowed.any(axis=2, keepdims=True)
        return np.where(hit[:, :, None], allowed, True)


def face_open(d: Dice, c: Card) -> np.ndarray:
    return np.take_along_axis(c.open[:, :6], np.broadcast_to(d.face - 1, (len(c.states), 252)), 1)


def only(box: int):
    def allowed(d, c, points):
        out = np.zeros(points.shape, bool)
        out[:, :, box] = True
        return out

    return allowed


def only_face(d, c, points):
    out = np.zeros(points.shape, bool)
    np.put_along_axis(out, np.broadcast_to(d.face - 1, points.shape[:2])[:, :, None], True, 2)
    return out


def all_but(*boxes: int):
    def allowed(d, c, points):
        out = np.ones(points.shape, bool)
        out[:, :, list(boxes)] = False
        return out

    return allowed


def lower_if_it_scores(d, c, points):
    """Upper boxes only when no open lower box scores."""
    lower = (points[:, :, 6:] > 0) & c.open[:, None, 6:]
    out = np.ones(points.shape, bool)
    out[:, :, :6] = ~lower.any(axis=2, keepdims=True)
    return out


def three_or_more_up(d, c, points):
    """Upper boxes only with three or more dice of their face, or none (a scratch)."""
    enough = (d.count >= 3) | (d.count == 0)  # (R, 6)
    out = np.ones(points.shape, bool)
    out[:, :, :6] = enough[None]
    return out


Y, C = BOX["yahtzee"], BOX["chance"]
ADVICE = [
    Advice(
        "Score four 4s, 5s or 6s in their upper box",
        "Cornell math department, Yahtzee strategy page",
        "take the points in the upper section if they are available",
        (0,),
        lambda d, c, left: (d.top[None] == 4) & (d.face[None] >= 4) & face_open(d, c),
        boxes=only_face,
    ),
    Advice(
        "Aim for three or more of each face in the upper boxes",
        "Yahtzee Manifesto, strategy cheat sheet",
        "Aim for at least 3-of-a-kind in every Upper box.",
        (0,),
        lambda d, c, left: ~c.secured,
        boxes=three_or_more_up,
    ),
    Advice(
        "Prefer a lower box to an upper box when both score",
        "Straight Dope message board, 2024",
        "If a roll can fill a block on the top or the bottom, I generally choose the bottom.",
        (0,),
        lambda d, c, left: np.ones((1, 1), bool),
        boxes=lower_if_it_scores,
    ),
    Advice(
        "With three of a kind and its upper box open, go for the Yahtzee",
        "Cornell math department, Yahtzee strategy page",
        "you should definitely go for the Yahtzee",
        (2, 1),
        lambda d, c, left: (d.top[None] >= 3) & face_open(d, c) & c.open[:, Y : Y + 1],
        keep="most",
    ),
    Advice(
        "Keep the Yahtzee box open until the last turn",
        "Yahtzee Manifesto, strategy cheat sheet",
        "Keep it open as long as possible",
        (0,),
        lambda d, c, left: c.open[:, Y : Y + 1] & (c.turns_left > 1),
        boxes=all_but(Y),
    ),
    Advice(
        "Keep two pair for the full house",
        "Dinogame blog, 2026",
        "Two pair: keep both pairs and go for Full House",
        (2, 1),
        lambda d, c, left: d.two_pair[None] & c.open[:, BOX["full_house"] : BOX["full_house"] + 1],
        keep="pairs",
    ),
    Advice(
        "With a small straight after the first roll, chase the large straight",
        "Yahtzee Manifesto, strategy cheat sheet",
        "Use your remaining rolls to chase a Large Straight instead.",
        (2,),
        lambda d, c, left: (
            (d.run[None] == 4) & c.open[:, BOX["large_straight"] : BOX["large_straight"] + 1]
        ),
        keep="run",
    ),
    Advice(
        "Score Chance only with 21 or more, unless it is the last box",
        "Yahtzee Manifesto, strategy cheat sheet",
        "Save it for rolls of 21+ points",
        (0,),
        lambda d, c, left: (d.total[None] < 21) & (c.turns_left > 1),
        boxes=all_but(C),
    ),
    Advice(
        "Do not keep a pair of 1s or 2s after the first roll",
        "Cornell math department, Yahtzee strategy page",
        "Keeping the 1's or 2's to seek a 3 or 4 of a kind will not get you many points.",
        (2,),
        lambda d, c, left: (d.top[None] == 2) & (d.face[None] <= 2) & ~d.two_pair[None],
        avoid="most",
    ),
]


def costs(advice=ADVICE, verbose=True):
    """For each piece of advice: the best score while following it, and its cost."""
    t, best = solve(OFFICIAL)
    optimal = float(best[t.index(0, 0, 0)])
    out = []
    for a in advice:
        start = time.time()
        t, values = solve(OFFICIAL, rule=a.rule())
        v = float(values[t.index(0, 0, 0)])
        out.append((a, v, optimal - v))
        if verbose:
            print(f"{a.name}: {v:.4f}, cost {optimal - v:.4f} ({time.time() - start:.0f} s)")
    return out


def obeyed(advice=ADVICE, batch=2048):
    """Under optimal play: times per game each advice applies, and the share obeyed there."""
    t, values = solve(OFFICIAL)
    g = policy.graph(t, values)
    d = CARD.dice(t)
    applies = np.zeros(len(advice))
    follows = np.zeros(len(advice))
    for layer in g.layers[:-1]:
        for lo in range(0, len(layer.states), batch):
            s, occ = layer.states[lo : lo + batch], layer.occupancy[lo : lo + batch]
            tr = turn(t, values, s, choices=True)
            c = Card(t, s)
            nb, nk = len(s), len(t.keeps)
            rows = np.arange(nb)[:, None]
            q = [np.broadcast_to(t.first, (nb, len(t.rolls)))]
            for keep in tr.keeps:
                z = np.bincount((rows * nk + keep).ravel(), q[-1].ravel(), nb * nk)
                q.append(z.reshape(nb, nk) @ t.P)
            assert np.allclose(q[-1], final_distribution(t, tr))
            for i, a in enumerate(advice):
                rule = a.rule()
                for left in a.levels:
                    w = occ[:, None] * q[2 - left]
                    hit = np.broadcast_to(a.where(d, c, left), w.shape)
                    if left == 0:
                        allowed = rule.boxes(t, s)
                        ok = np.take_along_axis(allowed, tr.box[:, :, None], 2)[:, :, 0]
                    else:
                        choice = tr.keeps[2 - left]
                        menu = rule.keeps(t, s, left)
                        if menu.ndim == 3:
                            ok = (menu == choice[:, :, None]).any(axis=2)
                        else:
                            ok = (menu < 0) | (menu == choice)
                    applies[i] += (w * hit).sum()
                    follows[i] += (w * (hit & ok)).sum()
    return applies, follows / np.maximum(applies, 1e-300)


def main(argv: list[str]) -> None:
    applies, share = obeyed()
    for a, n, f in zip(ADVICE, applies, share, strict=True):
        print(f"{a.name}: applies {n:.4f} times per game; optimal play obeys {100 * f:.1f}%")
    costs()


if __name__ == "__main__":
    main(sys.argv[1:])
