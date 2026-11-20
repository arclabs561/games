"""Common Yahtzee advice, each piece priced exactly against the optimal strategy.

A piece of **advice** applies to some decisions (a keep with two or one rolls to come, or the
box for the final roll) and restricts what may be done there. Its **cost** is the optimal score
minus the best score of a player who follows the advice wherever it applies and plays optimally
everywhere else. Both scores are exact: the second comes from `solver.solve` with the advice as
a rule, so the rest of the play adapts to the advice.

How often optimal play follows the advice anyway is measured too: the expected number of times
per game the advice applies, and the share of those times the optimal move obeys it.

    uv run python -m yahtzee.advice cost 0     # the cost of the first piece; about 2 minutes
    uv run python -m yahtzee.advice obeyed     # how often optimal play obeys each; 6 minutes
    uv run python -m yahtzee.advice all        # both, written to results/advice.json; 40 minutes
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yahtzee import policy
from yahtzee.human import BOX, Card, Dice, Strategy, decision_key
from yahtzee.rules import OFFICIAL
from yahtzee.solver import Tables, final_distribution, solve, turn

CARD = Strategy()  # for its helpers: dice facts and keeps
HERE = Path(__file__).parent


@dataclass(frozen=True)
class Advice:
    """Where a piece of advice applies, and what it allows there.

    `levels` holds 2 or 1 for the keep with that many rolls to come, or 0 for the box. `where`
    maps (dice, card, rolls to come) to booleans that broadcast to (B, R). For a keep, `allow`
    maps (dice, card, kept) to the keeps allowed, where kept[b, r, j, f] counts the dice of face
    f + 1 in the j-th keep inside roll r (B, R, 32, 6). For a box, it maps (dice, card, points)
    to the boxes allowed, (B, R, 13). Where it allows nothing, anything is allowed.
    """

    name: str
    source: str
    url: str
    quote: str
    levels: tuple[int, ...]
    where: Callable
    allow: Callable

    @property
    def kind(self) -> str:
        return "box" if self.levels == (0,) else "keep"

    def rule(self) -> "Follow":
        return Follow(self)


class Follow:
    """The advice as a rule for the solver: obey it where it applies, else play optimally."""

    def __init__(self, advice: Advice):
        self.a = advice

    def keeps(self, t: Tables, states: np.ndarray, left: int):
        a = self.a
        if left not in a.levels:
            return None
        d, c = CARD.dice(t), Card(t, states)
        nb = len(states)
        hit = np.broadcast_to(a.where(d, c, left), (nb, len(t.rolls)))
        ok = np.broadcast_to(a.allow(d, c, kept_counts(t)), (nb, *t.subkeeps.shape)).copy()
        ok |= ~ok.any(axis=2, keepdims=True)
        ok |= ~hit[:, :, None]
        menu = np.broadcast_to(t.subkeeps[None], ok.shape)
        first = np.take_along_axis(menu, ok.argmax(axis=2)[:, :, None], axis=2)
        return np.where(ok, menu, first)  # a keep not allowed stands in as an allowed one

    def boxes(self, t: Tables, states: np.ndarray):
        a = self.a
        if a.levels != (0,):
            return None
        d, c = CARD.dice(t), Card(t, states)
        points = CARD.points(t, d, c)
        hit = np.broadcast_to(a.where(d, c, 0), (len(states), len(t.rolls)))
        allowed = np.broadcast_to(a.allow(d, c, points), points.shape) & c.open[:, None, :]
        allowed = allowed | ~allowed.any(axis=2, keepdims=True)
        return np.where(hit[:, :, None], allowed, True)


def kept_counts(t: Tables) -> np.ndarray:
    """(1, R, 32, 6): the dice of each face in each keep inside each roll."""
    counts = np.array([[k.count(f) for f in range(1, 7)] for k in t.keeps])
    return counts[t.subkeeps][None]


def face_open(d: Dice, c: Card) -> np.ndarray:
    """(B, R): whether the upper box of the roll's most common face is open."""
    return np.take_along_axis(c.open[:, :6], np.broadcast_to(d.face - 1, (len(c.states), 252)), 1)


def opened(c: Card, box: str) -> np.ndarray:
    return c.open[:, BOX[box] : BOX[box] + 1]


def most_face(d: Dice):
    """(1, R, 1, 1) index of the most common face, to read kept counts of that face."""
    return (d.face - 1)[None, :, None, None]


def kept_of_face(d: Dice, kept: np.ndarray) -> np.ndarray:
    """(1, R, 32): how many dice of the roll's most common face each keep holds."""
    return np.take_along_axis(kept, most_face(d), axis=3)[..., 0]


def only(*boxes: str):
    def allow(d, c, points):
        out = np.zeros(points.shape, bool)
        out[:, :, [BOX[b] for b in boxes]] = True
        return out

    return allow


def only_face(d, c, points):
    out = np.zeros(points.shape, bool)
    np.put_along_axis(out, np.broadcast_to(d.face - 1, points.shape[:2])[:, :, None], True, 2)
    return out


def all_but(*boxes: str):
    def allow(d, c, points):
        out = np.ones(points.shape, bool)
        out[:, :, [BOX[b] for b in boxes]] = False
        return out

    return allow


def lower_scores(c: Card, points: np.ndarray) -> np.ndarray:
    """(B, R): whether some open lower box other than Chance scores."""
    return ((points[:, :, 6:12] > 0) & c.open[:, None, 6:12]).any(axis=2)


def no_upper(d, c, points):
    out = np.ones(points.shape, bool)
    out[:, :, :6] = False
    return out


def three_or_more_up(d, c, points):
    """Upper boxes only with three or more dice of their face, or none (a scratch)."""
    enough = (d.count >= 3) | (d.count == 0)  # (R, 6)
    out = np.ones(points.shape, bool)
    out[:, :, :6] = enough[None]
    return out


def nothing_scores(d, c):
    points = CARD.points(c.t, d, c)
    return ~((points > 0) & c.open[:, None, :]).any(axis=2) & c.open[:, :6].any(axis=1)[:, None]


def upper_only(d, c, points):
    out = np.zeros(points.shape, bool)
    out[:, :, :6] = True
    return out


def every(d, c, left):
    return np.ones((1, 1), bool)


CORNELL = (
    "Cornell math department, Yahtzee strategy page",
    "https://pi.math.cornell.edu/~mec/2006-2007/Probability/Yahtzee.htm",
)
MANIFESTO = (
    "Yahtzee Manifesto, strategy cheat sheet",
    "https://www.yahtzeemanifesto.com/yahtzee-strategy.php",
)
BGA = ("Board Game Arena forum, 2026", "https://forum.boardgamearena.com/viewtopic.php?t=50222")
DOPE_2024 = (
    "Straight Dope message board, 2024",
    "https://boards.straightdope.com/t/whats-the-best-yahtzee-strategy/1006177",
)
DOPE_2020 = (
    "Straight Dope message board, 2020",
    "https://boards.straightdope.com/t/is-there-an-optimum-strategy-to-yahtzee/855164",
)
UK = ("yahtzee.org.uk strategy page", "http://yahtzee.org.uk/strategy.html")
HASBRO = (
    "Hasbro rule booklet, tips and tactics",
    "https://www.hasbro.com/common/documents/dad2af551c4311ddbd0b0800200c9a66/8302f43150569047f57eb8d746ba9d86.pdf",
)
RULES_1967 = (
    "1967 rule booklet, as quoted by the Yahtzee Manifesto",
    "https://www.yahtzeemanifesto.com/yahtzee-strategy.php",
)
DICEWAR = ("dice-war.com strategy page", "https://dice-war.com/yahtzee/strategy.html")


def _keep_most(d, c, kept):
    """Keep every die of the most common face and nothing else."""
    n = kept_of_face(d, kept)
    return (n == d.top[None, :, None]) & (kept.sum(axis=3) == n)


def _keep_run(d, c, kept):
    run = np.array(RUN_COUNTS)[d.keep["run"]][None, :, None, :]
    return (kept == run).all(axis=3)


def _keep_pairs(d, c, kept):
    pairs = np.array(RUN_COUNTS)[d.keep["pairs"]][None, :, None, :]
    return (kept == pairs).all(axis=3)


def _keep_all(d, c, kept):
    return kept.sum(axis=3) == 5


RUN_COUNTS: list = []  # keep index -> dice per face; set by `setup`


def setup(t: Tables) -> None:
    RUN_COUNTS[:] = [[k.count(f) for f in range(1, 7)] for k in t.keeps]


ADVICE = [
    Advice(
        "Four 4s, 5s or 6s go in their upper box",
        *CORNELL,
        "take the points in the upper section if they are available",
        (0,),
        lambda d, c, left: (d.top[None] == 4) & (d.face[None] >= 4) & face_open(d, c),
        only_face,
    ),
    Advice(
        "Four of a kind goes in its upper box while the bonus is open",
        *BGA,
        "Always score 4 of a kind on the top if we need it there",
        (0,),
        lambda d, c, left: (d.top[None] == 4) & face_open(d, c) & ~c.secured,
        only_face,
    ),
    Advice(
        "Three or four 1s, 2s or 3s go in their upper box",
        *RULES_1967,
        "it is better to score them in the UPPER SECTION",
        (0,),
        lambda d, c, left: (
            (d.top[None] >= 3) & (d.top[None] <= 4) & (d.face[None] <= 3) & face_open(d, c)
        ),
        only_face,
    ),
    Advice(
        "Upper boxes take three or more of their face, or a zero",
        *MANIFESTO,
        "Aim for at least 3-of-a-kind in every Upper box.",
        (0,),
        lambda d, c, left: ~c.secured,
        three_or_more_up,
    ),
    Advice(
        "A roll that fits a lower box goes there, not in the upper section",
        *DOPE_2024,
        "I generally choose the bottom",
        (0,),
        lambda d, c, left: lower_scores(c, CARD.points(c.t, d, c)),
        no_upper,
    ),
    Advice(
        "A full house goes in Full House, not Three of a Kind",
        *DOPE_2020,
        "Take the full house over a high scoring 3-of-a-kind.",
        (0,),
        lambda d, c, left: d.full_house[None] & opened(c, "full_house"),
        only("full_house"),
    ),
    Advice(
        "Chance takes 21 or more, except in the last three turns",
        *MANIFESTO,
        "Save it for rolls of 21+ points",
        (0,),
        lambda d, c, left: (d.total[None] < 21) & (c.turns_left > 3),
        all_but("chance"),
    ),
    Advice(
        "Chance stays open for the first six turns",
        *HASBRO,
        "Leave your CHANCE roll until later in the game",
        (0,),
        lambda d, c, left: c.turns_left >= 8,
        all_but("chance"),
    ),
    Advice(
        "The Yahtzee box is not scratched before the last turn",
        *MANIFESTO,
        "Keep it open as long as possible",
        (0,),
        lambda d, c, left: opened(c, "yahtzee") & (c.turns_left > 1) & (d.top[None] < 5),
        all_but("yahtzee"),
    ),
    Advice(
        "A forced zero goes in the upper section",
        *MANIFESTO,
        "Scratch it in the Upper Section (e.g. Ones)",
        (0,),
        lambda d, c, left: nothing_scores(d, c),
        upper_only,
    ),
    Advice(
        "With three or four of a kind and its upper box open, go for the Yahtzee",
        *CORNELL,
        "you should definitely go for the Yahtzee",
        (2, 1),
        lambda d, c, left: (
            (d.top[None] >= 3) & (d.top[None] <= 4) & face_open(d, c) & opened(c, "yahtzee")
        ),
        _keep_most,
    ),
    Advice(
        "With three of a kind, keep them, except in the last three turns",
        *BGA,
        "If we throw 3 of something, always continue with it",
        (2, 1),
        lambda d, c, left: (d.top[None] == 3) & (c.turns_left > 3),
        lambda d, c, kept: kept_of_face(d, kept) >= 3,
    ),
    Advice(
        "With three of a kind and the bonus not made, stop rolling",
        *DICEWAR,
        "bank the sure points in Four of a Kind or Three of a Kind",
        (2, 1),
        lambda d, c, left: (d.top[None] == 3) & ~c.secured,
        _keep_all,
    ),
    Advice(
        "With a lone pair of 1s or 2s early on, do not keep the pair",
        *CORNELL,
        "will not get you many points",
        (2,),
        lambda d, c, left: (
            (d.top[None] == 2) & (d.face[None] <= 2) & ~d.two_pair[None] & (c.turns_left >= 9)
        ),
        lambda d, c, kept: kept_of_face(d, kept) < 2,
    ),
    Advice(
        "With two pair and Full House open, keep both pairs",
        *UK,
        "hold the two's and sixes",
        (2, 1),
        lambda d, c, left: d.two_pair[None] & opened(c, "full_house"),
        _keep_pairs,
    ),
    Advice(
        "With two pair before the last six turns, do not keep both pairs",
        *BGA,
        "Don't chase full houses unless we are late in the game",
        (2, 1),
        lambda d, c, left: d.two_pair[None] & (c.turns_left > 6),
        lambda d, c, kept: ~_keep_pairs(d, c, kept),
    ),
    Advice(
        "With a small straight after the first roll, chase the large straight",
        *MANIFESTO,
        "Use your remaining rolls to chase a Large Straight instead.",
        (2,),
        lambda d, c, left: (d.run[None] == 4) & opened(c, "large_straight"),
        _keep_run,
    ),
]


def optimal() -> tuple[Tables, np.ndarray, float]:
    t, values = solve(OFFICIAL)
    return t, values, float(values[t.index(0, 0, 0)])


def cost(i: int, best: float | None = None) -> dict:
    """The best score while following piece i, and its cost against optimal play."""
    t = Tables(OFFICIAL)
    setup(t)
    a = ADVICE[i]
    _, values = solve(OFFICIAL, rule=a.rule())
    v = float(values[t.index(0, 0, 0)])
    if best is None:
        best = optimal()[2]
    return {"index": i, "name": a.name, "value": v, "cost": best - v}


def obeyed(batch=2048) -> list[dict]:
    """Under optimal play: times per game each piece applies, and the share obeyed there."""
    t, values, _ = optimal()
    setup(t)
    g = policy.graph(t, values)
    d = CARD.dice(t)
    applies = np.zeros(len(ADVICE))
    follows = np.zeros(len(ADVICE))
    for layer in g.layers[:-1]:
        for lo in range(0, len(layer.states), batch):
            s, occ = layer.states[lo : lo + batch], layer.occupancy[lo : lo + batch]
            tr = turn(t, values, s, choices=True)
            nb, nk = len(s), len(t.keeps)
            rows = np.arange(nb)[:, None]
            q = [np.broadcast_to(t.first, (nb, len(t.rolls)))]
            for keep in tr.keeps:
                z = np.bincount((rows * nk + keep).ravel(), q[-1].ravel(), nb * nk)
                q.append(z.reshape(nb, nk) @ t.P)
            assert np.allclose(q[-1], final_distribution(t, tr))
            # The advice reads only what `decision_key` holds, so work each key out once.
            _, first, inv = np.unique(decision_key(t, s).ravel(), True, True)
            rep, c = s[first], Card(t, s[first])
            for i, a in enumerate(ADVICE):
                rule = a.rule()
                for left in a.levels:
                    w = occ[:, None] * q[2 - left]
                    hit = np.broadcast_to(a.where(d, c, left), (len(rep), w.shape[1]))[inv]
                    if left == 0:
                        allowed = rule.boxes(t, rep)[inv]
                        ok = np.take_along_axis(allowed, tr.box[:, :, None], 2)[:, :, 0]
                    else:
                        menu = rule.keeps(t, rep, left)[inv]
                        ok = (menu == tr.keeps[2 - left][:, :, None]).any(2)
                    applies[i] += (w * hit).sum()
                    follows[i] += (w * (hit & ok)).sum()
    share = follows / np.maximum(applies, 1e-300)
    return [
        {"index": i, "name": a.name, "applies": float(n), "obeyed": float(f)}
        for i, (a, n, f) in enumerate(zip(ADVICE, applies, share, strict=True))
    ]


def save(costs: list[dict], obeys: list[dict], path: Path = HERE / "results" / "advice.json"):
    """The table the README quotes: each piece with its source, quote, cost and obedience."""
    rows = []
    for a, c, o in zip(ADVICE, costs, obeys, strict=True):
        assert c["name"] == o["name"] == a.name
        rows.append(
            {"name": a.name, "source": a.source, "url": a.url, "quote": a.quote}
            | {k: c[k] for k in ("value", "cost")}
            | {k: o[k] for k in ("applies", "obeyed")}
        )
    path.write_text(json.dumps({"cost": rows}, indent=1) + "\n")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["cost", "obeyed", "all"])
    p.add_argument("index", nargs="?", type=int, default=-1, help="the piece; -1 for all")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.mode == "all":
        best = optimal()[2]
        save([cost(i, best) for i in range(len(ADVICE))], obeyed())
        return
    out = None
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        out = a.out / (f"cost-{a.index}.jsonl" if a.mode == "cost" else "obeyed.jsonl")
        out.write_text("")
    if a.mode == "cost":
        best = optimal()[2]
        rows = (cost(i, best) for i in (range(len(ADVICE)) if a.index < 0 else [a.index]))
    else:
        rows = iter(obeyed())
    for r in rows:
        print(json.dumps(r), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
