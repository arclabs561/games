"""Glenn's rule-based strategy for a human player (Glenn 2007, Tables V and VI), scored exactly.

Glenn ("Computer Strategies for Solitaire Yahtzee", IEEE CIG 2007) gives two lists of rules.
Each list is read from the top and the first rule that matches is followed. He reports an
average of 218.18 for the strategy. His words leave some choices open; the readings taken here:

- A **tripleton** is three or more dice of one face, a **doubleton** exactly two, a
  **singleton** one. "High preferred" picks the highest face that qualifies.
- "Small straight" as a keep keeps one die of each face of the four-long run.
- A scoring rule does not match when it would score zero (Glenn says so). "If it earns the
  bonus" means the upper total reaches 63 with this score and had not before.
- "Doubletons in an upper category (lower preferred)" scores two of a face in its upper box.
- With no rule matching, the first open box in the order 1, C, 2, Y, 4K, LS, SS, 3, FH, 4,
  3K, 5, 6 is filled.

    uv run python -m yahtzee.glenn      # about half a minute
"""

import sys

import numpy as np

from yahtzee.human import BOX, Card, Strategy, expected_score
from yahtzee.rules import OFFICIAL
from yahtzee.solver import Tables

FALLBACK = ["aces", "chance", "twos", "yahtzee", "four_kind", "large_straight"]
FALLBACK += ["small_straight", "threes", "full_house", "fours", "three_kind", "fives", "sixes"]
FACES = np.arange(1, 7)


class Glenn(Strategy):
    """The rules of Glenn's Tables V and VI as a strategy the evaluator can follow."""

    dedupe = False  # "if it earns the bonus" reads the upper total itself

    def keeps(self, t: Tables, states: np.ndarray, left: int) -> np.ndarray:
        d, c = self.dice(t), Card(t, states)
        o = c.open  # (B, 13)
        k = t.keep_index
        nb, nr = len(states), len(t.rolls)
        cnt = d.count[None]  # (1, R, 6)
        up = o[:, None, :6]  # (B, 1, 6)

        def any_open(*names):
            return np.logical_or.reduce([o[:, BOX[n] : BOX[n] + 1] for n in names])

        def keep_face(mask, high=True):
            """Keep every die of the highest (or lowest) face where `mask` (B, R, 6) holds."""
            score = np.where(mask, FACES if high else 7 - FACES, 0)
            face = score.argmax(axis=2)
            return mask.any(axis=2), d.of_face[face, np.arange(nr)[None, :]]

        def keep_one(mask, high=True):
            score = np.where(mask, FACES if high else 7 - FACES, 0)
            face = score.argmax(axis=2) + 1
            return mask.any(axis=2), single[face]

        single = np.array([k[()]] + [k[(f,)] for f in range(1, 7)])
        triple = cnt >= 3
        y = o[:, BOX["yahtzee"] : BOX["yahtzee"] + 1]
        face_filled = ~np.take_along_axis(o[:, :6], np.broadcast_to(d.face - 1, (nb, nr)), 1)
        joker = ~y & face_filled
        run = d.run[None]
        r345 = np.array([k[(3, 4, 5)] if {3, 4, 5} <= set(r) else -1 for r in t.rolls])
        r234 = np.array([k[(2, 3, 4)] if {2, 3, 4} <= set(r) else -1 for r in t.rolls])
        mid = np.where(r345 >= 0, r345, r234)
        n_up = o[:, :6].sum(axis=1, keepdims=True)
        rules = [
            ((d.top[None] == 5) & (y | joker), d.keep["all"]),
            ((run == 5) & any_open("large_straight", "small_straight"), d.keep["all"]),
            (
                (run >= 4)
                & (any_open("small_straight") | any_open("large_straight") & any_open("chance")),
                d.keep["run"],
            ),
            keep_face(triple & up),
            (
                (d.top[None] >= 3) & any_open("three_kind", "four_kind", "full_house", "chance"),
                d.keep["most"],
            ),
            keep_face((cnt == 2) & up),
            (
                (mid[None] >= 0)
                & (any_open("small_straight") | any_open("large_straight") & any_open("chance")),
                mid,
            ),
            (lambda m: (m[0] & any_open("three_kind", "chance"), m[1]))(keep_face(cnt == 2)),
            ((d.top[None] >= 3) & (y | c.flag), d.keep["most"]),
            (lambda m: (m[0] & (n_up <= 4), m[1]))(keep_one((cnt >= 1) & up, high=False)),
            keep_face(cnt == 2),
            (lambda m: (m[0] & any_open("three_kind", "four_kind", "chance"), m[1]))(
                keep_one((cnt >= 1) & (FACES >= 4))
            ),
        ]
        out = np.full((nb, nr), k[()])
        done = np.zeros((nb, nr), bool)
        for hit, keep in rules:
            hit = np.broadcast_to(hit, (nb, nr)) & ~done
            out = np.where(hit, np.broadcast_to(keep, (nb, nr)), out)
            done |= hit
        return out

    def box_choice(self, t: Tables, states: np.ndarray):
        d, c = self.dice(t), Card(t, states)
        nb, nr = len(states), len(t.rolls)
        pts = self.points(t, d, c)  # (B, R, 13), Joker included
        o = c.open[:, None, :]
        scores = (pts > 0) & o
        cnt = d.count[None]  # (1, R, 6)
        upper = pts[:, :, :6]
        reach = (c.total[:, :, None] < 63) & (c.total[:, :, None] + upper >= 63)
        total = d.total[None]

        def box(name, cond=True):
            i = BOX[name]
            return np.broadcast_to(scores[:, :, i] & cond, (nb, nr)), np.full((nb, nr), i)

        def up(mask, high=True):
            mask = mask & scores[:, :, :6]
            score = np.where(mask, FACES if high else 7 - FACES, 0)
            return mask.any(axis=2), score.argmax(axis=2)

        rules = [
            box("yahtzee"),
            box("large_straight"),
            box("small_straight"),
            up((cnt >= 3) & reach),
            up((cnt >= 4) & (FACES >= 5)),
            box("four_kind"),
            up((cnt >= 3) & (FACES >= 5)),
            box("full_house"),
            box("three_kind", total >= 22),
            up(cnt >= 3),
            box("three_kind"),
            box("chance", total >= 22),
            up(cnt == 2, high=False),
        ]
        order = np.array([BOX[n] for n in FALLBACK])
        first_open = order[np.argmax(c.open[:, order], axis=1)]
        out = np.broadcast_to(first_open[:, None], (nb, nr)).copy()
        done = np.zeros((nb, nr), bool)
        for hit, b in rules:
            hit = hit & ~done
            out = np.where(hit, b, out)
            done |= hit
        return out, None


GLENN = Glenn()
GLENN_REPORTED = 218.18


def main(argv: list[str]) -> None:
    t = Tables(OFFICIAL)
    print(
        f"Glenn's rules (Tables V and VI): {expected_score(GLENN, t):.4f} (reported {GLENN_REPORTED})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
