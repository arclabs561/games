"""Why the three-player information strategy here trails hanabi.rs by 2.5 points.

`information.py` generalizes Cox et al.'s strategy to P players with 2(P - 1) hint values: a
rank or a color hint to one of the other players. With three players that is 4 values, so
each hint tells every other player which of 4 sets holds the targeted card. hanabi.rs
(github.com/WuTheFWasThat/hanabi.rs) reads a hint by which cards it touches too, and gets 3
or 4 values per other player, 6 to 8 in all; it also asks different questions with them.

This module measures the first of those two differences inside `information.py`. It plays
the same strategy with **relaxed hint values**: each hint carries a number modulo m, for m
from 4 to 16, as if the hinter could send any of m messages. That is not a legal strategy
for m > 4; it is a measurement of how much the number of hint values limits this strategy.
The second difference is measured on hanabi.rs's side, by capping its values per player at
2 (see `results/hat-gap.md`; its code is not copied here).

    uv run python -m hanabi.hat_gap -n 10000 --out hanabi/results   # about 25 minutes
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from hanabi.information import InfoTeam, play_info_game
from hanabi.rules import Rules, seeded_deck

WIDTHS = (4, 6, 8, 12, 16)


class RelaxedTeam(InfoTeam):
    """The information strategy with each hint read as a number modulo `width`: the legal
    hint for the number mod 2(P - 1) is played, and the full number is passed beside it."""

    width = 8

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.hats = self.width
        self.sent = 0

    def encode(self, giver, number, state):
        self.sent = number
        return super().encode(giver, number % (2 * (self.rules.players - 1)), state)

    def decode(self, giver, action):
        return self.sent


def scores(players: int, width: int, n: int) -> list[int]:
    """Scores on seeded deals 0 to n - 1, HanSim's discard rule as in the README. A width of
    2(P - 1) is the strategy itself."""
    rules = Rules(players=players, discard_when_full=True)
    team = InfoTeam
    if width != 2 * (players - 1):
        team = type(f"Relaxed{width}", (RelaxedTeam,), {"width": width})
    return [play_info_game(rules, seeded_deck(rules, s), team_class=team).score for s in range(n)]


def summary(xs: list[int], top: int = 25) -> dict:
    n = len(xs)
    m = mean(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    p = sum(x == top for x in xs) / n
    return {
        "deals": n,
        "mean": round(m, 4),
        "half_width": round(1.96 * sd / math.sqrt(n), 4),
        "perfect": round(p, 4),
        "perfect_half_width": round(1.96 * math.sqrt(p * (1 - p) / n), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=10000)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = {"players": args.players, "seeds": f"0 to {args.n - 1}", "widths": {}}
    for w in WIDTHS:
        xs = scores(args.players, w, args.n)
        out["widths"][str(w)] = {**summary(xs), "scores": "".join(chr(65 + x) for x in xs)}
        print(w, summary(xs), flush=True)
    if args.out:
        text = json.dumps(out, indent=1, sort_keys=True) + "\n"
        (args.out / "hat-gap.json").write_text(text)


if __name__ == "__main__":
    main()
