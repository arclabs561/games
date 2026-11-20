"""Verdicts for a range of Microsoft deals with 0 to 4 free cells, by Solver A or Solver B.

For each deal the levels go up from 0 free cells and stop at the first win. A win with k
cells is a win with more (the same line is legal), and a deal lost with k cells is lost with
fewer (every position reachable with fewer cells is reachable with k, and so is every move
between them). So the first winning level is the deal's **least number of free cells**, and
it fixes the verdict at every level. Each winning line is replayed by `rules.replay`, with the
cells of its level and with four.

One JSON record per deal and level goes to `<out>/survey-<solver>-<start>-<stop>.jsonl`.
Deals can be split into slices for parallel jobs: `--slice k --slices n` takes the k-th of
n equal parts of the range.

    uv run python -m freecell.survey --solver b --start 1 --stop 1001 --out runs/
    uv run python -m freecell.survey --solver a --slice 3 --slices 40 --out OUT --workers 4
    uv run python -m freecell.survey --solver b --levels 4 --start 1 --stop 1000001 --out OUT
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

from freecell import native
from freecell.deals import ms_deal, name
from freecell.rules import Rules, replay
from freecell.solver import decide

LEVELS = (0, 1, 2, 3, 4)


def record(deal: int, cells: int, verdict: str, states: int, **extra) -> dict:
    """One line of the survey."""
    return {"deal": deal, "cells": cells, "verdict": verdict, "states": states, **extra}


def _won(deal: int, cells: int, states: int, line) -> dict:
    ok = replay(Rules(cells=cells), ms_deal(deal), line)
    ok4 = replay(Rules(cells=4), ms_deal(deal), line)
    return record(deal, cells, "won", states, moves=len(line), replay=ok and ok4)


def _lost(deal: int, cells: int, states: int, most_home: int, frozen) -> dict:
    names = [name(c) for c in sorted(frozen)]
    return record(deal, cells, "lost", states, most_home=most_home, frozen=names)


def survey_a(deal: int, cap: int | None = None, levels=LEVELS) -> list[dict]:
    """Solver A on one deal, level by level."""
    out = []
    for cells in levels:
        res = decide(Rules(cells=cells), ms_deal(deal), cap=cap)
        if res.verdict == "won":
            out.append(_won(deal, cells, res.states, res.line))
            break
        if res.verdict == "lost":
            out.append(_lost(deal, cells, res.states, res.most_home, res.frozen))
        else:
            out.append(record(deal, cells, "unknown", res.states))
            break
    return out


def _survey_a(args) -> list[dict]:
    return survey_a(*args)


def survey_b(deals: list[int], cap: int = 0, levels=LEVELS, autoplay: bool = False) -> list[dict]:
    """Solver B on many deals, one level at a time for all of them."""
    out = []
    todo = list(deals)
    for cells in levels:
        problems = [native.ms_problem(d, cells) for d in todo]
        answers = native.decide(problems, cap=cap, autoplay=autoplay)
        nxt = []
        for deal, ans in zip(todo, answers, strict=True):
            if ans.verdict == "won":
                out.append(_won(deal, cells, ans.states, ans.line))
            elif ans.verdict == "lost":
                out.append(_lost(deal, cells, ans.states, ans.most_home, ans.frozen))
                nxt.append(deal)
            else:
                out.append(record(deal, cells, "unknown", ans.states))
        todo = nxt
    return sorted(out, key=lambda r: (r["deal"], r["cells"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", choices="ab", default="b")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=32001, help="one past the last deal")
    ap.add_argument("--slice", type=int, help="which slice, from 0")
    ap.add_argument("--slices", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="processes for Solver A")
    ap.add_argument("--cap", type=int, default=0, help="most positions in the last search")
    ap.add_argument("--levels", default="01234", help="free cells to try, in order")
    ap.add_argument("--deals", help="comma-separated deal numbers, in place of a range")
    ap.add_argument("--autoplay", action="store_true", help="Solver B plays safe cards home")
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    deals = list(range(args.start, args.stop))
    if args.deals:
        deals = [int(d) for d in args.deals.split(",")]
    if args.slice is not None:
        size = -(-len(deals) // args.slices)
        deals = deals[args.slice * size : (args.slice + 1) * size]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = ("" if args.levels == "01234" else f"-cells{args.levels}") + (
        "-auto" if args.autoplay else ""
    )
    path = args.out / f"survey-{args.solver}{tag}-{deals[0]}-{deals[-1] + 1}.jsonl"
    t0 = time.time()
    cap = args.cap or None
    levels = tuple(int(x) for x in args.levels)
    with path.open("w") as f:
        if args.solver == "b":
            for i in range(0, len(deals), 500):
                chunk = deals[i : i + 500]
                for rec in survey_b(chunk, cap=args.cap, levels=levels, autoplay=args.autoplay):
                    f.write(json.dumps(rec) + "\n")
                f.flush()
        else:
            with Pool(args.workers) as pool:
                jobs = [(d, cap, levels) for d in deals]
                for recs in pool.imap(_survey_a, jobs, chunksize=4):
                    for rec in recs:
                        f.write(json.dumps(rec) + "\n")
                    f.flush()
    print(f"{len(deals)} deals in {time.time() - t0:.0f} s -> {path}")


if __name__ == "__main__":
    main()
