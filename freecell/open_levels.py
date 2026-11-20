"""Runs on the eight deal-levels that the survey left undecided, one job per solver and level.

The survey's Solver B searches depth first and stores every position in memory, and its
table filled a 16 GB runner on eight levels. Two searches are run on them here:

- **Solver C** (`solver_c.c`) searches every reachable position, one layer of foundations at
  a time, so only one layer is held in memory. It decides a level: lost when no layer is
  left, won when every card reaches home. It gives no line of moves.
- **Solver B** with a larger budget and another seed for the order of its moves, to find a
  line that wins; every line is replayed by `rules.replay`.

A job is named `<solver>:<deal>:<cells>` or `b:<deal>:<cells>:<seed>`. Each writes
`<out>/open-<job with '-' for ':'>.json`; Solver C also writes the positions of each layer
to `<out>/layers-<deal>-<cells>.tsv` and its progress to `<out>/progress-<deal>-<cells>.log`.

    uv run python -m freecell.open_levels c:11982:4 --out runs/
    uv run python -m freecell.open_levels b:9499:1:3 --budget 600000000 --out runs/
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from freecell import native
from freecell.deals import ms_deal, name
from freecell.rules import Rules, replay

SOLVER_C = Path(__file__).with_name("solver_c.c")

# The undecided levels: (deal, free cells, the verdict in Pringle's list).
OPEN = [
    (9499, 1, "won"),
    (13167, 1, "won"),
    (13599, 1, "won"),
    (20410, 1, "lost"),
    (24966, 1, "lost"),
    (27708, 1, "lost"),
    (982, 2, "lost"),
    (17760, 2, "won"),
]


def solve_c(
    problems: list[str], folder: Path, cap: int = 0, layers: bool = False, log: Path | None = None
) -> list[dict]:
    """Solver C on each problem; with `layers`, each answer lists its layers' positions."""
    args = [str(native.binary(SOLVER_C)), str(folder), str(cap), str(1 << 30)]
    args += ["layers"] if layers else []
    env = dict(os.environ, SOLVER_C_PROGRESS="1") if log else None
    with open(log or os.devnull, "w") as err:
        proc = subprocess.run(
            args,
            input="".join(p + "\n" for p in problems),
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            env=env,
        )
    if proc.returncode:
        raise RuntimeError(f"Solver C failed with code {proc.returncode}")
    lines = iter(proc.stdout.splitlines())
    out = []
    for text in lines:
        w = text.split()
        ans: dict = {"verdict": w[0], "states": int(w[1])}
        if w[0] in ("won", "lost"):
            ans["most_home"] = int(w[2])
        if w[0] == "lost":
            ans["frozen"] = sorted(int(x) for x in w[3:])
        if w[0] not in ("won", "lost", "unknown"):
            raise RuntimeError("Solver C: " + text)
        if layers:
            ans["layers"] = []
            for row in lines:
                if row == "end":
                    break
                _, *nums = row.split()
                ans["layers"].append(tuple(map(int, nums)))
        out.append(ans)
    if len(out) != len(problems):
        raise RuntimeError("Solver C gave too few answers")
    return out


def run_job(job: str, out: Path, budget: int, cap: int) -> dict:
    """Run one job and write its record."""
    solver, deal, cells, *rest = job.split(":")
    deal_n, cells_n = int(deal), int(cells)
    t0 = time.time()
    rec: dict = {"job": job, "deal": deal_n, "cells": cells_n, "solver": solver}
    if solver == "c":
        log = out / f"progress-{deal}-{cells}.log"
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            (ans,) = solve_c([native.ms_problem(deal_n, cells_n)], Path(tmp), cap, True, log)
        layers = ans.pop("layers")
        rows = "".join("\t".join(map(str, r)) + "\n" for r in layers)
        (out / f"layers-{deal}-{cells}.tsv").write_text("home\tlayer positions\n" + rows)
        rec |= ans | {"layer_count": len(layers), "largest_layer": max(r[4] for r in layers)}
        if "frozen" in rec:
            rec["frozen"] = [name(c) for c in rec["frozen"]]
    else:
        seed = int(rest[0]) if rest else 0
        (ans,) = native.run(
            [native.ms_problem(deal_n, cells_n)], budget=budget, seed=seed, autoplay=True
        )
        rec |= {"seed": seed, "budget": budget, "verdict": ans.verdict, "states": ans.states}
        if ans.verdict == "won":
            layout = ms_deal(deal_n)
            rec["moves"] = len(ans.line)
            rec["replay"] = replay(Rules(cells=cells_n), layout, ans.line)
            rec["replay_four_cells"] = replay(Rules(cells=4), layout, ans.line)
            rec["line"] = [[c, d] for c, d in ans.line]
        if ans.verdict == "lost":
            rec |= {"most_home": ans.most_home, "frozen": [name(c) for c in sorted(ans.frozen)]}
    rec["seconds"] = round(time.time() - t0, 1)
    (out / f"open-{job.replace(':', '-')}.json").write_text(json.dumps(rec) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+", help="jobs such as c:982:2 or b:9499:1:3")
    ap.add_argument("--budget", type=int, default=0, help="Solver B: most positions stored")
    ap.add_argument("--cap", type=int, default=0, help="Solver C: most positions in a layer")
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for job in args.jobs:
        rec = run_job(job, args.out, args.budget, args.cap)
        print(json.dumps({k: v for k, v in rec.items() if k != "line"}), flush=True)


if __name__ == "__main__":
    main()
