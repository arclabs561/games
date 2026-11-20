"""Runs on the eight deal-levels that the survey left undecided, one job per solver and level.

The survey's Solver B searches depth first and stores every position in memory, and its
table filled a 16 GB runner on eight levels. Two searches are run on them here:

- **Solver C** (`solver_c.c`) searches every reachable position, one layer of foundations at
  a time, so only one layer is held in memory. It decides a level: lost when no layer is
  left, won when every card reaches home. It gives no line of moves.
- **Solver B by layers** (`solver_b.c` with "layers"): the same plan of one foundation at a
  time, written in Solver B's own code, with its own positions, moves and safe-card rule. On
  a lost level it must count the same positions as Solver C, layer by layer.
- **Solver C hunting a win** ("win="): the layer with most cards home first, and a line of
  moves rebuilt at the end; every line is replayed by `rules.replay`.
- **Solver B** with a larger budget and another seed for the order of its moves, to find a
  line that wins.

Solvers `ch` and `blh` are Solver C and Solver B by layers with Horne's rule for safe cards
(a card of rank r goes home when the other color's foundations are at r - 2 or more and the
other foundation of its color at r - 3 or more), the rule of the solvers in Fish's reports,
so that their counts of positions can be compared with his.

A job is named `<solver>-<deal>-<cells>`, with solver c, bl, ch, blh or w, or
`b-<deal>-<cells>-<seed>`
(':' also separates). Each writes `<out>/open-<job>.json`; the layered searches also write
the positions of each layer to `<out>/layers-<solver>-<deal>-<cells>.tsv`, and Solver C its
progress to `<out>/progress-<solver>-<deal>-<cells>.log`.

    uv run python -m freecell.open_levels c-11982-4 bl-11982-4 --out runs/
    uv run python -m freecell.open_levels w-164-0 --budget 50000000 --out runs/
    uv run python -m freecell.open_levels b-9499-1-3 --budget 600000000 --out runs/
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
    problems: list[str],
    folder: Path,
    cap: int = 0,
    layers: bool = False,
    log: Path | None = None,
    memory: int = 1 << 30,
    horne: bool = False,
) -> list[dict]:
    """Solver C on each problem; with `layers`, each answer lists its layers' positions.
    Waiting positions go to files in `folder` when they take more than `memory` bytes.
    With `horne`, cards go home by Horne's rule."""
    args = [str(native.binary(SOLVER_C)), str(folder), str(cap), str(memory)]
    args += (["layers"] if layers else []) + (["horne"] if horne else [])
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


def solve_b_layers(problems: list[str], horne: bool = False) -> list[dict]:
    """Solver B, one foundation at a time, on each problem (Horne's rule with `horne`)."""
    proc = subprocess.run(
        [str(native.binary()), "0", "0", "layers", *(["horne"] if horne else [])],
        input="".join(p + "\n" for p in problems),
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(f"Solver B failed: {proc.stderr.strip()}")
    lines = iter(proc.stdout.splitlines())
    out = []
    for text in lines:
        w = text.split()
        ans: dict = {"verdict": w[0], "states": int(w[1]), "layers": []}
        if w[0] == "lost":
            ans |= {"most_home": int(w[2]), "frozen": sorted(int(x) for x in w[3:])}
        for row in lines:
            if row == "end":
                break
            ans["layers"].append(tuple(map(int, row.split()[1:])))
        out.append(ans)
    return out


def hunt_c(problem: str, folder: Path, budget: int, log: Path | None = None) -> native.Answer:
    """Solver C's search for a win, as a Solver B answer (won, unknown or exhausted)."""
    args = [str(native.binary(SOLVER_C)), str(folder), "0", str(1 << 30), f"win={budget}"]
    env = dict(os.environ, SOLVER_C_PROGRESS="1") if log else None
    with open(log or os.devnull, "w") as err:
        proc = subprocess.run(
            args, input=problem + "\n", stdout=subprocess.PIPE, stderr=err, text=True, env=env
        )
    if proc.returncode:
        raise RuntimeError(f"Solver C failed with code {proc.returncode}")
    words = proc.stdout.split()
    if words[0] == "exhausted":
        return native.Answer("exhausted", int(words[1]))
    return native._parse(words)


def run_job(job: str, out: Path, budget: int, cap: int) -> dict:
    """Run one job and write its record."""
    solver, deal, cells, *rest = job.replace(":", "-").split("-")
    deal_n, cells_n = int(deal), int(cells)
    tag = f"{solver}-{deal}-{cells}"
    t0 = time.time()
    rec: dict = {"job": job, "deal": deal_n, "cells": cells_n, "solver": solver}
    if solver in ("c", "bl", "ch", "blh"):
        problem = native.ms_problem(deal_n, cells_n)
        horne = solver.endswith("h")
        if solver.startswith("c"):
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                log = out / f"progress-{tag}.log"
                (ans,) = solve_c([problem], Path(tmp), cap, True, log, horne=horne)
        else:
            (ans,) = solve_b_layers([problem], horne)
        layers = ans.pop("layers")
        rows = "".join("\t".join(map(str, r)) + "\n" for r in layers)
        (out / f"layers-{tag}.tsv").write_text("C\tD\tH\tS\tpositions\n" + rows)
        rec |= ans | {"layer_count": len(layers), "largest_layer": max(r[4] for r in layers)}
        if "frozen" in rec:
            rec["frozen"] = [name(c) for c in rec["frozen"]]
    elif solver == "w":
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            ans = hunt_c(
                native.ms_problem(deal_n, cells_n), Path(tmp), budget, out / f"progress-{tag}.log"
            )
        rec |= {"budget": budget, "verdict": ans.verdict, "states": ans.states}
        if ans.verdict == "won":
            layout = ms_deal(deal_n)
            rec["moves"] = len(ans.line)
            rec["replay"] = replay(Rules(cells=cells_n), layout, ans.line)
            rec["replay_four_cells"] = replay(Rules(cells=4), layout, ans.line)
            rec["line"] = [[c, d] for c, d in ans.line]
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
    ap.add_argument(
        "jobs", nargs="+", help="jobs such as c-982-2, bl-982-2, w-9499-1 or b-9499-1-3"
    )
    ap.add_argument(
        "--budget", type=int, default=0, help="Solver B and the hunt: most positions stored"
    )
    ap.add_argument("--cap", type=int, default=0, help="Solver C: most positions in a layer")
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for job in args.jobs:
        rec = run_job(job, args.out, args.budget, args.cap)
        print(json.dumps({k: v for k, v in rec.items() if k != "line"}), flush=True)


if __name__ == "__main__":
    main()
