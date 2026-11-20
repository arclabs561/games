"""How hard is a set to solve? The size of the search tree of a fixed, natural solver.

The solver is `solver.recur`. It puts a nut in the center, then fills the six outer places
one after another around the center, starting at the side the center's first number faces.
Each outer nut is turned to face the center with the right number, so it fits only if its side
touching the previous nut matches. Before placing the first outer nut and each later one, it
also checks the side that will touch the next nut: that side cannot show the number the next
nut turns toward the center, since every nut shows each number once (a **look-ahead**). It
tries every center and every nut, and so finds every solution.

The **search size** of a set is the number of calls of `recur` in a complete search, counting
the first: one per partial placement that passes the checks, dead ends included. It does not
depend on the order in which nuts are tried. It does depend on the side the search starts
from, so the measure used here adds up the six complete searches, one from each side of the
center. That sum does not change when the numbers are renamed, the puzzle is turned, or the
nuts are read in the other direction, so it is the same for every set of a class.

A second measure drops the look-ahead (`look_ahead=False`): a plain search that checks only
the sides of nuts already placed. Both are reported, with whether they agree.

To compare the original with a uniformly random set that has one solution, the sets are drawn
through their solution: a uniformly random ring around C0 = (1, 2, 3, 4, 5, 6) with seven
distinct nuts, kept when its set has exactly one solution. A one-solution set has one ring, so
this is uniform over the one-solution sets whose solution has C0 in the center, and renaming
gives every center the same distribution of search sizes.

The exact pass visits every one-solution set with the C program `difficulty.c`, a second
implementation of the same search.

    uv run python -m drive_ya_nuts.difficulty sample --samples 20000 --seed 1 --out DIR
    uv run python -m drive_ya_nuts.difficulty exact --part 0 --parts 8 --out DIR
    uv run python -m drive_ya_nuts.difficulty report --runs DIR     # writes difficulty.json

Sampling 20,000 sets takes a few minutes; the exact pass about half an hour of CPU in all.
Both run on GitHub Actions, ten seeds and eight parts, and `report` combines them.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from drive_ya_nuts.exact import candidates, edge_sequences
from drive_ya_nuts.solver import nuts as original_nuts

Order = tuple[int, ...]
HERE = Path(__file__).parent
C0: Order = (1, 2, 3, 4, 5, 6)
EDGES = list(edge_sequences(6))


def neighbors(order: Order) -> dict[int, tuple[int, int]]:
    """For each number on the nut, the numbers before and after it (as `Nut.left`, `Nut.right`)."""
    return {v: (order[i - 1], order[(i + 1) % len(order)]) for i, v in enumerate(order)}


def search(orders: list[Order], look_ahead: bool = True) -> tuple[int, int]:
    """(calls of `recur`, solutions) for one complete search, centers read as given."""
    near = [neighbors(o) for o in orders]
    n = len(orders)
    calls = 1  # the first call, with nothing placed
    solutions = 0
    for c in range(n):
        calls += 1  # the center placed
        cl = near[c]
        faces = orders[c]

        def place(used: int, k: int, prev: int | None, first: int, faces=faces, cl=cl) -> None:
            nonlocal calls, solutions
            middle = faces[k - 1]
            left, right = cl[middle]
            for j in range(n):
                if used >> j & 1:
                    continue
                a, b = near[j][middle]  # the nut's numbers beside the one facing the center
                if look_ahead and a == right:
                    continue
                if prev is None:
                    if look_ahead and b == left:
                        continue
                elif b != prev:
                    continue
                calls += 1
                if k == n - 1:  # the ring is full: the last nut must match the first
                    solutions += a == near[first][faces[0]][1]
                else:
                    place(used | 1 << j, k + 1, a, j if k == 1 else first)

        place(1 << c, 1, None, -1)
    return calls, solutions


def turned(orders: list[Order], s: int) -> list[Order]:
    """Every nut's order turned by s, so that each search starts s sides further on."""
    return [tuple(o[s:] + o[:s]) for o in orders]


def size(orders: list[Order], look_ahead: bool = True) -> int:
    """The search size: calls of complete searches from each of the six sides, added up."""
    return sum(search(turned(orders, s), look_ahead)[0] for s in range(len(orders[0])))


def solutions(orders: list[Order]) -> int:
    return search(orders)[1]


def random_one_solution_set(rng: random.Random) -> list[Order]:
    """A uniformly random set with one solution (in the sense of the docstring)."""
    while True:
        e = rng.choice(EDGES)
        outer = [rng.choice(candidates(6, i, e[i - 1], e[i])) for i in range(6)]
        orders = [C0, *outer]
        if len(set(orders)) == 7 and solutions(orders) == 1:
            return orders


def original() -> list[Order]:
    return sorted(n.canonical_order() for n in original_nuts)


def sample(samples: int, seed: int) -> dict:
    """Search sizes of seeded random one-solution sets, against the original's."""
    rng = random.Random(seed)
    mine = {k: size(original(), k) for k in (True, False)}
    below = {k: 0 for k in (True, False)}
    equal = {k: 0 for k in (True, False)}
    hist: dict[bool, Counter] = {k: Counter() for k in (True, False)}
    for _ in range(samples):
        orders = random_one_solution_set(rng)
        for k in (True, False):
            t = size(orders, k)
            hist[k][t] += 1
            below[k] += t < mine[k]
            equal[k] += t == mine[k]
    names = {True: "look_ahead", False: "plain"}
    return {
        "samples": samples,
        "seed": seed,
        **{
            names[k]: {
                "original": mine[k],
                "below": below[k],
                "equal": equal[k],
                "sizes": sorted(hist[k].items()),
            }
            for k in (True, False)
        },
    }


def binary() -> Path:
    """The compiled `difficulty.c`, built once per version of the source."""
    source = HERE / "difficulty.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"drive-ya-nuts-difficulty-{digest}"
    if not out.exists():
        cc = compiler()
        if cc is None:
            raise RuntimeError("difficulty.c needs a C compiler (cc)")
        tmp = out.with_suffix(f".{os.getpid()}")
        subprocess.run([cc, "-O2", "-o", str(tmp), str(source)], check=True)
        tmp.replace(out)
    return out


def compiler() -> str | None:
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")


def c_sizes(orders: list[Order]) -> tuple[int, int, int]:
    """(search size with look-ahead, plain search size, solutions) from the C program."""
    args = ["".join(map(str, o)) for o in orders]
    out = subprocess.run(
        [str(binary()), "sizes", *args], capture_output=True, text=True, check=True
    )
    a, b, c = map(int, out.stdout.split())
    return a, b, c


def c_all(part: int, parts: int) -> dict:
    """The C program's counts and histograms over one part of the edge sequences."""
    out = subprocess.run(
        [str(binary()), "all", str(part), str(parts)], capture_output=True, text=True, check=True
    )
    data: dict = {"part": part, "parts": parts}
    for line in out.stdout.splitlines():
        key, *rest = line.split()
        if key in ("look_ahead", "plain"):
            data[key] = sorted((int(t), int(c)) for t, c in (x.split(":") for x in rest))
        elif key == "solutions":
            data[key] = [int(x) for x in rest]
        else:
            data[key] = int(rest[0])
    return data


def percentile(original: int, sizes: list[tuple[int, int]]) -> dict:
    """How many sizes fall below, at and above the original's."""
    below = sum(c for t, c in sizes if t < original)
    equal = sum(c for t, c in sizes if t == original)
    total = sum(c for _, c in sizes)
    return {"below": below, "equal": equal, "total": total}


def merge(hists: list[list]) -> list[tuple[int, int]]:
    total: Counter = Counter()
    for h in hists:
        for t, c in h:
            total[t] += c
    return sorted(total.items())


def report(runs: Path) -> dict:
    """Combine the Actions outputs (one file per seed or part) into difficulty.json."""
    samples = [json.loads(p.read_text()) for p in sorted(runs.rglob("difficulty-sample-*.json"))]
    parts = [json.loads(p.read_text()) for p in sorted(runs.rglob("difficulty-exact-*.json"))]
    assert sorted(p["part"] for p in parts) == list(range(parts[0]["parts"]))
    out: dict = {"exact": {}, "sampled": {}}
    for key in ("edge_sequences", "rings", "distinct", "one"):
        out["exact"][key] = sum(p[key] for p in parts)
    out["exact"]["solutions"] = [sum(c) for c in zip(*(p["solutions"] for p in parts), strict=True)]
    out["sampled"]["seeds"] = sorted(s["seed"] for s in samples)
    out["sampled"]["samples"] = sum(s["samples"] for s in samples)
    for k in ("look_ahead", "plain"):
        mine = size(original(), k == "look_ahead")
        exact = merge([p[k] for p in parts])
        sampled = merge([s[k]["sizes"] for s in samples])
        assert all(s[k]["original"] == mine for s in samples)
        out["exact"][k] = {"original": mine, **percentile(mine, exact), "sizes": exact}
        out["sampled"][k] = {"original": mine, **percentile(mine, sampled), "sizes": sampled}
    return out


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["sample", "exact", "report"])
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runs", type=Path, help="for report: the downloaded outputs")
    args = parser.parse_args(argv)
    start = time.time()
    if args.what == "report":
        data = report(args.runs)
        (HERE / "difficulty.json").write_text(json.dumps(data, indent=1) + "\n")
        for k in ("look_ahead", "plain"):
            print(k, {x: data["exact"][k][x] for x in ("original", "below", "equal", "total")})
        return
    if args.what == "sample":
        data, name = sample(args.samples, args.seed), f"difficulty-sample-{args.seed}.json"
    else:
        data, name = c_all(args.part, args.parts), f"difficulty-exact-{args.part}.json"
    data["seconds"] = round(time.time() - start)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / name).write_text(json.dumps(data) + "\n")
    print({k: v for k, v in data.items() if k not in ("look_ahead", "plain")})


if __name__ == "__main__":
    main(sys.argv[1:])
