"""Merge the survey runs, compare the two solvers with each other and with published results.

    uv run python -m freecell.report --a <folder of Solver A runs> --b <folder of Solver B runs>

reads every `survey-a-*.jsonl` and `survey-b-*.jsonl` below the folders, writes `freecell/results/f1.json` (the
summary) and `freecell/results/least_cells.tsv` (one line per deal: the least number of
free cells found by Solver A and by Solver B, '?' where a solver did not decide), and prints
the tables of the README.

The published per-deal list is `data/msfc32k-min-fc` from
github.com/shlomif/MicrosoftFreeCell32000-minimal-freecells (Theodore Pringle, 2016, MIT
license): the least number of free cells that wins each of the first 32,000 deals.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from freecell.certificates import blocked_at_start, static_stuck
from freecell.deals import ms_deal, name

HERE = Path(__file__).parent
DEALS = range(1, 32001)

# The 69 deals of the first 32,000 that the FreeCell FAQ (Keller) lists as winnable with
# zero free cells, from "a complete analysis of the 32,000 standard deals using four
# different solvers".
FAQ_ZERO_CELLS = [
    164, 892, 1012, 1081, 1150, 1529, 2508, 2514, 3178, 3225, 3250, 4929, 5055, 5152, 5213,
    5300, 5814, 5877, 5907, 6749, 6893, 7018, 7058, 7167, 7807, 8355, 8471, 8961, 9998, 10772,
    11863, 11987, 12392, 12411, 12676, 13214, 13464, 13532, 14014, 14624, 14826, 15140, 15196,
    17772, 17871, 18026, 18150, 18427, 19951, 20533, 21657, 21900, 22663, 23328, 24176, 24919,
    25001, 25904, 26719, 27121, 27853, 28856, 30329, 30418, 30584, 30755, 30849, 31185, 31316,
]  # fmt: skip
# The FAQ's deals below 1000 that need all four free cells.
FAQ_FOUR_BELOW_1000 = [169, 178, 285, 454, 575, 598, 617, 657, 775, 829, 988]
# Published counts of lost deals among the first 32,000, by free cells, and their sources.
PUBLISHED_LOST = {
    4: (1, "FreeCell FAQ (Keller): every deal but 11982 solved"),
    3: (199, "FreeCell FAQ (Keller)"),
    2: (6619, "Fish, fc-solve blog, 2 September 2012"),
    1: (25711, "FreeCell FAQ: at least 6,289 won, so at most 25,711 lost"),
    0: (31931, "FreeCell FAQ: 69 won with zero free cells"),
}


def published_least() -> dict[int, int]:
    """Least free cells per deal from Pringle's list."""
    out = {}
    for line in (HERE / "data" / "msfc32k-min-fc").read_text().splitlines():
        deal, cells = map(int, line.split())
        out[deal] = cells
    return out


def load(folders: list[Path] | None, solver: str) -> list[dict]:
    """Every record of one solver ('a' or 'b') below the folders, each deal and level once.

    A later run of the same deal and level replaces an earlier one only if the earlier one
    was left undecided. Records from runs of chosen levels only (`--levels`) are marked
    `partial`: they give verdicts at those levels but not the least number of free cells."""
    out: dict[tuple[int, int], dict] = {}
    for folder in folders or []:
        for p in sorted(folder.rglob(f"survey-{solver}-*.jsonl")):
            for x in p.open():
                r = json.loads(x)
                r["autoplay"] = "-auto" in p.name
                r["partial"] = "-cells" in p.name
                k = r["deal"], r["cells"]
                if k not in out or out[k]["verdict"] == "unknown":
                    out[k] = r
    return [out[k] for k in sorted(out)]


def least(records: list[dict]) -> dict[int, int | None]:
    """Least winning number of free cells per deal: 5 for a deal lost with 0 to 4, None if
    a level was left open before a win."""
    out: dict[int, int | None] = {}
    for r in sorted(records, key=lambda r: (r["deal"], r["cells"])):
        if r["partial"]:
            continue
        if r["verdict"] == "won":
            out.setdefault(r["deal"], r["cells"])
        elif r["verdict"] == "unknown":
            out.setdefault(r["deal"], None)
        elif r["cells"] == 4:
            out.setdefault(r["deal"], 5)
    return out


def by_level(records: list[dict], deals=DEALS) -> dict[int, dict[str, int]]:
    """For each number of free cells: deals won, lost and left undecided. A deal won with
    fewer free cells is won; a deal left undecided with fewer is undecided."""
    won_at: dict[int, int] = {}
    lost = {(r["deal"], r["cells"]) for r in records if r["verdict"] == "lost"}
    for r in records:
        if r["verdict"] == "won":
            won_at[r["deal"]] = min(r["cells"], won_at.get(r["deal"], 9))
    out = {}
    for k in range(5):
        row = Counter(
            "won" if won_at.get(d, 9) <= k else "lost" if (d, k) in lost else "undecided"
            for d in deals
        )
        out[k] = {v: row[v] for v in ("won", "lost", "undecided")}
    return out


FENCE_BANDS = (100, 10_000, 1_000_000)


def quantiles(xs: list[int]) -> dict:
    """Median, 90th percentile and largest."""
    xs = sorted(xs)
    if not xs:
        return {}
    return {
        "median": statistics.median(xs),
        "p90": xs[int(0.9 * (len(xs) - 1))],
        "max": xs[-1],
    }


def certificates(records: list[dict], least_cells: dict) -> dict:
    """For each level, how many lost deals each kind of reason explains."""
    out = {}
    lost_by_level: dict[int, list[dict]] = {c: [] for c in range(5)}
    for r in records:
        if r["verdict"] == "lost":
            lost_by_level[r["cells"]].append(r)
    for cells, rows in lost_by_level.items():
        frozen = [r for r in rows if r["frozen"]]
        sizes = [r["states"] for r in rows]
        row = {
            "lost": len(rows),
            "fence at most": {n: sum(1 for x in sizes if x <= n) for n in FENCE_BANDS},
            "fences counted with autoplay": sum(1 for r in rows if r.get("autoplay")),
            "fence size": quantiles([r["states"] for r in rows]),
            "largest fence": max(rows, key=lambda r: r["states"])["deal"] if rows else None,
            "with a frozen card": len(frozen),
            "without a frozen card": sorted(r["deal"] for r in rows if not r["frozen"]),
            "most cards home": dict(sorted(Counter(r["most_home"] for r in rows).items())),
            "frozen cards": quantiles([len(r["frozen"]) for r in frozen]),
        }
        if cells == 0:
            blocked = [r["deal"] for r in rows if blocked_at_start(ms_deal(r["deal"]), 0)]
            stuck = {r["deal"]: static_stuck(ms_deal(r["deal"])) for r in rows}
            names = {r["deal"]: set(r["frozen"]) for r in rows}
            row["blocked at the start"] = len(blocked)
            row["first blocked"] = blocked[:6]
            row["with stuck cards"] = sum(1 for v in stuck.values() if v)
            row["stuck or blocked"] = len({d for d, v in stuck.items() if v} | set(blocked))
            row["stuck cards that the solver moves"] = sorted(
                d for d, v in stuck.items() if not {name(c) for c in v} <= names[d]
            )
            row["stuck cards per deal"] = quantiles([len(v) for v in stuck.values() if v])
        row["without a frozen card"] = row["without a frozen card"][:50]
        out[cells] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", type=Path, nargs="+", help="folders of Solver A runs")
    ap.add_argument("--b", type=Path, nargs="+", help="folders of Solver B runs")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    pub = published_least()
    rec_a, rec_b = load(args.a, "a"), load(args.b, "b")
    la, lb = least(rec_a), least(rec_b)
    summary: dict = {"published": {}, "solvers": {}}
    for label, recs, lc in (("A", rec_a, la), ("B", rec_b, lb)):
        decided = {d: c for d, c in lc.items() if c is not None}
        summary["solvers"][label] = {
            "deals": len(lc),
            "decided": len(decided),
            "undecided": sorted(d for d, c in lc.items() if c is None),
            "least cells": dict(sorted(Counter(decided.values()).items())),
            "by level": by_level(recs),
            "differ from the published list": sorted(d for d, c in decided.items() if pub[d] != c),
            "lines that fail replay": sorted(
                r["deal"] for r in recs if r["verdict"] == "won" and not r["replay"]
            ),
            "positions searched": sum(r["states"] for r in recs),
        }
    both = [d for d in DEALS if la.get(d) is not None and lb.get(d) is not None]
    fa = {(r["deal"], r["cells"]): r["frozen"] for r in rec_a if r["verdict"] == "lost"}
    fb = {(r["deal"], r["cells"]): r["frozen"] for r in rec_b if r["verdict"] == "lost"}
    summary["A and B"] = {
        "deals both decide": len(both),
        "disagree": sorted(d for d in both if la[d] != lb[d]),
        "lost levels both decide": len(fa.keys() & fb.keys()),
        "frozen cards differ": sorted(k for k in fa.keys() & fb.keys() if fa[k] != fb[k]),
    }
    summary["either solver"] = by_level(rec_a + rec_b)
    summary["published"] = {
        "least cells": dict(sorted(Counter(pub.values()).items())),
        "lost by level": {k: sum(1 for c in pub.values() if c > k) for k in range(5)},
        "counts": {k: {"lost": v[0], "source": v[1]} for k, v in PUBLISHED_LOST.items()},
    }
    best = lb
    summary["FAQ checks (Solver B)"] = {
        "zero-cell wins equal the FAQ's 69": sorted(d for d, c in best.items() if c == 0)
        == FAQ_ZERO_CELLS,
        "four-cell deals below 1000 equal the FAQ's 11": sorted(
            d for d, c in best.items() if c == 4 and d < 1000
        )
        == FAQ_FOUR_BELOW_1000,
    }
    summary["certificates (Solver B)"] = certificates(rec_b, lb)
    if rec_a:
        summary["certificates (Solver A)"] = {
            c: {k: v for k, v in row.items() if k in ("lost", "fence size", "with a frozen card")}
            for c, row in certificates(rec_a, la).items()
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "f1.json").write_text(json.dumps(summary, indent=1) + "\n")
    with (args.out / "least_cells.tsv").open("w") as f:
        f.write("deal\tsolver_a\tsolver_b\tpublished\n")
        for d in DEALS:
            a, b = la.get(d), lb.get(d)
            f.write(f"{d}\t{'?' if a is None else a}\t{'?' if b is None else b}\t{pub[d]}\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
