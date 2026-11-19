"""Plan H1: merge the per-block records, check the two solvers agree, and summarize.

    uv run python -m hanabi.h1 --runs <folder> --out hanabi/results/h1.json

A block is a JSON list written by `python -m hanabi.upper_bound --json`. A record has the
verdict of Solver A (`hanabi.solver`), of Solver B (`hanabi.dfs_solver`), and, for deals
Solver A does not find won, the free-order test. Records from a later, larger budget for
the same deal (the separate 5,000,000-node jobs) replace Solver A's part of the earlier
record.

The summary gives, for each player count, the winnable share as a range (undecided deals
counted as lost, then as won) with a 95% Wilson interval around both ends, the agreement
of the solvers, the certificates of the lost deals, and the bound on the mean best score.
Any deal the two solvers decide differently stops the summary.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from hanabi.upper_bound import wilson


def load(folder: str) -> dict[tuple[int, int], dict]:
    """Every record under the folder, keyed by (players, seed). Files named `*rerun*`
    update Solver A's verdict of deals already present."""
    recs: dict[tuple[int, int], dict] = {}
    files = sorted(Path(folder).rglob("*.json"))
    for f in [f for f in files if "rerun" not in f.name] + [f for f in files if "rerun" in f.name]:
        for r in json.loads(f.read_text()):
            key = (r["players"], r["seed"])
            if "rerun" in f.name:
                old = recs[key]
                assert old["verdict"] == "undecided", key
                for k in ("verdict", "certificates", "budget", "nodes", "seconds", "best"):
                    if k in r:
                        old[k] = r[k]
                old["nodes"] = r["nodes"]
            else:
                assert key not in recs, key
                recs[key] = r
    return recs


def first_proof(r: dict) -> str:
    """The certificate that first proves a lost deal, in the README's order."""
    for c in ("time", "storage", "pooled"):
        if c in r["certificates"]:
            return c
    if r.get("free_order") == "lost":
        return "free-order"
    return "search"


def share(won: int, undecided: int, n: int) -> dict:
    lo, _ = wilson(won, n)
    _, hi = wilson(won + undecided, n)
    return {
        "low": round(won / n, 6),
        "high": round((won + undecided) / n, 6),
        "interval": [round(lo, 4), round(hi, 4)],
    }


def mean_bound(scores: list[int]) -> dict:
    n = len(scores)
    m = sum(scores) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in scores) / (n - 1))
    return {"mean": round(m, 4), "half_width": round(1.96 * sd / math.sqrt(n), 4)}


def summarize(recs: dict, seeds: range) -> dict:
    out = {}
    for p in sorted({p for p, _ in recs}):
        rows = [recs[p, s] for s in seeds if (p, s) in recs]
        n = len(rows)
        if not n:
            continue
        va = Counter(r["verdict"] for r in rows)
        both = [r for r in rows if r["verdict"] != "undecided" and r["verdict_b"] != "undecided"]
        disagree = [r["seed"] for r in both if r["verdict"] != r["verdict_b"]]
        assert not disagree, f"solvers disagree on {p} players, seeds {disagree}"
        # settled by either solver
        settled = Counter(
            r["verdict"] if r["verdict"] != "undecided" else r["verdict_b"] for r in rows
        )
        # and with the free-order test, which is not part of the plan
        with_free = Counter(
            "lost"
            if settled_one(r) == "undecided" and r.get("free_order") == "lost"
            else settled_one(r)
            for r in rows
        )
        lost = [r for r in rows if settled_one(r) == "lost"]
        # a deal lost by Solver B alone has no best score from Solver A; 24 bounds it
        best = Counter(r.get("best", 24) for r in lost)
        scores = [r.get("best", 24) if settled_one(r) == "lost" else 25 for r in rows]
        out[p] = {
            "deals": n,
            "solver_a": dict(va),
            "solver_b": dict(Counter(r["verdict_b"] for r in rows)),
            "both_decide": len(both),
            "agree": len(both) - len(disagree),
            "won": settled["won"],
            "lost": settled["lost"],
            "undecided": settled["undecided"],
            "undecided_seeds": [r["seed"] for r in rows if settled_one(r) == "undecided"],
            "share": share(settled["won"], settled["undecided"], n),
            "first_proof": dict(sorted(Counter(first_proof(r) for r in lost).items())),
            "free_order_on_lost": dict(
                sorted(Counter(r.get("free_order", "not run") for r in lost).items())
            ),
            "lost_without_older_certificate": [r["seed"] for r in lost if not r["certificates"]],
            "best_scores_of_lost": {str(k): v for k, v in sorted(best.items())},
            "mean_best_score_at_most": mean_bound(scores),
            "with_free_order": {
                "lost": with_free["lost"],
                "undecided": with_free["undecided"],
                "share": share(with_free["won"], with_free["undecided"], n),
            },
            "largest_a_budget": dict(Counter(str(r["budget"]) for r in rows)),
            "seconds_a": round(sum(r["seconds"] for r in rows), 1),
            "seconds_b": round(sum(r["seconds_b"] for r in rows), 1),
        }
    return out


def settled_one(r: dict) -> str:
    return r["verdict"] if r["verdict"] != "undecided" else r["verdict_b"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--label", default="confirmatory, plan H1")
    args = parser.parse_args()
    recs = load(args.runs)
    summary = {
        "label": args.label,
        "seeds 0 to 1,999": summarize(recs, range(2000)),
        f"seeds 2,000 to {args.deals - 1:,}": summarize(recs, range(2000, args.deals)),
        "all seeds": summarize(recs, range(args.deals)),
    }
    rows = [
        {
            k: r[k]
            for k in (
                "players",
                "seed",
                "verdict",
                "verdict_b",
                "certificates",
                "best",
                "free_order",
            )
            if k in r
        }
        for _, r in sorted(recs.items())
    ]
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "deals": rows}, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print(json.dumps(summary["all seeds"], indent=1))


if __name__ == "__main__":
    main()
