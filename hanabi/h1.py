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

from hanabi import dfs_solver
from hanabi.rules import Rules, seeded_deck
from hanabi.upper_bound import wilson


def load(folder: str) -> dict[tuple[int, int], dict]:
    """Every record under the folder, keyed by (players, seed). Files named `*rerun*`
    update Solver A's verdict of deals already present."""
    recs: dict[tuple[int, int], dict] = {}
    files = sorted(Path(folder).rglob("*.json")) + sorted(Path(folder).rglob("*.jsonl"))
    for f in [f for f in files if "rerun" not in f.name] + [f for f in files if "rerun" in f.name]:
        text = f.read_text()
        for r in (
            [json.loads(x) for x in text.splitlines()] if f.suffix == ".jsonl" else json.loads(text)
        ):
            key = (r["players"], r["seed"])
            if "rerun" in f.name:
                old = recs[key]
                assert old["verdict"] == "undecided", key
                for k in ("verdict", "certificates", "budget", "nodes", "seconds", "best", "extra"):
                    if k in r:
                        old[k] = r[k]
            else:
                assert key not in recs, key
                recs[key] = r
    return recs


def add_last_card(recs: dict) -> None:
    """Run the last-card test of `hanabi.dfs_solver` on every deal not found won."""
    for (p, seed), r in recs.items():
        if r["verdict"] != "won" and "last_card" not in r:
            rules = Rules(players=p)
            r["last_card"] = dfs_solver.last_card_test(rules, seeded_deck(rules, seed))


def first_proof(r: dict) -> str:
    """The certificate that first proves a lost deal: the README's, then the new ones."""
    for c in ("time", "storage", "pooled"):
        if c in r["certificates"]:
            return c
    if r.get("last_card"):
        return "last card"
    if "lost" in (r.get("free_order"), r.get("extra", {}).get("free_order")):
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
        both = [
            r
            for r in rows
            if r["verdict"] != "undecided" and r.get("verdict_b", "undecided") != "undecided"
        ]
        disagree = [r["seed"] for r in both if r["verdict"] != r["verdict_b"]]
        assert not disagree, f"solvers disagree on {p} players, seeds {disagree}"
        # settled by either solver
        settled = Counter(settled_one(r) for r in rows)
        # and with the attempts outside the plan
        ext = Counter(extended(r) for r in rows)
        lost = [r for r in rows if settled_one(r) == "lost"]
        # a deal lost by Solver B alone has no best score from Solver A; 24 bounds it
        best = Counter(r.get("best", 24) for r in lost)
        scores = [r.get("best", 24) if settled_one(r) == "lost" else 25 for r in rows]
        out[p] = {
            "deals": n,
            "solver_a": dict(va),
            "solver_b": dict(Counter(r.get("verdict_b", "not run") for r in rows)),
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
            "outside_the_plan": {
                "won": ext["won"],
                "lost": ext["lost"],
                "undecided": ext["undecided"],
                "undecided_seeds": [r["seed"] for r in rows if extended(r) == "undecided"],
                "share": share(ext["won"], ext["undecided"], n),
                "mean_best_score_at_most": mean_bound(
                    [r.get("best", 24) if extended(r) == "lost" else 25 for r in rows]
                ),
            },
            "largest_a_budget": dict(Counter(str(r["budget"]) for r in rows)),
            "seconds_a": round(sum(r["seconds"] for r in rows), 1),
            "seconds_b": round(sum(r.get("seconds_b", 0) for r in rows), 1),
        }
    return out


def settled_one(r: dict) -> str:
    """The verdict of the plan: Solver A's, or Solver B's where A has none."""
    return r["verdict"] if r["verdict"] != "undecided" else r.get("verdict_b", "undecided")


def extended(r: dict) -> str:
    """The verdict with the attempts outside the plan: Solver B with all its cuts, and the
    free-order test, which proves a loss but never a win."""
    v = settled_one(r)
    extra = r.get("extra", {})
    if v == "undecided":
        v = extra.get("b_all_cuts", "undecided")
    if v == "undecided" and (
        r.get("last_card") or "lost" in (r.get("free_order"), extra.get("free_order"))
    ):
        v = "lost"
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--label", default="confirmatory, plan H1")
    parser.add_argument(
        "--all-rows", action="store_true", help="a row for every deal, not only those not won"
    )
    args = parser.parse_args()
    recs = load(args.runs)
    add_last_card(recs)
    summary = {
        "label": args.label,
        "seeds 0 to 1,999": summarize(recs, range(2000)),
        f"seeds 2,000 to {args.deals - 1:,}": summarize(recs, range(2000, args.deals)),
        "all seeds": summarize(recs, range(args.deals)),
    }
    cols = ["players", "seed", "verdict", "verdict_b", "certificates", "best"]
    cols += ["free_order", "last_card", "extra"]
    keep = [r for _, r in sorted(recs.items()) if args.all_rows or r["verdict"] != "won"]
    rows = [[r.get(k) for k in cols] for r in keep]
    with open(args.out, "w") as f:
        # the summary indented, then one line per deal
        head = json.dumps({"summary": summary, "columns": cols}, indent=1, sort_keys=True)
        f.write(head[:-2] + ',\n "deals": [\n')
        f.write(",\n".join(json.dumps(row, separators=(",", ":")) for row in rows))
        f.write("\n ]\n}\n")
    print(json.dumps(summary["all seeds"], indent=1))


if __name__ == "__main__":
    main()
