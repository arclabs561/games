"""The open-game ceiling with six colors: merge the records, summarize, compare.

    uv run python -m hanabi.six_suits --runs <folder> --out hanabi/results/six-suits.json

The records are those of `python -m hanabi.upper_bound --sample P N --colors 6 --solver both
--json OUT`, run on seeded deals 0 to 19,999 of `Rules(players=P, colors=6)` for two to five
players, with the budgets of plan H1, and the reruns at 5,000,000 nodes of the deals both
solvers left undecided (`--rerun`). The sixth color has the same ten cards as the others, so
a deck has 60 cards and the top score is 30. This is the purple suit of van Doorn's hat bot,
the "6 suits" column of its documentation. The summary is `hanabi.h1.summarize` with top
score 30; any deal the two solvers decide differently stops it.

This run was not planned in advance as H1 was: it is an exact computation per deal with no
cost and no learned model, and its seeds are new (the six-color deck differs from the
five-color one for every seed).
"""

from __future__ import annotations

import argparse
import json

from hanabi import h1

TOP = 30

# Published win rates with six suits, van Doorn's hat bot (github.com/fpvandoorn/hanabi,
# doc_hat_player.md): percent of games at 30, and points lost against 30.
HAT_BOT = {4: (93.9, 0.09), 5: (95.1, 0.07)}


def compare(summary: dict) -> dict:
    """The hat bot against the ceiling: its win rate must lie below the winnable share's
    upper end, and its mean below the bound on the mean best score."""
    out = {}
    for p, (wins, lost) in HAT_BOT.items():
        s = summary[p]["outside_the_plan"]
        share_high = s["share"]["interval"][1]
        bound = s["mean_best_score_at_most"]
        out[p] = {
            "hat bot win rate": round(wins / 100, 4),
            "hat bot mean": round(TOP - lost, 2),
            "winnable share, upper end of the interval": share_high,
            "mean best score at most": bound,
            "below the ceiling": wins / 100 <= share_high
            and TOP - lost <= bound["mean"] + bound["half_width"],
        }
    return out


def _share(x: float) -> str:
    """A share as the README writes it: 1, or five decimals without a trailing zero."""
    if x == 1:
        return "1"
    text = f"{x:.5f}"
    return text[:-1] if text.endswith("0") else text


def readme_rows(summary: dict) -> list[str]:
    """The rows of the README's six-color table, from the summary of `six-suits.json`."""
    out = []
    for p in sorted(summary, key=int):
        s = summary[p]["outside_the_plan"]
        lo, hi = s["share"]["low"], s["share"]["high"]
        share = _share(lo) if lo == hi else f"{_share(lo)} to {_share(hi)}"
        a, b = (_share(round(x, 4)) for x in s["share"]["interval"])
        m = s["mean_best_score_at_most"]
        out.append(
            f"| {p} | {s['lost']} | {s['undecided']} | {share} | {a} to {b} |"
            f" {m['mean']:.4f} ± {m['half_width']:.4f} |"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deals", type=int, default=20000)
    args = parser.parse_args()
    recs = h1.load(args.runs)
    assert all(r.get("colors") == 6 for r in recs.values())
    h1.add_last_card(recs)
    summary = h1.summarize(recs, range(args.deals), top=TOP)
    result = {
        "label": "six colors, seeds 0 to 19,999, H1's budgets and solvers",
        "all seeds": summary,
        "hat bot": compare(summary),
    }
    cols = ["players", "seed", "verdict", "verdict_b", "certificates", "best"]
    cols += ["free_order", "last_card", "extra"]
    keep = [r for _, r in sorted(recs.items()) if r["verdict"] != "won"]
    rows = [[r.get(k) for k in cols] for r in keep]
    with open(args.out, "w") as f:
        head = json.dumps({"summary": result, "columns": cols}, indent=1, sort_keys=True)
        f.write(head[:-2] + ',\n "deals": [\n')
        f.write(",\n".join(json.dumps(row, separators=(",", ":")) for row in rows))
        f.write("\n ]\n}\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
