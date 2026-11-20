"""Write results.json: every number that the README, its shared fragments in docs/ and the
paper quote, each computed here by the scripts that own it.

    python3 erase/report.py                      # every section; over an hour, mostly targets
    python3 erase/report.py deals countdown      # only these sections; the rest are kept
    python3 erase/report.py targets --targets 14 # only these targets of the targets section

Sections and their cost on one core: deals and no_zero (about 35 s and 5 s, count.py),
wild and certificates (a few seconds), twenty_four (about 10 s), countdown (about 3 min),
simulated (about 4 min, simulate.py with its fixed seeds) and targets (about an hour for
targets 1 to 30 and up to 11 cards). The tests recompute the cheap sections in full and the
others in part.
"""

import argparse
import json
from math import comb
from pathlib import Path

import count
import countdown
import simulate
import targets
import twenty_four

from erase import in_families, multisets

RESULTS = Path(__file__).parent / "results.json"
TARGETS = range(1, 31)
MAX_CARDS = 11


def one_in(stuck: int, total: int) -> int:
    """'1 in k', with k rounded to a whole number."""
    return round(total / stuck)


def deals(zero: bool = True) -> dict:
    stuck, total = count.stuck(zero)
    return {"stuck": stuck, "total": total, "one_in": one_in(stuck, total)}


def wild(stuck: int) -> dict:
    """With four wild kings, only deals without a king can be stuck (proved in the paper), and
    those are the ordinary deals, so the stuck count is unchanged and the total grows."""
    total = comb(48, 7) * comb(41, 4)
    return {"stuck": stuck, "total": total, "one_in": one_in(stuck, total)}


def certificates() -> dict:
    """The cases of the big-hands theorem: multisets of 7 to 11 nonzero cards, at most four of
    each value, outside both families. make_certs.py writes one expression for each."""
    by_size = {
        n: sum(not in_families(m) for m in multisets(n, [0] + [4] * 10)) for n in range(7, 12)
    }
    return {"cases": sum(by_size.values()), "by_size": by_size}


def twenty_four_counts() -> dict:
    hands = {rule: twenty_four.solvable(rule) for rule in twenty_four.RULES}
    fraction = hands["fraction"]
    p = {rule: float(twenty_four.deck_probability(h)) for rule, h in hands.items()}
    return {
        "hands": len(twenty_four.HANDS),
        "make": {rule: len(h) for rule, h in hands.items()},
        "deck_chance": {rule: round(x, 4) for rule, x in p.items()},
        "up_to_9": [sum(max(h) <= 9 for h in fraction), comb(9 + 3, 4)],
        "up_to_10": [sum(max(h) <= 10 for h in fraction), comb(10 + 3, 4)],
        "need_fraction": sorted(set(fraction) - set(hands["integer"])),
    }


def countdown_counts() -> dict:
    r = countdown.survey()
    problems = r["selections"] * len(countdown.TARGETS)
    exact = r["near_range"][0]
    return {
        "selections": r["selections"],
        "targets": len(countdown.TARGETS),
        "problems": problems,
        "exact": exact,
        "exact_percent": round(100 * exact / problems, 2),
        "near_range": [r["near_range"][d] for d in range(1, 5)],
        "near_any_1": r["near_any"][1],
        "reach_every_target": len(r["hits_all"]),
        "reach_none": [list(s) for s in r["hits_none"]],
    }


def simulated() -> dict:
    return {"seeds": [1, 2, 3, 4, 5], **simulate.run()}


def targets_section(old: dict | None, only: list[int] | None) -> dict:
    by = dict(old["by_target"]) if old else {}
    for t in only or TARGETS:
        by[str(t)] = targets.summary(t, MAX_CARDS)
        print(f"  target {t}: threshold {by[str(t)]['threshold']}", flush=True)
    by = {str(t): by[str(t)] for t in TARGETS if str(t) in by}
    thresholds = [x["threshold"] for x in by.values()]
    return {
        "max_cards": MAX_CARDS,
        "threshold_counts": {str(s): thresholds.count(s) for s in sorted(set(thresholds))},
        "by_target": by,
    }


SECTIONS = {
    "deals": lambda: deals(True),
    "no_zero": lambda: deals(False),
    "wild": None,  # from the deals section
    "certificates": certificates,
    "twenty_four": twenty_four_counts,
    "countdown": countdown_counts,
    "simulated": simulated,
}
ORDER = [*SECTIONS, "targets"]


def dump(data: dict) -> str:
    """Sections one per block and targets one per line, so that a change is a short diff."""
    out = []
    for k in ORDER:
        if k not in data:
            continue
        v = data[k]
        if k == "targets":
            rows = ",\n".join(f'   "{t}": {json.dumps(x)}' for t, x in v["by_target"].items())
            body = (
                f'  "max_cards": {v["max_cards"]},\n'
                f'  "threshold_counts": {json.dumps(v["threshold_counts"])},\n'
                f'  "by_target": {{\n{rows}\n  }}'
            )
            out.append(f' "targets": {{\n{body}\n }}')
        else:
            out.append(f" {json.dumps(k)}: {json.dumps(v)}")
    return "{\n" + ",\n".join(out) + "\n}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sections", nargs="*", help=f"any of {', '.join(ORDER)}; default all")
    ap.add_argument("--targets", type=int, nargs="+", help="recompute only these targets")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args()
    if unknown := set(a.sections) - set(ORDER):
        ap.error(f"unknown sections {sorted(unknown)}")
    data = json.loads(a.out.read_text()) if a.out.exists() else {}
    for name in a.sections or ORDER:
        print(f"{name} ...", flush=True)
        if name == "targets":
            data[name] = targets_section(data.get("targets"), a.targets)
        elif name == "wild":
            data[name] = wild(data["deals"]["stuck"])
        else:
            data[name] = SECTIONS[name]()
    a.out.write_text(dump(data))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
