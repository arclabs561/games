"""Write the numerical results used by the Erase paper and README.

    python3 erase/report.py                      # every section; over an hour, mostly targets
    python3 erase/report.py deals countdown      # only these sections; the rest are kept
    python3 erase/report.py targets --targets 14 # only these targets of the targets section

Sections and their cost on one core: deals and no_zero (about 35 s and 5 s, count.py),
wild, certificates, ten_free and pairs (a few seconds), addition (about 4 min) and alone (about 10 s),
twenty_four (about 10 s), countdown (about 3 min), simulated (about 4 min, simulate.py with
its fixed seeds), plus_minus (instant, from simulated) and targets (about an hour for
targets 1 to 30 and up to 11 cards). The tests recompute the cheap sections in full and the
others in part. exact_play and variants exhaust the rule variants. whole_deck uses
a C++20 compiler and cross-checks the published base rows before extending to 40 cards.
"""

import argparse
import json
from collections import Counter
from math import comb, exp, lgamma, log, prod
from pathlib import Path

import count
import countdown
import deck_study
import exact_play
import exact_variants
import simulate
import targets
import twenty_four

from erase import in_families, make, multisets

RESULTS = Path(__file__).parent / "results.json"
TARGETS = range(1, 31)
MAX_CARDS = 11


def one_in(stuck: int, total: int) -> int:
    """'1 in k', with k rounded to a whole number."""
    return round(total / stuck)


def deals(zero: bool = True) -> dict:
    stuck, total = count.stuck(zero)
    return {"stuck": stuck, "total": total, "one_in": one_in(stuck, total)}


def wild(stuck: int, stuck_no_zero: int) -> dict:
    """With four wild kings, only deals without a king can be stuck (proved in the paper), and
    those are the ordinary deals, so the stuck count is unchanged and the total grows. The
    same holds for the deck of 1 to 10 with four kings."""
    total = comb(48, 7) * comb(41, 4)
    total_no_zero = comb(44, 7) * comb(37, 4)
    return {
        "stuck": stuck,
        "total": total,
        "one_in": one_in(stuck, total),
        "no_zero_one_in": one_in(stuck_no_zero, total_no_zero),
    }


def certificates() -> dict:
    """The cases of the big-hands theorem: base multisets of 7 or 8 nonzero cards, at most four of
    each value, outside both families. make_certs.py writes one expression for each."""
    by_size = {n: sum(not in_families(m) for m in multisets(n, [0] + [4] * 10)) for n in (7, 8)}
    return {"cases": sum(by_size.values()), "by_size": by_size}


def addition() -> dict:
    """Deals that addition alone cannot clear, for both decks (count.stuck_addition)."""
    out = {}
    for name, zero in (("zero", True), ("no_zero", False)):
        stuck, total = count.stuck_addition(zero)
        out[name] = {"stuck": stuck, "total": total, "percent": round(100 * stuck / total, 2)}
    return out


def ten_free(max_cards: int = 7) -> dict:
    """Nonzero multisets (at most four of each value) in which no part of two or more cards
    makes 10, counted by size outside both families, and the six-card ones listed. Built
    upward: a multiset is ten-free when it does not make 10 and every multiset one card
    smaller is ten-free."""
    level, by_size, six = {()}, [], []
    for n in range(1, max_cards + 1):
        grown = set()
        for m in level:
            for v in range(m[-1] if m else 1, 11):
                new = m + (v,)
                if m.count(v) == 4 or (n >= 2 and make(new, 10)):
                    continue
                if all(new[:i] + new[i + 1 :] in level for i in range(n)):
                    grown.add(new)
        level = grown
        outside = sorted(m for m in level if not in_families(m))
        by_size.append(len(outside))
        if n == 6:
            six = [list(m) for m in outside]
    return {"by_size": by_size, "six": six}


def pairs() -> dict:
    """Deals that the printed rules (pairs that sum to 10) cannot clear (count.stuck_pairs);
    the last entry lets a 10 be erased by itself, with cards 1 to 10."""
    out = {}
    for name, zero, lone in (
        ("zero", True, False),
        ("no_zero", False, False),
        ("no_zero_lone_ten", False, True),
    ):
        stuck, total = count.stuck_pairs(zero, lone)
        out[name] = {"stuck": stuck, "total": total, "percent": round(100 * stuck / total, 2)}
    return out


def alone() -> dict:
    """Seven-card hands that cannot be cleared with no center cards (the original version)."""
    stuck, total = count.stuck_alone()
    return {"stuck": stuck, "total": total, "one_in": one_in(stuck, total)}


def parity_blocked(zero: bool = True) -> tuple[int, int]:
    """(deals with an odd number of odd hand cards and no odd center card, all deals)."""
    odd, even = 20, 24 if zero else 20
    blocked = sum(
        comb(odd, k) * comb(even, 7 - k) * comb(even - (7 - k), 4) for k in range(1, 8, 2)
    )
    return blocked, comb(odd + even, 7) * comb(odd + even - 7, 4)


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in logs."""
    terms = [
        lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p)
        for i in range(k + 1)
    ]
    top = max(terms)
    return exp(top) * sum(exp(t - top) for t in terms)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """The exact (Clopper-Pearson) interval for a binomial share k of n, by bisection."""

    def solve(f, want):  # f increases in p
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < want else (lo, mid)
        return (lo + hi) / 2

    a = (1 - level) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - binomial_tail(k - 1, n, p), a)
    high = 1.0 if k == n else solve(lambda p: 1 - binomial_tail(k, n, p), 1 - a)
    return low, high


def plus_minus(sim: dict) -> dict:
    """Deals stuck with + and - only: the parity share is exact, and only the deals that
    parity does not block are estimated, from the simulated deals without the obstruction."""
    out = {}
    for name, zero in (("zero", True), ("no_zero", False)):
        x = sim[name]
        blocked, total = parity_blocked(zero)
        p = blocked / total
        sampled, other = x["deals"] - x["parity"], x["plus_minus"] - x["parity"]
        low, high = clopper_pearson(other, sampled)
        out[name] = {
            "parity": blocked,
            "total": total,
            "parity_percent": round(100 * p, 3),
            "sampled": sampled,
            "other": other,
            "percent": round(100 * (p + (1 - p) * other / sampled), 2),
            "low": round(100 * (p + (1 - p) * low), 2),
            "high": round(100 * (p + (1 - p) * high), 2),
        }
    return out


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


def targets_summary(by: dict) -> dict:
    """The targets section, from one summary per target (targets.summary)."""
    by = {str(t): by[str(t)] for t in TARGETS if str(t) in by}
    thresholds = [x["threshold"] for x in by.values()]
    return {
        "max_cards": MAX_CARDS,
        "thresholds": sorted(set(thresholds)),
        "threshold_counts": {str(s): thresholds.count(s) for s in sorted(set(thresholds))},
        "by_target": by,
    }


def targets_section(data: dict, only: list[int] | None = None) -> dict:
    by = dict(data["targets"]["by_target"]) if "targets" in data else {}
    for t in only or TARGETS:
        by[str(t)] = targets.summary(t, MAX_CARDS)
        print(f"  target {t}: threshold {by[str(t)]['threshold']}", flush=True)
    return targets_summary(by)


def hand_features() -> dict:
    """Physical seven-card hands and the closed-form any-center family count."""
    total = comb(44, 7)
    equal = consecutive = 0
    for hand in multisets(7, [4] * 11):
        counts = Counter(hand)
        ways = prod(comb(4, n) for n in counts.values())
        equal += ways * (len(counts) < 7)
        consecutive += ways * any(v + 1 in counts for v in counts)
    # Choose the distinct eleven cards first, then which seven belong to the hand.
    partitions = comb(11, 7)
    return {
        "total": total,
        "zero": total - comb(40, 7),
        "equal": equal,
        "consecutive": consecutive,
        "any_center": {
            "zero": (comb(16, 11) + comb(12, 11) + 20 * comb(12, 10)) * partitions,
            "no_zero": comb(12, 11) * partitions,
        },
    }


def whole_deck(data: dict) -> dict:
    # A standalone run with a fresh --out still compares with the printed table.
    published = data.get("targets") or json.loads(RESULTS.read_text())["targets"]
    if set(published["by_target"]) != {str(t) for t in TARGETS}:
        raise ValueError("whole_deck requires the complete 30-target comparison table")
    result = deck_study.study(published["by_target"])
    example = tuple(published["by_target"]["14"]["example"])
    values = exact_play.values(example)
    if 14 in values:
        raise ValueError("the published exception for target 14 makes 14")
    result["example_14_values"] = len(values)
    return result


# Each section is computed from the results so far (wild reuses the two counts of deals).
SECTIONS = {
    "deals": lambda data: deals(True),
    "no_zero": lambda data: deals(False),
    "wild": lambda data: wild(data["deals"]["stuck"], data["no_zero"]["stuck"]),
    "certificates": lambda data: certificates(),
    "ten_free": lambda data: ten_free(),
    "pairs": lambda data: pairs(),
    "addition": lambda data: addition(),
    "alone": lambda data: alone(),
    "twenty_four": lambda data: twenty_four_counts(),
    "countdown": lambda data: countdown_counts(),
    "simulated": lambda data: simulated(),
    "plus_minus": lambda data: plus_minus(data["simulated"]),
    "targets": targets_section,
    "exact_play": lambda data: exact_play.survey(),
    "variants": lambda data: exact_variants.survey(),
    "make_ten": lambda data: exact_variants.make_ten(),
    "hand_features": lambda data: hand_features(),
    "whole_deck": whole_deck,
}
ORDER = list(SECTIONS)


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
                f'  "thresholds": {json.dumps(v["thresholds"])},\n'
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
            data[name] = targets_section(data, a.targets)
        else:
            data[name] = SECTIONS[name](data)
    a.out.write_text(dump(data))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
