"""Independent checker for targets.py's thresholds. Shares no code with the search.

For each published target t and threshold s in results.json: every multiset of s to `max` nonzero
cards (values 1 to 10, at most four of each) either makes t with +, - and x using every card
once, or is blocked by a remainder: for some modulus m from 2 to 12, no such expression leaves
the remainder of t on division by m. The file gives an expression for each multiset that makes
t; this program lists the multisets itself, checks each expression with `ast`, and decides the
rest by its own search over remainders, which repeatedly replaces two remainders by one.

    python3 erase/targets.py --max 11 --certs     # writes erase/targets_certs.jsonl.gz
    python3 erase/verify_targets.py      # checks all 30 targets through 11 cards
    python3 erase/verify_targets.py small.gz --target 7 --max 7  # check a smaller run
"""

import argparse
import ast
import gzip
import json
from collections import Counter
from functools import cache
from itertools import combinations
from itertools import combinations_with_replacement as multisets
from pathlib import Path

OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b}


def value(expr, leaves):
    def ev(n):
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant) and type(n.value) is int and 1 <= n.value <= 10:
            leaves.append(n.value)
            return n.value
        raise ValueError("illegal expression")

    return ev(ast.parse(expr, mode="eval").body)


@cache
def remainders(state, m):
    """Remainders mod m left when the remainders in `state` are combined into one."""
    if len(state) == 1:
        return frozenset(state)
    out = set()
    for i, j in combinations(range(len(state)), 2):
        rest = state[:i] + state[i + 1 : j] + state[j + 1 :]
        x, y = state[i], state[j]
        for v in {(x + y) % m, (x - y) % m, (y - x) % m, (x * y) % m}:
            out |= remainders(tuple(sorted(rest + (v,))), m)
    return frozenset(out)


def blocked(cards, t):
    return any(
        t % m not in remainders(tuple(sorted(c % m for c in cards)), m) for m in range(2, 13)
    )


@cache
def exact_values(cards):
    """Values of whole expressions. Enumerate their last operation across a bipartition."""
    if len(cards) == 1:
        return frozenset(cards)
    out = set()
    for n in range(1, len(cards) // 2 + 1):
        for left in set(combinations(cards, n)):
            right = list(cards)
            for card in left:
                right.remove(card)
            for a in exact_values(left):
                for b in exact_values(tuple(right)):
                    out.update((a + b, a - b, b - a, a * b))
    return frozenset(out)


def check_target(t, row, top, certs):
    s = row["threshold"]
    if s > 1:
        cards = tuple(row["example"] or ())
        if (
            len(cards) != s - 1
            or not all(type(c) is int and 1 <= c <= 10 for c in cards)
            or max(Counter(cards).values()) > 4
            or blocked(cards, t)
            or t in exact_values(tuple(sorted(cards)))
        ):
            raise ValueError(f"target {t}: invalid witness below threshold {s}")
        exact_values.cache_clear()

    made = by_remainder = 0
    for n in range(s, top + 1):
        for m in multisets(range(1, 11), n):
            if max(Counter(m).values()) > 4:
                continue
            if m in certs:
                leaves = []
                if value(certs.pop(m), leaves) != t or Counter(leaves) != Counter(m):
                    raise ValueError(f"target {t}: wrong expression for {m}")
                made += 1
            else:
                if not blocked(m, t):
                    raise ValueError(f"target {t}: missing expression for {m}")
                by_remainder += 1
    if certs:
        raise ValueError(f"target {t}: unexpected multiset {next(iter(certs))}")
    print(f"t={t:2}: sizes {s} to {top}, {made:,} expressions, {by_remainder:,} blocked")
    remainders.cache_clear()


def main(path, *, targets=range(1, 31), max_cards=11):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run targets.py --max 11 --certs first")
    published = json.loads(Path(__file__).with_name("results.json").read_text())["targets"]
    expected = set(targets)
    if (
        not expected
        or not expected <= set(range(1, 31))
        or not 1 <= max_cards <= published["max_cards"]
    ):
        raise ValueError("select targets 1 to 30 and a maximum within the published study")
    rows = {t: published["by_target"][str(t)] for t in expected}
    seen = set()
    certs = {}
    current = None
    with gzip.open(path, "rt") as f:
        for rec in map(json.loads, f):
            t = rec["target"]
            if "threshold" in rec:
                if t not in rows or t in seen:
                    raise ValueError(f"unexpected or duplicate target {t}")
                s = rows[t]["threshold"]
                if rec["threshold"] != s or rec["max"] != max_cards or not 1 <= s <= max_cards:
                    raise ValueError(f"target {t}: expected sizes {s} to {max_cards}")
                if current is not None:
                    check_target(current, rows[current], max_cards, certs)
                seen.add(t)
                current = t
            else:
                if current is None or t != current:
                    raise ValueError(f"expression outside its target {t} section")
                cards = tuple(rec["cards"])
                if cards in certs:
                    raise ValueError(f"target {t}: duplicate expression for {cards}")
                certs[cards] = rec["expr"]
    if current is None or seen != expected:
        raise ValueError(f"missing targets: {sorted(expected - seen)}")
    check_target(current, rows[current], max_cards, certs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path(__file__).with_name("targets_certs.jsonl.gz")
    )
    parser.add_argument(
        "--target", type=int, action="append", help="target for a partial check (repeatable)"
    )
    parser.add_argument("--max", type=int, default=11, help="largest hand size to check")
    args = parser.parse_args()
    main(args.path, targets=args.target or range(1, 31), max_cards=args.max)
