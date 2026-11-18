"""Independent checker for targets.py's thresholds. Shares no code with the search.

Claim, for each target t and threshold s in the file: every multiset of s to `max` nonzero
cards (values 1 to 10, at most four of each) either makes t with +, - and x using every card
once, or is blocked by a remainder: for some modulus m from 2 to 12, no such expression leaves
the remainder of t on division by m. The file gives an expression for each multiset that makes
t; this program lists the multisets itself, checks each expression with `ast`, and decides the
rest by its own search over remainders, which repeatedly replaces two remainders by one.

    python3 erase/targets.py --certs     # writes erase/targets_certs.jsonl.gz
    python3 erase/verify_targets.py      # checks it
"""

import ast
import gzip
import json
import sys
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


def main(path):
    if not Path(path).exists():
        sys.exit(f"{path} not found; run targets.py --certs first")
    headers, certs = {}, {}
    with gzip.open(path, "rt") as f:
        for rec in map(json.loads, f):
            if "threshold" in rec:
                headers[rec["target"]] = (rec["threshold"], rec["max"])
            else:
                certs[rec["target"], tuple(rec["cards"])] = rec["expr"]
    for t, (s, top) in sorted(headers.items()):
        made = by_remainder = 0
        for n in range(s, top + 1):
            for m in multisets(range(1, 11), n):
                if max(Counter(m).values()) > 4:
                    continue
                if (t, m) in certs:
                    leaves = []
                    assert value(certs[t, m], leaves) == t, (t, m)
                    assert Counter(leaves) == Counter(m), (t, m)
                    made += 1
                else:
                    assert blocked(m, t), (t, m)
                    by_remainder += 1
        print(f"t={t:2}: sizes {s} to {top}, {made:,} expressions, {by_remainder:,} blocked")
        remainders.cache_clear()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "targets_certs.jsonl.gz")
