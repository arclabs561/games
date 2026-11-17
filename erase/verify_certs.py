"""Independent checker for the big-hands theorem (Theorem 15 of the paper). Shares no code with the search.

Theorem. Let M be a multiset of 7 to 11 nonzero cards (values 1..10, at most four of
each), in neither the 3-family (all multiples of 3) nor the 4-family (no 2, 6 or 10,
at most one odd card). Then M makes 10 using every card, except M = {1,6,6,7,7,7,7},
which makes 10 with a sub-collection.

    python3 erase/make_certs.py      # writes erase/certificates.jsonl.gz
    python3 erase/verify_certs.py    # checks it
"""

import ast
import gzip
import json
import sys
from collections import Counter
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


def in_families(m):
    three = all(v % 3 == 0 for v in m)
    four = not {2, 6, 10} & set(m) and sum(v % 2 for v in m) <= 1
    return three or four


def check(m, cert):
    leaves = []
    assert value(cert["expr"], leaves) == 10, m
    if cert["uses"] == "all":
        assert Counter(leaves) == Counter(m), m
    else:  # the one exception
        assert m == (1, 6, 6, 7, 7, 7, 7)
        assert not Counter(leaves) - Counter(m)


def main(path):
    if not Path(path).exists():
        sys.exit(f"{path} not found; run make_certs.py first")
    with gzip.open(path, "rt") as f:
        certs = {tuple(c["cards"]): c for c in map(json.loads, f)}
    checked = 0
    for n in range(7, 12):
        for m in multisets(range(1, 11), n):
            if max(Counter(m).values()) > 4 or in_families(m):
                continue
            check(m, certs[m])  # a missing case raises KeyError
            checked += 1
    print(f"all {checked:,} cases certified")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "certificates.jsonl.gz")
