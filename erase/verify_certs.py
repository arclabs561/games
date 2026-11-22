"""Independent checker for the big-hands theorem (Theorem 15 of the paper). Shares no code with the search.

Theorem. The base cases use 7 or 8 nonzero cards (values 1..10, at most four of
each), in neither the 3-family (all multiples of 3) nor the 4-family (no 2, 6 or 10,
at most one odd card). Such a multiset M makes 10 using every card, except M = {1,6,6,7,7,7,7},
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
    if value(cert["expr"], leaves) != 10:
        raise ValueError(f"expression does not make ten: {m}")
    used, available = Counter(leaves), Counter(m)
    if cert["uses"] == "all" and used == available:
        return
    if cert["uses"] == "subset" and m == (1, 6, 6, 7, 7, 7, 7) and used < available:
        return
    raise ValueError(f"wrong cards or use mode: {m}")


def main(path):
    # Older archives have sizes 9–11 as well. Check every supplied witness and
    # require complete coverage of every supplied layer in addition to both bases.
    seen = set()
    sizes = {7, 8}
    with gzip.open(path, "rt") as f:
        for cert in map(json.loads, f):
            m = tuple(cert["cards"])
            if (
                not 7 <= len(m) <= 11
                or any(type(v) is not int or not 1 <= v <= 10 for v in m)
                or m != tuple(sorted(m))
                or max(Counter(m).values()) > 4
                or in_families(m)
            ):
                raise ValueError(f"unexpected case: {m}")
            if m in seen:
                raise ValueError(f"duplicate case: {m}")
            check(m, cert)
            seen.add(m)
            sizes.add(len(m))
    for n in sorted(sizes):
        for m in multisets(range(1, 11), n):
            if max(Counter(m).values()) <= 4 and not in_families(m) and m not in seen:
                raise ValueError(f"missing certificate: {m}")
    print(f"all {len(seen):,} cases certified")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "certificates.jsonl.gz")
