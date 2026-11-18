"""Is 10 special? For each target t, the hand size beyond which only remainders block.

A multiset of nonzero cards (values 1 to 10, at most four of each) **makes t** if one
expression with +, - and x that uses every card once has the value t. It is **blocked by a
remainder** if for some modulus m, no such expression has a value that leaves the same
remainder as t on division by m; then it cannot make t. For each size, this script lists the
multisets that neither make t nor are blocked by a remainder (the **exceptions**). The
**threshold** of t is one more than the largest size with an exception, among the sizes
checked.

For t = 10 the two families of the paper are exactly the multisets blocked by 3 and by 4, and
the only exception of 7 to 11 cards is {1, 6, 6, 7, 7, 7, 7} (Theorem 15).

Remainders are tried for moduli 2 to 12 (`MODULI`). A modulus larger than every value the
cards can make would block trivially, so the range has to stop somewhere; the exceptions of the
largest size are rechecked against the moduli 13 to 60, and any that one of those blocks is
printed. With `--certs`, the expressions for every multiset from the
threshold up are written to targets_certs.jsonl.gz, for verify_targets.py to check.

    python3 erase/targets.py                  # targets 1 to 30, sizes 1 to 10
    python3 erase/targets.py 10 --max 11      # one target, sizes 1 to 11
"""

import argparse
import gzip
import io
import json
import time
from collections import Counter
from functools import cache
from itertools import combinations
from pathlib import Path

from erase import make, multisets

MODULI = range(2, 13)
OUT = Path(__file__).parent / "targets_certs.jsonl.gz"


@cache
def residues(rs: tuple[int, ...], m: int) -> int:
    """The remainders mod m of the values of expressions using every card of rs once
    (cards given by their sorted remainders), as a bit mask."""
    if len(rs) == 1:
        return 1 << rs[0]
    out = 0
    n = len(rs)
    seen = set()
    for k in range(1, n // 2 + 1):
        for idx in combinations(range(n), k):
            a = tuple(rs[i] for i in idx)
            if a in seen:
                continue
            seen.add(a)
            b = tuple(rs[i] for i in range(n) if i not in idx)
            ra, rb = residues(a, m), residues(b, m)
            xs = [x for x in range(m) if ra >> x & 1]
            ys = [y for y in range(m) if rb >> y & 1]
            for x in xs:
                for y in ys:
                    out |= 1 << (x + y) % m | 1 << (x - y) % m | 1 << (y - x) % m
                    out |= 1 << (x * y) % m
    return out


def blocking_moduli(cards: tuple[int, ...], t: int, moduli=MODULI) -> list[int]:
    """The moduli m for which no value of the cards leaves the remainder of t mod m."""
    return [
        m for m in moduli if not residues(tuple(sorted(c % m for c in cards)), m) >> (t % m) & 1
    ]


def study(t: int, max_size: int, certs=None) -> dict:
    """Exceptions by size; how many multisets of each size make t; and how often each
    modulus is the least that blocks, by size."""
    exceptions: dict[int, list[tuple[int, ...]]] = {}
    least: dict[int, Counter[int]] = {}
    made: dict[int, int] = {}
    for n in range(1, max_size + 1):
        exceptions[n] = []
        made[n] = 0
        least[n] = Counter()
        for cards in multisets(n, [0] + [4] * 10):
            quick = blocking_moduli(cards, t, range(2, 7))  # cheap, and spares a long search
            e = None if quick else make(cards, t)
            if e is not None:
                made[n] += 1
                if certs is not None:
                    certs.append((n, cards, e))
            elif blocked := quick or blocking_moduli(cards, t):
                least[n][blocked[0]] += 1
            else:
                exceptions[n].append(cards)
    return {"exceptions": exceptions, "least_moduli": least, "made": made}


def threshold(exceptions: dict[int, list]) -> int:
    """One more than the largest size with an exception."""
    return 1 + max((n for n, ex in exceptions.items() if ex), default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", type=int, nargs="*", default=list(range(1, 31)))
    ap.add_argument("--max", type=int, default=10, help="largest number of cards")
    ap.add_argument("--certs", type=Path, nargs="?", const=OUT, help="write certificates")
    a = ap.parse_args()
    out = io.TextIOWrapper(gzip.GzipFile(a.certs, "wb", mtime=0)) if a.certs else None
    for t in a.targets:
        start = time.process_time()
        certs: list | None = [] if out else None
        r = study(t, a.max, certs)
        s = threshold(r["exceptions"])
        last = r["exceptions"].get(s - 1, [])
        wider = {c: blocking_moduli(c, t, range(13, 61)) for c in last}
        wider = {c: ms for c, ms in wider.items() if ms}
        counts = [len(r["exceptions"][n]) for n in range(1, a.max + 1)]
        beyond = Counter()
        for n in range(s, a.max + 1):
            beyond += r["least_moduli"][n]
        moduli = dict(sorted(beyond.items()))
        made = sum(r["made"][n] for n in range(s, a.max + 1))
        print(
            f"t={t:2}  threshold {s:2}  exceptions by size {counts}  "
            f"from size {s} on: {made:,} make t, least blocking moduli {moduli}  "
            f"last {last[:3]}{' ...' if len(last) > 3 else ''}"
            f"{f'  blocked by larger moduli: {wider}' if wider else ''}"
            f"  ({time.process_time() - start:.0f} s)",
            flush=True,
        )
        if out and certs is not None:
            out.write(json.dumps({"target": t, "threshold": s, "max": a.max}) + "\n")
            for n, c, e in certs:
                if n >= s:
                    out.write(json.dumps({"target": t, "cards": c, "expr": e}) + "\n")
        make.cache_clear()
    if out:
        out.close()


if __name__ == "__main__":
    main()
