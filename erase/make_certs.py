"""Write certificates.jsonl.gz: one expression for each base case of the big-hands theorem
(Theorem 15 of the paper).

Cases: multisets of 7 or 8 nonzero cards (at most four of each value) outside both
families. Check the output with verify_certs.py.
"""

import gzip
import io
import json
from pathlib import Path

from erase import in_families, make, multisets

EXCEPTION = (1, 6, 6, 7, 7, 7, 7)
OUT = Path(__file__).parent / "certificates.jsonl.gz"


def main():
    # mtime=0 keeps the file byte-for-byte reproducible
    with io.TextIOWrapper(gzip.GzipFile(OUT, "wb", mtime=0)) as f:
        for n in (7, 8):
            count = 0
            for m in multisets(n, [0] + [4] * 10):
                if in_families(m):
                    continue
                e = make(m, 10)
                if e is None:  # the single exception makes 10 only with a sub-collection
                    if m != EXCEPTION:
                        raise ValueError(f"missing whole-hand witness: {m}")
                    rec = {"cards": m, "expr": "(1+(((7+(7+7))-6)-6))", "uses": "subset"}
                else:
                    rec = {"cards": m, "expr": e, "uses": "all"}
                f.write(json.dumps(rec) + "\n")
                count += 1
            print(n, count)


if __name__ == "__main__":
    main()
