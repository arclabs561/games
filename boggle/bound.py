"""The best 3 by 3 board, by branch and bound over classes of boards (`bound.c`).

`bound.c` splits the boards into classes, bounds the best score in each class from above,
and splits a class further until its bound falls below a threshold or it is a single board.
Every board that reaches the threshold is found, up to rotation and reflection. The search
starts from Vanderkam's four groups of letters, and ran in 20 slices on GitHub Actions:

    uv run python -m boggle.bound run --shard 0 --shards 20 --out runs/   # one slice
    uv run python -m boggle.bound summary runs/*/bound-*.txt              # writes results/
    uv run python -m boggle.bound check   # the search agrees with scoring every board

The checks: on 3 by 3 boards over a few letters, and on all 2 by 2 boards, the boards the
search finds are exactly those that scoring every board finds (`anneal.c exhaust`, a separate
program); and the bounds hold on random small classes, against the best board of each.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

from boggle import anneal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "bound.c"
RESULTS = HERE / "results"
LISTS = {"enable1": HERE / "enable1.txt", "enable2k": HERE / "enable2k.txt"}
GROUPS = "lnrsy aeiou chkmpt bdfgjqvwxz"  # Vanderkam's four groups, with q added to the last
THRESHOLD = 500


def binary() -> Path:
    """The compiled search, built once per version of the source."""
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"boggle-bound-{digest}"
    if not out.exists():
        cc = anneal.compiler()
        if cc is None:
            raise RuntimeError("bound.c needs a C compiler (cc)")
        tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}")
        subprocess.run([cc, "-O3", "-o", str(tmp), str(SOURCE)], check=True)
        tmp.replace(out)
    return out


def _run(*args: str) -> str:
    return subprocess.run([str(binary()), *args], capture_output=True, text=True, check=True).stdout


def c_bounds(words: Path, rows: int, cols: int, cells: Sequence[str]) -> tuple[int, int]:
    """The max and union bounds of a class, each cell given as a string of its letters."""
    mx, un = map(int, _run("bound", str(words), str(rows), str(cols), *cells).split())
    return mx, un


def c_score(words: Path, rows: int, cols: int, letters: str) -> tuple[int, int]:
    """Points and number of words of one board ('q' is Qu)."""
    pts, count = map(int, _run("score", str(words), str(rows), str(cols), letters).split())
    return pts, count


def solve(words: Path, rows: int, cols: int, threshold: int, groups: str = GROUPS,
          shard: int = 0, shards: int = 1) -> str:  # fmt: skip
    """The raw output of one slice of the search."""
    return _run(
        "solve", str(words), str(rows), str(cols), str(threshold), groups, str(shard), str(shards)
    )


def parse(text: str) -> tuple[dict[str, int], dict[str, int]]:
    """The boards found (letters to points) and the counts line of a search's output."""
    boards, counts = {}, {}
    for line in text.splitlines():
        if line.startswith("board "):
            _, letters, pts = line.split()
            boards[letters] = int(pts)
        elif line.startswith("classes "):
            parts = line.split()
            counts = {parts[i]: int(parts[i + 1]) for i in range(0, len(parts), 2)}
    return boards, counts


def canonical(letters: str, rows: int, cols: int) -> str:
    """The least of the board's images under the symmetries of the grid."""
    grid = [letters[i * cols : (i + 1) * cols] for i in range(rows)]
    images = [grid, [r[::-1] for r in grid], grid[::-1], [r[::-1] for r in grid[::-1]]]
    if rows == cols:
        t = ["".join(r[i] for r in grid) for i in range(cols)]
        images += [t, [r[::-1] for r in t], t[::-1], [r[::-1] for r in t[::-1]]]
    return min("".join(g) for g in images)


def exhaust(words: Path, n: int, alphabet: str, threshold: int) -> dict[str, int]:
    """Every n by n board over the alphabet that reaches the threshold, up to symmetry, by
    scoring each board with anneal.c."""
    out = subprocess.run(
        [str(anneal.binary()), "exhaust", str(words), str(n), alphabet, str(threshold)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return {canonical(b, n, n): p for b, p in parse(out)[0].items()}


def run(
    shard: int, shards: int, out: Path, groups: str = GROUPS, lists: Sequence[str] = tuple(LISTS)
) -> None:
    """One slice of the search for each word list, written to out/bound-<list>-<shard>.txt."""
    out.mkdir(parents=True, exist_ok=True)
    for name in lists:
        text = solve(LISTS[name], 3, 3, THRESHOLD, groups, shard, shards)
        (out / f"bound-{name}-{shard:02d}.txt").write_text(text)


def summary(paths: list[Path]) -> dict:
    """The boards of 500 points or more for each word list, up to symmetry, best first, with
    the search's counts summed over the slices; each board is rescored by anneal.c."""
    result: dict = {"groups": GROUPS, "threshold": THRESHOLD}
    for name, words in LISTS.items():
        files = sorted(p for p in paths if re.fullmatch(rf"bound-{name}-\d+\.txt", p.name))
        boards: dict[str, int] = {}
        totals: dict[str, int] = {}
        for p in files:
            found, counts = parse(p.read_text())
            boards |= {canonical(b, 3, 3): s for b, s in found.items()}
            for k, v in counts.items():
                k = {"boards": "boards scored", "found": "boards found, with repeats"}.get(k, k)
                totals[k] = totals.get(k, 0) + v
        top = sorted(boards.items(), key=lambda kv: (-kv[1], kv[0]))
        for b, s in top:
            assert anneal.c_score(b, words)[0] == s, b
        result[name] = {
            "slices": len(files),
            **totals,
            "boards found": len(top),
            "best": {
                "board": top[0][0],
                "points": top[0][1],
                "words": anneal.c_score(top[0][0], words)[1],
            },
            "boards": [{"board": b, "points": s} for b, s in top],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--shards", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--groups", default=GROUPS, help="letter groups, separated by commas")
    r.add_argument("--lists", nargs="+", default=list(LISTS), choices=list(LISTS))
    s = sub.add_parser("summary")
    s.add_argument("paths", type=Path, nargs="+")
    sub.add_parser("check")
    args = parser.parse_args()
    if args.command == "run":
        run(args.shard, args.shards, args.out, args.groups.replace(",", " "), args.lists)
    elif args.command == "summary":
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / "best-3x3.json"
        out.write_text(json.dumps(summary(args.paths), indent=1) + "\n")
        print("wrote", out)
    else:
        words = LISTS["enable1"]
        found = {
            canonical(b, 3, 3): p for b, p in parse(solve(words, 3, 3, 300, "ae lrst"))[0].items()
        }
        assert found == exhaust(words, 3, "aelrst", 300), "3 by 3 over aelrst"
        print(f"3 by 3 over aelrst: {len(found)} boards of 300 points or more, found both ways")


if __name__ == "__main__":
    main()
