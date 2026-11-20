"""Search for high-scoring 5 by 5 boards with `anneal.c`, and summarize the runs.

The C program scores a board as `board.score` does, with the same trie walk and the same
points, about a thousand times faster; `check` compares the two. A search is simulated
annealing from a random board followed by a full climb (every one-square change to any of
the 26 faces, and every swap of two squares, until none raises the score), so each run ends
on a board that no single change or swap improves. The settings were fixed before the main
runs from two pilot runs with seed 900, which the main runs do not use:

    STEPS = 300,000 changes, temperature from 300 down to 5 points, geometrically.

    uv run python -m boggle.anneal run --seed 1 --runs 25 --procs 4 --out runs/
    uv run python -m boggle.anneal summary runs/*/anneal-*.jsonl   # writes results/
    uv run python -m boggle.anneal check
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from boggle.board import get_words, make_trie, orientations, parse_board, points, wilson

HERE = Path(__file__).parent
SOURCE = HERE / "anneal.c"
WORDS = HERE / "enable1.txt"
RESULTS = HERE / "results"
STEPS, T0, T1 = 300_000, 300.0, 5.0


def compiler() -> str | None:
    """The C compiler, if there is one."""
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")


def compiler_missing() -> bool:
    """True when no C compiler is found (the tests that need one are skipped)."""
    return compiler() is None


def binary() -> Path:
    """The compiled annealer, built once per version of the source."""
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"boggle-anneal-{digest}"
    if not out.exists():
        cc = compiler()
        if cc is None:
            raise RuntimeError("anneal.c needs a C compiler (cc)")
        tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}")
        subprocess.run([cc, "-O3", "-o", str(tmp), str(SOURCE), "-lm"], check=True)
        tmp.replace(out)
    return out


def c_score(letters: str, words: Path = WORDS) -> tuple[int, int]:
    """Points and number of words of a square board, by the C program ('q' is Qu)."""
    n = round(len(letters) ** 0.5)
    out = subprocess.run(
        [str(binary()), "score", str(words), str(n), letters],
        capture_output=True,
        text=True,
        check=True,
    )
    pts, count = map(int, out.stdout.split())
    return pts, count


def c_climb(letters: str, words: Path = WORDS) -> tuple[int, int, str]:
    """Climb from a board by single changes and swaps; points, words and the final board."""
    n = round(len(letters) ** 0.5)
    out = subprocess.run(
        [str(binary()), "climb", str(words), str(n), letters],
        capture_output=True,
        text=True,
        check=True,
    )
    pts, count, board = out.stdout.split()
    return int(pts), int(count), board


def py_score(letters: str, trie) -> tuple[int, int]:
    """Points and number of words by board.get_words."""
    found = set(get_words(parse_board(letters), trie))
    return sum(map(points, found)), len(found)


def anneal(seed: int, runs: int, steps: int = STEPS, words: Path = WORDS) -> list[dict]:
    """`runs` searches with one seed; one record per search."""
    out = subprocess.run(
        [
            str(binary()),
            "anneal",
            str(words),
            "5",
            str(seed),
            str(runs),
            str(steps),
            str(T0),
            str(T1),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    recs = []
    for line in out.stdout.splitlines():
        _, s, i, pts, count, board = line.split()
        recs.append(
            {"seed": int(s), "run": int(i), "points": int(pts), "words": int(count), "board": board}
        )
    return recs


def canonical(letters: str) -> str:
    """The board up to rotation and reflection: the least of its eight orientations."""
    n = round(len(letters) ** 0.5)
    board = [list(letters[i : i + n]) for i in range(0, n * n, n)]
    return min("".join("".join(row) for row in b) for b in [board, *orientations(board)])


def summary(path: Path = RESULTS / "anneal-5x5.jsonl") -> dict:
    """The boards the searches ended on, up to symmetry, best first, with how many searches
    ended on each; the best is rescored by board.get_words."""
    recs = [json.loads(x) for x in path.open()]
    boards: dict[str, dict] = {}
    for r in recs:
        b = boards.setdefault(
            canonical(r["board"]),
            {"board": canonical(r["board"]), "points": r["points"], "words": r["words"]},
        )
        b["searches"] = b.get("searches", 0) + 1
    ranked = sorted(boards.values(), key=lambda b: (-b["points"], b["board"]))
    best = ranked[0]
    trie = make_trie(w for w in WORDS.read_text().split() if len(w) >= 3)
    assert py_score(best["board"], trie) == (best["points"], best["words"])
    lo, hi = wilson(best["searches"], len(recs))
    points = sorted(r["points"] for r in recs)
    return {
        "word_list": WORDS.name,
        "settings": {"steps": STEPS, "t0": T0, "t1": T1},
        "searches": len(recs),
        "seeds": sorted({r["seed"] for r in recs}),
        "best": best,
        "share of searches ending on the best": [
            round(best["searches"] / len(recs), 4),
            round(lo, 4),
            round(hi, 4),
        ],
        "searches above Boyan's 10,112": sum(p > 10112 for p in points),
        "points quartiles": [points[len(points) * q // 4] for q in (1, 2, 3)],
        "distinct final boards": len(boards),
        "top boards": ranked[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--runs", type=int, default=25, help="searches per process")
    r.add_argument("--procs", type=int, default=1, help="processes, seeds 10*seed+k")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--out", type=Path, default=Path("."))
    s = sub.add_parser("summary")
    s.add_argument("records", nargs="*", type=Path, help="anneal-*.jsonl files to merge first")
    sub.add_parser("check")
    args = ap.parse_args()
    if args.cmd == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        seeds = [10 * args.seed + k for k in range(args.procs)]
        binary()  # build once, before the threads start

        def one(seed: int) -> None:
            recs = anneal(seed, args.runs, args.steps)
            path = args.out / f"anneal-{seed}.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in recs))
            print(seed, max(x["points"] for x in recs), flush=True)

        with ThreadPoolExecutor(args.procs) as pool:
            list(pool.map(one, seeds))
    elif args.cmd == "summary":
        path = RESULTS / "anneal-5x5.jsonl"
        if args.records:  # merge run outputs into the committed records, in seed order
            recs = [json.loads(x) for p in args.records for x in p.open()]
            recs.sort(key=lambda r: (r["seed"], r["run"]))
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        (RESULTS / "best-5x5.json").write_text(json.dumps(summary(path), indent=1) + "\n")
        print((RESULTS / "best-5x5.json").read_text())
    else:
        trie = make_trie(w for w in WORDS.read_text().split() if len(w) >= 3)
        for letters in ("rstcsdeiaegnlrpeatesmssid", "ligdrmanesietildsracsepes"):
            assert c_score(letters) == py_score(letters, trie), letters
        print("C and Python scores agree")


if __name__ == "__main__":
    main()
