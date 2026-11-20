"""Plan M1 (experiments/plans/M1.md): do Fallout screens of common words change the table?

Two steps:

    python -m mastermind.m1 run --slice K --out OUT     # one slice, on a GitHub Actions job
    uv run python -m mastermind.m1 score                # here: results/m1.json from the slices

A slice is one word list, seed and set of rows of `fallout3.study` (SLICES). `run` scores
every strategy on every password of every screen, exactly, and writes `slice-K.json`: per
row and strategy, how many screens had each number of passwords cracked within four
attempts. That histogram is all the plan's statistics need: a mean, a normal interval,
the share of screens with a miss, and a bootstrap over screens, which draws screens with
replacement and so is a multinomial draw from the histogram. Random guessing, whose value
on a screen is a fraction, is kept as a sum and a sum of squares. Committed copies of the
slices are in `results/m1-slices/`, and `score` reads them.

The ENABLE screens use the README's word list with a new seed (31), and also its own seed
(3, 1,000 screens), which `score` checks against the README's table. Common screens draw
from ENABLE words with a Zipf frequency of at least 3.0 in wordfreq 3.1.1 (seed 32), and at
2.5 and 3.5 for the 20-word Very Easy row (sensitivity). No word list derived from wordfreq
is written, only counts, and the hardest common screen found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from mastermind import fallout3 as f3

HERE = Path(__file__).parent
SLICES_DIR = HERE / "results" / "m1-slices"
OUT = HERE / "results" / "m1.json"
PLAN = "experiments/plans/M1.md"
SCREENS = 5000
STRATEGIES = [*f3.STRATEGIES, "best possible"]  # whose value on a screen is an integer
RANDOM = "random candidate"
HEADLINE = ("Very Easy", 20)
ROWS = [(level, n) for level in f3.LEVELS for n in f3.COUNTS[level]]
RESAMPLES, BOOT_SEED = 10_000, 0
MDE_Z = 2.80

# slice: (words, zipf or None, seed, screens per row, rows)
SLICES: dict[int, tuple[str, float | None, int, int, list[tuple[str, int]]]] = {
    0: ("enable", None, 31, SCREENS, ROWS),
    1: ("common", 3.0, 32, SCREENS, ROWS),
    2: ("common", 2.5, 32, SCREENS, [HEADLINE]),
    3: ("common", 3.5, 32, SCREENS, [HEADLINE]),
    4: ("enable", None, f3.SEED, 1000, ROWS),  # the README's own run, recomputed
}


def pool_of(words: str, zipf: float | None) -> dict[int, list[str]]:
    return f3.word_pool(words, zipf if zipf is not None else f3.ZIPF)


def pool_sizes(words: str, zipf: float | None) -> dict[str, int]:
    pool = pool_of(words, zipf)
    return {str(n): len(pool[n]) for n in range(4, 16)}


def score_row(level: str, n: int, screens: int, seed: int, pool) -> dict:
    """Every strategy on every screen of one row, as histograms and sums."""
    hist: dict[str, Counter[int]] = {s: Counter() for s in STRATEGIES}
    rand_sum, rand_sq = Fraction(0), Fraction(0)
    like: dict[int, list[float]] = {}
    no_sure_win = 0
    hardest: tuple[int, list[str]] | None = None
    for screen in f3.row_screens(level, n, screens, seed, pool):
        scores = f3.score_screen(screen)
        for s in STRATEGIES:
            hist[s][int(scores[s][0])] += 1
        r = Fraction(scores[RANDOM][0]) / n
        rand_sum += r
        rand_sq += r * r
        best = int(scores["best possible"][0])
        no_sure_win += best < n
        if hardest is None or best < hardest[0]:
            hardest = (best, screen)
        m = f3.mean_likeness(screen)
        like.setdefault(len(screen[0]), []).append(m)
    assert hardest is not None
    return {
        "level": level,
        "words": n,
        "screens": screens,
        "cracked_histogram": {
            s: {str(k): v for k, v in sorted(h.items())} for s, h in hist.items()
        },
        "random_share_sum": float(rand_sum),
        "random_share_sum_sq": float(rand_sq),
        "screens_no_sure_win": no_sure_win,
        "mean_likeness_by_length": {
            str(k): {"screens": len(v), "sum": math.fsum(v), "sum_sq": math.fsum(x * x for x in v)}
            for k, v in sorted(like.items())
        },
        "hardest": {"cracked": hardest[0], "screen": [w.upper() for w in hardest[1]]},
    }


def run_slice(k: int) -> dict:
    words, zipf, seed, screens, rows = SLICES[k]
    pool = pool_of(words, zipf)
    out: dict = {
        "plan": PLAN,
        "slice": k,
        "words": words,
        "zipf": zipf,
        "seed": seed,
        "screens": screens,
        "wordfreq": f3.WORDFREQ_VERSION if words == "common" else None,
        "enable_sha256": f3.ENABLE_SHA256,
        "pool_sizes": pool_sizes(words, zipf),
        "rows": [],
    }
    for level, n in rows:
        out["rows"].append(score_row(level, n, screens, seed, pool))
        print(k, level, n, flush=True)
    return out


# Scoring (here).


def hist_moments(h: dict[str, int], n: int) -> tuple[int, float, float]:
    """Screens, mean and sample standard deviation of the share cracked."""
    m = sum(h.values())
    xs = {int(k) / n: v for k, v in h.items()}
    mean = sum(x * v for x, v in xs.items()) / m
    var = sum(v * (x - mean) ** 2 for x, v in xs.items()) / (m - 1)
    return m, mean, math.sqrt(var)


def normal(mean: float, sd: float, m: int) -> dict:
    """Percent, with the half-width of a normal 95% interval, as the README reports."""
    return {"percent": 100 * mean, "half_width": 100 * 1.96 * sd / math.sqrt(m)}


def row_summary(row: dict) -> dict:
    n = row["words"]
    out: dict = {}
    for s in STRATEGIES:
        h = row["cracked_histogram"][s]
        m, mean, sd = hist_moments(h, n)
        out[s] = {**normal(mean, sd, m), "screens_missed": m - h.get(str(n), 0)}
    m = row["screens"]
    mean = row["random_share_sum"] / m
    var = (row["random_share_sum_sq"] - m * mean * mean) / (m - 1)
    out[RANDOM] = normal(mean, math.sqrt(max(var, 0.0)), m)
    out["screens_no_sure_win_percent"] = 100 * row["screens_no_sure_win"] / m
    out["mean_likeness_by_length"] = {
        k: v["sum"] / v["screens"] for k, v in row["mean_likeness_by_length"].items()
    }
    return out


def boot_means(h: dict[str, int], n: int, rng) -> list[float]:
    """Bootstrap means of the share cracked: resampling screens with replacement is a
    multinomial draw from the histogram."""
    import numpy as np

    values = np.array([int(k) / n for k in h])
    counts = np.array(list(h.values()))
    m = int(counts.sum())
    draws = rng.multinomial(m, counts / m, size=RESAMPLES)
    return list(draws @ values / m)


def difference(a: dict[str, int], b: dict[str, int], n: int) -> dict:
    """Mean share cracked, a minus b, in points, with a 95% percentile bootstrap interval
    over screens within each list (independent lists), and the detectable effect."""
    import numpy as np

    rng = np.random.default_rng(BOOT_SEED)
    da, db = np.array(boot_means(a, n, rng)), np.array(boot_means(b, n, rng))
    d = 100 * (da - db)
    ma, mean_a, sd_a = hist_moments(a, n)
    mb, mean_b, sd_b = hist_moments(b, n)
    se = 100 * math.sqrt(sd_a**2 / ma + sd_b**2 / mb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "points": 100 * (mean_a - mean_b),
        "low": float(lo),
        "high": float(hi),
        "resamples": RESAMPLES,
        "seed": BOOT_SEED,
        "excludes_0": bool(lo > 0 or hi < 0),
        "detectable_effect_points": MDE_Z * se,
        "sd_points": {"a": 100 * sd_a, "b": 100 * sd_b},
    }


def row_key(level: str, n: int) -> str:
    return f"{level} {n}"


# The README's table (seed 3, 1,000 screens), as printed there: percent, half-width or None.
README_TABLE: dict[str, dict[str, tuple[str, str | None]]] = {
    "Very Easy 5": {
        "most parts": ("99.7", "0.1"),
        "expected size": ("99.7", "0.1"),
        "entropy": ("99.7", "0.1"),
        "minimax": ("99.7", "0.1"),
        "most in common": ("99.6", "0.2"),
        "random candidate": ("99.3", "0.1"),
        "best possible": ("99.7", "0.1"),
    },
    "Very Easy 20": {
        "most parts": ("79.5", "0.6"),
        "expected size": ("88.1", "0.5"),
        "entropy": ("86.9", "0.5"),
        "minimax": ("87.6", "0.5"),
        "most in common": ("85.1", "0.5"),
        "random candidate": ("67.8", "0.4"),
        "best possible": ("92.5", "0.4"),
    },
    "Easy 20": {
        "most parts": ("93.7", "0.4"),
        "expected size": ("98.1", "0.2"),
        "entropy": ("97.7", "0.2"),
        "minimax": ("98.4", "0.2"),
        "most in common": ("93.5", "0.3"),
        "random candidate": ("82.9", "0.3"),
        "best possible": ("99.78", "0.07"),
    },
    "Average 20": {
        "most parts": ("98.5", "0.2"),
        "expected size": ("99.95", None),
        "entropy": ("99.85", None),
        "minimax": ("99.95", None),
        "most in common": ("96.8", "0.2"),
        "random candidate": ("91.2", "0.2"),
        "best possible": ("100", None),
    },
    "Hard 20": {
        "most parts": ("99.64", None),
        "expected size": ("99.99", None),
        "entropy": ("99.98", None),
        "minimax": ("100.00", None),
        "most in common": ("97.9", "0.2"),
        "random candidate": ("94.7", "0.1"),
        "best possible": ("100", None),
    },
    "Very Hard 13": {
        "most parts": ("100", None),
        "expected size": ("100", None),
        "entropy": ("100", None),
        "minimax": ("100", None),
        "most in common": ("99.6", "0.1"),
        "random candidate": ("99.13", None),
        "best possible": ("100", None),
    },
}


def readme_check(summary: dict[str, dict]) -> dict:
    """The recomputed seed-3 run against the README's printed table: each value, printed
    with as many decimals as the README prints, must match, and so must each half-width.
    "100" must be exactly every password."""
    bad = []
    for row, cells in README_TABLE.items():
        for s, (pct, half) in cells.items():
            got = summary[row][s]
            places = len(pct.partition(".")[2])
            ok = got["percent"] == 100 if pct == "100" else f"{got['percent']:.{places}f}" == pct
            if half is not None:
                ok = ok and f"{got['half_width']:.{len(half.partition('.')[2])}f}" == half
            if not ok:
                bad.append({"row": row, "strategy": s, "readme": [pct, half], "got": got})
    return {"cells": sum(len(c) for c in README_TABLE.values()), "mismatches": bad}


def replicate_check(new: dict[str, dict], old: dict[str, dict]) -> dict:
    """Seed 31 (5,000 screens) against seed 3 (1,000) on ENABLE: the difference of each
    cell, and whether it is inside 1.96 times its standard error (both runs are
    independent draws of the same simulation)."""
    out, outside = {}, 0
    for row in old:
        for s in [*STRATEGIES, RANDOM]:
            a, b = new[row][s], old[row][s]
            se = math.hypot(a["half_width"], b["half_width"])  # half-widths are 1.96 se
            d = a["percent"] - b["percent"]
            inside = abs(d) <= se + 1e-9
            outside += not inside
            out[f"{row} | {s}"] = {"difference_points": d, "allowed": se, "inside": inside}
    return {"cells": len(out), "outside": outside, "cells_detail": out}


def load_slices(folder: Path = SLICES_DIR) -> dict[int, dict]:
    return {
        int(p.stem.split("-")[1]): json.loads(p.read_text())
        for p in sorted(folder.glob("slice-*.json"))
    }


def score(folder: Path = SLICES_DIR) -> dict:
    slices = load_slices(folder)
    assert sorted(slices) == sorted(SLICES), sorted(slices)
    rows = {k: {row_key(r["level"], r["words"]): r for r in s["rows"]} for k, s in slices.items()}
    summ = {k: {key: row_summary(r) for key, r in rs.items()} for k, rs in rows.items()}
    head = row_key(*HEADLINE)
    n = HEADLINE[1]
    best = "best possible"
    result: dict = {
        "plan": PLAN,
        "slices_sha256": {
            str(k): hashlib.sha256((folder / f"slice-{k}.json").read_bytes()).hexdigest()
            for k in sorted(slices)
        },
        "settings": {
            str(k): {x: slices[k][x] for x in ("words", "zipf", "seed", "screens", "wordfreq")}
            for k in sorted(slices)
        },
        "pool_sizes": {
            "enable": slices[0]["pool_sizes"],
            **{f"zipf {slices[k]['zipf']}": slices[k]["pool_sizes"] for k in (1, 2, 3)},
        },
        "primary": {
            "h_common": difference(
                rows[1][head]["cracked_histogram"][best],
                rows[0][head]["cracked_histogram"][best],
                n,
            ),
            "row": head,
            "metric": "best play, percent of passwords cracked within four attempts, common - ENABLE",
        },
    }
    sens = {}
    for k in (2, 3):
        sens[f"zipf {slices[k]['zipf']}"] = {
            "common": summ[k][head],
            "minus_enable": difference(
                rows[k][head]["cracked_histogram"][best],
                rows[0][head]["cracked_histogram"][best],
                n,
            ),
        }
    diffs = {}
    for key in rows[0]:
        diffs[key] = {
            s: summ[1][key][s]["percent"] - summ[0][key][s]["percent"]
            for s in [*STRATEGIES, RANDOM]
        }
    result["secondary"] = {
        "enable": summ[0],
        "common": summ[1],
        "common_minus_enable_points": diffs,
        "sensitivity_headline_row": sens,
        "hardest_common_screen": min(
            (r["hardest"] | {"row": key, "words": r["words"]} for key, r in rows[1].items()),
            key=lambda h: (h["cracked"] / h["words"], h["row"]),
        ),
        "hardest_enable_screen": min(
            (r["hardest"] | {"row": key, "words": r["words"]} for key, r in rows[0].items()),
            key=lambda h: (h["cracked"] / h["words"], h["row"]),
        ),
    }
    result["checks"] = {
        "readme_seed_3": readme_check(summ[4]),
        "replicate_seed_31_vs_3": replicate_check(
            {k: v for k, v in summ[0].items() if k in summ[4]}, summ[4]
        ),
    }
    return result


def rounded(x, places: int = 6):
    if isinstance(x, float):
        return round(x, places)
    if isinstance(x, dict):
        return {k: rounded(v, places) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, places) for v in x]
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["run", "score"])
    parser.add_argument("--slice", type=int, help="run: the slice, a key of SLICES")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        data = run_slice(args.slice)
        (args.out / f"slice-{args.slice}.json").write_text(json.dumps(data, indent=1) + "\n")
        return
    result = rounded(score())
    (args.out or OUT).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result["primary"], indent=1))


if __name__ == "__main__":
    main()
