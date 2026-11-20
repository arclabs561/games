"""How many words can a Boggle board show? Estimators of unseen words, scored against the truth.

Each sampled board is a sampling unit, and a word is either on it or not, so the data are
**incidence data**: for each word, the number of boards that show it. `Q[k]` is the number of
words seen on exactly k boards, `T` the number of boards and `U` the number of word sightings
(the sum of k Q[k]). The estimators below use only these counts. The truth is known here:
160,719 words of `enable1.txt` can appear on some board with the new cubes (`board.possible`).

- **Good–Turing.** Good's (1953) sample coverage, 1 − Q1/U, estimates the share of sightings
  on a new board that are of words already seen. Dividing the words seen by it gives
  S_obs / (1 − Q1/U), the estimate if unseen words were on average as common as seen ones.
- **Chao2.** Chao's (1987) lower bound for incidence data, S_obs + (T − 1)/T · Q1² / (2 Q2),
  the incidence form of Chao1, with Chao's variance and log-normal interval (Chao and Colwell
  2017, equations 2b, 3a and 3c).
- **iChao2.** Chiu, Wang, Walther and Chao (2014): Chao2 plus a correction from Q3 and Q4
  (Chao and Colwell 2017, equation 6d).
- **Jackknife.** The second-order jackknife for incidence data (Burnham and Overton 1978),
  S_obs + Q1 (2T − 3)/T − Q2 (T − 2)² / (T (T − 1)).

Chao2's interval uses Chao's variance. The others take their standard error from resampling
the boards (200 resamples, seeded by the number of boards) into the same log-normal interval.
Percentiles of the resampled estimates would not do: a board drawn twice turns each of its
words seen once into a word seen twice, so the resampled estimates sit far below the estimate. The boards are the post's 15,000, from one
generator seeded with 20161126; the estimates use the first T of them.

    uv run python -m boggle.unseen          # writes results/unseen.json, about 2 minutes
"""

import argparse
import json
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import exp, log, sqrt
from pathlib import Path

import numpy as np

from boggle import chance
from boggle.board import (
    _board_words,
    _set_trie,
    load_words,
    make_trie,
    possible,
    sample_boards,
    wilson,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORDS = HERE / "enable1.txt"
ENABLE1_SHA256 = "a392640f14602fcbb36779eb6671d170260c06d43f41c16383e7f0e35b221b4a"
NBOARDS, SEED = 15_000, 20161126
SIZES = (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 15_000)
BLOCKS = (1_000, 3_000)  # block sizes for the spread over disjoint samples
RESAMPLES = 200
ESTIMATORS = ("good_turing", "chao2", "ichao2", "jackknife2")


def frequency_counts(counts: Sequence[int] | np.ndarray, upto: int = 4) -> dict[str, int]:
    """S_obs, U and Q1 to Q`upto` from the number of boards that show each word."""
    c = np.asarray(counts, dtype=np.int64)
    c = c[c > 0]
    out = {"S_obs": int(c.size), "U": int(c.sum())}
    out |= {f"Q{k}": int((c == k).sum()) for k in range(1, upto + 1)}
    return out


def good_turing(s_obs: int, q1: int, u: int) -> float:
    """Words seen divided by Good's sample coverage 1 - Q1/U."""
    return s_obs / (1 - q1 / u)


def chao2(s_obs: int, q1: int, q2: int, t: int) -> float:
    """Chao's (1987) lower bound for incidence data (Chao and Colwell 2017, eq. 2b)."""
    a = (t - 1) / t
    return s_obs + (a * q1 * q1 / (2 * q2) if q2 > 0 else a * q1 * (q1 - 1) / 2)


def chao2_interval(s_obs: int, q1: int, q2: int, t: int, z: float = 1.96) -> tuple[float, float]:
    """Chao's variance (eq. 3a, for Q1, Q2 > 0) and log-normal interval (eq. 3c)."""
    a, r = (t - 1) / t, q1 / q2
    var = q2 * (a * a * r**4 / 4 + a * a * r**3 + a * r * r / 2)
    return log_interval(s_obs, chao2(s_obs, q1, q2, t), var, z)


def log_interval(s_obs: int, estimate: float, var: float, z: float = 1.96) -> tuple[float, float]:
    """Chao's (1987) interval: log(estimate - S_obs) taken as normal (Chao and Colwell 2017,
    eq. 3c), so the interval never reaches below the words seen."""
    extra = estimate - s_obs
    k = exp(z * sqrt(log(1 + var / extra**2)))
    return s_obs + extra / k, s_obs + extra * k


def ichao2(s_obs: int, q1: int, q2: int, q3: int, q4: int, t: int) -> float:
    """Chiu et al. (2014): Chao2 plus a correction from Q3 and Q4 (Chao and Colwell, eq. 6d)."""
    base = chao2(s_obs, q1, q2, t)
    if q4 == 0:
        q4 = 1  # Chiu et al.'s replacement when no word is seen exactly four times
    return base + (t - 3) / (4 * t) * q3 / q4 * max(q1 - (t - 3) / (2 * (t - 1)) * q2 * q3 / q4, 0)


def jackknife2(s_obs: int, q1: int, q2: int, t: int) -> float:
    """Second-order jackknife for incidence data (Burnham and Overton 1978)."""
    return s_obs + q1 * (2 * t - 3) / t - q2 * (t - 2) ** 2 / (t * (t - 1))


def estimates(counts: Sequence[int] | np.ndarray, t: int) -> dict[str, float]:
    """Each estimator of the number of possible words, from the boards per word."""
    f = frequency_counts(counts)
    s, u, q1, q2, q3, q4 = (f[k] for k in ("S_obs", "U", "Q1", "Q2", "Q3", "Q4"))
    return {
        "good_turing": good_turing(s, q1, u),
        "chao2": chao2(s, q1, q2, t),
        "ichao2": ichao2(s, q1, q2, q3, q4, t),
        "jackknife2": jackknife2(s, q1, q2, t),
    }


def incidences(board_words: Sequence[Sequence[str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The board and word index of every sighting, and the words in index order."""
    vocab = sorted({w for ws in board_words for w in ws})
    index = {w: i for i, w in enumerate(vocab)}
    boards = np.repeat(np.arange(len(board_words)), [len(ws) for ws in board_words])
    words = np.fromiter((index[w] for ws in board_words for w in ws), dtype=np.int64)
    return boards, words, vocab


def counts_for(
    boards: np.ndarray, words: np.ndarray, nwords: int, weights: np.ndarray
) -> np.ndarray:
    """Boards per word when board b is counted weights[b] times."""
    return np.bincount(words, weights=weights[boards], minlength=nwords).astype(np.int64)


def row(boards: np.ndarray, words: np.ndarray, nwords: int, t: int, truth: int) -> dict:
    """The estimates from the first t boards, with intervals, and their errors."""
    keep = boards < t
    b, w = boards[keep], words[keep]
    counts = counts_for(b, w, nwords, np.ones(t))
    f = frequency_counts(counts)
    point = estimates(counts, t)
    rng = np.random.default_rng([SEED, t])
    boot = {k: [] for k in ESTIMATORS}
    for _ in range(RESAMPLES):
        weights = np.bincount(rng.integers(0, t, t), minlength=t).astype(float)
        for k, v in estimates(counts_for(b, w, nwords, weights), t).items():
            boot[k].append(v)
    out: dict = {"boards": t, **f}
    for k in ESTIMATORS:
        se = float(np.std(boot[k], ddof=1))
        lo, hi = log_interval(f["S_obs"], point[k], se * se)
        out[k] = {"estimate": round(point[k], 1), "se": round(se, 1), "low": round(lo, 1),
                  "high": round(hi, 1), "resample_mean": round(float(np.mean(boot[k])), 1),
                  "share_of_truth": round(point[k] / truth, 4)}  # fmt: skip
    lo, hi = chao2_interval(f["S_obs"], f["Q1"], f["Q2"], t)
    out["chao2"] |= {"analytic_low": round(lo, 1), "analytic_high": round(hi, 1)}
    return out


def block_spread(boards: np.ndarray, words: np.ndarray, nwords: int, size: int) -> dict:
    """Each estimator on disjoint blocks of `size` boards: independent samples of that size."""
    per = {k: [] for k in ESTIMATORS}
    for start in range(0, NBOARDS - size + 1, size):
        keep = (boards >= start) & (boards < start + size)
        counts = np.bincount(words[keep], minlength=nwords)
        for k, v in estimates(counts, size).items():
            per[k].append(v)
    return {
        "boards": size,
        "blocks": len(per["chao2"]),
        **{k: {"mean": round(float(np.mean(v)), 1), "sd": round(float(np.std(v, ddof=1)), 1),
               "min": round(min(v), 1), "max": round(max(v), 1)} for k, v in per.items()},
    }  # fmt: skip


def explain(found: Sequence[Sequence[str]], possible: set[str], bounds: dict[str, float]) -> dict:
    """Why the estimates fall short, from the whole sample. Chao2 divides the Good–Turing
    estimate of the unseen words' total chance per board, Q1/T, by an estimate of their mean
    chance, 2 Q2 / ((T - 1) Q1), which is the mean chance of the words seen once (Chao and
    Colwell 2017, eq. 5e). The bounds of chance.word_bounds cap the unseen words' true total
    and mean."""
    t = len(found)
    counts: dict[str, int] = {}
    for ws in found:
        for w in ws:
            counts[w] = counts.get(w, 0) + 1
    q1 = sum(c == 1 for c in counts.values())
    q2 = sum(c == 2 for c in counts.values())
    unseen = sorted(possible - counts.keys())
    total = sum(bounds[w] for w in unseen)
    return {
        "boards": t,
        "unseen possible words": len(unseen),
        "Chao2 unseen words": round((t - 1) / t * q1 * q1 / (2 * q2), 1),
        "unseen total chance, Good-Turing": round(q1 / t, 4),
        "unseen total chance, at most": round(total, 4),
        "mean chance of words seen once, estimated": float(f"{2 * q2 / ((t - 1) * q1):.4g}"),
        "mean chance of unseen words, at most": float(f"{total / len(unseen):.4g}"),
        "unseen words with bound below 1e-6": sum(bounds[w] < 1e-6 for w in unseen),
    }


def short_word_check(found: Sequence[Sequence[str]], exact: Mapping[str, Fraction]) -> dict:
    """How the sample's share of boards for each word of two or three faces compares with
    its exact chance (chance.word_chance): the words whose exact chance lies outside the 95%
    Wilson interval of the sample, and the mean and spread of the standardized differences."""
    t = len(found)
    counts = {w: 0 for w in exact}
    for ws in found:
        for w in ws:
            if w in counts:
                counts[w] += 1
    z, outside = [], 0
    for w, chance_w in exact.items():
        p = float(chance_w)
        lo, hi = wilson(counts[w], t)
        outside += not lo <= p <= hi
        z.append((counts[w] / t - p) / sqrt(p * (1 - p) / t))
    return {
        "words": len(exact),
        "outside the 95% interval": outside,
        "mean difference": round(float(np.mean(z)), 3),
        "spread of differences": round(float(np.std(z)), 3),
    }


def possible_words(path: Path = WORDS) -> set[str]:
    """The different words of three or more letters that some shake of the cubes shows."""
    return {w for w in load_words(path, ENABLE1_SHA256) if len(w) >= 3 and possible(w)}


def board_words(n: int = NBOARDS, processes: int | None = None) -> list[list[str]]:
    """The words on each of the first n boards of the seeded sample, in a pool of processes."""
    trie = make_trie(w for w in load_words(WORDS, ENABLE1_SHA256) if len(w) >= 3)
    with mp.Pool(processes, initializer=_set_trie, initargs=(trie, 3)) as pool:
        return pool.map(_board_words, sample_boards(n, SEED), chunksize=250)


def summary(found: Sequence[Sequence[str]], possible: set[str], bounds: dict[str, float],
            exact: Mapping[str, Fraction], sizes: Sequence[int] = SIZES) -> dict:  # fmt: skip
    """The table that results/unseen.json holds."""
    boards, words, vocab = incidences(found)
    truth = len(possible)
    return {
        "truth": truth,
        "seed": SEED,
        "resamples": RESAMPLES,
        "rows": [row(boards, words, len(vocab), t, truth) for t in sizes],
        "blocks": [block_spread(boards, words, len(vocab), s) for s in BLOCKS if s <= len(found)],
        "explanation": explain(found, possible, bounds),
        "short words against the sample": short_word_check(found, exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "unseen.json")
    parser.add_argument("--procs", type=int, default=None)
    args = parser.parse_args()
    possible = possible_words()
    bounds = chance.word_bounds(sorted(possible))
    result = summary(
        board_words(processes=args.procs), possible, bounds, chance.short_chances(possible)
    )
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
