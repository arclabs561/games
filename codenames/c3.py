"""Experiment C3 (experiments/plans/C3.md): population training across seeds and held-out
guessers.

There are 14 tasks: two givers (potion and Qwen3-Embedding-0.6B), each with every other
embedding model held out in turn. Each task runs with training seeds 1 to 5, which
makes 70 runs of codenames.rl. All of them use the test boards of seed 3003 and write
per-board values. A run is named by one number, its slice, so a launcher can start
them as a list:

    slice = 5 * task + (seed - 1),   task = 7 * giver index + held-out index

The analysis pairs the population policy with the self-play policy on each test board
and seed. The primary statistic is the mean over the 14 tasks of the mean held-out
value, population minus self-play. Its 95% interval is a stratified bootstrap with
10,000 resamples and seed 0. Each resample draws the 200 boards with replacement, the
same draw for every task, and five seeds with replacement within each task.

    uv run python -m codenames.c3 --slice 0 --out OUT          # one run (3 to 5 min)
    uv run python -m codenames.c3 --runs DIR --out codenames/results/c3.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from codenames import rl
from codenames.study import models

GIVERS = ("potion", "Qwen/Qwen3-Embedding-0.6B")
MODELS = (
    "potion",
    "wordllama",
    "granite",
    "bge",
    "Qwen/Qwen3-Embedding-0.6B",
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-0.6b",
    "google/embeddinggemma-2",
)
SEEDS = (1, 2, 3, 4, 5)
TEST_SEED = 3003
BOOT_SEED = 0
DRAWS = 10_000
POLICIES = ("start", "self", "population", "population/rlvr")


def tasks() -> list[tuple[str, str]]:
    """(giver, held-out model) for the 14 tasks, in slice order."""
    return [(g, m) for g in GIVERS for m in MODELS if m != g]


def slice_of(n: int) -> tuple[str, str, int]:
    """(giver, held-out model, training seed) of slice n, for n in 0..69."""
    giver, held = tasks()[n // len(SEEDS)]
    return giver, held, SEEDS[n % len(SEEDS)]


def run_name(giver: str, held: str, seed: int) -> str:
    return f"{giver.replace('/', '--')}__{held.replace('/', '--')}__{seed}"


def run_slice(n: int, out: Path) -> Path:
    """Run one slice with the plan's settings and write its JSON into out."""
    assert tuple(models()) == MODELS, f"vectors missing: models() is {models()}"
    giver, held, seed = slice_of(n)
    result = rl.run(giver, held, seed=seed, test_seed=TEST_SEED, per_board=True)
    result["slice"] = n
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_name(giver, held, seed)}.json"
    path.write_text(json.dumps(result) + "\n")
    return path


# The analysis.


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile mean: drop the lowest and highest quarter (rounded down) and average
    the rest, as scipy.stats.trim_mean(x, 0.25) does."""
    x = np.sort(np.asarray(x, dtype=float), axis=axis)
    n = x.shape[axis]
    cut = int(0.25 * n)
    return np.take(x, np.arange(cut, n - cut), axis=axis).mean(axis=axis)


class Draws:
    """The bootstrap's resamples, drawn once and shared by every statistic: board counts
    (draws, boards), the same for every task, and for each task the indices of the
    seeds drawn (draws, seeds)."""

    def __init__(self, boards: int, seeds: list[int], draws: int = DRAWS, seed: int = BOOT_SEED):
        rng = np.random.default_rng(seed)
        self.boards = rng.multinomial(boards, np.full(boards, 1 / boards), size=draws)
        self.seeds = [rng.integers(0, s, size=(draws, s)) for s in seeds]


def run_means(d: np.ndarray, draws: Draws) -> np.ndarray:
    """Per-run mean over the resampled boards: d is (seeds, boards), result (draws, seeds)."""
    return draws.boards @ d.T / d.shape[1]


def task_reps(d: np.ndarray, draws: Draws, t: int) -> np.ndarray:
    """Resampled mean over seeds and boards of one task: (draws,)."""
    r = run_means(d, draws)
    return np.take_along_axis(r, draws.seeds[t], axis=1).mean(axis=1)


def interval(point: float, reps: np.ndarray) -> dict:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"mean": round(float(point), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}


def summarize(per_task: list[np.ndarray], draws: Draws) -> dict:
    """Mean over tasks of the per-task mean of d (each (seeds, boards)), with the
    stratified interval; the per-task means and intervals; the interquartile mean of the
    run means; and the share of runs whose mean is above 0 (ties count half)."""
    reps = np.stack([task_reps(d, draws, t) for t, d in enumerate(per_task)])  # (tasks, draws)
    point = float(np.mean([d.mean() for d in per_task]))
    runs = [run_means(d, draws) for d in per_task]
    pooled = np.concatenate(
        [np.take_along_axis(r, draws.seeds[t], axis=1) for t, r in enumerate(runs)], axis=1
    )
    flat = np.concatenate([d.mean(axis=1) for d in per_task])

    def above(x):
        return (x > 0).mean(axis=-1) + 0.5 * (x == 0).mean(axis=-1)

    return {
        "mean_over_tasks": interval(point, reps.mean(axis=0)),
        "iqm_over_runs": interval(float(iqm(flat)), iqm(pooled)),
        "share_of_runs_above_0": interval(float(above(flat)), above(pooled)),
        "runs": len(flat),
        "per_task": [interval(d.mean(), reps[t]) for t, d in enumerate(per_task)],
    }


def load_runs(folder: Path) -> dict[tuple[str, str, int], dict]:
    """Every run JSON under folder (searched recursively), keyed by (giver, held, seed)."""
    out = {}
    for p in sorted(folder.rglob("*.json")):
        r = json.loads(p.read_text())
        if "per_board" not in r:
            continue
        c = r["config"]
        key = (c["giver"], c["held_out"], c["seed"])
        assert key not in out, f"two runs for {key}"
        defaults = {"train_boards": 300, "test_boards": 200, "shortlist": 30, "steps": 400}
        assert all(c[k] == v for k, v in defaults.items()) and c["lr"] == 0.05, p
        assert c["test_seed"] == TEST_SEED, p
        assert r["slice"] == tasks().index(key[:2]) * len(SEEDS) + SEEDS.index(key[2]), p
        out[key] = r
    return out


def checks(runs: dict) -> dict:
    """Properties the design implies. The start policy is not trained, so its values do
    not depend on the seed or the held-out model. The self-play policy does not depend
    on the held-out model, since its population is the giver alone."""
    start_same, self_same = True, True
    for g in GIVERS:
        group = [r for (gg, _, _), r in runs.items() if gg == g]
        ref = group[0]["per_board"]["start"]
        start_same &= all(r["per_board"]["start"] == ref for r in group)
        for s in SEEDS:
            th = [r["theta"]["self"] for (gg, _, ss), r in runs.items() if gg == g and ss == s]
            self_same &= all(t == th[0] for t in th)
    return {"start_independent_of_seed_and_task": start_same, "self_independent_of_task": self_same}


def analyze(folder: Path) -> dict:
    runs = load_runs(folder)
    present = [[s for s in SEEDS if (g, m, s) in runs] for g, m in tasks()]
    missing = [run_name(g, m, s) for (g, m) in tasks() for s in SEEDS if (g, m, s) not in runs]
    draws = Draws(200, [len(p) for p in present])

    def values(policy: str, guesser) -> list[np.ndarray]:
        """(seeds, boards) per task, for guesser(giver, held) of each task."""
        return [
            np.array([runs[(g, m, s)]["per_board"][policy][guesser(g, m)] for s in seeds])
            for (g, m), seeds in zip(tasks(), present, strict=True)
        ]

    held = {p: values(p, lambda g, m: m) for p in POLICIES}
    eat = {p: values(p, lambda g, m: "eat") for p in POLICIES}
    own = {p: values(p, lambda g, m: g) for p in POLICIES}

    def diff(a, b):
        return [x - y for x, y in zip(a, b, strict=True)]

    def gap(p):
        return diff(own[p], held[p])

    result = {
        "plan": "experiments/plans/C3.md",
        "tasks": [
            {"giver": g, "held_out": m, "seeds": s}
            for (g, m), s in zip(tasks(), present, strict=True)
        ],
        "missing_runs": missing,
        "test_seed": TEST_SEED,
        "bootstrap": {"draws": DRAWS, "seed": BOOT_SEED},
        "checks": checks(runs),
        "primary": summarize(diff(held["population"], held["self"]), draws),
        "secondary": {
            "eat_population_minus_self": summarize(diff(eat["population"], eat["self"]), draws),
            "own_population_minus_self": summarize(diff(own["population"], own["self"]), draws),
            "held_rlvr_minus_population": summarize(
                diff(held["population/rlvr"], held["population"]), draws
            ),
            "eat_rlvr_minus_population": summarize(
                diff(eat["population/rlvr"], eat["population"]), draws
            ),
            "gap_self": summarize(gap("self"), draws),
            "gap_population": summarize(gap("population"), draws),
        },
        "values": {
            guesser: {p: summarize(v[p], draws) for p in POLICIES}
            for guesser, v in (("held_out", held), ("eat", eat), ("own", own))
        },
        "theta": {
            "features": list(rl.FEATURES),
            "runs": [
                {"giver": g, "held_out": m, "seed": s, **runs[(g, m, s)]["theta"]}
                for (g, m), seeds in zip(tasks(), present, strict=True)
                for s in seeds
            ],
        },
        "per_board": {
            "note": "values per task, seed (in the order of tasks[].seeds) and test board",
            "held_out": {p: [x.round(4).tolist() for x in held[p]] for p in POLICIES},
            "eat": {p: [x.round(4).tolist() for x in eat[p]] for p in POLICIES},
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--slice", type=int, help="run one slice, 0 to 69")
    which.add_argument("--runs", type=Path, help="analyze the run JSONs under this folder")
    parser.add_argument("--out", type=Path, required=True, help="folder (--slice) or file")
    args = parser.parse_args()
    if args.slice is not None:
        print(run_slice(args.slice, args.out))
        return
    result = analyze(args.runs)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    p = result["primary"]["mean_over_tasks"]
    print(f"primary: {p['mean']:+.3f} ({p['ci'][0]:+.3f} to {p['ci'][1]:+.3f})")


if __name__ == "__main__":
    main()
