"""Experiment C1 (experiments/plans/C1.md): the Cultural Codes results under a split by game.

Runs every arm of the plan on the turns of the ``game`` split of codenames.split, and
writes results/calibrate-game.json, results/train-<model>-game.json and results/c1.json.

- Untrained models: temperature fitted on train, template chosen on val
  (study.calibrate), top-1 on test.
- The linear map of codenames.train for potion (bare) and Qwen3-Embedding-0.6B (query),
  trained on the train turns only (no EAT), seeds 1 to 5; each test turn's hit rate is
  the mean over the five maps.
- Primary metric: trained potion minus untrained potion, paired by turn, with a 95%
  interval from 10,000 resamples of test games (seed 0).
- Leakage: each test game's turns split in half by codenames.split.half. Arm A is the
  map trained on the train games; arm B the same with as many random training turns
  (seeded by the training seed) swapped out for the "sibling" half. Both scored on the
  "held" half. B minus A is the gain from seeing other turns of the same game.
- Player overlap: test games whose players also play in train games, by profile string
  (as the plan says) and by survey answers; the primary metric on the other test games.
- Kendall's tau-b between the untrained models' order on the new test set and their
  order on the published test set.

    uv run python -m codenames.c1 --dry-run   # everything scored on val instead of test
    uv run python -m codenames.c1             # the confirmatory run (once)
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from codenames.evaluate import all_words
from codenames.players import Space, space
from codenames.split import half, published_overlap, turns
from codenames.stats import DRAWS, cluster_draws, clustered, clustered_kendall
from codenames.study import calibrate, load, models, short
from codenames.train import adapted, by_game, items, train

RESULTS = Path(__file__).parent / "results"
SEEDS = (1, 2, 3, 4, 5)
MAPS = (("potion", "bare"), ("Qwen/Qwen3-Embedding-0.6B", "query"))
PUBLISHED_ORDER_FROM = "calibrate"  # results/calibrate.json, the published split


def as_hits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s])


def top1_hits(sp: Space, data) -> np.ndarray:
    return np.array(
        [
            int(
                t.guess.words[int(np.argmax(sp.sims(t.guess.clue, t.guess.words)))]
                == t.guess.picked
            )
            for t in data
        ]
    )


def leakage(model: str, template: str, score: str, mean_a: np.ndarray, test) -> dict:
    """Arm B of the plan, scored on the held half of each scored game, next to arm A."""
    sp = space(model, template, "bare", all_words())
    held = [i for i, t in enumerate(test) if half(t) == "held"]
    sibling = [t for t in test if half(t) == "sibling"]
    base = list(turns("game:train"))
    hits_b = []
    for seed in SEEDS:
        rng = random.Random(seed)
        drop = set(rng.sample(range(len(base)), len(sibling)))
        guesses = tuple(t.guess for i, t in enumerate(base) if i not in drop)
        guesses += tuple(t.guess for t in sibling)
        a, _ = train(sp, items(sp, seed, eat=False, split=guesses), seed=seed)
        hits_b.append(top1_hits(adapted(sp, a, f"{model}+B"), [test[i] for i in held]))
        print("leakage B", seed, round(float(hits_b[-1].mean()), 3), flush=True)
    b = np.mean(hits_b, axis=0)
    a = mean_a[held]
    index, counts = cluster_draws([test[i].game for i in held])
    return {
        "scored": score,
        "held_turns": len(held),
        "sibling_turns": len(sibling),
        "held_games": int(counts.shape[1]),
        "arm_a": clustered(a, index, counts),
        "arm_b": clustered(b, index, counts),
        "b_minus_a": clustered(b - a, index, counts),
        "arm_b_by_seed": [float(h.mean()) for h in hits_b],
    }


def overlap(test, diff: np.ndarray) -> dict:
    """Test games whose players also play in train games, and the primary metric on the
    test games with no such player."""
    train_set = turns("game:train")
    out = {}
    for name, ids in (
        ("profile_string", lambda t: set(t.profiles)),
        ("survey_id", lambda t: {t.giver, t.guesser}),
    ):
        seen = set().union(*(ids(t) for t in train_set))
        shared = {t.game for t in test if ids(t) & seen}
        fresh = [i for i, t in enumerate(test) if t.game not in shared]
        row = {
            "test_games": len({t.game for t in test}),
            "test_games_with_a_player_in_train": len(shared),
            "turns_in_other_games": len(fresh),
        }
        if fresh:
            index, counts = cluster_draws([test[i].game for i in fresh])
            row["primary_on_other_games"] = clustered(diff[fresh], index, counts)
        out[name] = row
    return out


def commit() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "codenames")),
    }


def run(score: str) -> dict:
    test = turns(f"game:{score}")
    index, counts = cluster_draws([t.game for t in test])
    out: dict = {
        "plan": "experiments/plans/C1.md",
        "commit": commit(),
        "scored": score,
        "draws": DRAWS,
        "bootstrap_seed": 0,
        "training_seeds": list(SEEDS),
        "published_overlap": published_overlap(),
        "turns": {s: len(turns(f"game:{s}")) for s in ("train", "val", "test")},
        "games": {s: len({t.game for t in turns(f"game:{s}")}) for s in ("train", "val", "test")},
    }
    cal = calibrate("game", score)
    untrained: dict[str, np.ndarray] = {}
    out["untrained"] = {}
    for m in models():
        b = cal[m]["best"]
        untrained[m] = as_hits(b["test"]["hits"])
        out["untrained"][m] = {
            "template": [b["clue"], b["word"]],
            "tau": b["tau"],
            "loglik": b["test"]["loglik"],
            "top1": clustered(untrained[m], index, counts),
        }
    eat = as_hits(cal["eat"]["hits"])
    out["eat"] = {"eps": cal["eat"]["eps"], "top1": clustered(eat, index, counts)}
    published = load(PUBLISHED_ORDER_FROM)
    names = models()
    ref = [published[m]["best"]["test"]["top1"] for m in names]
    out["kendall_vs_published"] = {
        "models": names,
        "published_top1": ref,
        **clustered_kendall(np.stack([untrained[m] for m in names], axis=1), ref, index, counts),
    }
    out["maps"] = {}
    for model, template in MAPS:
        res = by_game(model, template, "game", SEEDS, 3, score)
        each = [as_hits(r["guess"]["hits"]) for r in res["runs"].values()]
        mean = np.mean(each, axis=0)
        diff = mean - untrained[model]
        row = {
            "template": template,
            "top1": clustered(mean, index, counts),
            "top1_by_seed": [float(h.mean()) for h in each],
            "minus_untrained": clustered(diff, index, counts),
            "untrained_template": out["untrained"][model]["template"],
        }
        if model == "potion":
            row["leakage"] = leakage(model, template, score, mean, test)
            row["player_overlap"] = overlap(test, diff)
        out["maps"][model] = row
        print(short(model), "map minus untrained", row["minus_untrained"], flush=True)
    out["primary"] = {
        "metric": "top-1 of the trained potion map minus untrained potion, paired by turn",
        **out["maps"]["potion"]["minus_untrained"],
        "published_estimate": 0.082,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="score val, not test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    score = "val" if args.dry_run else "test"
    out = run(score)
    path = (
        Path(args.out) if args.out else RESULTS / ("c1.json" if score == "test" else "c1-val.json")
    )
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["primary"]))


if __name__ == "__main__":
    main()
