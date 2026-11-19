"""Exploratory: human-match scores on Cultural Codes split by game and by player, with
intervals from a bootstrap over games.

This was run before experiments/plans/C1.md was written, on splits of its own
(``shuffled`` and ``player`` in codenames.split), and its numbers are exploratory. The
plan's confirmatory run is codenames.c1. It does not score the plan's ``game`` split.

For each split scheme (published, shuffled, player; see codenames.split), each model and
template: the temperature fitted on that scheme's train turns, the template chosen by
log-likelihood on its val turns, and on its test turns the share of turns where the
model's most similar word is the person's pick. Also the association-count guesser and
the linear clue map of codenames.train, trained on that scheme's train turns only.

Intervals resample whole games with replacement (10,000 draws, seed 0) and take the
2.5th and 97.5th percentiles of the pooled share, so turns of one game move together.
Paired differences between two models use the same draws on the same turns. Each
scheme also has intervals that resample guessers instead of games, since one person's
turns are not independent either.

    uv run python -m codenames.heldout              # all schemes (about 20 min here)
    uv run python -m codenames.heldout --quick      # two models, published and shuffled
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from codenames.evaluate import TAUS, all_words
from codenames.luce import luce_weights
from codenames.norms import cultural_guesses, eat_counts
from codenames.players import Space, norm_weigher, space
from codenames.split import Turn, turns
from codenames.stats import DRAWS, cluster_draws, clustered
from codenames.study import eat_eps, models, short, templates

RESULTS = Path(__file__).parent / "results"
ADAPTED = (("potion", "bare"), ("Qwen/Qwen3-Embedding-0.6B", "query"))


def sim_rows(sp: Space, data: tuple[Turn, ...]) -> list[tuple[np.ndarray, int]]:
    return [
        (sp.sims(t.guess.clue, t.guess.words), t.guess.words.index(t.guess.picked)) for t in data
    ]


def loglik(rows, tau: float) -> float:
    total = 0.0
    for s, k in rows:
        w = luce_weights(s, tau)[0]
        total += math.log(w[k] / w.sum())
    return total / len(rows)


def hits(rows) -> np.ndarray:
    """1 where the most similar word (the first, on a tie) is the person's pick."""
    return np.array([int(int(np.argmax(s)) == k) for s, k in rows])


def eat_hits(data: tuple[Turn, ...], eps: float) -> tuple[np.ndarray, float]:
    weigh = norm_weigher(eat_counts(), eps)
    out, total = [], 0.0
    for t in data:
        g = t.guess
        w = weigh(g.clue, g.words)
        out.append(int(g.words[int(np.argmax(w))] == g.picked and w.max() > eps))
        total += math.log(w[g.words.index(g.picked)] / w.sum())
    return np.array(out), total / len(data)


def score_space(sp: Space, scheme: str, test_rows=None) -> dict:
    """Temperature fitted on train, val log-likelihood, and test hits of one space."""
    train = sim_rows(sp, turns(f"{scheme}:train"))
    tau = max(TAUS, key=lambda t: loglik(train, t))
    val = sim_rows(sp, turns(f"{scheme}:val"))
    test = test_rows if test_rows is not None else sim_rows(sp, turns(f"{scheme}:test"))
    return {
        "tau": tau,
        "val_loglik": loglik(val, tau),
        "test_loglik": loglik(test, tau),
        "hits": hits(test),
    }


def adapter(sp: Space, scheme: str, source: str, seed: int = 0, exclude=frozenset()) -> Space:
    """The space with the clue map of codenames.train fitted on one scheme's train turns
    (source "cultural"), on EAT (source "eat"), or both. Games in exclude are left out."""
    from codenames.train import adapted, items, train

    guesses = tuple(t.guess for t in turns(f"{scheme}:train") if t.game not in exclude)
    data = items(sp, seed, eat=source != "cultural", cultural=source != "eat", split=guesses)
    a, _ = train(sp, data, seed=seed)
    return adapted(sp, a, f"{sp.name}+{source}")


def run(schemes: tuple[str, ...], model_list: list[str], adapted_models) -> dict:
    words = all_words()
    out: dict = {"label": "exploratory", "draws": DRAWS, "seed": 0, "schemes": {}}
    for scheme in schemes:
        t0 = time.perf_counter()
        test = turns(f"{scheme}:test")
        index, counts = cluster_draws([t.game for t in test])
        by_guesser = cluster_draws([t.guesser for t in test])
        res: dict = {"test_turns": len(test), "test_games": int(counts.shape[1]), "models": {}}
        best_hits: dict[str, np.ndarray] = {}
        for model in model_list:
            rows, hs = [], []
            for ct, wt in templates(model):
                s = score_space(space(model, ct, wt, words), scheme)
                hs.append(s.pop("hits"))
                rows.append({"clue": ct, "word": wt, **s, "top1": clustered(hs[-1], index, counts)})
            k = max(range(len(rows)), key=lambda i: rows[i]["val_loglik"])
            best, best_hits[model] = rows[k], hs[k]
            res["models"][model] = {"settings": rows, "best": [best["clue"], best["word"]]}
            print(scheme, short(model), best["clue"], f"{best['top1']['mean']:.3f}", flush=True)
        eps = eat_eps("train", scheme)
        h, ll = eat_hits(test, eps)
        res["eat"] = {"eps": eps, "test_loglik": ll, "top1": clustered(h, index, counts)}
        best_hits["eat"] = h
        for model, ct in adapted_models:
            if model not in model_list:
                continue
            sp = space(model, ct, "bare", words)
            res.setdefault("adapter", {})[model] = {"template": ct}
            base = score_space(sp, scheme)["hits"]
            for source in ("cultural", "eat", "both"):
                s = score_space(adapter(sp, scheme, source), scheme)
                h = s.pop("hits")
                name = f"{model}+{source}"
                best_hits[name] = h
                res["adapter"][model][source] = {
                    **s,
                    "top1": clustered(h, index, counts),
                    "minus_untrained": clustered(h - base, index, counts),
                    "minus_untrained_by_guesser": clustered(h - base, *by_guesser),
                }
                print(scheme, name, f"{h.mean():.3f}", flush=True)
            if scheme == "published":
                full = best_hits[f"{model}+cultural"]
                res["adapter"][model]["leakage"] = leakage(sp, full, index, counts)
            if scheme == "shuffled":
                res["adapter"][model]["smaller"] = smaller(sp, base, index, counts)
        res["paired"] = paired(best_hits, index, counts)
        res["paired_by_guesser"] = paired(best_hits, *by_guesser)
        res["top1_by_guesser"] = {m: clustered(h, *by_guesser) for m, h in best_hits.items()}
        res["seconds"] = round(time.perf_counter() - t0)
        out["schemes"][scheme] = res
    return out


def leakage(sp: Space, full: np.ndarray, index, counts) -> dict:
    """The clue map trained on published train without the games that have a turn in
    published test, scored on published test, paired with the map trained on all of
    published train."""
    test = turns("published:test")
    games = frozenset(t.game for t in test)
    kept = sum(t.game not in games for t in turns("published:train"))
    s = score_space(adapter(sp, "published", "cultural", exclude=games), "published")
    h = s.pop("hits")
    return {
        "train_turns_kept": kept,
        **s,
        "top1": clustered(h, index, counts),
        "full_minus_without": clustered(full - h, index, counts),
    }


def smaller(sp: Space, base: np.ndarray, index, counts, seed: int = 0) -> dict:
    """The clue map trained on a random subset of game-split training games with as many
    turns as the player split's training set, so that the two splits can be compared
    at one training size."""
    import random

    target = len(turns("player:train"))
    ids = sorted({t.game for t in turns("shuffled:train")})
    random.Random(seed).shuffle(ids)
    size = {g: 0 for g in ids}
    for t in turns("shuffled:train"):
        size[t.game] += 1
    keep, total = set(), 0
    for g in ids:
        if total >= target:
            break
        keep.add(g)
        total += size[g]
    drop = frozenset(ids) - keep
    s = score_space(adapter(sp, "shuffled", "cultural", exclude=drop), "shuffled")
    h = s.pop("hits")
    return {
        "train_turns": total,
        **s,
        "top1": clustered(h, index, counts),
        "minus_untrained": clustered(h - base, index, counts),
    }


def paired(hits_by: dict[str, np.ndarray], index, counts) -> dict:
    """Clustered intervals for the difference of every pair of the scored guessers."""
    names = sorted(hits_by, key=lambda m: -hits_by[m].mean())
    return {
        f"{a} - {b}": clustered(hits_by[a] - hits_by[b], index, counts)
        for a, b in itertools.combinations(names, 2)
    }


def all_turns_zero_shot(model_list: list[str]) -> dict:
    """Top-1 on all 6,088 turns for each untrained model and template. No step of these
    scores is fitted to the turns, so every turn is out of sample for them."""
    words = all_words()
    every = turns("all")
    index, counts = cluster_draws([t.game for t in every])
    out: dict = {"turns": len(every), "games": int(counts.shape[1]), "models": {}}
    keep: dict[str, np.ndarray] = {}
    for model in model_list:
        for ct, wt in templates(model):
            h = hits(sim_rows(space(model, ct, wt, words), every))
            out["models"].setdefault(model, {})[f"{ct}/{wt}"] = clustered(h, index, counts)
            keep[f"{model} {ct}/{wt}"] = h
        print("all", short(model), flush=True)
    bare = {m: keep[f"{m} bare/bare"] for m in model_list}
    out["paired_bare"] = paired(bare, index, counts)
    by_guesser = cluster_draws([t.guesser for t in every])
    out["paired_bare_by_guesser"] = paired(bare, *by_guesser)
    out["bare_by_guesser"] = {m: clustered(h, *by_guesser) for m, h in bare.items()}
    out["prompt_minus_bare"] = {
        k: clustered(h - keep[k.split(" ")[0] + " bare/bare"], index, counts)
        for k, h in keep.items()
        if not k.endswith(" bare/bare")
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(RESULTS / "heldout.json"))
    args = parser.parse_args()
    ms = models()
    if args.quick:
        ms = [m for m in ms if m in ("potion", "google/embeddinggemma-2")]
        schemes: tuple[str, ...] = ("published", "shuffled")
    else:
        schemes = ("published", "shuffled", "player")
    assert len(turns("published:test")) == len(cultural_guesses("test"))
    out = run(schemes, ms, ADAPTED)
    out["all_turns"] = all_turns_zero_shot(ms)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
