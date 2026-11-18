"""How well each model and template predicts human associations and human guesses.

Two tests, both on data the models never saw:

- **Human guesses** (Cultural Codes). For each turn: the clue, the unrevealed words, and
  the word a person picked. A model predicts the word most similar to the clue. We report
  how often that is the person's word (top-1 accuracy) and the mean log-probability of
  the person's word under the model's Luce guesser, with tau fitted on the train split.
- **Associations** (EAT, held-out cues only). For each cue: its most frequent one-word
  response and 24 board words that are not responses to the cue, drawn with a fixed
  seed. We report how often the response is the most similar of the 25.

    uv run python -m codenames.evaluate           # all models and templates, a few minutes
"""

import argparse
import json
import math
import random

import numpy as np

from codenames.luce import luce_weights
from codenames.norms import Counts, cultural_guesses, eat_counts, held_out
from codenames.players import Space, space
from codenames.words import board_words, vocabulary

TAUS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2)
SETTINGS = [
    (model, clue_t, word_t)
    for model in ("potion", "wordllama", "granite", "bge")
    for clue_t, word_t in (
        ("bare", "bare"),
        ("word", "word"),
        ("related", "bare"),
        ("query", "bare"),
        *((("bge_query", "bare"),) if model == "bge" else ()),
    )
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def guess_scores(sp: Space, split: str, tau: float | None = None) -> dict:
    """Top-1 accuracy and mean log-likelihood on human guesses; tau fitted on train if None."""
    if tau is None:
        tau = max(TAUS, key=lambda t: guess_loglik(sp, "train", t))
    data = cultural_guesses(split)
    hits = sum(int(g.words[int(np.argmax(sp.sims(g.clue, g.words)))] == g.picked) for g in data)
    lo, hi = wilson(hits, len(data))
    return {
        "n": len(data),
        "top1": hits / len(data),
        "top1_ci": [lo, hi],
        "tau": tau,
        "loglik": guess_loglik(sp, split, tau),
        "chance_loglik": float(np.mean([-math.log(len(g.words)) for g in data])),
    }


def guess_loglik(sp: Space, split: str, tau: float) -> float:
    total = 0.0
    data = cultural_guesses(split)
    for g in data:
        w = luce_weights(sp.sims(g.clue, g.words), tau)[0]
        total += math.log(w[g.words.index(g.picked)] / w.sum())
    return total / len(data)


def association_items(counts: Counts, vocab: set[str], seed: int = 0):
    """(cue, response, 24 distractors) for each held-out cue whose top response is usable."""
    rng = random.Random(seed)
    board = sorted(board_words())
    items = []
    for cue in sorted(counts):
        if not held_out(cue) or cue not in vocab:
            continue
        row = counts[cue]
        best = max(sorted(row), key=lambda r: row[r])
        if best == cue or best not in vocab:
            continue
        pool = [w for w in board if w not in row and w != cue]
        items.append((cue, best, rng.sample(pool, 24)))
    return items


def association_top1(sp: Space, items) -> dict:
    hits = 0
    for cue, resp, others in items:
        words = (resp, *others)
        hits += int(np.argmax(sp.sims(cue, words)) == 0)
    lo, hi = wilson(hits, len(items))
    return {"n": len(items), "top1": hits / len(items), "top1_ci": [lo, hi]}


def all_words() -> list[str]:
    words = list(vocabulary())
    extra = {
        w for s in ("train", "val", "test") for g in cultural_guesses(s) for w in (g.clue, *g.words)
    }
    return words + sorted(extra - set(words))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="write the table as JSON here")
    args = parser.parse_args()
    words = all_words()
    items = association_items(eat_counts(), set(words))
    rows: list[dict] = []
    for model, ct, wt in SETTINGS:
        sp = space(model, ct, wt, words)
        g = guess_scores(sp, "test")
        a = association_top1(sp, items)
        rows.append({"model": model, "clue": ct, "word": wt, "guess": g, "assoc": a})
        print(
            f"{model:9} {ct:9} {wt:5}  guesses {g['top1']:.3f} "
            f"[{g['top1_ci'][0]:.3f}, {g['top1_ci'][1]:.3f}] tau {g['tau']:<5} "
            f"loglik {g['loglik']:.3f}   EAT {a['top1']:.3f} "
            f"[{a['top1_ci'][0]:.3f}, {a['top1_ci'][1]:.3f}]",
            flush=True,
        )
    print(f"human guesses: {rows[0]['guess']['n']} test turns; EAT items: {len(items)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
