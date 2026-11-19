"""Experiment C2 (experiments/plans/C2.md): scoring a tuned embedding model on human guesses.

The functions here run inside the GPU job, right after `codenames.lora` has trained, since
the machine that writes the README cannot reach the Hugging Face Hub. They need numpy
only. For each model the job writes `scores.json` with:

- per template: the temperature τ fitted on train and the log-likelihood on val;
- the chosen template (best val log-likelihood) and, for that template, the hit and the
  log-probability of the person's word for every turn of the evaluated split, in the
  order of `norms.cultural_guesses`, so arms can be paired by turn afterwards on CPU;
- top-1 on the held-out EAT cues;
- with `--evaluate test`, the clue and count the model gives on 200 fresh boards, so its
  expected score with every other guesser can be computed afterwards on CPU.
"""

import math
import random

import numpy as np

from codenames.evaluate import TAUS, association_items, association_top1
from codenames.game import deal
from codenames.luce import luce_weights
from codenames.norms import Guess, cultural_guesses, eat_counts, held_out
from codenames.players import Space, expected_giver
from codenames.words import board_words, clue_words

BOARD_SEED = 2002
N_BOARDS = 200


def guesses(split: str, scheme: str = "published") -> tuple[Guess, ...]:
    """Human guesses of one split, under the published split or C1's game split."""
    return cultural_guesses(split, scheme)


def turn_sims(sp: Space, data: tuple[Guess, ...]) -> list[tuple[np.ndarray, int]]:
    """For each turn, the similarity of the clue to each unrevealed word, and the index
    of the person's word."""
    return [(sp.sims(g.clue, g.words), g.words.index(g.picked)) for g in data]


def hits(table: list[tuple[np.ndarray, int]]) -> list[int]:
    """1 where the most similar word (the first, on a tie) is the person's word."""
    return [int(int(np.argmax(s)) == k) for s, k in table]


def log_probs(table: list[tuple[np.ndarray, int]], tau: float) -> list[float]:
    """Log-probability of the person's word under the Luce guesser at temperature tau."""
    out = []
    for s, k in table:
        w = luce_weights(s, tau)[0]
        out.append(math.log(w[k] / w.sum()))
    return out


def fit_tau(table: list[tuple[np.ndarray, int]]) -> float:
    """The temperature in evaluate.TAUS with the best mean log-likelihood (first on a tie)."""
    return max(TAUS, key=lambda t: float(np.mean(log_probs(table, t))))


def score(spaces: dict[str, Space], splits: dict[str, tuple[Guess, ...]], evaluate: str) -> dict:
    """τ on train and the template on val for one model; per-turn scores on `evaluate`.

    spaces maps a clue-side template to its space (word side bare). splits has train,
    val and, when evaluate is "test", test.
    """
    settings = []
    for template, sp in spaces.items():
        tau = fit_tau(turn_sims(sp, splits["train"]))
        val = float(np.mean(log_probs(turn_sims(sp, splits["val"]), tau)))
        settings.append((val, template, tau))
    val, template, tau = max(settings, key=lambda r: r[0])  # the first template on a tie
    table = turn_sims(spaces[template], splits[evaluate])
    hit, logp = hits(table), log_probs(table, tau)
    data = splits[evaluate]
    return {
        "settings": [
            {"clue": t, "word": "bare", "tau": u, "val_loglik": v} for v, t, u in settings
        ],
        "chosen": {"clue": template, "word": "bare", "tau": tau, "val_loglik": val},
        "split": evaluate,
        "n": len(data),
        "top1": float(np.mean(hit)),
        "loglik": float(np.mean(logp)),
        "turns": [
            {"clue": g.clue, "picked": g.picked, "hit": h, "logp": round(p, 6)}
            for g, h, p in zip(data, hit, logp, strict=True)
        ],
    }


def giver_pool() -> tuple[str, ...]:
    """Clue words that are held-out EAT cues, as in train.codenames_scores."""
    counts = eat_counts()
    return tuple(c for c in clue_words() if c in counts and held_out(c))


def boards(n: int = N_BOARDS, seed: int = BOARD_SEED):
    rng = random.Random(seed)
    return [deal(board_words(), rng) for _ in range(n)]


def giver_clues(sp: Space, tau: float, pool: tuple[str, ...], n: int = N_BOARDS) -> list:
    """The clue and count the expected-value giver picks on each fresh board."""
    give = expected_giver(sp, tau, candidates=pool)
    return [[c.word, c.count] for c in map(give, boards(n))]


def full_scores(
    spaces: dict[str, Space],
    words: list[str],
    scheme: str,
    evaluate: str,
    n_boards: int = N_BOARDS,
) -> dict:
    """Everything `scores.json` holds for one model."""
    splits = {
        s: guesses(s, scheme) for s in ("train", "val", *(("test",) if evaluate == "test" else ()))
    }
    out = score(spaces, splits, evaluate)
    sp = spaces[out["chosen"]["clue"]]
    out["assoc"] = association_top1(sp, association_items(eat_counts(), set(words)))
    if evaluate == "test":
        out["giver"] = {
            "boards": n_boards,
            "seed": BOARD_SEED,
            "clues": giver_clues(sp, out["chosen"]["tau"], giver_pool(), n_boards),
        }
    return out
