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


# Analysis, on the machine that writes the README, from the jobs' summary.json files.

TEST_RUNS = {"lora": ("lora-s1", "lora-s2", "lora-s3"), "lora+eat": ("lora-eat-s1",)}
MAPS = {"potion map": "train-potion-game", "Qwen3 map": "train-Qwen--Qwen3-Embedding-0.6B-game"}
QWEN = "Qwen/Qwen3-Embedding-0.6B"


def as_hits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s])


def turn_digest(pairs: list[tuple[str, str]]) -> str:
    """A digest of turns' clues and picks in order, so that per-turn hits written on one
    machine can be checked against the turns on another."""
    import hashlib

    return hashlib.sha256("\n".join(f"{c}|{p}" for c, p in pairs).encode()).hexdigest()


def giver_scores(clue_sets: dict[str, list[list]], seed: int = BOARD_SEED) -> dict:
    """Expected team words found on each board, for each giver's clues, as heard by the
    EAT guesser and by each calibrated embedding guesser other than Qwen3 (C1's
    temperatures and templates). A giver with several clue lists (seeds) is averaged per
    board."""
    from codenames.game import Clue
    from codenames.players import embedding_weigher, expected_found, norm_weigher
    from codenames.study import chosen_spaces, load

    cal = load("calibrate-game")
    weighers = {
        m: embedding_weigher(sp, tau) for m, (sp, tau) in chosen_spaces(cal).items() if m != QWEN
    }
    weighers["eat"] = norm_weigher(eat_counts(), cal["eat"]["eps"])
    n = len(next(iter(clue_sets.values()))[0])
    bs = boards(n, seed)
    out: dict = {}
    for giver, lists in clue_sets.items():
        out[giver] = {
            g: np.mean(
                [
                    [expected_found(w, b, Clue(c, k)) for b, (c, k) in zip(bs, clues, strict=True)]
                    for clues in lists
                ],
                axis=0,
            )
            for g, w in weighers.items()
        }
    return out


def analyze(summaries: dict[str, dict], lr_choice: dict) -> dict:
    """The primary and secondary metrics of experiments/plans/C2.md."""
    from codenames.split import turns
    from codenames.stats import cluster_draws, clustered
    from codenames.study import load

    test = turns("game:test")
    data = tuple(t.guess for t in test)
    index, counts = cluster_draws([t.game for t in test])
    digest = turn_digest([(g.clue, g.picked) for g in data])
    for name, s in summaries.items():
        for key in ("before", "after"):
            assert s[key]["turns_sha256"] == digest, (name, key)
            assert s[key]["split"] == "test" and s["config"]["scheme"] == "game"

    hits: dict[str, np.ndarray] = {}
    by_seed: dict[str, list[float]] = {}
    for arm, name in MAPS.items():
        each = [as_hits(r["guess"]["hits"]) for r in load(name)["runs"].values()]
        hits[arm], by_seed[arm] = np.mean(each, axis=0), [float(h.mean()) for h in each]
    hits["Qwen3 untrained"] = as_hits(summaries["lora-s1"]["before"]["hits"])
    by_seed["Qwen3 untrained"] = [
        float(as_hits(summaries[r]["before"]["hits"]).mean()) for r in TEST_RUNS["lora"]
    ]
    for arm, runs in TEST_RUNS.items():
        each = [as_hits(summaries[r]["after"]["hits"]) for r in runs]
        hits[arm], by_seed[arm] = np.mean(each, axis=0), [float(h.mean()) for h in each]

    arms = {}
    for arm in ("potion map", "Qwen3 untrained", "Qwen3 map", "lora", "lora+eat"):
        arms[arm] = {"top1": clustered(hits[arm], index, counts), "top1_by_seed": by_seed[arm]}
    for arm, runs, key in (
        ("Qwen3 untrained", ("lora-s1",), "before"),
        ("lora", TEST_RUNS["lora"], "after"),
        ("lora+eat", TEST_RUNS["lora+eat"], "after"),
    ):
        rs = [summaries[r][key] for r in runs]
        arms[arm]["loglik_by_seed"] = [r["loglik"] for r in rs]
        arms[arm]["assoc_top1_by_seed"] = [r["assoc"]["top1"] for r in rs]
        arms[arm]["chosen"] = [r["chosen"] for r in rs]

    def diff(a: str, b: str) -> dict:
        return clustered(hits[a] - hits[b], index, counts)

    clue_sets = {
        "Qwen3 untrained": [summaries["lora-s1"]["before"]["giver"]["clues"]],
        "lora": [summaries[r]["after"]["giver"]["clues"] for r in TEST_RUNS["lora"]],
    }
    found = giver_scores(clue_sets)
    giver = {}
    for g in found["lora"]:
        d = found["lora"][g] - found["Qwen3 untrained"][g]
        b_index, b_counts = cluster_draws([str(i) for i in range(len(d))])
        giver[g] = {
            "untrained": float(found["Qwen3 untrained"][g].mean()),
            "lora": float(found["lora"][g].mean()),
            "lora_minus_untrained": clustered(d, b_index, b_counts),
        }
    return {
        "plan": "experiments/plans/C2.md",
        "turns": len(data),
        "games": len({t.game for t in test}),
        "learning_rate": lr_choice,
        "runs": {r: summaries[r]["config"] for r in summaries},
        "primary": {
            "metric": "top-1 of LoRA (arm 4, mean of 3 seeds) minus the potion map "
            "(arm 1, mean of 5 seeds), paired by turn",
            **diff("lora", "potion map"),
        },
        "lora_minus_qwen3_map": diff("lora", "Qwen3 map"),
        "lora_eat_minus_lora": diff("lora+eat", "lora"),
        "lora_minus_untrained": diff("lora", "Qwen3 untrained"),
        "arms": arms,
        "giver": {"boards": N_BOARDS, "seed": BOARD_SEED, "guessers": giver},
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="C2's tables from the jobs' summary.json files")
    parser.add_argument("--runs", type=Path, required=True, help="folder of <run>/summary.json")
    parser.add_argument("--lr", type=Path, required=True, help="the learning-rate choice (JSON)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "c2.json")
    args = parser.parse_args()
    names = [r for runs in TEST_RUNS.values() for r in runs]
    summaries = {r: json.loads((args.runs / r / "summary.json").read_text()) for r in names}
    out = analyze(summaries, json.loads(args.lr.read_text()))
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    p = out["primary"]
    print(f"LoRA minus potion map: {p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
