"""The experiments behind the README, each writing a JSON file to codenames/results/.

    uv run python -m codenames.study calibrate   # temperatures and templates (~1 min)
    uv run python -m codenames.study crossplay   # giver x guesser expected scores (~10 min)
    uv run python -m codenames.study games       # whole games, Kim et al. style (~10 min)
    uv run python -m codenames.study all

Every random choice is seeded. Models: the four that run here, plus any whose vectors
are in codenames/vectors/ (computed on a runner that can reach Hugging Face).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from codenames.embed import COMPUTED
from codenames.evaluate import TAUS, all_words, guess_loglik, guess_scores
from codenames.game import deal, play
from codenames.norms import cultural_guesses, eat_counts
from codenames.players import (
    EnsembleGiver,
    Space,
    embedding_guesser,
    embedding_weigher,
    expected_found,
    expected_giver,
    expected_value,
    norm_guesser,
    norm_weigher,
    space,
    threshold_giver,
)
from codenames.words import board_words, clue_words

RESULTS = Path(__file__).parent / "results"
LOCAL = ("potion", "wordllama", "granite", "bge")
REMOTE = (
    "Qwen/Qwen3-Embedding-0.6B",
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-0.6b",
    "google/embeddinggemma-2",
)
EPS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def models() -> list[str]:
    """Local models, then the remote ones whose bare vectors are present."""
    have = [m for m in REMOTE if (COMPUTED / f"{m.replace('/', '--')}__bare.npz").exists()]
    return [*LOCAL, *have]


def templates(model: str) -> list[tuple[str, str]]:
    """(clue template, word template) pairs that have vectors for this model."""
    out = [("bare", "bare"), ("related", "bare"), ("query", "bare")]
    if model in LOCAL:
        out.insert(1, ("word", "word"))
    return out


def save(name: str, data) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


def mean_se(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def eat_eps() -> float:
    """The eps that makes the EAT guesser fit the human guesses (train split) best."""
    counts = eat_counts()

    def loglik(eps: float) -> float:
        total = 0.0
        data = cultural_guesses("train")
        for g in data:
            w = norm_weigher(counts, eps)(g.clue, g.words)
            total += math.log(w[g.words.index(g.picked)] / w.sum())
        return total / len(data)

    return max(EPS, key=loglik)


def calibrate() -> dict:
    """For each model: the template pair with the best fit to human guesses on the
    validation split, its temperature (fitted on train), and its test scores."""
    words = all_words()
    out: dict = {}
    for model in models():
        best: dict = {}
        best_val = -math.inf
        for ct, wt in templates(model):
            sp = space(model, ct, wt, words)
            tau = max(TAUS, key=lambda t: guess_loglik(sp, "train", t))
            val = guess_loglik(sp, "val", tau)
            row: dict = {"clue": ct, "word": wt, "tau": tau, "val_loglik": val}
            row["test"] = guess_scores(sp, "test", tau)
            out.setdefault(model, {"settings": []})["settings"].append(row)
            if val > best_val:
                best, best_val = row, val
            print(model, ct, wt, tau, round(val, 3), round(row["test"]["top1"], 3), flush=True)
        out[model]["best"] = best
    eps = eat_eps()
    counts = eat_counts()
    data = cultural_guesses("test")
    hits = 0
    total = 0.0
    for g in data:
        w = norm_weigher(counts, eps)(g.clue, g.words)
        hits += int(g.words[int(np.argmax(w))] == g.picked and w.max() > eps)
        total += math.log(w[g.words.index(g.picked)] / w.sum())
    covered = sum(1 for g in data if g.clue in counts)
    out["eat"] = {"eps": eps, "test_top1": hits / len(data), "test_loglik": total / len(data)}
    out["eat"]["test_clues_in_eat"] = covered
    save("calibrate", out)
    return out


def chosen_spaces(cal: dict) -> dict[str, tuple[Space, float]]:
    words = all_words()
    out = {}
    for model in models():
        b = cal[model]["best"]
        out[model] = (space(model, b["clue"], b["word"], words), b["tau"])
    return out


def crossplay(n_boards: int = 200, seed: int = 0) -> dict:
    """Each giver plans one clue per board under its own Luce guesser; each guesser's
    exact expected score for that clue is then computed. Clues are limited to EAT cues,
    so that the association-count guesser has data for every clue."""
    cal = load("calibrate")
    spaces = chosen_spaces(cal)
    counts = eat_counts()
    pool = tuple(c for c in clue_words() if c in counts)
    weighers = {m: embedding_weigher(sp, tau) for m, (sp, tau) in spaces.items()}
    weighers["eat"] = norm_weigher(counts, cal["eat"]["eps"])
    rng = random.Random(seed)
    boards = [deal(board_words(), rng) for _ in range(n_boards)]
    table = {}
    givers = {m: expected_giver(sp, tau, candidates=pool) for m, (sp, tau) in spaces.items()}
    # Ensemble givers average over every model but one; the held-out model then tests them
    # as a guesser they never planned for.
    for out_model in ("bge", "google/embeddinggemma-2"):
        if out_model not in spaces:
            continue
        members = [spaces[m] for m in spaces if m != out_model]
        givers[f"ensemble-without-{short(out_model)}"] = EnsembleGiver(members, candidates=pool)
    for giver, give in givers.items():
        t = time.perf_counter()
        clues = [give(b) for b in boards]
        row = {}
        for guesser, weigh in weighers.items():
            found = [expected_found(weigh, b, c) for b, c in zip(boards, clues, strict=True)]
            value = [expected_value(weigh, b, c) for b, c in zip(boards, clues, strict=True)]
            row[guesser] = {"found": mean_se(found), "value": mean_se(value)}
        row["mean_count"] = float(np.mean([c.count for c in clues]))
        row["examples"] = [[c.word, c.count] for c in clues[:5]]
        table[giver] = row
        print(
            giver,
            f"{time.perf_counter() - t:.0f} s",
            {g: round(v["found"][0], 3) for g, v in row.items() if isinstance(v, dict)},
            flush=True,
        )
    out = {"n_boards": n_boards, "seed": seed, "pool": len(pool), "table": table}
    save("crossplay", out)
    return out


def games(n_games: int = 50, seed: int = 1, threshold: float = 0.4) -> dict:
    """Whole games between threshold givers and greedy guessers of every model."""
    cal = load("calibrate")
    spaces = chosen_spaces(cal)
    guessers = {m: embedding_guesser(sp) for m, (sp, _) in spaces.items()}
    guessers["eat"] = norm_guesser(eat_counts())
    counts = eat_counts()
    pool = tuple(c for c in clue_words() if c in counts)
    out = {}
    for giver, (sp, _) in spaces.items():
        give = threshold_giver(sp, threshold, candidates=pool)
        row = {}
        for name, guess in guessers.items():
            rng = random.Random(seed)
            results = [play(deal(board_words(), rng), give, guess) for _ in range(n_games)]
            wins = [r for r in results if r.won]
            row[name] = {
                "wins": len(wins),
                "games": n_games,
                "mean_turns_when_won": float(np.mean([r.turns for r in wins])) if wins else None,
            }
        out[giver] = row
        print(giver, {k: v["wins"] for k, v in row.items()}, flush=True)
    save("games", {"n_games": n_games, "seed": seed, "threshold": threshold, "table": out})
    return out


def peaks(n_boards: int = 50, seed: int = 2) -> dict:
    """How often the expected value of a turn, as a function of the count n = 1..4, has
    more than one local maximum, over every shortlisted clue on n_boards boards; and
    the same for 100,000 random weightings of 4 team and 3 other words."""
    from codenames.luce import luce_weights, turn_odds
    from codenames.players import ClueTable, costs_for, shortlist_rows

    def multi_peaked(values: np.ndarray) -> np.ndarray:
        d = np.diff(values, axis=0)  # (3, C): R(n+1) - R(n)
        down_then_up = (d[:-1] < -1e-12) & (d[1:] > 1e-12)
        return down_then_up.any(axis=0)

    cal = load("calibrate")
    spaces = chosen_spaces(cal)
    rng = random.Random(seed)
    boards = [deal(board_words(), rng) for _ in range(n_boards)]
    checked = bad = 0
    for sp, tau in spaces.values():
        for b in boards:
            words = b.unrevealed()
            kinds = tuple(b.kind(w) for w in words)
            tab = ClueTable.build(sp, words, clue_words())
            keep = shortlist_rows(tab.sims, kinds, 300)
            odds = turn_odds(luce_weights(tab.sims[keep], tau), kinds, 4)
            values = np.stack([odds.value(n, costs_for(b)) for n in range(1, 5)])
            checked += values.shape[1]
            bad += int(multi_peaked(values).sum())
    gen = np.random.default_rng(seed)
    kinds = ("T",) * 4 + ("O", "N", "A")
    w = np.exp(gen.normal(0, 3, (100_000, 7)))
    odds = turn_odds(w, kinds, 4)
    random_bad = 0
    for cost in range(1, 10):
        values = np.stack([odds.value(n, {"O": 1, "N": 0, "A": cost}) for n in range(1, 5)])
        random_bad += int(multi_peaked(values).sum())
    out = {
        "boards": n_boards,
        "clues_checked": checked,
        "multi_peaked": bad,
        "random_weightings": 100_000,
        "assassin_costs": list(range(1, 10)),
        "random_multi_peaked": random_bad,
    }
    save("peaks", out)
    print(out)
    return out


def short(model: str) -> str:
    return model.split("/")[-1]


def report() -> None:
    """Print the README's tables from the JSON files in codenames/results/."""
    cal = load("calibrate")
    print(
        "| Model | Template (clue / word) | tau | Top-1 on human guesses (95% CI) | Log-likelihood |"
    )
    print("|---|---|---|---|---|")
    for m, v in cal.items():
        if m == "eat":
            continue
        b = v["best"]
        t = b["test"]
        lo, hi = t["top1_ci"]
        print(
            f"| {short(m)} | {b['clue']} / {b['word']} | {b['tau']} | "
            f"{t['top1']:.3f} ({lo:.3f}-{hi:.3f}) | {t['loglik']:.2f} |"
        )
    e = cal["eat"]
    print(f"| EAT counts | eps {e['eps']} | | {e['test_top1']:.3f} | {e['test_loglik']:.2f} |")
    print()
    print(
        "| Model | "
        + " | ".join(s["clue"] + "/" + s["word"] for s in cal["potion"]["settings"])
        + " |"
    )
    print("|---|" + "---|" * len(cal["potion"]["settings"]))
    for m, v in cal.items():
        if m == "eat":
            continue
        cells = {s["clue"] + "/" + s["word"]: s["test"]["top1"] for s in v["settings"]}
        keys = [s["clue"] + "/" + s["word"] for s in cal["potion"]["settings"]]
        print(
            f"| {short(m)} | "
            + " | ".join(f"{cells[k]:.3f}" if k in cells else "" for k in keys)
            + " |"
        )
    print()
    x = load("crossplay")
    guessers = [g for g in next(iter(x["table"].values())) if g not in ("mean_count", "examples")]
    print(
        "| Giver \\ guesser | "
        + " | ".join(short(g) for g in guessers)
        + " | own - mean of others |"
    )
    print("|---|" + "---|" * (len(guessers) + 1))
    for giver, row in x["table"].items():
        cells = [f"{row[g]['found'][0]:.2f}" for g in guessers]
        if giver in row:
            others = [row[g]["found"][0] for g in guessers if g not in (giver, "eat")]
            gap = f"{row[giver]['found'][0] - np.mean(others):.2f}"
        else:
            gap = ""
        print(f"| {short(giver)} | " + " | ".join(cells) + f" | {gap} |")
    print()
    g = load("games")
    names = list(next(iter(g["table"].values())))
    print("| Giver \\ guesser | " + " | ".join(short(n) for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for giver, row in g["table"].items():
        print(f"| {short(giver)} | " + " | ".join(str(row[n]["wins"]) for n in names) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "what", choices=["calibrate", "crossplay", "games", "peaks", "report", "all"]
    )
    parser.add_argument("--boards", type=int, default=200)
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()
    if args.what in ("calibrate", "all"):
        calibrate()
    if args.what in ("crossplay", "all"):
        crossplay(args.boards)
    if args.what in ("games", "all"):
        games(args.games)
    if args.what in ("peaks", "all"):
        peaks()
    if args.what == "report":
        report()


if __name__ == "__main__":
    main()
