"""Scoring for C6 (experiments/plans/C6.md): every number in results/c6.json.

Each board has the 120 actions of codenames.actions. Every action is scored by its exact
expected turn value (codenames.luce, the costs of players.costs_for) under:

- the seven calibrated embedding guessers other than EmbeddingGemma 2 (the
  **strangers**); a board's stranger value is their mean;
- the potion guesser with the linear map of C1 (`train.py`, game split), one map per
  training seed 1 to 5, rebuilt here and checked against the hits C1 recorded; a
  board's **human-fitted** value is the mean over the five;
- the EAT association guesser (secondary);
- EmbeddingGemma 2 itself, the proposer, which defines the first choice and the rank.

Arms on each board: each model's pick (the first choice when its reply had none), the
**first choice** (highest value under EmbeddingGemma 2), **uniform** (the mean over
all 120 actions, exact), and the **oracles** (the highest stranger value, and the
highest human-fitted value). Differences are paired by board, with 95% percentile
bootstrap intervals over boards (10,000 resamples, seed 0) from codenames.c4score.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from codenames.actions import PROPOSER, expand, read_action_lines, sha256
from codenames.actions import check as check_actions
from codenames.actions import values as action_values
from codenames.c4 import boards, clue_lines
from codenames.c4score import RESAMPLES, bootstrap, holm, rounded, score_crossguess
from codenames.players import Space, embedding_weigher, norm_weigher
from codenames.rerank import ACTIONS, PLAN, action_index, read_choice_lines
from codenames.words import zipf

RESULTS = Path(__file__).parent / "results"
VAL, TEST = 6005, 6004
MAP_SEEDS = (1, 2, 3, 4, 5)
BASE = "first-choice"  # the giver name of the first choice in the crossguess lines


@cache
def guessers():
    """{name: weigher} for the proposer, the strangers, the five potion maps and EAT,
    with the lists of stranger and map names."""
    from codenames.c4score import spaces_with
    from codenames.norms import eat_counts
    from codenames.study import load

    spaces = spaces_with([])
    assert PROPOSER in spaces and len(spaces) == 8, sorted(spaces)
    strangers = [m for m in spaces if m != PROPOSER]
    out = {m: embedding_weigher(*spaces[m]) for m in spaces}
    maps = []
    for seed, (sp, tau) in potion_maps().items():
        out[f"potion map s{seed}"] = embedding_weigher(sp, tau)
        maps.append(f"potion map s{seed}")
    out["eat"] = norm_weigher(eat_counts(), load("calibrate-game")["eat"]["eps"])
    spaces_used = [spaces[m][0] for m in spaces]
    return out, strangers, maps, spaces_used


def potion_maps() -> dict[int, tuple[Space, float]]:
    """C1's potion maps, retrained (seconds each) and checked against the per-turn hits
    and temperature that results/train-potion-game.json recorded for each seed."""
    from codenames.evaluate import all_words, guess_scores
    from codenames.players import space
    from codenames.train import adapted, items, train

    recorded = json.loads((RESULTS / "train-potion-game.json").read_text())
    sp = space("potion", recorded["template"], "bare", all_words())
    out = {}
    for seed in MAP_SEEDS:
        data = items(sp, seed, eat=False, split="train", scheme="game")
        a, _ = train(sp, data, recorded["epochs"], seed=seed)
        tuned = adapted(sp, a, "potion+cultural")
        run = recorded["runs"][str(seed)]["guess"]
        tau = run["tau"]
        again = guess_scores(tuned, "test", tau, "game")
        assert again["hits"] == run["hits"], f"potion map seed {seed} does not reproduce C1"
        out[seed] = (tuned, tau)
    return out


def board_values(board, actions) -> dict[str, np.ndarray]:
    weighers, *_ = guessers()
    return {g: action_values(w, board, actions) for g, w in weighers.items()}


def rank(proposer: np.ndarray, i: int) -> int:
    """1 + the number of actions the proposer values strictly above action i."""
    return int((proposer > proposer[i]).sum()) + 1


def rarity(clue: str) -> float:
    return max(0.0, 3.5 - zipf(clue))


def score_seed(seed: int, clue_lists: list[list[str]], chosen: dict[str, list[dict]]) -> dict:
    """Per board: the arms' actions and values; and the summary over boards."""
    _, strangers, maps, spaces = guessers()
    bs = boards(seed, len(clue_lists))
    order = [PROPOSER, *strangers, *maps, "eat"]
    rows = []
    for i, (b, clues) in enumerate(zip(bs, clue_lists, strict=True)):
        acts = expand(clues)
        check_actions(b, acts, spaces)
        v = board_values(b, acts)
        s = np.mean([v[g] for g in strangers], axis=0)
        h = np.mean([v[g] for g in maps], axis=0)
        first = int(np.argmax(v[PROPOSER]))
        arms = {"first": first, "oracle_s": int(np.argmax(s)), "oracle_h": int(np.argmax(h))}
        why = {}
        for m, picks in chosen.items():
            j = action_index(picks[i], clues)
            why[m] = picks[i]["why"]
            arms[m] = first if j is None else j
        row = {
            "board": i,
            "actions": {a: list(acts[j]) for a, j in arms.items()},
            "value": {a: [s[j], h[j], v["eat"][j]] for a, j in arms.items()},
            "uniform": [float(s.mean()), float(h.mean()), float(v["eat"].mean())],
            "rank": {m: rank(v[PROPOSER], arms[m]) for m in chosen},
            "fallback": {m: w for m, w in why.items() if w},
            "by_guesser": {a: [v[g][j] for g in order] for a, j in arms.items()},
        }
        rows.append(row)
    return {"n": len(bs), "guessers": order, "boards": rows, "summary": summarize(rows, chosen)}


def ratio_bootstrap(num, den, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """mean(num) / mean(den) and its 95% interval, from the resamples of c4score.bootstrap
    (the same generator, seed and draws)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (resamples, len(num)))
    r = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"mean": float(num.mean() / den.mean()), "ci": [float(lo), float(hi)]}


def column(rows, arm: str, k: int) -> np.ndarray:
    if arm == "uniform":
        return np.array([r["uniform"][k] for r in rows])
    return np.array([r["value"][arm][k] for r in rows])


MEASURES = ("strangers", "human", "eat")


def summarize(rows: list[dict], chosen: dict) -> dict:
    out: dict = {"arms": {}}
    first = {k: column(rows, "first", k) for k in range(3)}
    oracle = {0: column(rows, "oracle_s", 0), 1: column(rows, "oracle_h", 1)}
    for arm in ["first", "uniform", "oracle_s", "oracle_h", *chosen]:
        a: dict = {}
        for k, name in enumerate(MEASURES):
            x = column(rows, arm, k)
            a[name] = bootstrap(x)
            if arm != "first":
                a[f"{name}_minus_first"] = bootstrap(x - first[k])
            if k in oracle and arm not in ("first", f"oracle_{'sh'[k]}"):
                a[f"{name}_gap_closed"] = ratio_bootstrap(x - first[k], oracle[k] - first[k])
        if arm not in ("uniform",):
            acts = [tuple(r["actions"][arm]) for r in rows]
            a["mean_count"] = float(np.mean([n for _, n in acts]))
            a["rarity"] = float(np.mean([rarity(c) for c, _ in acts]))
            a["distinct_clues"] = len({c for c, _ in acts})
        if arm in chosen:
            ranks = [r["rank"][arm] for r in rows]
            a["rank_mean"] = float(np.mean(ranks))
            a["rank_median"] = float(np.median(ranks))
            a["is_first"] = float(
                np.mean([r["actions"][arm] == r["actions"]["first"] for r in rows])
            )
            reasons = [r["fallback"].get(arm) for r in rows if r["fallback"].get(arm)]
            a["fallbacks"] = {w: reasons.count(w) for w in sorted(set(reasons))}
        out["arms"][arm] = a
    gap = {k: column(rows, f"oracle_{'sh'[k]}", k) - first[k] for k in (0, 1)}
    out["oracle_gap"] = {MEASURES[k]: bootstrap(gap[k]) for k in (0, 1)}
    return out


def picks_for_crossguess(seed: int, clue_lists, chosen_rows: list[dict], model: str) -> str:
    """Clue lines (codenames.c4 format) for the language-model guessers: the chosen model's
    pick, with its fallback, and the first choice."""
    rows = score_seed_actions(seed, clue_lists, {model: chosen_rows})
    given = {
        seed: {
            model: [{"clue": c, "count": n} for c, n in (r[model] for r in rows)],
            BASE: [{"clue": c, "count": n} for c, n in (r["first"] for r in rows)],
        }
    }
    return clue_lines(given)


def score_seed_actions(seed, clue_lists, chosen) -> list[dict]:
    """Only the first choice and each model's action per board (needs the proposer only)."""
    from codenames.actions import proposer

    sp, tau = proposer()
    w = embedding_weigher(sp, tau)
    out = []
    for b, clues in zip(boards(seed, len(clue_lists)), clue_lists, strict=True):
        acts = expand(clues)
        first = int(np.argmax(action_values(w, b, acts)))
        row = {"first": acts[first]}
        i = len(out)
        for m, picks in chosen.items():
            j = action_index(picks[i], clues)
            row[m] = acts[first if j is None else j]
        out.append(row)
    return out


def main(runs: Path | None, out: Path) -> None:
    """Score the choices files in runs (default results/): c6-choices.txt, and
    c6-crossguess.txt when present."""
    runs = runs or RESULTS
    lists = read_action_lines(ACTIONS.read_text())
    text = (runs / "c6-choices.txt").read_text()
    chosen = read_choice_lines(text)
    result: dict = {
        "plan": "experiments/plans/C6.md",
        "actions_sha256": sha256(ACTIONS.read_text()),
        "choices_sha256": sha256(text),
        "seeds": {},
    }
    for seed in sorted({s for _, s in chosen}):
        by_model = {m: rows for (m, s), rows in chosen.items() if s == seed}
        by_model = {m: by_model[m] for m in PLAN if m in by_model}
        n = len(next(iter(by_model.values())))
        result["seeds"][seed] = score_seed(seed, lists[seed][:n], by_model)
    if VAL in result["seeds"]:
        val = result["seeds"][VAL]["summary"]["arms"]
        models = [m for m in PLAN if m in val]
        best = max(models, key=lambda m: val[m]["strangers"]["mean"])
        result["selected"] = {"model": best, "on": VAL, "by": "mean stranger value"}
        if TEST in result["seeds"] and best in result["seeds"][TEST]["summary"]["arms"]:
            rows = result["seeds"][TEST]["boards"]
            tests: dict[str, dict] = {}
            for k, name in ((0, "h_strangers"), (1, "h_human")):
                d = column(rows, best, k) - column(rows, "first", k)
                tests[name] = {"model": best, "seed": TEST, **bootstrap(d)}
            ps = {k: float(t["p"]) for k, t in tests.items()}
            reject = holm(ps)
            for k, t in tests.items():
                t["holds"] = bool(reject[k] and t["mean"] > 0)
            result["primary"] = tests
    cross = runs / "c6-crossguess.txt"
    if cross.exists():
        result["crossguess"] = score_crossguess(cross.read_text(), base=BASE)
        result["crossguess"]["per_board"] = cross.read_text().splitlines()
    out.write_text(json.dumps(rounded(result, 4), indent=1, sort_keys=True) + "\n")
