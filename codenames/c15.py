"""Experiment C15 (experiments/plans/C15.md): has gpt-oss-120b memorized Cultural Codes?

The same human turns are shown three ways (`contamination.arms`): O, the board as
published; S, its words shuffled; W, shuffled with every word but the person's pick
swapped for a noun of about the same frequency. A guesser hears C4's guesser prompt with
count 1, and agrees with the person when its first board word is the person's pick.

    python -m codenames.c15 guess --models gpt-oss-120b --turns 2000 --cap 0.40 --out OUT
    python -m codenames.c15 guess --models llama-3.3-70b --turns 1000 --cap 0.20 --out OUT
    python -m codenames.c15 repeat --models gpt-oss-120b,llama-3.3-70b --turns 20 \\
        --times 2 --cap 0.03 --out OUT                  # in Hugging Face jobs
    uv run python -m codenames.c15 score               # here, with the word vectors

**Turns.** C1's training games (`splits.json`, scheme game), shuffled by seed 15001, taken
whole until there are at least 2,000 turns; a model given --turns N takes the first games
of that order until it has at least N turns. Training turns, not test turns: C1's test
turns have four confirmatory uses, and memorization is as likely on either split.

**Replies.** One reply per arm and turn; an empty, failed or unpinned reply is sent once
more at the end of the job. A reply that names no board word is not read as a guess.
`guess` writes `guesses.txt`, one line per model, arm and send, with one character per
turn: 1 (agrees), 0 (does not), or the reason there is no answer (e empty, f failed,
u unpinned, n no board word, x stopped by the cap), upper case when the reply reached the
token cap. On W it also writes a line `W-orig` marking turns whose reply named a word of
the original board that the swapped board does not have. Boards stay in the job's output
(the private dataset); only these lines and aggregates are committed.

**Scoring.** EmbeddingGemma 2 (untrained, C1's calibrated templates) and potion with C1's
linear map (five training seeds, the mean of their hits) are computed here on the same
arms. H-memory is, per turn, [model on O - model on W] - [EmbeddingGemma 2 on O - on W],
averaged over the turns the model answered in both arms, with a 95% interval from
resampling games (10,000 draws, seed 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

from codenames import c4
from codenames.contamination import Arm, arms

HERE = Path(__file__).parent
RESULTS = HERE / "results"
GUESSES = RESULTS / "c15-guesses.txt"
RUNS = RESULTS / "c15-runs.json"
OUT = RESULTS / "c15.json"
PLAN = "experiments/plans/C15.md"
TURN_SEED = 15001
N_TURNS, N_LLAMA = 2000, 1000
ARMS = ("O", "S", "W")
GUESSERS = ("gpt-oss-120b", "llama-3.3-70b")
PRIMARY_CONTROL = "google/embeddinggemma-2"
MAP_MODEL, MAP_TEMPLATE, MAP_SEEDS = "potion", "bare", (1, 2, 3, 4, 5)
RESENT = ("e", "f", "u")
DELTA = 0.03  # the plan's equivalence margin
MDE_Z = 2.80


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# The turns and their arms.


@cache
def chosen(n: int = N_TURNS) -> tuple:
    """(turn, game) pairs: C1's training games in the order of seed 15001, whole, until
    there are at least n turns."""
    import random

    from codenames.split import turns

    data = turns("game:train")
    games = sorted({t.game for t in data})
    random.Random(TURN_SEED).shuffle(games)
    by_game: dict[str, list] = {}
    for t in data:
        by_game.setdefault(t.game, []).append(t)
    out: list = []
    for g in games:
        out += by_game[g]
        if len(out) >= n:
            break
    return tuple(out)


def first_games(n: int) -> int:
    """How many of chosen()'s turns the first whole games holding at least n turns cover."""
    ts = chosen()
    count = 0
    for i, t in enumerate(ts):
        if i >= n and t.game != ts[i - 1].game:
            return i
        count = i + 1
    return count


@cache
def all_arms(n: int = N_TURNS) -> tuple[dict[str, Arm], ...]:
    return tuple(
        arms(t.guess.clue, t.guess.words, t.guess.picked, i) for i, t in enumerate(chosen(n))
    )


def pilot_arms(n: int = 20) -> tuple[dict[str, Arm], ...]:
    """The pilot's turns: the n turns after the scored ones, from the next games in the
    same order. Nothing from them is scored."""
    k = len(chosen())
    return all_arms(N_TURNS + 50 + n)[k : k + n]


# Guessing (in a job).

CODES = {"empty": "e", "failed": "f", "unpinned": "u", "noword": "n", "stopped": "x"}


def cell(rec: dict | None, arm: Arm) -> str:
    """One turn's character: 1 or 0, upper case for a reply cut at the cap... see the
    module's docstring."""
    from codenames.c11 import status_of

    st = status_of(rec, arm.words)
    if st not in ("ok", "cut"):
        return CODES[st]
    assert rec is not None
    first = c4.read_ranking(rec["reply"], arm.words)["ranking"][0]
    return (
        ("1" if first == arm.picked else "0")
        if st == "ok"
        else ("H" if first == arm.picked else "M")
    )


def named_original(rec: dict | None, o: Arm, w: Arm) -> str:
    """1 if the reply names a word of the original board that W does not show."""
    if rec is None or not rec.get("reply"):
        return "-"
    gone = set(o.words) - set(w.words)
    said = set(re.findall(r"[a-z]+", rec["reply"].lower()))
    return "1" if said & gone else "0"


def send(hosted, prompts: list[tuple[str, dict]], workers: int) -> list[dict | None]:
    return c4.run_many(hosted.call, prompts, workers)


def guess_model(
    name: str, n: int, meter, out: Path, workers: int, times: int = 0, pilot: bool = False
) -> tuple[list[str], dict]:
    """Every arm of the first n turns (whole games) for one model, or with times > 0 the
    repeat check: the first n turns of each arm sent `times` more times."""
    from codenames.c11 import status_of
    from codenames.hosted import Hosted
    from codenames.llm import guesser_prompt

    k = first_games(n) if not (times or pilot) else n
    turns_ = pilot_arms(n) if pilot else all_arms()[:k]
    hosted = Hosted(c4.MODELS[name], meter, out / f"calls-{name}.jsonl", c4.tokens(name, "guess"))
    w = c4.WORKERS.get(name, workers)
    lines: list[str] = []
    summary: dict = {"turns": k, "max_tokens": c4.tokens(name, "guess"), "arms": {}}
    for arm in ARMS:
        shown = [t[arm] for t in turns_]
        for r in range(1 if not times else times):
            tag = "guess" if not times else f"repeat-{r + 1}"
            jobs = [
                (guesser_prompt(a.clue, 1, a.words), {"part": tag, "arm": arm, "turn": i})
                for i, a in enumerate(shown)
            ]
            recs = send(hosted, jobs, w)
            first = [status_of(x, a.words) for x, a in zip(recs, shown, strict=True)]
            again = [i for i, s in enumerate(first) if CODES.get(s) in RESENT]
            for i, rec in zip(again, send(hosted, [jobs[i] for i in again], w), strict=True):
                recs[i] = rec
            cells = "".join(cell(x, a) for x, a in zip(recs, shown, strict=True))
            lines.append(f"{name} {arm} {0 if not times else r + 1} {cells}")
            if arm == "W" and not times:
                orig = [t["O"] for t in turns_]
                marks = "".join(
                    named_original(x, o, a) for x, o, a in zip(recs, orig, shown, strict=True)
                )
                lines.append(f"{name} W-orig 0 {marks}")
            stats = reply_counts(recs, first, cells, len(again))
            summary["arms"][f"{arm}-{tag}"] = stats
            print(name, arm, tag, stats["final_cells"], f"${meter.spent:.4f}", flush=True)
    return lines, summary


def reply_counts(recs: list[dict | None], first: list[str], cells: str, resent: int) -> dict:
    """First and final statuses, sends, cut replies, latency and who served the replies."""
    done = [r for r in recs if r is not None]
    secs = [r["seconds"] for r in done if isinstance(r.get("seconds"), int | float)]
    return {
        "prompts": len(recs),
        "first_status": dict(Counter(first)),
        "final_cells": dict(sorted(Counter(cells).items())),
        "resent": resent,
        "cut_share": sum(c in "HM" for c in cells) / len(cells) if cells else None,
        "seconds_median": statistics.median(secs) if secs else None,
        "pin_held": dict(Counter(str(r.get("pin_held")) for r in done)),
        "served_providers": dict(
            Counter(str((r.get("served_by") or {}).get("provider")) for r in done)
        ),
    }


def run(
    what: str, models, n: int, times: int, meter, out: Path, workers: int = 8, pilot: bool = False
) -> dict:
    lines: list[str] = []
    summary: dict = {"what": what, "pilot": pilot, "turn_seed": TURN_SEED, "models": {}}
    shown = pilot_arms(n) if pilot else all_arms()[: max(first_games(n), n)]
    boards = [
        {"turn": i, **{a: list(t[a].words) for a in ARMS}, "widened": t["W"].widened}
        for i, t in enumerate(shown)
    ]
    (out / "boards.jsonl").write_text("".join(json.dumps(b) + "\n" for b in boards))
    for name in models:
        got, s = guess_model(name, n, meter, out, workers, times if what == "repeat" else 0, pilot)
        lines += got
        log = out / f"calls-{name}.jsonl"
        s["calls"] = c4.call_stats(log) if log.exists() else {"calls": 0}
        summary["models"][name] = s
    text = "\n".join(lines) + "\n"
    (out / "guesses.txt").write_text(text)
    summary["guesses_sha256"] = sha256(text)
    summary["spent"], summary["cap"] = meter.spent, meter.cap
    (out / "guess.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("guesses.txt sha256", summary["guesses_sha256"], f"spent ${meter.spent:.4f}")
    return summary


# Scoring (here).


def read_lines(text: str) -> dict[tuple[str, str, int], str]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        model, arm, send_, cells = line.split()
        key = (model, arm, int(send_))
        assert key not in out, key
        out[key] = cells
    return out


def hits(cells: str) -> np.ndarray:
    """1 or 0 per turn, nan where there is no answer."""
    return np.array([1.0 if c in "1H" else 0.0 if c in "0M" else np.nan for c in cells])


def embedding_hits(n: int) -> dict[str, dict[str, np.ndarray]]:
    """{guesser: {arm: hits}} for EmbeddingGemma 2 (untrained) and potion with C1's map
    (mean over five training seeds), on the first n turns."""
    from codenames.evaluate import all_words
    from codenames.players import space
    from codenames.study import load
    from codenames.train import adapted, items, train

    cal = load("calibrate-game")
    words = all_words()
    shown = all_arms()[:n]

    def top1(sp) -> dict[str, np.ndarray]:
        return {
            a: np.array(
                [
                    float(t[a].words[int(np.argmax(sp.sims(t[a].clue, t[a].words)))] == t[a].picked)
                    for t in shown
                ]
            )
            for a in ARMS
        }

    b = cal[PRIMARY_CONTROL]["best"]
    out = {PRIMARY_CONTROL: top1(space(PRIMARY_CONTROL, b["clue"], b["word"], words))}
    sp = space(MAP_MODEL, MAP_TEMPLATE, "bare", words)
    per_seed = []
    for seed in MAP_SEEDS:
        a, _ = train(sp, items(sp, seed, eat=False, split="train", scheme="game"), 3, seed=seed)
        per_seed.append(top1(adapted(sp, a, f"{MAP_MODEL}+cultural")))
    out["potion+map"] = {a: np.mean([s[a] for s in per_seed], axis=0) for a in ARMS}
    return out


def wilson(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    n, k, z = len(x), float(x.sum()), 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"mean": p, "ci": [c - h, c + h], "n": n}


def did(model: dict[str, np.ndarray], control: dict[str, np.ndarray], games, other: str) -> dict:
    """[model on O - on `other`] - [control on O - on other], per turn, over the turns the
    model answered in both arms; clustered bootstrap over games."""
    from codenames.stats import cluster_draws, clustered

    d = (model["O"] - model[other]) - (control["O"] - control[other])
    keep = ~np.isnan(d)
    g = [x for x, k in zip(games, keep, strict=True) if k]
    index, counts = cluster_draws(g)
    out = clustered(d[keep], index, counts)
    sizes = Counter(g)
    cluster_means = [float(np.mean(d[keep][index == i])) for i in range(len(sizes))]
    sd = float(np.std(cluster_means, ddof=1))
    out["turns_dropped"] = int((~keep).sum())
    out["games"] = len(sizes)
    out["detectable_effect"] = MDE_Z * sd / np.sqrt(len(sizes))
    out["inside_margin"] = bool(-DELTA < out["ci"][0] and out["ci"][1] < DELTA)
    out["model_drop"] = float(np.nanmean((model["O"] - model[other])[keep]))
    out["control_drop"] = float(np.mean((control["O"] - control[other])[keep]))
    return out


def score(guesses: Path = GUESSES, runs: Path = RUNS) -> dict:
    from codenames.c4score import rounded

    g = read_lines(guesses.read_text())
    games = [t.game for t in chosen()]
    result: dict = {
        "plan": PLAN,
        "guesses_sha256": sha256(guesses.read_text()),
        "turn_seed": TURN_SEED,
        "turns": len(chosen()),
        "games": len(set(games)),
        "margin": DELTA,
    }
    widened = [t["W"].widened for t in all_arms()]
    result["swap"] = {
        "replacements_outside_window": int(sum(widened)),
        "turns_with_one": int(sum(w > 0 for w in widened)),
    }
    emb = embedding_hits(len(chosen()))
    models = {}
    for m in GUESSERS:
        if (m, "O", 0) not in g:
            continue
        n = len(g[(m, "O", 0)])
        mh = {a: hits(g[(m, a, 0)]) for a in ARMS}
        ctrl = {k: {a: v[a][:n] for a in ARMS} for k, v in emb.items()}
        row: dict = {
            "turns": n,
            "agreement": {a: wilson(mh[a]) for a in ARMS},
            "replies": {a: dict(sorted(Counter(g[(m, a, 0)]).items())) for a in ARMS},
            "named_original_word_on_W": int(g[(m, "W-orig", 0)].count("1")),
            "did": {
                f"{other} vs {c}": did(mh, ctrl[c], games[:n], other)
                for other in ("W", "S")
                for c in ctrl
            },
        }
        reps = sorted(s for (mm, a, s) in g if mm == m and a == "O" and s > 0)
        if reps:
            row["repeat"] = {}
            for a in ARMS:
                cols = [g[(m, a, s)] for s in reps]
                k = len(cols[0])
                base = g[(m, a, 0)][:k]
                changed = sum(len({base[i], *(c[i] for c in cols)}) > 1 for i in range(k))
                row["repeat"][a] = {"prompts": k, "sends": 1 + len(reps), "changed": changed}
        models[m] = row
    result["models"] = models
    result["controls"] = {
        k: {a: wilson(v[a]) if k == PRIMARY_CONTROL else {"mean": float(v[a].mean())} for a in ARMS}
        for k, v in emb.items()
    }
    if "gpt-oss-120b" in models:
        result["primary"] = {"h_memory": models["gpt-oss-120b"]["did"][f"W vs {PRIMARY_CONTROL}"]}
    if runs.exists():
        result["runs"] = json.loads(runs.read_text())
    return rounded(result, 4)


def main() -> None:
    from codenames.hosted import Meter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["guess", "repeat", "score", "check"])
    parser.add_argument("--models", default=",".join(GUESSERS))
    parser.add_argument("--turns", type=int, default=N_TURNS)
    parser.add_argument("--times", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--pilot", action="store_true", help="the turns after the scored ones")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "score":
        result = score()
        (args.out or OUT).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result.get("primary"), indent=1))
        return
    if args.what == "check":
        ts = chosen()
        print(len(ts), "turns in", len({t.game for t in ts}), "games; Llama:", first_games(N_LLAMA))
        w = [t["W"].widened for t in all_arms()]
        print("replacements outside the window:", sum(w), "on", sum(x > 0 for x in w), "turns")
        print("median words per board:", statistics.median(len(t.guess.words) for t in ts))
        return
    models = args.models.split(",")
    assert all(m in GUESSERS for m in models), models
    args.out.mkdir(parents=True, exist_ok=True)
    c4.price_models(models, args.out)
    run(
        args.what,
        models,
        args.turns,
        args.times,
        Meter(args.cap),
        args.out,
        args.workers,
        args.pilot,
    )


if __name__ == "__main__":
    main()
