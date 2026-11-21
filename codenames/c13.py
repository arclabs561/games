"""Experiment C13 (experiments/plans/C13.md): C3's trained givers and the count-1 rule,
scored by language-model guessers.

    uv run python -m codenames.c13 lists      # here, with the word vectors
    python -m codenames.c13 guess --models gpt-oss-120b --seed 13004 --boards 0-400 \\
        --cap 0.30 --out OUT                   # in a Hugging Face job
    python -m codenames.c13 repeat --models gpt-oss-120b,llama-3.3-70b --seed 13004 \\
        --boards 0-2 --times 2 --cap 0.02 --out OUT
    uv run python -m codenames.c13 score      # here

`lists` deals the pilot boards (seed 13999) and the test boards (seed 13004) and, for each
of C3's two givers on its headline task (potion with bge held out, Qwen3-Embedding-0.6B
with EmbeddingGemma 2 held out), picks the action of every arm among C3's 30 shortlisted
clues times counts 1 to 4 (`rl.actions`): start, self-play and population for each of the
five training seeds (theta from results/c3.json, greedy, with the feature scale of that
seed's 300 training boards, as `diagnose_c3` does), and the count-1 rule
(`cautious.rule`). It writes `results/c13-picks.txt` (clue lines) and
`results/c13-lists.json` (each pick's exact value under the task's held-out guesser and
under EAT).

`guess` and `repeat` send C4's guesser prompt with the action's count, once per distinct
(board, clue, count), and play the turn out (`c11.send_all`, `c11.cell`); `score` reads
the committed lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from codenames import c4

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PICKS = RESULTS / "c13-picks.txt"
LISTS = RESULTS / "c13-lists.json"
GUESSES = RESULTS / "c13-guesses.txt"
RUNS = RESULTS / "c13-runs.json"
OUT = RESULTS / "c13.json"
PLAN = "experiments/plans/C13.md"
PILOT, TEST = 13999, 13004
N_PILOT, N_TEST = 10, 400
TASKS = {"potion": "bge", "Qwen/Qwen3-Embedding-0.6B": "google/embeddinggemma-2"}
SEEDS = (1, 2, 3, 4, 5)
TRAINED = ("self", "population")
GUESSERS = ("gpt-oss-120b", "llama-3.3-70b")
SHORTLIST, TRAIN_BOARDS = 30, 300
RESAMPLES = 10_000
DELTA = 0.10  # H-rule's equivalence margin
HOLM_Z, MDE_Z = 3.08, 2.80


def short(giver: str) -> str:
    return giver.split("/")[-1]


def arm_names(giver: str) -> list[str]:
    g = short(giver)
    return [f"{g}|start", f"{g}|count1"] + [f"{g}|{p}|{s}" for p in TRAINED for s in SEEDS]


def all_arms() -> list[str]:
    return [a for g in TASKS for a in arm_names(g)]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Picks (lists, here).


def thetas() -> dict[tuple[str, int, str], np.ndarray]:
    runs = json.loads((RESULTS / "c3.json").read_text())["theta"]["runs"]
    return {
        (r["giver"], r["seed"], p): np.array(r[p])
        for r in runs
        if TASKS.get(r["giver"]) == r["held_out"]
        for p in ("start", *TRAINED)
    }


def picks_for(giver: str, boards, scales: dict[int, tuple], th: dict, sp, tau, pool) -> list[dict]:
    """Per board: {arm: (clue, count)} and the actions' features, for one giver."""
    from codenames import rl
    from codenames.cautious import rule
    from codenames.diagnose_c3 import pick

    out = []
    for b in boards:
        clues, phi = rl.actions(sp, tau, b, pool, SHORTLIST)
        row = {}
        mu, sd = scales[SEEDS[0]]
        row[f"{short(giver)}|start"] = clues[pick(th[(giver, SEEDS[0], "start")], mu, sd, phi)]
        row[f"{short(giver)}|count1"] = clues[rule(phi)]
        for p in TRAINED:
            for s in SEEDS:
                mu, sd = scales[s]
                row[f"{short(giver)}|{p}|{s}"] = clues[pick(th[(giver, s, p)], mu, sd, phi)]
        out.append(row)
    return out


def setting():
    from codenames.norms import eat_counts
    from codenames.players import embedding_weigher, norm_weigher
    from codenames.study import chosen_spaces, load
    from codenames.words import clue_words

    cal = load("calibrate")
    spaces = chosen_spaces(cal)
    counts = eat_counts()
    pool = tuple(c for c in clue_words() if c in counts)
    eat = norm_weigher(counts, cal["eat"]["eps"])
    return spaces, pool, eat, embedding_weigher


def scales_for(giver: str, spaces, pool) -> dict[int, tuple]:
    from codenames import rl
    from codenames.diagnose_c3 import boards

    sp, tau = spaces[giver]
    return {
        s: rl.feature_scale(
            [(rl.actions(sp, tau, b, pool, SHORTLIST)[1], None) for b in boards(s, TRAIN_BOARDS)]
        )
        for s in SEEDS
    }


def check_c3(n: int = 3) -> dict:
    """The picks on C3's first n test boards equal the committed regenerated picks
    (results/diagnostics/c3-picks.txt, whose values match C3's per-board values)."""
    from codenames.c3 import TEST_SEED
    from codenames.diagnose_c3 import boards, read_picks

    committed = read_picks((RESULTS / "diagnostics" / "c3-picks.txt").read_text())
    spaces, pool, _, _ = setting()
    th = thetas()
    for giver, held in TASKS.items():
        sp, tau = spaces[giver]
        got = picks_for(
            giver, boards(TEST_SEED, n), scales_for(giver, spaces, pool), th, sp, tau, pool
        )
        for p in ("start", *TRAINED):
            for s in SEEDS:
                want = committed[(short(giver), short(held), s, p)][:n]
                key = f"{short(giver)}|{p}|{s}" if p != "start" else f"{short(giver)}|start"
                if p == "start" and s != SEEDS[0]:
                    continue
                assert [tuple(r[key]) for r in got] == [tuple(x) for x in want], (giver, p, s)
    return {"c3_test_boards": n, "passed": True}


def lists() -> tuple[str, str]:
    from codenames import rl

    spaces, pool, eat, weigher = setting()
    th = thetas()
    checks = check_c3()
    given: dict[int, dict[str, list[dict]]] = {}
    values: list[dict] = []
    for seed, n in ((PILOT, N_PILOT), (TEST, N_TEST)):
        bs = c4.boards(seed, n)
        rows: list[dict] = [{"seed": seed, "board": i, "values": {}} for i in range(n)]
        for giver, held in TASKS.items():
            sp, tau = spaces[giver]
            scales = scales_for(giver, spaces, pool)
            got = picks_for(giver, bs, scales, th, sp, tau, pool)
            held_w = weigher(*spaces[held])
            for i, (b, r) in enumerate(zip(bs, got, strict=True)):
                acts = list(r.items())
                v_held = rl.rewards(held_w, b, [a for _, a in acts])
                v_eat = rl.rewards(eat, b, [a for _, a in acts])
                for (name, (c, k)), vh, ve in zip(acts, v_held, v_eat, strict=True):
                    given.setdefault(seed, {}).setdefault(name, []).append(
                        {"clue": c, "count": int(k)}
                    )
                    rows[i]["values"][name] = {"held_out": float(vh), "eat": float(ve)}
            print(seed, giver, "done", flush=True)
        values += rows
    head = {"plan": PLAN, "tasks": TASKS, "seeds": list(SEEDS), "checks": checks}
    text = (
        "{\n"
        + "".join(f'"{k}": {json.dumps(v, sort_keys=True)},\n' for k, v in head.items())
        + '"boards": [\n'
        + ",\n".join(json.dumps(r, sort_keys=True) for r in values)
        + "\n]}\n"
    )
    return c4.clue_lines(given), text


# Guessing (in a job).


def run(
    models, picks: str, seed: int, boards: range, times: int, meter, out: Path, workers: int = 8
) -> dict:
    """Every distinct prompt of every arm on the boards, once (times = 0), or the repeat
    check (times > 0). Writes guesses.txt and guess.json."""
    from codenames import c11
    from codenames.hosted import Hosted

    lines: list[str] = []
    summary: dict = {"seed": seed, "boards": [boards.start, boards.stop], "models": {}}
    by_arm = c11.targets(picks, seed, boards, all_arms())
    keys = {k: b for rows in by_arm.values() for k, b in rows}
    for name in models:
        hosted = Hosted(
            c4.MODELS[name], meter, out / f"calls-{name}.jsonl", c4.tokens(name, "guess")
        )
        w = c4.WORKERS.get(name, workers)
        s: dict = {"max_tokens": c4.tokens(name, "guess"), "distinct_prompts": len(keys)}
        for r in range(times + 1 if times else 1):
            send_no = r + 1 if times else 0
            recs = c11.send_all(hosted, keys, w, "guess" if not times else f"repeat-{send_no}")
            lines += c11.guess_lines(name, by_arm, recs, send_no)
            s[f"send-{send_no}"] = c11.reply_stats(list(recs.values()), list(recs), keys)
            print(
                name,
                send_no,
                s[f"send-{send_no}"]["final_status"],
                f"${meter.spent:.4f}",
                flush=True,
            )
            if times and r + 1 == times:
                break
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


# Inputs from the private dataset (in a job), for files that are not committed.

DATASET = "arclabs561/games-runs"


def fetch_text(spec: str) -> str:
    """The text of a committed path, or of `hf:<path>` files in the private dataset
    (several, comma-separated, joined in order)."""
    if not spec.startswith("hf:"):
        return Path(spec).read_text()
    import os

    from huggingface_hub import hf_hub_download

    parts = []
    for item in spec.removeprefix("hf:").split(","):
        item = item.removeprefix("hf:")
        local = hf_hub_download(
            DATASET, item, repo_type="dataset", token=os.environ.get("HF_TOKEN")
        )
        parts.append(Path(local).read_text())
    return "".join(p if p.endswith("\n") else p + "\n" for p in parts)


# Scoring (here).


def played(g, seed: int, send: int = 0) -> dict[tuple[str, str], np.ndarray]:
    from codenames.c11 import column

    return {
        (a, m): column(g[(a, m, seed, send)])
        for a in all_arms()
        for m in GUESSERS
        if (a, m, seed, send) in g
    }


def arm_value(v: dict, arm: str) -> np.ndarray:
    """Per board: the mean over the two guessers (nan if either did not answer)."""
    return np.mean([v[(arm, m)] for m in GUESSERS], axis=0)


def stratified(
    v: dict, boards: np.ndarray, rng, resamples: int = RESAMPLES
) -> dict[str, np.ndarray]:
    """Bootstrap replicates of H-self and H-rule: boards resampled (shared by every arm and
    giver), and within each giver its five seeds resampled, population and self-play of
    one seed as a pair."""
    n = len(boards)
    per_giver = {}
    for giver in TASKS:
        g = short(giver)
        pop = np.stack([arm_value(v, f"{g}|population|{s}")[boards] for s in SEEDS])  # (5, n)
        slf = np.stack([arm_value(v, f"{g}|self|{s}")[boards] for s in SEEDS])
        c1 = arm_value(v, f"{g}|count1")[boards]
        per_giver[g] = (pop, slf, c1)
    out = {"h_self": np.empty(resamples), "h_rule": np.empty(resamples)}
    for r in range(resamples):
        b = rng.integers(0, n, n)
        hs, hr = [], []
        for pop, slf, c1 in per_giver.values():
            s = rng.integers(0, len(SEEDS), len(SEEDS))
            p = pop[s][:, b].mean()
            hs.append(p - slf[s][:, b].mean())
            hr.append(p - c1[b].mean())
        out["h_self"][r], out["h_rule"][r] = np.mean(hs), np.mean(hr)
    return out


def point(v: dict, keep: np.ndarray) -> dict[str, np.ndarray]:
    """Per board: H-self's and H-rule's differences, averaged over seeds and givers."""
    hs, hr = [], []
    for giver in TASKS:
        g = short(giver)
        pop = np.mean([arm_value(v, f"{g}|population|{s}") for s in SEEDS], axis=0)
        slf = np.mean([arm_value(v, f"{g}|self|{s}") for s in SEEDS], axis=0)
        hs.append(pop - slf)
        hr.append(pop - arm_value(v, f"{g}|count1"))
    return {"h_self": np.mean(hs, axis=0)[keep], "h_rule": np.mean(hr, axis=0)[keep]}


def summarize(d: np.ndarray, reps: np.ndarray) -> dict:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    mean = float(d.mean())
    wrong = float(np.mean(reps <= 0)) if mean > 0 else float(np.mean(reps >= 0))
    sd = float(np.std(d, ddof=1))
    return {
        "mean": mean,
        "ci": [float(lo), float(hi)],
        "p": min(1.0, 2 * wrong),
        "n": len(d),
        "sd": sd,
        "mde_at_0.05": MDE_Z * sd / np.sqrt(len(d)),
        "mde_at_holm": HOLM_Z * sd / np.sqrt(len(d)),
    }


def score(lists_path: Path = LISTS, guesses: str = str(GUESSES), runs: Path = RUNS) -> dict:
    from codenames.c4score import bootstrap, holm, rounded
    from codenames.c11 import column, read_guesses, reply_counts

    data = json.loads(lists_path.read_text())
    text = fetch_text(guesses)
    g = read_guesses(text)
    rows = [r for r in data["boards"] if r["seed"] == TEST]
    v = played(g, TEST)
    every = np.stack([v[k] for k in v])
    keep = ~np.isnan(every).any(axis=0)
    idx = np.flatnonzero(keep)
    d = point(v, keep)
    reps = stratified(v, idx, np.random.default_rng(0))
    res = {h: summarize(d[h], reps[h]) for h in d}
    reject = holm({h: res[h]["p"] for h in res})
    for h in res:
        res[h]["holm_rejects"] = bool(reject[h])
    res["h_rule"]["inside_margin"] = bool(
        -DELTA < res["h_rule"]["ci"][0] and res["h_rule"]["ci"][1] < DELTA
    )
    exact = {
        m: {a: np.array([r["values"][a][m] for r in rows]) for a in all_arms()}
        for m in ("held_out", "eat")
    }
    sec: dict = {}
    for giver in TASKS:
        gname = short(giver)

        by: dict = {}
        for m in GUESSERS:
            vm = {a: v[(a, m)] for a in all_arms()}
            by[m] = {
                k: bootstrap(mean_arm(vm, gname, k)[keep]) for k in ("start", "count1", *TRAINED)
            }
            by[m]["population_minus_self"] = bootstrap(
                (mean_arm(vm, gname, "population") - mean_arm(vm, gname, "self"))[keep]
            )
            by[m]["population_minus_count1"] = bootstrap(
                (mean_arm(vm, gname, "population") - mean_arm(vm, gname, "count1"))[keep]
            )
        for m in ("held_out", "eat"):
            em = exact[m]
            by[m] = {k: bootstrap(mean_arm(em, gname, k)) for k in ("start", "count1", *TRAINED)}
            by[m]["population_minus_self"] = bootstrap(
                mean_arm(em, gname, "population") - mean_arm(em, gname, "self")
            )
            by[m]["population_minus_count1"] = bootstrap(
                mean_arm(em, gname, "population") - mean_arm(em, gname, "count1")
            )
        given = c4.read_clue_lines(PICKS.read_text())
        by["mean_count"] = {
            k: float(
                np.mean(
                    [
                        c
                        for a in arm_names(giver)
                        if a.split("|")[1] == k
                        for _, c, _ in given[(a, TEST)]
                    ]
                )
            )
            for k in ("start", "count1", *TRAINED)
        }
        sec[gname] = by
    sec["replies"] = {
        m: reply_counts([c for a in all_arms() for c in g[(a, m, TEST, 0)]]) for m in GUESSERS
    }
    sec["distinct_actions_per_board"] = float(
        np.mean(
            [len({tuple(given[(a, TEST)][i][:2]) for a in all_arms()}) for i in range(len(rows))]
        )
    )
    result: dict = {
        "plan": PLAN,
        "seed": TEST,
        "lists_sha256": sha256(lists_path.read_text()),
        "guesses_sha256": sha256(text),
        "guesses": guesses,
        "boards_kept": int(keep.sum()),
        "boards_dropped": int((~keep).sum()),
        "primary": res,
        "margin_h_rule": DELTA,
        "secondary": sec,
        "pilot_replies": {
            m: reply_counts(
                [c for (a, mm, sd, s), cs in g.items() if mm == m and sd == PILOT for c in cs]
            )
            for m in GUESSERS
        },
    }
    reps_keys = sorted({s for (_, _, sd, s) in g if sd == TEST and s > 0})
    if reps_keys:
        rep: dict = {}
        for m in GUESSERS:
            changed = total = 0
            for a in all_arms():
                if (a, m, TEST, reps_keys[0]) not in g:
                    continue
                cols = [column(g[(a, m, TEST, s)]) for s in reps_keys]
                base = column(g[(a, m, TEST, 0)])[: len(cols[0])]
                for i in range(len(cols[0])):
                    vals = [base[i], *(c[i] for c in cols)]
                    if any(np.isnan(vals)):
                        continue
                    total += 1
                    changed += len(set(vals)) > 1
            rep[m] = {"answered_every_send": total, "value_changed": changed}
        result["repeat"] = rep
    if runs.exists():
        result["runs"] = json.loads(runs.read_text())
    return rounded(result, 4)


def mean_arm(src: dict, gname: str, kind: str) -> np.ndarray:
    """An arm's per-board values; for a trained arm, the mean over the five seeds."""
    if kind in TRAINED:
        return np.mean([src[f"{gname}|{kind}|{s}"] for s in SEEDS], axis=0)
    return src[f"{gname}|{kind}"]


def main() -> None:
    from codenames.c11 import parse_range
    from codenames.hosted import Meter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["lists", "guess", "repeat", "score"])
    parser.add_argument("--models", default=",".join(GUESSERS))
    parser.add_argument("--seed", type=int, default=TEST)
    parser.add_argument("--boards", type=parse_range)
    parser.add_argument("--times", type=int, default=2)
    parser.add_argument("--picks", type=Path, default=PICKS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--guesses", default=str(GUESSES), help="score: a file, or hf:path,...")
    args = parser.parse_args()
    if args.what == "lists":
        picks, text = lists()
        PICKS.write_text(picks)
        LISTS.write_text(text)
        print(PICKS, sha256(picks), LISTS, sha256(text))
        return
    if args.what == "score":
        result = score(guesses=args.guesses)
        if args.out and args.out.suffix != ".json":
            args.out.mkdir(parents=True, exist_ok=True)
            args.out = args.out / OUT.name
        (args.out or OUT).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result["primary"], indent=1))
        return
    models = args.models.split(",")
    assert all(m in GUESSERS for m in models), models
    args.out.mkdir(parents=True, exist_ok=True)
    c4.price_models(models, args.out)
    times = args.times if args.what == "repeat" else 0
    run(
        models,
        args.picks.read_text(),
        args.seed,
        args.boards,
        times,
        Meter(args.cap),
        args.out,
        args.workers,
    )


if __name__ == "__main__":
    main()
