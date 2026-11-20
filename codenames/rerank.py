"""Experiment C6 (experiments/plans/C6.md): a hosted language model picks one of the legal
clues that an embedding giver proposes.

The 120 actions of each board (30 clues times counts 1 to 4) come from
codenames.actions, which needs word vectors; they are computed here and committed as
`results/c6-actions.txt`. The parts that call models run inside a Hugging Face Job
(experiments/hfrun.py) and read that file:

    python -m codenames.rerank choose --models all --seed 6005 --n 50 --repeat 20 --cap 0.2 --out OUT
    python -m codenames.rerank picks --models qwen3.8-27b --seed 6004 --out c6-picks-6004.txt  # here
    python -m codenames.rerank crossguess --clues codenames/results/c6-picks-6004.txt --seed 6004 --cap 0.3 --out OUT

`choose` writes choose-<model>-<seed>.json (every reply and the action read from it),
calls-<model>.jsonl (every request) and choices.txt (one line per model and seed, all
that scoring needs). Scoring needs the vectors and runs where they are:

    python -m codenames.rerank score --runs DIR --out codenames/results/c6.json
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from codenames import c4
from codenames.actions import COUNTS, read_action_lines, sha256
from codenames.game import Board
from codenames.hosted import Hosted, Meter, Model
from codenames.llm import giver_prompt

HERE = Path(__file__).parent
ACTIONS = HERE / "results" / "c6-actions.txt"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
# The plan's six choosers, in its order. Each is pinned to one provider that the Hub
# listed as live for it on 2026-11-19.
MODELS = {
    "gpt-oss-120b": c4.MODELS["gpt-oss-120b"],
    "llama-3.3-70b": c4.MODELS["llama-3.3-70b"],
    "qwen3-32b": c4.MODELS["qwen3-32b"],
    "deepseek-v3.2": c4.MODELS["deepseek-v3.2"],
    "qwen3.8-27b": Model("Qwen/Qwen3.8-27B", "deepinfra", NOTHINK),
    "gemma-4-31b": Model("google/gemma-4-31B-it", "deepinfra", NOTHINK),
}
PLAN = tuple(MODELS)
GUESSERS = ["llama-3.3-70b", "deepseek-v3.2", "gpt-oss-120b"]  # C4's language-model guessers
TOKENS = {"gpt-oss-120b": 1024}  # its reasoning counts against max_tokens (C4)
CHOOSER_TOKENS = 64
WORKERS = {"qwen3-32b": 2}  # deepinfra answered Qwen3-32B with 429 at 8 (C4)
WORD, NUMBER = re.compile(r"[A-Za-z]+"), re.compile(r"\d+")
# Gemma 4 writes an empty thought channel before its answer when thinking is off.
CHANNEL = re.compile(r"<\|channel>.*?<channel\|>", re.S)


def rerank_prompt(board: Board, clues: list[str]) -> str:
    """The giver prompt of codenames.llm, then the clues, numbered, in the list's order."""
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(clues, 1))
    return (
        giver_prompt(board)
        + "\n\nChoose your clue from this list:\n"
        + listed
        + f"\n\nAnswer on one line as: CLUE NUMBER, with a clue from the list and a number "
        f"from {COUNTS[0]} to {COUNTS[-1]}."
    )


def read_choice(reply: str, clues: list[str], finish: str | None = None) -> dict:
    """The action in a reply, read from its last line that names a listed clue with a
    number after it: that clue, and the first number after it. Otherwise the reason
    there is none: empty, truncated (the provider stopped the reply at its token cap),
    unlisted (no line names a listed clue followed by a number), or a count outside 1
    to 4."""
    text = CHANNEL.sub("", reply)
    if not text.strip():
        return {"clue": None, "count": None, "why": "empty"}
    if finish == "length":
        return {"clue": None, "count": None, "why": "truncated"}
    listed = set(clues)
    for line in reversed(text.splitlines()):
        for m in WORD.finditer(line):
            word = m[0].lower()
            number = NUMBER.search(line, m.end())
            if word in listed and number:
                n = int(number[0])
                if n in COUNTS:
                    return {"clue": word, "count": n, "why": None}
                why = "count out of range"
                return {"clue": None, "count": None, "why": why, "said": [word, n]}
    return {"clue": None, "count": None, "why": "unlisted"}


def action_index(row: dict, clues: list[str]) -> int | None:
    """Position of the chosen action in actions.expand(clues), or None for no choice."""
    if not row.get("clue"):
        return None
    return clues.index(row["clue"]) * len(COUNTS) + COUNTS.index(row["count"])


def tokens(name: str) -> int:
    return TOKENS.get(name, CHOOSER_TOKENS)


def choose_one(name, seed, bs, lists, meter, out, workers) -> list[dict]:
    hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name))
    jobs = [
        (rerank_prompt(b, c), {"part": "choose", "seed": seed, "board": i})
        for i, (b, c) in enumerate(zip(bs, lists, strict=True))
    ]
    records = c4.run_many(hosted.call, jobs, WORKERS.get(name, workers))
    rows = []
    for i, (rec, c) in enumerate(zip(records, lists, strict=True)):
        if rec is None:
            rows.append({"board": i, "stopped": True})
            continue
        choice = read_choice(rec["reply"], c, rec.get("finish"))
        rows.append({"board": i, "reply": rec["reply"], "finish": rec.get("finish"), **choice})
    record = {"model": name, **vars(MODELS[name]), "seed": seed, "rows": rows}
    (out / f"choose-{name}-{seed}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(name, seed, Counter(r.get("why") for r in rows), f"${meter.spent:.4f}", flush=True)
    return rows


def choose(models, seed, n, lists, meter, out, workers=8) -> dict[str, list[dict]]:
    """Each model picks one action on each of n boards; the models run side by side."""
    bs = c4.boards(seed, n)
    assert len(lists) >= n, (seed, len(lists), n)
    with ThreadPoolExecutor(len(models)) as pool:
        futures = {
            m: pool.submit(choose_one, m, seed, bs, lists[:n], meter, out, workers) for m in models
        }
        return {m: f.result() for m, f in futures.items()}


def repeat(name, seed, bs, lists, rows, meter, out, times: int = 2) -> dict:
    """Send these boards' prompts `times` more times; count the prompts whose replies
    were not all the same as the first."""
    hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name))
    jobs = [
        (rerank_prompt(b, c), {"part": "repeat", "seed": seed, "board": i})
        for i, (b, c) in enumerate(zip(bs, lists, strict=True))
        for _ in range(times)
    ]
    recs = c4.run_many(hosted.call, jobs, WORKERS.get(name, 4))
    replies = [r and r["reply"] for r in recs]
    finishes = [r and r.get("finish") for r in recs]
    changed = sum(
        1
        for i in range(len(bs))
        if any(
            x is not None and x != rows[i].get("reply")
            for x in replies[i * times : (i + 1) * times]
        )
    )

    def action(i, reply, finish):
        r = read_choice(reply, lists[i], finish)
        return r["clue"], r["count"], r["why"]

    changed_action = sum(
        1
        for i in range(len(bs))
        if any(
            x is not None and action(i, x, f) != action(i, rows[i]["reply"], rows[i].get("finish"))
            for x, f in zip(
                replies[i * times : (i + 1) * times],
                finishes[i * times : (i + 1) * times],
                strict=True,
            )
        )
    )
    return {
        "prompts": len(bs),
        "times": times,
        "sent": sum(1 for r in recs if r),
        "changed": changed,
        "changed_action": changed_action,
    }


def choice_lines(chosen: dict[int, dict[str, list[dict]]]) -> str:
    """One line per model and seed: clue/count per board, or -/reason."""
    lines = []
    for seed, by_model in chosen.items():
        for name, rows in by_model.items():
            cells = [
                f"{r['clue']}/{r['count']}"
                if r.get("clue")
                else "-/" + ("stopped" if r.get("stopped") else r["why"].replace(" ", "-"))
                for r in rows
            ]
            lines.append(" ".join([name, str(seed), *cells]))
    return "\n".join(lines) + "\n"


def read_choice_lines(text: str) -> dict[tuple[str, int], list[dict]]:
    """The inverse of choice_lines: {(model, seed): [{clue, count, why}] per board}."""
    out = {}
    for line in text.splitlines():
        name, seed, *cells = line.split()
        rows = []
        for cell in cells:
            word, rest = cell.split("/")
            if word == "-":
                rows.append({"clue": None, "count": None, "why": rest.replace("-", " ")})
            else:
                rows.append({"clue": word, "count": int(rest), "why": None})
        out[(name, int(seed))] = rows
    return out


def price(models, out: Path) -> None:
    """Attach each pinned provider's listed price, and save the prices (as codenames.c4)."""
    try:
        prices = c4.listed_prices()
    except (OSError, ValueError, KeyError) as e:
        print("no price list; the meter uses its fallback price:", e, file=sys.stderr)
        prices = {}
    for name in models:
        m = MODELS[name]
        MODELS[name] = replace(m, price=prices.get(m.repo, {}).get(m.provider))
    listed = {MODELS[n].repo: prices.get(MODELS[n].repo, {}) for n in models}
    (out / "prices.json").write_text(json.dumps(listed, indent=1) + "\n")
    c4.record_serving([MODELS[n] for n in models], out)


def model_list(arg: str) -> list[str]:
    if arg == "all":
        return list(PLAN)
    names = arg.split(",")
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; known: {list(MODELS)}")
    return names


def call_stats(log: Path) -> dict:
    return c4.call_stats(log) if log.exists() else {"calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["choose", "crossguess", "picks", "score"])
    parser.add_argument("--models", default="all", help="'all' or names, comma-separated")
    parser.add_argument("--seed", type=int, nargs="+", default=[6005])
    parser.add_argument("--n", type=int, nargs="+", default=[50])
    parser.add_argument("--repeat", type=int, default=0, help="boards of the first seed sent 3x")
    parser.add_argument("--workers", type=int, default=8, help="calls in flight per model")
    parser.add_argument("--actions", type=Path, default=ACTIONS)
    parser.add_argument("--clues", type=Path, help="crossguess: clue lines of codenames.c4")
    parser.add_argument("--runs", type=Path, help="score: folder with the jobs' outputs")
    parser.add_argument(
        "--distinct", action="store_true", help="crossguess: send a shared clue once per board"
    )
    parser.add_argument("--cap", type=float, default=0.0, help="dollars this job may spend")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.what == "score":
        from codenames import c6score

        c6score.main(args.runs, args.out)
        return
    if args.what == "picks":  # here, with vectors: the clues the guessers of crossguess hear
        from codenames.c6score import RESULTS, picks_for_crossguess

        chosen = read_choice_lines((args.runs or RESULTS).joinpath("c6-choices.txt").read_text())
        seed, (model,) = args.seed[0], model_list(args.models)
        lists = read_action_lines(args.actions.read_text())[seed]
        args.out.write_text(picks_for_crossguess(seed, lists, chosen[(model, seed)], model))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    meter = Meter(args.cap)
    if args.what == "crossguess":
        guessers = GUESSERS if args.models == "all" else model_list(args.models)
        c4.price_models(guessers, args.out)
        text = c4.crossguess(
            guessers, args.clues.read_text(), args.seed[0], meter, args.out, args.distinct
        )
        (args.out / "crossguess.txt").write_text(text)
        stats = {m: call_stats(args.out / f"calls-{m}.jsonl") for m in guessers}
        record = {"spent": meter.spent, "cap": args.cap, "sha256": sha256(text), "calls": stats}
        (args.out / "crossguess.json").write_text(json.dumps(record, indent=1) + "\n")
    else:
        models = model_list(args.models)
        price(models, args.out)
        lists = read_action_lines(args.actions.read_text())
        chosen = {
            seed: choose(models, seed, n, lists[seed], meter, args.out, args.workers)
            for seed, n in zip(args.seed, args.n, strict=True)
        }
        first = args.seed[0]
        reps = {}
        if args.repeat:
            bs = c4.boards(first, args.repeat)
            ls = lists[first][: args.repeat]
            reps = {m: repeat(m, first, bs, ls, chosen[first][m], meter, args.out) for m in models}
        text = choice_lines(chosen)
        (args.out / "choices.txt").write_text(text)
        stats = {m: call_stats(args.out / f"calls-{m}.jsonl") for m in models}
        record = {
            "spent": meter.spent,
            "cap": args.cap,
            "actions_sha256": sha256(args.actions.read_text()),
            "choices_sha256": sha256(text),
            "repeat": reps,
            "calls": stats,
            "mean_count": {
                f"{m} {s}": statistics.mean(r["count"] for r in rows if r.get("count"))
                if any(r.get("count") for r in rows)
                else None
                for s, by in chosen.items()
                for m, rows in by.items()
            },
        }
        (args.out / "choose.json").write_text(json.dumps(record, indent=1) + "\n")
        print("choices.txt sha256", record["choices_sha256"])
    print(f"spent ${meter.spent:.4f} of ${args.cap:.2f} in {meter.calls} calls", flush=True)


if __name__ == "__main__":
    main()
