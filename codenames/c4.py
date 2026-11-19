"""Experiment C4 (experiments/plans/C4.md): hosted language models as givers and guessers.

The parts that call models run inside a Hugging Face Job (experiments/hfrun.py), which
holds the token; they need only the board words and write every reply to OUT:

    python -m codenames.c4 pilot --models all --n 50 --seed 4999 --cap 0.25 --out OUT
    python -m codenames.c4 give --models all --seed 4005 --n 50 --cap 0.5 --out OUT

`give` deals n boards with random.Random(seed), as codenames.study does, and asks each
model for one clue per board with the prompt of codenames.llm. It writes give-<model>-
<seed>.json with every reply and the clue read from it, and calls-<model>.jsonl with
every request. Scoring the clues needs word vectors and runs where they are:

    python -m codenames.c4 newwords --runs DIR           # clue words without vectors
    python -m codenames.c4 score --runs DIR --out codenames/results/c4-give.json

The human-guess part (`guess`) waits for C1's game split.
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codenames.game import Board, deal
from codenames.hosted import Hosted, Meter, Model, OverBudgetError
from codenames.llm import giver_prompt, guesser_prompt, parse_clue, parse_ranking
from codenames.words import board_words, legal

# The plan's four models, in its order, then its substitutes. Each is pinned to one
# provider that the Hub listed as live for it on 2026-11-18.
MODELS = {
    "llama-3.3-70b": Model("meta-llama/Llama-3.3-70B-Instruct", "novita"),
    "qwen3-32b": Model(
        "Qwen/Qwen3-32B", "deepinfra", {"chat_template_kwargs": {"enable_thinking": False}}
    ),
    "gpt-oss-120b": Model("openai/gpt-oss-120b", "deepinfra", {"reasoning_effort": "low"}),
    "deepseek-v3.2": Model("deepseek-ai/DeepSeek-V3.2", "deepinfra"),
    "gemma-3-27b": Model("google/gemma-3-27b-it", "deepinfra"),
    "mistral-small-3.1": Model("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "featherless-ai"),
}
PLAN = ("llama-3.3-70b", "qwen3-32b", "gpt-oss-120b", "deepseek-v3.2")
SUBSTITUTES = ("gemma-3-27b", "mistral-small-3.1")
GIVER_TOKENS, GUESSER_TOKENS = 64, 256
CLUE = re.compile(r"([A-Za-z]+)\W+(\d+)")


def boards(seed: int, n: int) -> list[Board]:
    rng = random.Random(seed)
    return [deal(board_words(), rng) for _ in range(n)]


def read_clue(reply: str, board: Board) -> dict:
    """The clue in a giver's reply, or the reason there is none: empty, unparsed (no word
    followed by a number) or illegal (a board word, or a count below 1)."""
    if not reply.strip():
        return {"clue": None, "count": None, "why": "empty"}
    clue = parse_clue(reply, board)
    if clue is not None:
        return {"clue": clue.word, "count": clue.count, "why": None}
    m = CLUE.search(reply)
    if not m:
        return {"clue": None, "count": None, "why": "unparsed"}
    word, n = m[1].lower(), int(m[2])
    why = "board word" if not legal(word, board.unrevealed()) else "count below 1"
    return {"clue": None, "count": None, "why": why, "said": [word, n]}


def read_ranking(reply: str, words: tuple[str, ...]) -> dict:
    """The guesser's ranking, and how many board words the reply itself named."""
    named = len({t for t in re.findall(r"[a-z]+", reply.lower()) if t in words})
    return {"ranking": parse_ranking(reply, words), "named": named}


def run_many(call: Callable[[str, dict], dict], jobs: list[tuple[str, dict]], workers: int):
    """Call each (prompt, tag) in parallel; None for a call the budget did not allow."""

    def one(job):
        try:
            return call(*job)
        except OverBudgetError:
            return None

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, jobs))


def give(models, seed: int, n: int, meter: Meter, out: Path, workers: int = 8) -> dict:
    """One clue per board from each model; returns {model: list of per-board results}."""
    bs = boards(seed, n)
    results = {}
    for name in models:
        hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", GIVER_TOKENS)
        jobs = [
            (giver_prompt(b), {"part": "give", "seed": seed, "board": i}) for i, b in enumerate(bs)
        ]
        records = run_many(hosted.call, jobs, workers)
        rows = []
        for i, (b, rec) in enumerate(zip(bs, records, strict=True)):
            if rec is None:
                rows.append({"board": i, "stopped": True})
                continue
            rows.append({"board": i, "reply": rec["reply"], **read_clue(rec["reply"], b)})
        results[name] = rows
        record = {"model": name, **vars(MODELS[name]), "seed": seed, "n": n, "rows": rows}
        (out / f"give-{name}-{seed}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(name, seed, Counter(r.get("why") for r in rows), f"${meter.spent:.4f}", flush=True)
    return results


def call_stats(log: Path) -> dict:
    """Cost, tokens, failures and truncations over every call in one model's log."""
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    done = [r for r in recs if "usage" in r]
    costs = [r["cost"] for r in done]
    out_tokens = [r["usage"].get("completion_tokens") or 0 for r in done]
    reasoning = [
        (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        for r in done
    ]
    return {
        "calls": len(recs),
        "failed": len(recs) - len(done),
        "retried": sum(1 for r in done if r["attempts"] > 1),
        "cost": sum(costs),
        "cost_per_call": statistics.mean(costs) if costs else None,
        "cost_max": max(costs, default=None),
        "cost_sources": dict(Counter(r["cost_source"] for r in done)),
        "truncated": sum(1 for r in done if r["finish"] == "length"),
        "completion_tokens_mean": statistics.mean(out_tokens) if done else None,
        "reasoning_tokens_mean": statistics.mean(reasoning) if done else None,
        "think_in_text": sum(1 for r in done if "<think>" in r["raw"]),
        "seconds_mean": statistics.mean(r["seconds"] for r in done) if done else None,
        "served_as": dict(Counter(str(r.get("served")) for r in done)),
        "errors": sorted({e[:120] for r in recs for e in r["errors"]})[:5],
    }


def pilot(models, seed: int, n: int, meter: Meter, out: Path, repeats: int = 5) -> dict:
    """A small run of every part, to measure cost per call and parse failures.

    Each model gives one clue on each of n boards. Then each model guesses on n clues
    from the next model in the list (with that giver's count), so no model guesses its
    own clues. Last, the first `repeats` giver prompts are sent twice more."""
    bs = boards(seed, n)
    given = give(models, seed, n, meter, out)
    summary: dict = {"seed": seed, "n": n, "models": {}}
    for k, name in enumerate(models):
        clues = []
        for i in range(len(bs)):
            for other in models[k + 1 :] + models[:k]:
                row = given[other][i]
                if row.get("clue"):
                    clues.append((i, other, row["clue"], row["count"]))
                    break
        hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", GUESSER_TOKENS)
        jobs = [
            (
                guesser_prompt(c, m, bs[i].unrevealed()),
                {"part": "guess", "seed": seed, "board": i, "giver": g},
            )
            for i, g, c, m in clues
        ]
        recs = run_many(hosted.call, jobs, 8)
        named = [
            (read_ranking(r["reply"], bs[i].unrevealed())["named"], len(bs[i].unrevealed()))
            for r, (i, *_) in zip(recs, clues, strict=True)
            if r is not None
        ]
        giver = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", GIVER_TOKENS)
        again = [
            (giver_prompt(bs[i]), {"part": "repeat", "seed": seed, "board": i})
            for i in range(repeats)
            for _ in range(2)
        ]
        recs = run_many(giver.call, again, 4)
        first = [given[name][i].get("reply") for i in range(repeats) for _ in range(2)]
        changed = sum(
            1 for r, f in zip(recs, first, strict=True) if r is not None and r["reply"] != f
        )
        rows = given[name]
        summary["models"][name] = {
            **vars(MODELS[name]),
            "give": {
                "boards": len(rows),
                "clues": sum(1 for r in rows if r.get("clue")),
                "no_clue": dict(Counter(r["why"] for r in rows if r.get("why"))),
                "stopped": sum(1 for r in rows if r.get("stopped")),
                "mean_count": statistics.mean(r["count"] for r in rows if r.get("clue"))
                if any(r.get("clue") for r in rows)
                else None,
            },
            "guess": {
                "turns": len(named),
                "named_none": sum(1 for x, _ in named if x == 0),
                "named_all": sum(1 for x, total in named if x == total),
                "named_mean": statistics.mean(x for x, _ in named) if named else None,
            },
            "repeat": {"sent": sum(1 for r in recs if r is not None), "changed": changed},
            "calls": call_stats(out / f"calls-{name}.jsonl"),
        }
    summary["spent"] = meter.spent
    (out / "pilot.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def model_list(arg: str) -> list[str]:
    if arg == "all":
        return list(PLAN)
    names = arg.split(",")
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; known: {list(MODELS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["pilot", "give"])
    parser.add_argument(
        "--models", default="all", help="'all' or names from MODELS, comma-separated"
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[4999])
    parser.add_argument("--n", type=int, nargs="+", default=[50])
    parser.add_argument("--cap", type=float, required=True, help="dollars this job may spend")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meter = Meter(args.cap)
    models = model_list(args.models)
    if args.what == "pilot":
        pilot(models, args.seed[0], args.n[0], meter, args.out)
    else:
        for seed, n in zip(args.seed, args.n, strict=True):
            give(models, seed, n, meter, args.out)
        stats = {m: call_stats(args.out / f"calls-{m}.jsonl") for m in models}
        (args.out / "give.json").write_text(
            json.dumps({"spent": meter.spent, "cap": args.cap, "calls": stats}, indent=1) + "\n"
        )
    print(f"spent ${meter.spent:.4f} of ${args.cap:.2f} in {meter.calls} calls", flush=True)


if __name__ == "__main__":
    main()
