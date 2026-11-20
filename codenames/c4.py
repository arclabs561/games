"""Experiment C4 (experiments/plans/C4.md): hosted language models as givers and guessers.

The parts that call models run inside a Hugging Face Job (experiments/hfrun.py), which
holds the token; they need only the board words and write every reply to OUT:

    python -m codenames.c4 pilot --models all --n 50 --seed 4999 --cap 0.25 --out OUT
    python -m codenames.c4 give --models all --seed 4005 --n 50 --cap 0.5 --out OUT

`give` deals n boards with random.Random(seed), as codenames.study does, and asks each
model for one clue per board with the prompt of codenames.llm. It writes give-<model>-
<seed>.json with every reply and the clue read from it, and calls-<model>.jsonl with
every request. Scoring the clues needs word vectors and runs where they are:

    python -m codenames.c4 newwords --clues clues.txt --out words.txt  # no vectors yet
    python -m codenames.c4 score --clues clues.txt --guesses guesses.txt --out c4.json

The human-guess part runs in a job too, on the turns of C1's game split
(codenames.split), val first and test once the model is chosen:

    python -m codenames.c4 guess --models all --split val --cap 1.0 --out OUT

It writes guesses.txt, one character per turn (see guess_lines).
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from codenames.game import Board, deal
from codenames.hosted import (
    Hosted,
    Meter,
    Model,
    OverBudgetError,
    listed_prices,
    record_serving,
)
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
# gpt-oss counts its reasoning against max_tokens: with 64 tokens every giver reply in
# the pilot was cut off before the answer (amendment of 2026-11-19 in the plan).
TOKENS = {"gpt-oss-120b": (1024, 1024)}
# Calls in flight per model. deepinfra answered Qwen3-32B with 429 at 8 (the pilot).
WORKERS = {"qwen3-32b": 2}


def tokens(name: str, role: str) -> int:
    giver, guesser = TOKENS.get(name, (GIVER_TOKENS, GUESSER_TOKENS))
    return giver if role == "give" else guesser


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
        hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name, "give"))
        jobs = [
            (giver_prompt(b), {"part": "give", "seed": seed, "board": i}) for i, b in enumerate(bs)
        ]
        records = run_many(hosted.call, jobs, WORKERS.get(name, workers))
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


def guess(models, split: str, meter: Meter, out: Path, workers: int = 8) -> dict:
    """Part 1: each model hears the clue of every human turn in C1's game split, with
    count 1, and its guess is the first word of llm.parse_ranking (the first board word
    when the reply names none)."""
    from codenames.split import turns

    data = turns(f"game:{split}")
    results = {}
    for name in models:
        hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name, "guess"))
        jobs = [
            (
                guesser_prompt(t.guess.clue, 1, t.guess.words),
                {"part": "guess", "split": split, "turn": i, "game": t.game},
            )
            for i, t in enumerate(data)
        ]
        records = run_many(hosted.call, jobs, WORKERS.get(name, workers))
        rows = []
        for i, (t, rec) in enumerate(zip(data, records, strict=True)):
            if rec is None:
                rows.append({"turn": i, "game": t.game, "stopped": True})
                continue
            r = read_ranking(rec["reply"], t.guess.words)
            first = r["ranking"][0]
            rows.append(
                {
                    "turn": i,
                    "game": t.game,
                    "reply": rec["reply"],
                    "guess": first,
                    "named": r["named"],
                    "hit": int(first == t.guess.picked),
                }
            )
        results[name] = rows
        record = {"model": name, **vars(MODELS[name]), "split": split, "rows": rows}
        (out / f"guess-{name}-{split}.json").write_text(json.dumps(record, indent=1) + "\n")
        done = [r for r in rows if not r.get("stopped")]
        top1 = sum(int(r["hit"]) for r in done) / len(done) if done else None
        print(name, split, len(done), "turns, top-1", top1, f"${meter.spent:.4f}", flush=True)
    return results


def play_turn(board: Board, ranking: list[str], count: int, nmax: int = 4) -> tuple[int, float]:
    """Team words found and the turn's value when a guesser picks in this order, up to
    min(count, nmax) words, stopping at the first word that is not the team's. The costs
    are those of codenames.players.costs_for: 1 for an opponent word, 0 for a neutral
    word, and the team words still to find for the assassin."""
    found, left = 0, board.left("T")
    for word in ranking[: min(count, nmax)]:
        kind = board.kind(word)
        if kind == "T":
            found += 1
            continue
        cost = {"O": 1.0, "N": 0.0, "A": float(left)}[kind]
        return found, found - cost
    return found, float(found)


def crossguess(
    guessers, clues_text: str, seed: int, meter: Meter, out: Path, distinct: bool = False
) -> str:
    """Language-model guessers on clues from givers of other families: each guesser hears
    every board's clue from every giver in the file except itself, once, with the giver's
    count. With distinct, a (board, clue, count) that two givers share is sent once and
    its reply is used for both. Returns one line per (giver, guesser): found/value per
    board."""
    given = read_clue_lines(clues_text)
    lines = []
    heard: dict[tuple, dict | None] = {}
    for (giver, s), rows in sorted(given.items()):
        if s != seed:
            continue
        bs = boards(seed, len(rows))
        for name in guessers:
            if name == giver:
                continue
            hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name, "guess"))
            jobs = [
                (
                    guesser_prompt(word, count, b.unrevealed()),
                    {"part": "crossguess", "seed": seed, "board": i, "giver": giver},
                )
                for i, (b, (word, count, _)) in enumerate(zip(bs, rows, strict=True))
                if word is not None and count is not None
            ]
            assert len(jobs) == len(rows), f"{giver} has boards without a clue"
            if distinct:
                keys = [(name, i, w, c) for i, (w, c, _) in enumerate(rows)]
                new = [n for n, key in enumerate(keys) if key not in heard]
                for n, rec in zip(
                    new,
                    run_many(hosted.call, [jobs[n] for n in new], WORKERS.get(name, 8)),
                    strict=True,
                ):
                    heard[keys[n]] = rec
                recs = [heard[key] for key in keys]
            else:
                recs = run_many(hosted.call, jobs, WORKERS.get(name, 8))
            cells = []
            for b, (_, count, _), rec in zip(bs, rows, recs, strict=True):
                if rec is None:
                    cells.append("-")
                    continue
                ranking = read_ranking(rec["reply"], b.unrevealed())["ranking"]
                found, value = play_turn(b, ranking, count or 1)
                cells.append(f"{found}/{value:g}")
            lines.append(" ".join([giver, name, str(seed), *cells]))
            print(giver, name, f"${meter.spent:.4f}", flush=True)
    return "\n".join(lines) + "\n"


def guess_lines(guessed: dict[str, dict[str, list[dict]]]) -> str:
    """One line per model and split, one character per turn: 1 or 0 for a hit or a miss
    when the reply named a board word, b or a when it named none (the guess is then the
    first board word), and - when the budget stopped the run."""
    lines = []
    for split, by_model in guessed.items():
        for name, rows in by_model.items():
            chars = "".join(
                "-" if r.get("stopped") else ("10" if r["named"] else "ba")[1 - r["hit"]]
                for r in rows
            )
            lines.append(f"{name} {split} {chars}")
    return "\n".join(lines) + "\n"


def repeat(name, bs, rows, seed: int, meter: Meter, out: Path, times: int = 2) -> dict:
    """Send the giver prompts of these boards `times` more times, and count the prompts
    whose replies were not all the same as the first."""
    hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name, "give"))
    jobs = [
        (giver_prompt(b), {"part": "repeat", "seed": seed, "board": i})
        for i, b in enumerate(bs)
        for _ in range(times)
    ]
    recs = run_many(hosted.call, jobs, WORKERS.get(name, 4))
    replies = [r and r["reply"] for r in recs]
    changed = sum(
        1
        for i in range(len(bs))
        if any(
            x is not None and x != rows[i].get("reply")
            for x in replies[i * times : (i + 1) * times]
        )
    )
    return {
        "prompts": len(bs),
        "times": times,
        "sent": sum(1 for r in recs if r),
        "changed": changed,
    }


def clue_lines(given: dict[int, dict[str, list[dict]]]) -> str:
    """One line per model and seed: the clue and count read from each board's reply, or
    '-' and the reason there is none. This is all that scoring needs."""
    lines = []
    for seed, by_model in given.items():
        for name, rows in by_model.items():
            cells = [
                f"{r['clue']}/{r['count']}"
                if r.get("clue")
                else "-/" + ("stopped" if r.get("stopped") else r["why"].replace(" ", "-"))
                for r in rows
            ]
            lines.append(" ".join([name, str(seed), *cells]))
    return "\n".join(lines) + "\n"


def read_clue_lines(
    text: str,
) -> dict[tuple[str, int], list[tuple[str | None, int | None, str | None]]]:
    """The inverse of clue_lines: {(model, seed): [(clue, count, why) per board]}."""
    out = {}
    for line in text.splitlines():
        name, seed, *cells = line.split()
        rows: list[tuple[str | None, int | None, str | None]] = []
        for cell in cells:
            word, rest = cell.split("/")
            rows.append((None, None, rest) if word == "-" else (word, int(rest), None))
        out[(name, int(seed))] = rows
    return out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
        hosted = Hosted(MODELS[name], meter, out / f"calls-{name}.jsonl", tokens(name, "guess"))
        jobs = [
            (
                guesser_prompt(c, m, bs[i].unrevealed()),
                {"part": "guess", "seed": seed, "board": i, "giver": g},
            )
            for i, g, c, m in clues
        ]
        recs = run_many(hosted.call, jobs, WORKERS.get(name, 8))
        named = [
            (read_ranking(r["reply"], bs[i].unrevealed())["named"], len(bs[i].unrevealed()))
            for r, (i, *_) in zip(recs, clues, strict=True)
            if r is not None
        ]
        rep = repeat(name, bs[:repeats], given[name], seed, meter, out)
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
            "repeat": rep,
            "calls": call_stats(out / f"calls-{name}.jsonl"),
        }
    summary["spent"] = meter.spent
    (out / "pilot.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def price_models(models, out: Path) -> None:
    """Attach each pinned provider's listed price to its model, and save the prices."""
    try:
        prices = listed_prices()
    except (OSError, ValueError, KeyError) as e:
        print("no price list; the meter uses its fallback price:", e, file=sys.stderr)
        prices = {}
    for name in models:
        m = MODELS[name]
        MODELS[name] = replace(m, price=prices.get(m.repo, {}).get(m.provider))
    listed = {MODELS[n].repo: prices.get(MODELS[n].repo, {}) for n in models}
    (out / "prices.json").write_text(json.dumps(listed, indent=1) + "\n")
    record_serving([MODELS[n] for n in models], out)


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
    parser.add_argument(
        "what", choices=["pilot", "give", "guess", "crossguess", "newwords", "score", "strict"]
    )
    parser.add_argument("--workers", type=int, default=8, help="calls in flight per model")
    parser.add_argument("--repeat", type=int, default=20, help="boards sent three times")
    parser.add_argument("--split", nargs="+", default=["val"], choices=["val", "test"])
    parser.add_argument(
        "--models", default="all", help="'all' or names from MODELS, comma-separated"
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[4999])
    parser.add_argument("--n", type=int, nargs="+", default=[50])
    parser.add_argument("--cap", type=float, default=0.0, help="dollars this job may spend")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clues", type=Path, help="clues.txt from a give job")
    parser.add_argument("--guesses", type=Path, help="guesses.txt from a guess job")
    parser.add_argument("--crossguess", type=Path, help="crossguess.txt lines, all guessers")
    parser.add_argument("--runs", help="strict: a folder of give-<model>-<seed>.json, or hf")
    args = parser.parse_args()
    if args.what == "strict":
        from codenames import c4strict

        c4strict.run(args.runs, args.out)
        return
    if args.what in ("newwords", "score"):
        from codenames import c4score

        if args.what == "newwords":
            args.out.write_text("\n".join(c4score.new_words(args.clues.read_text())) + "\n")
        else:
            c4score.main(args.clues, args.guesses, args.out, args.crossguess)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    meter = Meter(args.cap)
    models = model_list(args.models)
    price_models(models, args.out)
    if args.what == "pilot":
        pilot(models, args.seed[0], args.n[0], meter, args.out)
    elif args.what == "crossguess":
        text = crossguess(models, args.clues.read_text(), args.seed[0], meter, args.out)
        (args.out / "crossguess.txt").write_text(text)
        stats = {m: call_stats(args.out / f"calls-{m}.jsonl") for m in models}
        record = {"spent": meter.spent, "cap": args.cap, "sha256": sha256(text), "calls": stats}
        (args.out / "crossguess.json").write_text(json.dumps(record, indent=1) + "\n")
    elif args.what == "guess":
        guessed = {s: guess(models, s, meter, args.out, args.workers) for s in args.split}
        text = guess_lines(guessed)
        (args.out / "guesses.txt").write_text(text)
        stats = {m: call_stats(args.out / f"calls-{m}.jsonl") for m in models}
        record = {
            "spent": meter.spent,
            "cap": args.cap,
            "guesses_sha256": sha256(text),
            "calls": stats,
        }
        (args.out / "guess.json").write_text(json.dumps(record, indent=1) + "\n")
        print("guesses.txt sha256", record["guesses_sha256"])
    else:
        given = {
            seed: give(models, seed, n, meter, args.out)
            for seed, n in zip(args.seed, args.n, strict=True)
        }
        first = args.seed[0]
        rows = boards(first, args.n[0])[: args.repeat]
        reps = {m: repeat(m, rows, given[first][m], first, meter, args.out) for m in models}
        text = clue_lines(given)
        (args.out / "clues.txt").write_text(text)
        stats = {m: call_stats(args.out / f"calls-{m}.jsonl") for m in models}
        record = {
            "spent": meter.spent,
            "cap": args.cap,
            "clues_sha256": sha256(text),
            "repeat": reps,
            "calls": stats,
        }
        (args.out / "give.json").write_text(json.dumps(record, indent=1) + "\n")
        print("clues.txt sha256", record["clues_sha256"])
    print(f"spent ${meter.spent:.4f} of ${args.cap:.2f} in {meter.calls} calls", flush=True)


if __name__ == "__main__":
    main()
