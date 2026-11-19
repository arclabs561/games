"""Experiment C10 (experiments/plans/C10.md): a hosted language model as the listener inside
an embedding giver.

EmbeddingGemma 2 proposes 30 legal clues on a board (codenames.actions). The **prefilter**
keeps the K clues with the highest proposer value, the best of a clue's four actions
under the proposer's Luce guesser. gpt-oss-120b hears C4's guesser prompt once for each
kept clue, with count 1, and its reply becomes a Luce guesser:

- The **informative prefix** of a ranking is its shortest start after which the remaining
  words follow the prompt's order. Words in the prefix get ranks 0, 1, 2, ..., and every
  later word gets the prefix's length. A reply that names no board word, or names them
  in the prompt's order, gives every word the same rank.
- The **listener's guesser** picks with weights exp(-rank / T). T is fitted once, by
  maximum likelihood of the person's word on C1's validation turns, from C4's stored
  gpt-oss-120b replies on those turns.

The listener values all four counts of a clue from one reply. The arms are the first
choice (F), the listener's pick (L), the population's pick (P), their mixture (LP), C6's
chooser (C) and the same model asked to choose (S); codenames.c10score scores them.

Here, with word vectors:

    python -m codenames.listen lists --seed 9999 9005 9004 --n 5 100 300 --out codenames/results/c10-listen.txt
    python -m codenames.listen fit --replies codenames/results/c10-c4-val-prefixes.txt --out codenames/results/c10-listener.json
    python -m codenames.listen select --out codenames/results/c10-val.json
    python -m codenames.listen picks --out codenames/results/c10-picks-9004.txt
    python -m codenames.listen score --out codenames/results/c10.json

In a Hugging Face Job (experiments/hfrun.py), with no vectors:

    python -m codenames.listen prefixes --out OUT        # C4's replies, read with the tail rule
    python -m codenames.listen pilot --n 5 --cap 0.05 --out OUT   # listen and speak, seed 9999
    python -m codenames.listen listen --seed 9005 --boards 0-50 --k 20 --repeat 10 --cap 0.15 --out OUT
    python -m codenames.listen speak --seed 9004 --k K --cap 0.05 --out OUT

`listen` writes every request to calls-gpt-oss-120b.jsonl, every reply and its reading to
listen-<seed>-<A>-<B>.json, and rankings.txt, all that scoring needs: one line per board
and repeat, `seed board repeat cell_1 ... cell_k`, one cell per kept clue in prefilter
order (see `cell`).
"""

import argparse
import json
import statistics
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from codenames import c4
from codenames.actions import COUNTS, expand, read_action_lines, sha256
from codenames.game import Board
from codenames.hosted import Hosted, Meter
from codenames.llm import guesser_prompt
from codenames.luce import turn_odds
from codenames.players import Weigher, costs_for

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LISTS = RESULTS / "c10-listen.txt"
ACTIONS = RESULTS / "c10-actions.txt"
RANKINGS = RESULTS / "c10-rankings.txt"
PILOT, VAL, TEST = 9999, 9005, 9004
LISTENER = "gpt-oss-120b"
TOKENS = 1024  # C4's cap for gpt-oss-120b, whose reasoning counts against it
TOP = 20  # clues per board in c10-listen.txt
KS = (10, 15, 20)
WS = (1 / 7, 1 / 4, 1 / 2, 3 / 4)
T_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
# C4's gpt-oss-120b replies on the validation turns of C1's game split.
C4_REPLIES = ("runs/c4/guess-val-gpt-oss-120b", "guess-gpt-oss-120b-val.json")
DATASET = "arclabs561/games-runs"
LETTERS = string.ascii_lowercase  # a board has at most 25 words: a to y
# Cells with no prefix. All four give every word the same rank.
IN_ORDER, NAMED_NONE, FAILED, STOPPED = "=", "0", "!", "?"
MARKS = (IN_ORDER, NAMED_NONE, FAILED, STOPPED)


# The tail rule and the listener's guesser.


def informative_prefix(ranking: list[str], words: tuple[str, ...]) -> list[str]:
    """The shortest start of ranking after which the rest follows the order of words.

    The rest follows that order exactly when the words' positions in `words` increase
    along it, so the prefix ends just before the longest increasing run at the end."""
    pos = {w: i for i, w in enumerate(words)}
    assert sorted(ranking, key=pos.__getitem__) == list(words), "not a ranking of words"
    p = len(ranking)
    while p > 0 and (p == len(ranking) or pos[ranking[p - 1]] < pos[ranking[p]]):
        p -= 1
    return ranking[:p]


def prefix_by_loop(ranking: list[str], words: tuple[str, ...]) -> list[str]:
    """The same prefix, by trying every start from the shortest."""
    for p in range(len(ranking) + 1):
        rest = ranking[p:]
        if rest == [w for w in words if w in set(rest)]:
            return ranking[:p]
    raise AssertionError("unreachable: the empty rest is in order")


def ranks(prefix: list[str], words: tuple[str, ...]) -> np.ndarray:
    """Rank of each word: its place in the prefix, or the prefix's length."""
    at = {w: i for i, w in enumerate(prefix)}
    return np.array([at.get(w, len(prefix)) for w in words], dtype=float)


def listener_weights(r: np.ndarray, temp: float) -> np.ndarray:
    """exp(-rank / T): the largest weight is 1, for rank 0."""
    return np.exp(-np.asarray(r, dtype=float) / temp)


def cell(reply: str | None, words: tuple[str, ...], failed: bool = False) -> str:
    """One kept clue in rankings.txt: the prefix as letters (a is the board's first word),
    or a mark for a reply with no prefix: = (board words in the prompt's order), 0 (no
    board word), ! (the call failed or the reply was empty, after the retry), ? (the
    budget stopped the call)."""
    if reply is None:
        return STOPPED
    if failed or not reply.strip():
        return FAILED
    named = c4.read_ranking(reply, words)
    if named["named"] == 0:
        return NAMED_NONE
    prefix = informative_prefix(named["ranking"], words)
    if not prefix:
        return IN_ORDER
    return "".join(LETTERS[words.index(w)] for w in prefix)


def read_cell(c: str, words: tuple[str, ...]) -> list[str]:
    """The prefix in a cell (empty for the four marks)."""
    if c in MARKS:
        return []
    return [words[LETTERS.index(x)] for x in c]


def listener_weigher(prefixes: dict[str, list[str]], temp: float) -> Weigher:
    """A weigher for codenames.actions.values: each clue's weights from its prefix."""

    def weigh(clue: str, words: tuple[str, ...]) -> np.ndarray:
        return listener_weights(ranks(prefixes[clue], words), temp)

    return weigh


def listener_values(board: Board, kept: list[str], prefixes: dict[str, list[str]], temp: float):
    """Exact expected turn value of each of the 4K actions (expand(kept)) under the
    listener's guesser."""
    from codenames.actions import values

    return values(listener_weigher(prefixes, temp), board, expand(kept))


def listener_values_by_loop(board, kept, prefixes, temp) -> np.ndarray:
    """The same values one action at a time, from luce.turn_odds directly."""
    words = board.unrevealed()
    kinds = tuple(board.kind(w) for w in words)
    out = []
    for c, n in expand(kept):
        w = np.array([np.exp(-r / temp) for r in ranks(prefixes[c], words)])
        out.append(float(turn_odds(w, kinds, n).value(n, costs_for(board))[0]))
    return np.array(out)


# The fit of T.


def log_likelihood(turns: list[tuple[np.ndarray, int]], temp: float) -> float:
    """Mean log chance of the person's word, over turns of (ranks, index of the word)."""
    total = 0.0
    for r, picked in turns:
        z = -np.asarray(r, dtype=float) / temp
        total += z[picked] - (z.max() + np.log(np.exp(z - z.max()).sum()))
    return total / len(turns)


def fit_temperature(turns: list[tuple[np.ndarray, int]], grid=T_GRID) -> dict:
    """The T on the grid with the highest log-likelihood, the first of equal ones."""
    ll = np.array([log_likelihood(turns, temp) for temp in grid])
    i = int(np.argmax(ll))
    return {"T": float(grid[i]), "loglik_per_turn": float(ll[i]), "grid_index": i}


# The prefilter and the pickers.


def prefilter_values(values: np.ndarray, clues: list[str], k: int) -> list[str]:
    """The k clues whose best action is worth the most; ties keep the list's order.
    values are those of the 4 len(clues) actions of expand(clues)."""
    best = np.asarray(values).reshape(len(clues), len(COUNTS)).max(axis=1)
    order = sorted(range(len(clues)), key=lambda i: (-best[i], i))
    return [clues[i] for i in order[:k]]


def prefilter(board: Board, clues: list[str], k: int) -> list[str]:
    """prefilter_values with the proposer's values (needs its vectors)."""
    from codenames.actions import proposer, values
    from codenames.players import embedding_weigher

    sp, tau = proposer()
    return prefilter_values(values(embedding_weigher(sp, tau), board, expand(clues)), clues, k)


def kept_index(clues: list[str], kept: list[str]) -> list[int]:
    """Position in expand(clues) of each action of expand(kept)."""
    pos = {c: i for i, c in enumerate(clues)}
    return [pos[c] * len(COUNTS) + n - 1 for c, n in expand(kept)]


def argmax_first(x) -> int:
    """The first index of the largest value (np.argmax already takes the first)."""
    return int(np.argmax(np.asarray(x)))


def pick_population(stranger_values: np.ndarray, held_out: int) -> int:
    """P in one task: the action with the highest mean value over the strangers other
    than the held-out one. stranger_values has shape (strangers, actions)."""
    rest = np.delete(stranger_values, held_out, axis=0)
    return argmax_first(rest.mean(axis=0))


def pick_mixture(listener: np.ndarray, stranger_values: np.ndarray, held_out: int, w: float):
    """LP in one task: w times the listener's value plus 1 - w times the population's."""
    pop = np.delete(stranger_values, held_out, axis=0).mean(axis=0)
    return argmax_first(w * np.asarray(listener) + (1 - w) * pop)


# The job's side.


def list_lines(seed: int, lists: list[list[str]]) -> str:
    from codenames.actions import action_lines

    return action_lines(seed, lists)


def parse_range(text: str) -> range:
    a, b = text.split("-")
    return range(int(a), int(b))


def listen(seed: int, span: range, k: int, repeat: int, meter: Meter, out: Path, workers: int):
    """The listener hears each of the first k kept clues of each board in span; then the
    first `repeat` boards' prompts are sent twice more. Calls that failed or came back
    empty are sent once more at the end."""
    lists = read_action_lines(LISTS.read_text())[seed]
    bs = c4.boards(seed, span.stop)
    hosted = Hosted(c4.MODELS[LISTENER], meter, out / f"calls-{LISTENER}.jsonl", TOKENS)
    jobs, where = [], []
    for rep in range(3):
        for i in span:
            if rep and i >= span.start + repeat:
                break
            words = bs[i].unrevealed()
            for j, clue in enumerate(lists[i][:k]):
                tag = {"part": "listen", "seed": seed, "board": i, "clue": clue, "repeat": rep}
                jobs.append((guesser_prompt(clue, 1, words), tag))
                where.append((i, rep, j))
    records = c4.run_many(hosted.call, jobs, workers)
    again = [n for n, r in enumerate(records) if r is not None and not r["reply"].strip()]
    retried = c4.run_many(hosted.call, [jobs[n] for n in again], workers)
    for n, r in zip(again, retried, strict=True):
        if r is not None:
            records[n] = {**r, "retry": True}
    cells: dict[tuple[int, int], list[str]] = {}
    rows = []
    for (i, rep, j), rec in zip(where, records, strict=True):
        words = bs[i].unrevealed()
        reply = None if rec is None else rec["reply"]
        c = cell(reply, words)
        cells.setdefault((i, rep), [""] * len(lists[i][:k]))[j] = c
        rows.append(
            {
                "board": i,
                "repeat": rep,
                "clue": lists[i][j],
                "reply": reply,
                "finish": rec and rec.get("finish"),
                "retry": bool(rec and rec.get("retry")),
                "cell": c,
            }
        )
    text = "".join(f"{seed} {i} {rep} {' '.join(cs)}\n" for (i, rep), cs in sorted(cells.items()))
    (out / "rankings.txt").write_text(text)
    name = f"listen-{seed}-{span.start}-{span.stop}.json"
    record = {"seed": seed, "boards": [span.start, span.stop], "k": k, "rows": rows}
    (out / name).write_text(json.dumps(record, indent=1) + "\n")
    all_cells = [str(r["cell"]) for r in rows]
    summary = {
        "spent": meter.spent,
        "lists_sha256": sha256(LISTS.read_text()),
        "rankings_sha256": sha256(text),
        "cells": dict(Counter(c if c in MARKS else "prefix" for c in all_cells)),
        "retried": len(again),
        "truncated": sum(1 for r in rows if r["finish"] == "length"),
        "prefix_mean": statistics.mean(0 if c in MARKS else len(c) for c in all_cells),
        "calls": c4.call_stats(out / f"calls-{LISTENER}.jsonl"),
    }
    (out / "listen.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "calls"}), flush=True)


def speak(seed: int, n: int, k: int, meter: Meter, out: Path, workers: int) -> None:
    """S: the listener's model, asked with C6's prompt to choose among the same k clues."""
    from codenames.rerank import choice_lines, read_choice, rerank_prompt

    lists = [ls[:k] for ls in read_action_lines(LISTS.read_text())[seed][:n]]
    bs = c4.boards(seed, n)
    hosted = Hosted(c4.MODELS[LISTENER], meter, out / f"calls-{LISTENER}.jsonl", TOKENS)
    jobs = [
        (rerank_prompt(b, c), {"part": "speak", "seed": seed, "board": i, "k": k})
        for i, (b, c) in enumerate(zip(bs, lists, strict=True))
    ]
    records = c4.run_many(hosted.call, jobs, workers)
    rows = []
    for i, (rec, c) in enumerate(zip(records, lists, strict=True)):
        if rec is None:
            rows.append({"board": i, "stopped": True})
            continue
        choice = read_choice(rec["reply"], c, rec.get("finish"))
        rows.append({"board": i, "reply": rec["reply"], "finish": rec.get("finish"), **choice})
    record = {"model": LISTENER, "role": "speaker", "seed": seed, "k": k, "rows": rows}
    (out / f"speak-{seed}.json").write_text(json.dumps(record, indent=1) + "\n")
    text = choice_lines({seed: {"speaker": rows}})
    (out / "choices.txt").write_text(text)
    summary = {
        "spent": meter.spent,
        "lists_sha256": sha256(LISTS.read_text()),
        "choices_sha256": sha256(text),
        "why": dict(Counter(r.get("why") for r in rows)),
        "calls": c4.call_stats(out / f"calls-{LISTENER}.jsonl"),
    }
    (out / "speak.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "calls"}), flush=True)


def prefixes(out: Path) -> None:
    """C4's stored gpt-oss-120b replies on C1's validation turns, read with the tail rule:
    prefixes.txt has one line per turn, `turn cell`, with the cells of `cell`. No calls."""
    from huggingface_hub import hf_hub_download

    from codenames.split import turns

    folder, name = C4_REPLIES
    path = hf_hub_download(DATASET, f"{folder}/{name}", repo_type="dataset")
    raw = Path(path).read_text()
    rows = json.loads(raw)["rows"]
    data = turns("game:val")
    assert len(rows) == len(data), (len(rows), len(data))
    lines = []
    for i, (row, t) in enumerate(zip(rows, data, strict=True)):
        assert row["turn"] == i and row["game"] == t.game, (i, row["game"], t.game)
        lines.append(f"{i} {cell(row['reply'], t.guess.words)}\n")
    text = "".join(lines)
    (out / "prefixes.txt").write_text(text)
    record = {"source": f"{DATASET}/{folder}/{name}", "source_sha256": sha256(raw)}
    record["prefixes_sha256"] = sha256(text)
    (out / "prefixes.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record), flush=True)


# The local side.


def read_prefix_lines(text: str) -> list[str]:
    """Cells of prefixes.txt, in turn order."""
    out = []
    for line in text.splitlines():
        i, c = line.split()
        assert int(i) == len(out), line
        out.append(c)
    return out


def fit_turns(cells: list[str]):
    """(ranks, index of the person's word) for each validation turn, and the turns."""
    from codenames.split import turns

    data = turns("game:val")
    assert len(cells) == len(data)
    rows = []
    for c, t in zip(cells, data, strict=True):
        words = t.guess.words
        rows.append((ranks(read_cell(c, words), words), words.index(t.guess.picked)))
    return rows, data


def fit(replies: Path, out: Path) -> dict:
    """T, its log-likelihood, EmbeddingGemma 2's on the same turns, and the prefixes."""
    from codenames.actions import proposer
    from codenames.luce import luce_weights

    text = replies.read_text()
    cells = read_prefix_lines(text)
    rows, data = fit_turns(cells)
    best = fit_temperature(rows)
    sp, tau = proposer()
    eg = []
    for t in data:
        w = luce_weights(sp.sims(t.guess.clue, t.guess.words), tau)[0]
        eg.append(float(np.log(w[t.guess.words.index(t.guess.picked)] / w.sum())))
    lengths = [len(read_cell(c, t.guess.words)) for c, t in zip(cells, data, strict=True)]
    marks = Counter(c for c in cells if c in MARKS)
    result = {
        "plan": "experiments/plans/C10.md",
        "turns": len(rows),
        "prefixes_sha256": sha256(text),
        "grid": {"from": 0.05, "to": 20.0, "points": len(T_GRID), "spacing": "even in log T"},
        **best,
        "at_grid_edge": best["grid_index"] in (0, len(T_GRID) - 1),
        "loglik_half_T": log_likelihood(rows, best["T"] / 2),
        "loglik_double_T": log_likelihood(rows, best["T"] * 2),
        "loglik_uniform": float(np.mean([-np.log(len(t.guess.words)) for t in data])),
        "embeddinggemma_2": {"tau": tau, "loglik_per_turn": float(np.mean(eg))},
        "prefix_length": {
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
        "no_prefix": dict(marks),
        "top1_from_prefix": float(np.mean([r[p] == 0 and r.max() > 0 for r, p in rows])),
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result


def load_temperature() -> float:
    return json.loads((RESULTS / "c10-listener.json").read_text())["T"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "what",
        choices=[
            "lists",
            "fit",
            "prefixes",
            "pilot",
            "listen",
            "speak",
            "select",
            "picks",
            "score",
        ],
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[VAL])
    parser.add_argument("--n", type=int, nargs="+", default=[100])
    parser.add_argument("--boards", type=parse_range, help="listen: A-B, boards A to B - 1")
    parser.add_argument("--k", type=int, default=TOP)
    parser.add_argument("--repeat", type=int, default=0, help="listen: boards sent 3 times")
    parser.add_argument("--workers", type=int, default=16, help="calls in flight")
    parser.add_argument("--replies", type=Path, help="fit: prefixes.txt of the prefixes job")
    parser.add_argument("--runs", type=Path, default=RESULTS, help="folder of the inputs")
    parser.add_argument("--cap", type=float, default=0.0, help="dollars this job may spend")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.what == "lists":
        from codenames.actions import proposer, shortlist

        sp, tau = proposer()
        text = ""
        for seed, n in zip(args.seed, args.n, strict=True):
            kept = [prefilter(b, shortlist(b, sp, tau), TOP) for b in c4.boards(seed, n)]
            text += list_lines(seed, kept)
        args.out.write_text(text)
        print(args.out, sha256(text))
        return
    if args.what == "fit":
        print(json.dumps(fit(args.replies, args.out), indent=1))
        return
    if args.what in ("select", "picks", "score"):
        from codenames import c10score

        getattr(c10score, args.what)(args.runs, args.out)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.what == "prefixes":
        prefixes(args.out)
        return
    meter = Meter(args.cap)
    c4.price_models([LISTENER], args.out)
    if args.what == "pilot":  # the listen and speak parts on the pilot boards
        listen(PILOT, range(args.n[0]), TOP, 0, meter, args.out, args.workers)
        speak(PILOT, args.n[0], TOP, meter, args.out, args.workers)
    elif args.what == "listen":
        span = args.boards or range(args.n[0])
        listen(args.seed[0], span, args.k, args.repeat, meter, args.out, args.workers)
    else:
        speak(args.seed[0], args.n[0], args.k, meter, args.out, args.workers)
    print(f"spent ${meter.spent:.4f} of ${args.cap:.2f} in {meter.calls} calls", flush=True)
    if meter.spent > args.cap:
        sys.exit(1)


if __name__ == "__main__":
    main()
