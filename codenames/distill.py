"""Experiment C8 (experiments/plans/C8.md): turns labeled by gpt-oss-120b, for training the
LoRA embedding model of C2.

The teacher hears the person's clue on a board and names one word. Its turns are:

- **resampled**: for each of C1's 4,820 training turns, four new boards. Each word of the
  turn's board is kept, or with probability one half replaced by a word drawn uniformly
  from the words on boards of C1's training games that are not already on this board and
  against which the clue is legal (codenames.words.legal). The clue stays the person's.
- **relabeled**: the training turns themselves, with the teacher's word in place of the
  person's. These also give the teacher's agreement with people.
- **repeats**: the first 100 resampled turns, sent twice more, to see how often the
  teacher changes its answer at temperature 0 with a fixed seed.

The prompt and settings are C4's (`llm.guesser_prompt` with count 1; `c4.MODELS`). The
label is the first board word of `llm.parse_ranking`; a reply that names no board word
gives no label. The job runs inside a Hugging Face Job, which holds the token:

    python -m codenames.distill label --cap 2.5 --out OUT

It writes teacher.jsonl (one line per turn, the input of `codenames.lora --teacher`),
repeats.json, calls-gpt-oss-120b.jsonl (every request and reply) and label.json (cost,
counts, and the teacher's hit on each training turn as a string of 0s and 1s). The
labels are derived from Cultural Codes boards and from provider outputs, so they go to
the private dataset, not to git.
"""

import argparse
import hashlib
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from codenames.norms import Guess
from codenames.words import legal

K = 4
SEED = 8000
P_REPLACE = 0.5
REPEATS = 100
TEACHER = "gpt-oss-120b"


@dataclass(frozen=True)
class TeacherTurn:
    """One turn for the teacher. turn is the index of the training turn it came from, in
    the order of codenames.split.turns("game:train"); copy is 0 to K-1 for a resampled
    board and -1 for the training turn itself."""

    kind: str  # "resampled" or "relabeled"
    turn: int
    copy: int
    clue: str
    words: tuple[str, ...]


def board_vocab(turns: Sequence[Guess]) -> tuple[str, ...]:
    """The words on the boards of these turns, sorted."""
    return tuple(sorted({w for g in turns for w in g.words}))


def resample_board(
    clue: str,
    words: tuple[str, ...],
    vocab: Sequence[str],
    rng: random.Random,
    p: float = P_REPLACE,
) -> tuple[str, ...]:
    """The board with each word kept or, with probability p, replaced by a vocabulary
    word not on the board (as it stands at that point) and not ruled out by the clue.
    A word with no possible replacement is kept."""
    board = list(words)
    for i in range(len(board)):
        if rng.random() < p:
            pool = [v for v in vocab if v not in board and legal(clue, (v,))]
            if pool:
                board[i] = rng.choice(pool)
    return tuple(board)


def resample_boards(
    turns: Sequence[Guess], vocab: Sequence[str], k: int = K, seed: int = SEED
) -> list[TeacherTurn]:
    """k resampled boards for each turn, in the order of the turns."""
    rng = random.Random(seed)
    return [
        TeacherTurn("resampled", i, j, g.clue, resample_board(g.clue, g.words, vocab, rng))
        for i, g in enumerate(turns)
        for j in range(k)
    ]


def relabeled(turns: Sequence[Guess]) -> list[TeacherTurn]:
    return [TeacherTurn("relabeled", i, -1, g.clue, g.words) for i, g in enumerate(turns)]


def label_of(reply: str, words: tuple[str, ...]) -> tuple[str | None, int]:
    """The teacher's word (None when the reply names no board word), and how many board
    words the reply names."""
    from codenames.c4 import read_ranking

    r = read_ranking(reply, words)
    return (r["ranking"][0] if r["named"] else None), r["named"]


def teacher_rows(turns: Sequence[TeacherTurn], records: Sequence[dict | None]) -> list[dict]:
    """One row per turn: the turn, and the label, or why there is none."""
    out = []
    for t, rec in zip(turns, records, strict=True):
        row: dict = asdict(t) | {"words": list(t.words)}
        if rec is None:
            row |= {"label": None, "why": "stopped"}
        else:
            word, named = label_of(rec["reply"], t.words)
            row |= {
                "label": word,
                "named": named,
                "finish": rec.get("finish"),
                "why": None if word else "named no board word",
            }
        out.append(row)
    return out


def agreement(rows: list[dict], people: Sequence[Guess]) -> str:
    """The teacher's hit on each training turn: 1 where its word is the person's, 0
    where it is another word, - where there is no label."""
    by_turn = {r["turn"]: r for r in rows if r["kind"] == "relabeled"}
    out = []
    for i, g in enumerate(people):
        r = by_turn.get(i)
        if r is None or r["label"] is None:
            out.append("-")
        else:
            assert r["clue"] == g.clue and tuple(r["words"]) == g.words, i
            out.append("1" if r["label"] == g.picked else "0")
    return "".join(out)


def repeat_changes(first: list[dict], again: Sequence[Sequence[dict | None]]) -> dict:
    """How many of the repeated turns changed their label or their reply at least once."""
    label, reply, sent = 0, 0, 0
    for i, row in enumerate(first):
        recs = [r for a in again if (r := a[i]) is not None]
        sent += len(recs)
        words = tuple(row["words"])
        if any(label_of(r["reply"], words)[0] != row["label"] for r in recs):
            label += 1
        if any(r["reply"] != row["reply"] for r in recs):
            reply += 1
    return {
        "turns": len(first),
        "times": len(again),
        "sent": sent,
        "label_changed": label,
        "reply_changed": reply,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(args) -> None:
    from codenames.c4 import MODELS, call_stats, price_models, run_many, tokens
    from codenames.hosted import Hosted, Meter, OverBudgetError
    from codenames.llm import guesser_prompt
    from codenames.split import turns as split_turns

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    people = tuple(t.guess for t in split_turns("game:train"))
    games = [t.game for t in split_turns("game:train")]
    vocab = board_vocab(people)
    todo = resample_boards(people, vocab, K, args.seed) + relabeled(people)
    if args.limit:
        todo = todo[: args.limit]
    n_rep = min(args.repeats, sum(t.kind == "resampled" for t in todo))
    meter = Meter(args.cap)
    price_models([TEACHER], out)
    hosted = Hosted(
        MODELS[TEACHER], meter, out / f"calls-{TEACHER}.jsonl", tokens(TEACHER, "guess")
    )
    deadline = time.time() + 60 * args.stop_after

    def call(prompt: str, tag: dict) -> dict:
        if time.time() > deadline:
            raise OverBudgetError("past the time limit")
        return hosted.call(prompt, tag)

    jobs = [
        (guesser_prompt(t.clue, 1, t.words), {"part": t.kind, "turn": t.turn, "copy": t.copy})
        for t in todo
    ]
    jobs += [(p, tag | {"part": "repeat", "time": k}) for k in (1, 2) for p, tag in jobs[:n_rep]]
    start = time.time()
    records = run_many(call, jobs, args.workers)
    seconds = time.time() - start
    first = records[: len(todo)]
    rows = teacher_rows(todo, first)
    for row, rec in zip(rows, first, strict=True):
        row["reply"] = rec["reply"] if rec else None
    path = out / "teacher.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rest = records[len(todo) :]
    reps = repeat_changes(rows[:n_rep], [rest[:n_rep], rest[n_rep:]])
    (out / "repeats.json").write_text(json.dumps(reps, indent=1) + "\n")
    hits = agreement(rows, people)
    summary = {
        "plan": "experiments/plans/C8.md",
        "teacher": TEACHER,
        "model": vars(MODELS[TEACHER]),
        "params": hosted.params,
        "seed": args.seed,
        "k": K,
        "p_replace": P_REPLACE,
        "vocab": len(vocab),
        "turns": {
            kind: {
                "sent": sum(1 for r in rows if r["kind"] == kind),
                "labeled": sum(1 for r in rows if r["kind"] == kind and r["label"]),
                "named_none": sum(
                    1 for r in rows if r["kind"] == kind and r["why"] == "named no board word"
                ),
                "stopped": sum(1 for r in rows if r["kind"] == kind and r["why"] == "stopped"),
                "truncated": sum(
                    1 for r in rows if r["kind"] == kind and r.get("finish") == "length"
                ),
            }
            for kind in ("resampled", "relabeled")
        },
        "train_hits": hits,
        "train_games": games,
        "repeats": reps,
        "spent": meter.spent,
        "cap": args.cap,
        "seconds": round(seconds, 1),
        "calls": call_stats(out / f"calls-{TEACHER}.jsonl"),
        "teacher_sha256": sha256(path),
    }
    (out / "label.json").write_text(json.dumps(summary, indent=1) + "\n")
    labeled = sum(c in "01" for c in hits)
    if labeled:
        print(f"teacher agrees with people on {hits.count('1')}/{labeled} training turns")
    print(json.dumps({k: summary[k] for k in ("turns", "repeats", "spent", "teacher_sha256")}))
    print(f"spent ${meter.spent:.4f} of ${args.cap:.2f} in {meter.calls} calls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["label"])
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the board resampling")
    parser.add_argument("--limit", type=int, default=0, help="only the first N turns (a pilot)")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--workers", type=int, default=32, help="calls in flight")
    parser.add_argument("--cap", type=float, default=0.0, help="dollars this job may spend")
    parser.add_argument("--stop-after", type=float, default=150, help="minutes; then no new calls")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    label(args)


if __name__ == "__main__":
    main()
