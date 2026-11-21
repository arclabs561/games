"""Experiment C12 (experiments/plans/C12.md), part A: gpt-oss-120b as giver with the whole
clue rule, the costs, one retry and more reasoning.

    uv run python -m codenames.c12 lists      # here: E and the fallback on every board
    python -m codenames.c12 give --arms A0,A2,A3 --seed 12999 --n 10 --cap 0.03 --out OUT
    python -m codenames.c12 guess --models llama-3.3-70b,deepseek-v3.2 --arms E,A0,A3 \\
        --clues FILE --seed 12004 --boards 0-500 --cap 0.30 --out OUT
    python -m codenames.c12 repeat --models ... --arms E,A0,A3 --clues FILE --seed S \\
        --boards 0-7 --times 2 --cap C --out OUT
    uv run python -m codenames.c12 score      # here, with the word vectors

**Arms.** E is EmbeddingGemma 2's expected-value giver over the clue words that
`legality.strict_legal` allows on the board. A0 is C4's harness (`llm.giver_prompt`,
reasoning effort low, 1,024 tokens, no retry). A2 states the whole rule and the scoring
(`llm.giver_prompt_full`), and a clue that the strict check refuses gets one retry
(`llm.retry_message`); effort low, 1,024 tokens. A3 is A2 with effort medium and 4,096
tokens. Every arm's clue must pass the strict check. A reply with no strictly legal clue
after its tries gets the fallback: E's best strictly legal clue for one word.

`give` writes `give-<arm>-<seed>.json` (every reply, its reading and the retry) and
`clues.txt`: one line per arm and seed with the clue each board is played with (the
fallback substituted), and `status.txt`: a line `<arm>-status` per arm with why (ok,
retry, or the reason for the fallback). The validation job's summary applies the cap rule: an arm whose replies
reached the token cap on more than 5% of boards moves to the next cap in the test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from codenames import c4
from codenames.game import Board

HERE = Path(__file__).parent
RESULTS = HERE / "results"
E_LINES = RESULTS / "c12-e.txt"
CLUES = RESULTS / "c12-clues.txt"
STATUS = RESULTS / "c12-status.txt"
GUESSES = RESULTS / "c12-guesses.txt"
RUNS = RESULTS / "c12-runs.json"
OUT = RESULTS / "c12.json"
PLAN = "experiments/plans/C12.md"
PILOT, VAL, TEST = 12999, 12005, 12004
N = {PILOT: 10, VAL: 50, TEST: 500}
GIVER = "gpt-oss-120b"
EFFORT = {"A0": "low", "A2": "low", "A3": "medium"}
CAPS = {"A0": (1024, 2048), "A2": (1024, 2048), "A3": (4096, 8192)}
LLM_GUESSERS = ("llama-3.3-70b", "deepseek-v3.2")
CUT_RULE = 0.05
HOLM_Z, MDE_Z = 3.08, 2.80


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# E and the fallback (here, with the vectors).


def lists() -> str:
    """Clue lines: E (strict, counts 1 to 4) and F1 (strict, count 1) on every board."""
    from codenames.c4score import BASE, spaces_with
    from codenames.human_pilot import clue_pool
    from codenames.players import ExpectedGiver

    sp, tau = spaces_with([])[BASE]
    given: dict[int, dict[str, list[dict]]] = {}
    for seed in (PILOT, VAL, TEST):
        for b in c4.boards(seed, N[seed]):
            pool = clue_pool(b)
            for name, nmax in (("E", 4), ("F1", 1)):
                c = ExpectedGiver(sp, tau, nmax=nmax, candidates=pool)(b)
                given.setdefault(seed, {}).setdefault(name, []).append(
                    {"clue": c.word, "count": c.count}
                )
        print(seed, "done", flush=True)
    return c4.clue_lines(given)


# Giving (in a job).


def read_strict(reply: str, board: Board) -> dict:
    """The clue in a reply under the strict rule: status ok, empty, unparsed, count (a
    count below 1) or illegal (with the strict check's flags)."""
    from codenames.legality import flags, strict_legal

    if not reply.strip():
        return {"status": "empty"}
    m = c4.CLUE.search(reply)
    if not m:
        return {"status": "unparsed"}
    word, n = m[1].lower(), int(m[2])
    if n < 1:
        return {"status": "count", "word": word, "count": n}
    if not strict_legal(word, board.words):
        return {"status": "illegal", "word": word, "count": n, "flags": flags(word, board.words)}
    return {"status": "ok", "word": word, "count": n}


def giver_model(arm: str):
    m = c4.MODELS[GIVER]
    return replace(m, extra={**m.extra, "reasoning_effort": EFFORT[arm]})


def prompt_for(arm: str, board: Board) -> str:
    from codenames.llm import giver_prompt, giver_prompt_full

    return giver_prompt(board) if arm == "A0" else giver_prompt_full(board)


def give_one(hosted, arm: str, i: int, board: Board, seed: int, fallback: tuple[str, int]) -> dict:
    """One board: the first reply, the retry if the first clue is strictly illegal (A2, A3),
    and the clue played."""
    from codenames.llm import retry_message

    prompt = prompt_for(arm, board)
    rec = hosted.call(prompt, {"part": "give", "arm": arm, "seed": seed, "board": i})
    first = {
        "reply": rec.get("reply", ""),
        "finish": rec.get("finish"),
        **read_strict(rec.get("reply", ""), board),
    }
    row: dict = {"board": i, "first": first, "retry": None}
    final = first
    if first["status"] == "illegal" and arm != "A0":
        history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rec.get("reply", "")},
        ]
        msg = retry_message(first["word"], first["flags"])
        rec2 = hosted.call(msg, {"part": "retry", "arm": arm, "seed": seed, "board": i}, history)
        final = {
            "reply": rec2.get("reply", ""),
            "finish": rec2.get("finish"),
            **read_strict(rec2.get("reply", ""), board),
        }
        row["retry"] = final
    if final["status"] == "ok":
        row["clue"], row["count"] = final["word"], final["count"]
        row["why"] = "retry" if row["retry"] is not None else "ok"
    else:
        row["clue"], row["count"] = fallback
        row["why"] = f"fallback-{final['status']}"
    return row


def give(
    arms,
    seed: int,
    n: int,
    caps: dict[str, int],
    meter,
    out: Path,
    workers: int,
    repeat: int,
    times: int,
) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from codenames.hosted import Hosted, OverBudgetError

    bs = c4.boards(seed, n)
    fall = c4.read_clue_lines(E_LINES.read_text())[("F1", seed)]
    given: dict[str, list[dict]] = {}
    summary: dict = {"seed": seed, "n": n, "arms": {}}
    for arm in arms:
        model = giver_model(arm)
        hosted = Hosted(model, meter, out / f"calls-{GIVER}-{arm}.jsonl", caps[arm])

        def one(i: int, arm=arm, hosted=hosted):
            c, k, _ = fall[i]
            assert c is not None and k is not None
            try:
                return give_one(hosted, arm, i, bs[i], seed, (c, k))
            except OverBudgetError:
                return {"board": i, "stopped": True}

        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(one, range(n)))
        (out / f"give-{arm}-{seed}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "model": vars(model),
                    "max_tokens": caps[arm],
                    "seed": seed,
                    "rows": rows,
                },
                indent=1,
            )
            + "\n"
        )
        given[arm] = rows
        done = [r for r in rows if not r.get("stopped")]
        cut = sum(
            r["first"]["finish"] == "length" or (r["retry"] or {}).get("finish") == "length"
            for r in done
        )
        s = {
            "boards": n,
            "stopped": n - len(done),
            "why": dict(Counter(r["why"] for r in done)),
            "first_status": dict(Counter(r["first"]["status"] for r in done)),
            "cut_boards": cut,
            "cut_share": cut / len(done) if done else None,
            "max_tokens": caps[arm],
            "mean_count": statistics.mean(r["count"] for r in done) if done else None,
        }
        s["cap_rule"] = (
            "raise"
            if s["cut_share"] is not None
            and s["cut_share"] > CUT_RULE
            and caps[arm] == CAPS[arm][0]
            else "keep"
        )
        if repeat and times:
            s["repeat"] = repeat_givers(
                hosted, arm, bs[:repeat], seed, times, rows[:repeat], workers
            )
        summary["arms"][arm] = s
        print(arm, seed, s["why"], f"cut {s['cut_share']}", f"${meter.spent:.4f}", flush=True)
        log = out / f"calls-{GIVER}-{arm}.jsonl"
        s["calls"] = c4.call_stats(log) if log.exists() else {"calls": 0}
    lines, status = [], []
    for arm, rows in given.items():
        cells = [f"{r['clue']}/{r['count']}" if not r.get("stopped") else "-/stopped" for r in rows]
        lines.append(" ".join([arm, str(seed), *cells]))
        status.append(
            " ".join([f"{arm}-status", str(seed), *(r.get("why", "stopped") for r in rows)])
        )
    text = "\n".join(lines) + "\n"
    (out / "clues.txt").write_text(text)
    (out / "status.txt").write_text("\n".join(status) + "\n")
    summary["clues_sha256"] = sha256(text)
    summary["spent"], summary["cap"] = meter.spent, meter.cap
    (out / "give.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("clues.txt sha256", summary["clues_sha256"], f"spent ${meter.spent:.4f}")
    return summary


def repeat_givers(hosted, arm: str, bs, seed: int, times: int, rows, workers: int = 8) -> dict:
    """The first prompt of the first boards sent `times` more times, in parallel: how often
    the clue read from it changed, with empty and cut replies."""
    jobs = [
        (prompt_for(arm, b), {"part": f"repeat-{t + 1}", "arm": arm, "seed": seed, "board": i})
        for i, (b, row) in enumerate(zip(bs, rows, strict=True))
        if not row.get("stopped")
        for t in range(times)
    ]
    recs = c4.run_many(hosted.call, jobs, workers)
    seen: dict[int, set] = {}
    empty = cut = sent = 0
    for (_, tag), rec in zip(jobs, recs, strict=True):
        i = int(tag["board"])
        first = rows[i]["first"]
        seen.setdefault(i, {(first.get("word"), first.get("count"))})
        if rec is None:
            continue
        sent += 1
        r = read_strict(rec.get("reply", ""), bs[i])
        seen[i].add((r.get("word"), r.get("count")))
        empty += r["status"] == "empty"
        cut += rec.get("finish") == "length"
    changed = sum(len(v) > 1 for v in seen.values())
    return {
        "prompts": len(seen),
        "times": times,
        "sent": sent,
        "changed": changed,
        "empty": empty,
        "cut": cut,
    }


# Guessing (in a job).


def guess(
    models, arms, clues: str, seed: int, boards: range, times: int, meter, out: Path, workers: int
) -> dict:
    """C4's guesser prompt with each arm's clue and count, once per distinct (board, clue,
    count) and model, played out (`c11.send_all`); with times > 0, the repeat check."""
    from codenames import c11
    from codenames.hosted import Hosted

    by_arm = c11.targets(clues, seed, boards, arms)
    keys = {k: b for rows in by_arm.values() for k, b in rows}
    lines: list[str] = []
    summary: dict = {"seed": seed, "boards": [boards.start, boards.stop], "models": {}}
    for name in models:
        hosted = Hosted(
            c4.MODELS[name], meter, out / f"calls-{name}.jsonl", c4.tokens(name, "guess")
        )
        s: dict = {"max_tokens": c4.tokens(name, "guess"), "distinct_prompts": len(keys)}
        for send in range(1, times + 1) if times else [0]:
            recs = c11.send_all(
                hosted,
                keys,
                c4.WORKERS.get(name, workers),
                "guess" if not send else f"repeat-{send}",
            )
            lines += c11.guess_lines(name, by_arm, recs, send)
            s[f"send-{send}"] = c11.reply_stats(list(recs.values()), list(recs), keys)
            print(name, send, s[f"send-{send}"]["final_status"], f"${meter.spent:.4f}", flush=True)
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


def ensure_vectors() -> None:
    """In a job: copy the word vectors from the branch outputs/codenames-embed."""
    import shutil
    import subprocess
    import tempfile

    from codenames import embed

    if any(embed.COMPUTED.glob("*.npz")):
        return
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch", "outputs/codenames-embed",
         "https://github.com/arclabs561/games.git", str(tmp / "embed")],
        check=True,
    )  # fmt: skip
    embed.COMPUTED.mkdir(parents=True, exist_ok=True)
    for f in (tmp / "embed" / "codenames" / "vectors").glob("*.npz"):
        shutil.copy(f, embed.COMPUTED / f.name)


# Scoring (here).


def read_status(text: str) -> dict[tuple[str, int], list[str]]:
    out = {}
    for line in text.splitlines():
        name, seed, *cells = line.split()
        if name.endswith("-status"):
            out[(name.removesuffix("-status"), int(seed))] = cells
    return out


def exact_values(clues: str, seed: int, arms) -> dict:
    """{arm: {guesser: per-board exact value}} under the seven strangers and EmbeddingGemma
    2's own guesser (and words found under the strangers)."""
    from codenames.c4score import BASE, new_words, score_clue, spaces_with
    from codenames.game import Clue
    from codenames.players import embedding_weigher

    spaces = spaces_with(new_words(clues))
    weighers = {m: embedding_weigher(*spaces[m]) for m in spaces}
    given = c4.read_clue_lines(clues)
    bs = c4.boards(seed, N[seed])
    out: dict = {}
    for arm in arms:
        rows = given[(arm, seed)]
        vals = []
        for b, (c, k, why) in zip(bs, rows, strict=True):
            assert c is not None and k is not None, (arm, seed, why)
            vals.append(score_clue(weighers, b, Clue(c, k)))
        out[arm] = {
            "strangers": np.array(
                [np.mean([v["value"][m] for m in spaces if m != BASE]) for v in vals]
            ),
            "strangers_found": np.array(
                [np.mean([v["found"][m] for m in spaces if m != BASE]) for v in vals]
            ),
            "own": np.array([v["value"][BASE] for v in vals]),
        }
    return out


def holm_pair(ds: dict[str, np.ndarray]) -> dict:
    from codenames.c4score import bootstrap, holm

    res = {h: bootstrap(d) for h, d in ds.items()}
    rej = holm({h: r["p"] for h, r in res.items()})
    for h, d in ds.items():
        sd = float(np.std(d, ddof=1))
        res[h].update(
            holm_rejects=bool(rej[h]),
            sd=sd,
            mde_at_0_05=MDE_Z * sd / np.sqrt(len(d)),
            mde_at_holm=HOLM_Z * sd / np.sqrt(len(d)),
        )
    return res


def ratio_boot(num: np.ndarray, den: np.ndarray, seed: int = 0, resamples: int = 10_000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, (resamples, n))
    r = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"ratio": float(num.mean() / den.mean()), "ci": [float(lo), float(hi)], "n": n}


def score(
    clues: str = str(CLUES),
    guesses: str = str(GUESSES),
    status_spec: str = str(STATUS),
    runs: Path = RUNS,
) -> dict:
    """Every number of part A, from the test's clue, status and guess lines (committed
    files, or hf: paths in the private dataset)."""
    from codenames.c4score import bootstrap, rounded
    from codenames.c11 import column, read_guesses, reply_counts

    clue_text = fetch_text(clues)
    guess_text = fetch_text(guesses)
    text = clue_text + E_LINES.read_text()
    g = read_guesses(guess_text)
    status = read_status(fetch_text(status_spec))
    arms = ("E", "A0", "A2", "A3")
    ex = exact_values(text, TEST, arms)
    played = {
        (a, m): column(g[(a, m, TEST, 0)])
        for a in arms
        for m in (*LLM_GUESSERS, GIVER)
        if (a, m, TEST, 0) in g
    }
    lm = {a: np.mean([played[(a, m)] for m in LLM_GUESSERS], axis=0) for a in ("E", "A0", "A3")}
    keep = ~np.isnan(lm["A3"]) & ~np.isnan(lm["E"])
    primary = holm_pair(
        {
            "h_strangers": ex["A3"]["strangers"] - ex["E"]["strangers"],
            "h_models": (lm["A3"] - lm["E"])[keep],
        }
    )
    sec: dict = {
        "boards_kept_h_models": int(keep.sum()),
        "boards_dropped_h_models": int((~keep).sum()),
        "a0_minus_e": {
            "strangers": bootstrap(ex["A0"]["strangers"] - ex["E"]["strangers"]),
            "models": bootstrap((lm["A0"] - lm["E"])[~np.isnan(lm["A0"] - lm["E"])]),
        },
        "a2_minus_a0_strangers": bootstrap(ex["A2"]["strangers"] - ex["A0"]["strangers"]),
        "a3_minus_a2_strangers": bootstrap(ex["A3"]["strangers"] - ex["A2"]["strangers"]),
        "arm_means": {
            a: {
                "strangers": bootstrap(ex[a]["strangers"]),
                "strangers_found": bootstrap(ex[a]["strangers_found"]),
                **{
                    m: bootstrap(played[(a, m)][~np.isnan(played[(a, m)])])
                    for m in (*LLM_GUESSERS, GIVER)
                    if (a, m) in played
                },
            }
            for a in arms
        },
        "per_guesser_a3_minus_e": {
            m: bootstrap(
                (played[("A3", m)] - played[("E", m)])[
                    ~np.isnan(played[("A3", m)] - played[("E", m)])
                ]
            )
            for m in LLM_GUESSERS
        },
    }
    if ("A3", GIVER) in played:
        own = played[("A3", GIVER)]
        ok = ~np.isnan(own) & keep
        sec["retention"] = {
            "A3": ratio_boot(lm["A3"][ok], own[ok]),
            "E": ratio_boot(ex["E"]["strangers"], ex["E"]["own"]),
        }
    given = c4.read_clue_lines(text)
    sec["giving"] = {
        a: {
            "why": dict(Counter(status[(a, TEST)])),
            "fallback_rate": float(np.mean([w.startswith("fallback") for w in status[(a, TEST)]])),
            "mean_count": float(np.mean([k for _, k, _ in given[(a, TEST)]])),
        }
        for a in arms
        if (a, TEST) in status
    }
    fb_zero = {}
    for a in ("A0", "A2", "A3"):
        fb = np.array([w.startswith("fallback") for w in status[(a, TEST)]])
        fb_zero[a] = bootstrap(np.where(fb, 0.0, ex[a]["strangers"]) - ex["E"]["strangers"])
    sec["fallback_as_zero_minus_e"] = fb_zero
    sec["replies"] = {
        m: reply_counts([c for a in arms for c in g.get((a, m, TEST, 0), [])])
        for m in (*LLM_GUESSERS, GIVER)
    }
    result = {
        "plan": PLAN,
        "part": "A",
        "seed": TEST,
        "clues_sha256": sha256(clue_text),
        "guesses_sha256": sha256(guess_text),
        "inputs": {"clues": clues, "guesses": guesses, "status": status_spec},
        "primary": primary,
        "secondary": sec,
    }
    if runs.exists():
        result["runs"] = json.loads(runs.read_text())
    return rounded(result, 4)


def main() -> None:
    from codenames.c11 import parse_range
    from codenames.hosted import Meter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["lists", "give", "guess", "repeat", "score"])
    parser.add_argument("--arms", default="A0,A2,A3")
    parser.add_argument("--models", default=",".join(LLM_GUESSERS))
    parser.add_argument("--seed", type=int, default=PILOT)
    parser.add_argument("--n", type=int, help="give: boards")
    parser.add_argument("--boards", type=parse_range, help="guess: boards, as 0-500")
    parser.add_argument("--clues", help="guess, score: clue lines (a file, or hf:path)")
    parser.add_argument("--guesses", default=str(GUESSES), help="score: a file, or hf:path,...")
    parser.add_argument("--status", default=str(STATUS), help="score: a file, or hf:path")
    parser.add_argument("--max-tokens", default="", help="give: arm=tokens,... (the cap rule)")
    parser.add_argument("--repeat", type=int, default=0, help="give: boards whose prompt is resent")
    parser.add_argument("--times", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "lists":
        text = lists()
        E_LINES.write_text(text)
        print(E_LINES, sha256(text))
        return
    if args.what == "score":
        ensure_vectors()
        result = score(str(args.clues or CLUES), args.guesses, args.status)
        if args.out and args.out.suffix != ".json":
            args.out.mkdir(parents=True, exist_ok=True)
            args.out = args.out / OUT.name
        (args.out or OUT).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result["primary"], indent=1))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    meter = Meter(args.cap)
    if args.what == "give":
        caps = {a: CAPS[a][0] for a in arms}
        for item in filter(None, args.max_tokens.split(",")):
            a, t = item.split("=")
            assert int(t) in CAPS[a], (a, t)
            caps[a] = int(t)
        c4.price_models([GIVER], args.out)
        give(
            arms,
            args.seed,
            args.n or N[args.seed],
            caps,
            meter,
            args.out,
            args.workers,
            args.repeat,
            args.times,
        )
        return
    models = args.models.split(",")
    c4.price_models(models, args.out)
    clues = fetch_text(args.clues) + E_LINES.read_text()
    times = args.times if args.what == "repeat" else 0
    guess(models, arms, clues, args.seed, args.boards, times, meter, args.out, args.workers)


if __name__ == "__main__":
    main()
