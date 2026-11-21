"""Experiment C14 (experiments/plans/C14.md): C6 again, with reply caps that cut off fewer
than 5% of replies and strictly legal candidates.

    uv run python -m codenames.c14 lists      # here: the strict shortlists and F
    python -m codenames.c14 choose --models qwen3.8-27b,gpt-oss-120b --seed 14005 --n 50 \\
        --repeat 20 --cap 0.08 --out OUT       # in a Hugging Face job
    python -m codenames.c14 guess --picks FILE --seed 14004 --boards 0-300 --cap 0.15 --out OUT
    uv run python -m codenames.c14 score      # here, with the word vectors

**Candidates.** EmbeddingGemma 2's 30-clue shortlist (`actions.shortlist`) drawn from the
clue words that `legality.strict_legal` allows on the board, times counts 1 to 4, in C6's
order. F, the first choice, is the action of the 120 that EmbeddingGemma 2's own Luce
guesser values most. `lists` writes `results/c14-actions.txt` and `results/c14-f.txt`.

**Choosers.** Qwen3.8-27B (thinking off) and gpt-oss-120b (reasoning effort low), both
pinned to DeepInfra, with C6's prompt (`rerank.rerank_prompt`) and reading
(`rerank.read_choice`; a reply cut at the cap is not read as a pick). Caps: Qwen3.8-27B 256
tokens, gpt-oss-120b 1,024, each raised to 1,024 and 2,048 for the test if more than 5%
of its validation replies reach the cap (the summary's `cap_rule`). An empty reply is sent
once more at the end of the job; if more than 5% of a model's validation replies are
still empty, the summary says stop. A board without a pick is played with F.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from codenames import c4
from codenames.actions import action_lines, read_action_lines, sha256

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ACTIONS = RESULTS / "c14-actions.txt"
F_LINES = RESULTS / "c14-f.txt"
PICKS = RESULTS / "c14-picks.txt"
GUESSES = RESULTS / "c14-guesses.txt"
RUNS = RESULTS / "c14-runs.json"
OUT = RESULTS / "c14.json"
PLAN = "experiments/plans/C14.md"
PILOT, VAL, TEST = 14999, 14005, 14004
N = {PILOT: 10, VAL: 50, TEST: 300}
CHOOSERS = ("qwen3.8-27b", "gpt-oss-120b")
CAPS = {"qwen3.8-27b": (256, 1024), "gpt-oss-120b": (1024, 2048)}
LLM_GUESSERS = ("llama-3.3-70b", "deepseek-v3.2")
RULE = 0.05
DELTA = 0.05
HOLM_Z, MDE_Z = 3.08, 2.80


EXTRA: dict[str, dict] = {}  # request fields added per model (--extra), by amendment


def chooser_model(name: str):
    from dataclasses import replace

    from codenames.rerank import MODELS

    m = MODELS[name]
    assert m.provider == "deepinfra", (name, m.provider)
    return replace(m, extra={**m.extra, **EXTRA.get(name, {})})


# Lists (here).


def lists() -> tuple[str, str]:
    from codenames.actions import expand, proposer, shortlist, values
    from codenames.human_pilot import clue_pool
    from codenames.listen import argmax_first
    from codenames.players import embedding_weigher

    sp, tau = proposer()
    weigh = embedding_weigher(sp, tau)
    text, given = "", {}
    for seed in (PILOT, VAL, TEST):
        bs = c4.boards(seed, N[seed])
        clue_lists = [shortlist(b, sp, tau, pool=clue_pool(b)) for b in bs]
        text += action_lines(seed, clue_lists)
        rows = []
        for b, cl in zip(bs, clue_lists, strict=True):
            acts = expand(cl)
            c, k = acts[argmax_first(values(weigh, b, acts))]
            rows.append({"clue": c, "count": k})
        given[seed] = {"F": rows}
        print(seed, "done", flush=True)
    return text, c4.clue_lines(given)


# Choosing (in a job).


def choose_model(
    name: str, seed: int, n: int, cap: int, meter, out: Path, workers: int, repeat: int, times: int
) -> tuple[list[dict], dict]:
    from codenames.hosted import Hosted
    from codenames.rerank import read_choice, rerank_prompt

    bs = c4.boards(seed, n)
    lists_ = read_action_lines(ACTIONS.read_text())[seed][:n]
    hosted = Hosted(chooser_model(name), meter, out / f"calls-{name}.jsonl", cap)
    jobs = [
        (rerank_prompt(b, c), {"part": "choose", "seed": seed, "board": i})
        for i, (b, c) in enumerate(zip(bs, lists_, strict=True))
    ]
    recs = c4.run_many(hosted.call, jobs, workers)
    first = [
        None if r is None else read_choice(r["reply"], c, r.get("finish"))["why"]
        for r, c in zip(recs, lists_, strict=True)
    ]
    again = [i for i, w in enumerate(first) if w == "empty"]
    for i, rec in zip(
        again, c4.run_many(hosted.call, [jobs[i] for i in again], workers), strict=True
    ):
        recs[i] = rec
    rows = []
    for i, (rec, c) in enumerate(zip(recs, lists_, strict=True)):
        if rec is None:
            rows.append({"board": i, "stopped": True})
            continue
        ch = read_choice(rec["reply"], c, rec.get("finish"))
        pos = None if not ch["clue"] else c.index(ch["clue"])
        rows.append(
            {
                "board": i,
                "reply": rec["reply"],
                "finish": rec.get("finish"),
                "first_why": first[i],
                "sends": 1 + (i in again),
                "position": pos,
                **ch,
            }
        )
    done = [r for r in rows if not r.get("stopped")]
    cut = sum(r["finish"] == "length" for r in done)
    empty = sum(r["why"] == "empty" for r in done)
    secs = []
    if (out / f"calls-{name}.jsonl").exists():
        for line in (out / f"calls-{name}.jsonl").read_text().splitlines():
            x = json.loads(line)
            if isinstance(x.get("seconds"), int | float):
                secs.append(x["seconds"])
    s: dict = {
        "boards": n,
        "max_tokens": cap,
        "stopped": n - len(done),
        "why": dict(Counter(str(r["why"]) for r in done)),
        "cut": cut,
        "cut_share": cut / len(done) if done else None,
        "empty_after_resend": empty,
        "empty_share": empty / len(done) if done else None,
        "resent": len(again),
        "seconds_median": statistics.median(secs) if secs else None,
        "positions": dict(Counter(str(r["position"]) for r in done)),
        "count_1_share": float(np.mean([r["count"] == 1 for r in done if r["clue"]]))
        if done
        else None,
    }
    s["cap_rule"] = (
        "raise" if s["cut_share"] and s["cut_share"] > RULE and cap == CAPS[name][0] else "keep"
    )
    s["empty_rule"] = "stop" if s["empty_share"] and s["empty_share"] > RULE else "go on"
    if repeat and times:
        rjobs = [
            (jobs[i][0], {"part": f"repeat-{t + 1}", "seed": seed, "board": i})
            for i in range(min(repeat, n))
            for t in range(times)
        ]
        rrecs = c4.run_many(hosted.call, rjobs, workers)
        changed = 0
        for i in range(min(repeat, n)):
            base = (rows[i].get("clue"), rows[i].get("count"), rows[i].get("why"))
            got = [
                read_choice(r["reply"], lists_[i], r.get("finish"))
                for r in rrecs[i * times : (i + 1) * times]
                if r
            ]
            changed += any((x["clue"], x["count"], x["why"]) != base for x in got)
        s["repeat"] = {"prompts": min(repeat, n), "times": times, "changed": changed}
    (out / f"choose-{name}-{seed}.json").write_text(
        json.dumps({"model": name, "seed": seed, "max_tokens": cap, "rows": rows}, indent=1) + "\n"
    )
    log = out / f"calls-{name}.jsonl"
    s["calls"] = c4.call_stats(log) if log.exists() else {"calls": 0}
    print(name, seed, s["why"], f"cut {s['cut_share']}", f"${meter.spent:.4f}", flush=True)
    return rows, s


def choose(
    models,
    seed: int,
    n: int,
    caps: dict[str, int],
    meter,
    out: Path,
    workers: int,
    repeat: int,
    times: int,
) -> dict:
    from codenames.rerank import price

    price(list(models), out)
    with ThreadPoolExecutor(len(models)) as pool:
        futures = {
            m: pool.submit(choose_model, m, seed, n, caps[m], meter, out, workers, repeat, times)
            for m in models
        }
        got = {m: f.result() for m, f in futures.items()}
    fall = c4.read_clue_lines(F_LINES.read_text())[("F", seed)]
    lines = []
    for m, (rows, _) in got.items():
        cells = [
            f"{r['clue']}/{r['count']}"
            if r.get("clue")
            else "-/" + ("stopped" if r.get("stopped") else str(r["why"]).replace(" ", "-"))
            for r in rows
        ]
        lines.append(" ".join([m, str(seed), *cells]))
        played = [
            f"{r['clue']}/{r['count']}" if r.get("clue") else f"{c}/{k}"
            for r, (c, k, _) in zip(rows, fall, strict=False)
        ]
        lines.append(" ".join([f"{m}-played", str(seed), *played]))
    text = "\n".join(lines) + "\n"
    (out / "choices.txt").write_text(text)
    summary = {
        "seed": seed,
        "models": {m: s for m, (_, s) in got.items()},
        "choices_sha256": sha256(text),
        "spent": meter.spent,
        "cap": meter.cap,
    }
    (out / "choose.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("choices.txt sha256", summary["choices_sha256"], f"spent ${meter.spent:.4f}")
    return summary


# Guessing (in a job), as in C12.


def guess(
    models,
    picks: str,
    seed: int,
    boards: range,
    times: int,
    meter,
    out: Path,
    workers: int,
    arms=None,
) -> dict:
    """C4's guesser prompt with each chooser's played action and F, once per distinct
    (board, clue, count) and model, played out (`c11.send_all`); with times > 0, the
    repeat check."""
    from codenames import c11
    from codenames.hosted import Hosted

    arms = arms or [f"{m}-played" for m in CHOOSERS] + ["F"]
    by_arm = c11.targets(picks + F_LINES.read_text(), seed, boards, arms)
    keys = {k: b for rows in by_arm.values() for k, b in rows}
    lines: list[str] = []
    summary: dict = {"seed": seed, "boards": [boards.start, boards.stop], "models": {}}
    for name in models:
        hosted = Hosted(
            c4.MODELS[name], meter, out / f"calls-{name}.jsonl", c4.tokens(name, "guess")
        )
        s: dict = {"max_tokens": c4.tokens(name, "guess"), "distinct_prompts": len(keys)}
        for send in range(1, times + 1) if times else [0]:
            part = "guess" if not send else f"repeat-{send}"
            recs = c11.send_all(hosted, keys, c4.WORKERS.get(name, workers), part)
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


def score(picks_spec: str = str(PICKS), guesses: str | None = None, runs: Path = RUNS) -> dict:
    """Every number of the plan. A chooser whose test job was stopped by its cap on some
    board is not scored: the plan quotes no number from a stopped run."""
    from codenames.actions import expand, values
    from codenames.c4score import bootstrap, holm, rounded
    from codenames.c6score import guessers
    from codenames.c11 import column, read_guesses

    weighers, strangers, _, _ = guessers()
    picks_text = fetch_text(picks_spec)
    picks = c4.read_clue_lines(picks_text + F_LINES.read_text())
    stopped = {m: sum(w == "stopped" for _, _, w in picks[(m, TEST)]) for m in CHOOSERS}
    scored = [m for m in CHOOSERS if stopped[m] == 0]
    bs = c4.boards(TEST, N[TEST])
    lists_ = read_action_lines(ACTIONS.read_text())[TEST]
    st = []
    for b, cl in zip(bs, lists_, strict=True):
        acts = expand(cl)
        st.append((acts, np.mean([values(weighers[g], b, acts) for g in strangers], axis=0)))

    def value(arm: str) -> np.ndarray:
        out = []
        for (acts, v), (c, k, _) in zip(st, picks[(arm, TEST)], strict=True):
            assert c is not None and k is not None, arm
            out.append(v[acts.index((c, k))])
        return np.array(out)

    f = value("F")
    oracle = np.array([v.max() for _, v in st])
    prim: dict = {}
    for m in scored:
        d = value(f"{m}-played") - f
        prim[m] = bootstrap(d)
        sd = float(np.std(d, ddof=1))
        prim[m].update(
            sd=sd,
            mde_at_holm=HOLM_Z * sd / np.sqrt(len(d)),
            mde_at_0_05=MDE_Z * sd / np.sqrt(len(d)),
        )
    rej = holm({m: prim[m]["p"] for m in scored})
    for m in scored:
        prim[m]["holm_rejects"] = bool(rej[m])
        prim[m]["inside_margin"] = bool(-DELTA < prim[m]["ci"][0] and prim[m]["ci"][1] < DELTA)
    sec: dict = {
        "oracle_minus_f": bootstrap(oracle - f),
        "f": bootstrap(f),
        "stopped_boards": stopped,
    }
    for m in CHOOSERS:
        if m not in scored:
            why = dict(Counter(str(w) for _, _, w in picks[(m, TEST)]))
            sec[m] = {"why": why, "not_scored": "stopped by the cap"}
            continue
        named = np.array([c is not None for c, _, _ in picks[(m, TEST)]])
        d = value(f"{m}-played") - f
        sec[m] = {
            "gap_closed": float(d.mean() / (oracle - f).mean()),
            "named_a_listed_action": int(named.sum()),
            "conditional_on_named": bootstrap(d[named]) if named.sum() > 1 else None,
            "why": dict(Counter(str(w) for _, _, w in picks[(m, TEST)])),
        }
    if guesses:
        g = read_guesses(fetch_text(guesses))
        for m in scored:
            for gm in LLM_GUESSERS:
                a, b = column(g[(f"{m}-played", gm, TEST, 0)]), column(g[("F", gm, TEST, 0)])
                ok = ~np.isnan(a - b)
                sec[m][f"played_out_{gm}"] = bootstrap((a - b)[ok])
    result = {
        "plan": PLAN,
        "seed": TEST,
        "picks_sha256": sha256(picks_text),
        "inputs": {"picks": picks_spec, "guesses": guesses},
        "holm_family": scored,
        "primary": prim,
        "secondary": sec,
    }
    if runs.exists():
        result["runs"] = json.loads(runs.read_text())
    return rounded(result, 4)


def main() -> None:
    from codenames.c11 import parse_range
    from codenames.hosted import Meter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["lists", "choose", "guess", "repeat", "score"])
    parser.add_argument("--models", default=",".join(CHOOSERS))
    parser.add_argument("--seed", type=int, default=PILOT)
    parser.add_argument("--n", type=int)
    parser.add_argument("--boards", type=parse_range)
    parser.add_argument("--picks", default=str(PICKS), help="a file, or hf:path")
    parser.add_argument("--guesses", help="score: a file, or hf:path,...")
    parser.add_argument(
        "--arms", help="guess: arms, comma-separated (default: both choosers and F)"
    )
    parser.add_argument("--max-tokens", default="", help="choose: model=tokens,... (the cap rule)")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--extra", action="append", default=[], help="choose: model=JSON request fields"
    )
    parser.add_argument("--times", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "lists":
        a, f = lists()
        ACTIONS.write_text(a)
        F_LINES.write_text(f)
        print(ACTIONS, sha256(a), F_LINES, sha256(f))
        return
    if args.what == "score":
        ensure_vectors()
        result = score(args.picks, args.guesses)
        if args.out and args.out.suffix != ".json":
            args.out.mkdir(parents=True, exist_ok=True)
            args.out = args.out / OUT.name
        (args.out or OUT).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result["primary"], indent=1))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    meter = Meter(args.cap)
    models = args.models.split(",")
    for item in args.extra:
        name, _, js = item.partition("=")
        EXTRA[name] = json.loads(js)
    if args.what == "choose":
        caps = {m: CAPS[m][0] for m in models}
        for item in filter(None, args.max_tokens.split(",")):
            m, t = item.split("=")
            assert int(t) in CAPS[m], (m, t)
            caps[m] = int(t)
        choose(
            models,
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
    c4.price_models(models, args.out)
    times = args.times if args.what == "repeat" else 0
    arms = args.arms.split(",") if args.arms else None
    picks = fetch_text(args.picks)
    guess(models, picks, args.seed, args.boards, times, meter, args.out, args.workers, arms)


if __name__ == "__main__":
    main()
