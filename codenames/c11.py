"""Experiment C11, part A (experiments/plans/C11.md): does the population pick's lead over
the embedding giver's first choice hold with language-model guessers of other families?

Three steps, as the plan and its amendment of 2026-11-20 name them:

    uv run python -m codenames.c11 lists      # here, with the word vectors (~20 min)
    python -m codenames.c11 guess --models gpt-oss-120b --seed 11004 --boards 0-1000 \\
        --items --cap 0.21 --out OUT          # in a Hugging Face job
    python -m codenames.c11 repeat --models gpt-oss-120b,llama-3.3-70b --seed 11004 \\
        --boards 0-100 --times 2 --cap 0.06 --out OUT
    uv run python -m codenames.c11 score      # here, no vectors

`lists` deals the pilot boards (seed 11999) and the test boards (seed 11004) with
`c4.boards` and computes, on each, the three arms of C10 twice: once with every arm
choosing among the clues that `legality.strict_legal` allows (the primary arms of the
amendment) and once with C10's filter `words.legal` alone (the plan's original arms, kept
as a secondary). On each board and filter:

- F, the first choice: the action of the proposer's 120 that its Luce guesser values most;
- P: among the 40 actions of the K = 10 kept clues, the one with the highest mean exact
  value over the seven strangers less potion (C10's task that holds out potion), and the
  same pick in each of the seven tasks, for C10's stranger value;
- the oracle: the kept action best for the mean of all seven strangers.

It writes `results/c11-picks.txt` (clue lines of `c4.clue_lines`, one line per arm and
seed) and `results/c11-lists.json` (every arm's exact values, and the strict check's
level of every clue).

`guess` and `repeat` send C4's guesser prompt (`llm.guesser_prompt`) with the action's
own count to each guesser, once per distinct (board, clue, count), and play the turn out
from the reply's order with `c4.play_turn`. Unlike `c4.crossguess`, they keep the state of
each reply: a reply that is empty, that failed five times, that came from another
provider, or that named no board word is not played out (`c4.read_ranking` would put
the board in its printed order), and the first three kinds are sent once more at the end
of the job. They write `guesses.txt`, one line per arm, guesser, seed and send:

    arm guesser seed send cell_1 ... cell_n

where a cell is `found/value`, with `c` added when the reply reached the token cap, or
`-` and the reason there is no value. `score` reads the committed copies of these lines
and writes `results/c11.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from codenames import c4
from codenames.game import Board
from codenames.hosted import Hosted, Meter

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PICKS = RESULTS / "c11-picks.txt"
LISTS = RESULTS / "c11-lists.json"
GUESSES = RESULTS / "c11-guesses.txt"
RUNS = RESULTS / "c11-runs.json"
OUT = RESULTS / "c11.json"
ITEMS = RESULTS / "human-pilot" / "boards.json"
PLAN = "experiments/plans/C11.md"
PILOT, TEST, ITEM_SEED = 11999, 11004, 11010
N_PILOT, N_TEST = 20, 1000
K = 10  # C10's K, chosen on its validation boards
HELD_OUT = "potion"
GUESSERS = ("gpt-oss-120b", "llama-3.3-70b")
ARMS = ("F", "P", "oracle")
FILTERS = ("strict", "c10")  # strict: legality.strict_legal; c10: words.legal alone
MEASURES = ("strangers", "strangers_c10", "potion", "human", "eat", "proposer")
ANSWERED = ("ok", "cut")
RESENT = ("empty", "failed", "unpinned")  # sent once more at the end of a job
REPEAT_BOARDS = 100
HOLM_Z = 3.08  # 2.24 + 0.84: two-sided 0.025, Holm's first step, and 80% power
MDE_Z = 2.80  # 1.96 + 0.84: two-sided 0.05 and 80% power (METHOD.md, 2026-11-20)


def giver(arm: str, filt: str) -> str:
    """The name of an arm in the clue lines: F, P, oracle (strict) or F-c10, ... ."""
    return arm if filt == "strict" else f"{arm}-{filt}"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Picks and exact values (lists).


def pick_indices(strangers: np.ndarray, proposer: np.ndarray, kidx, potion: int) -> dict:
    """Each arm's index in the 120 actions. strangers has shape (7, 120); kidx are the
    indices of the kept actions, in prefilter order. Argmaxes take the first action."""
    from codenames.listen import argmax_first, pick_population

    idx = np.asarray(kidx)
    s = strangers[:, idx]
    tasks = [int(idx[pick_population(s, t)]) for t in range(len(s))]
    return {
        "F": argmax_first(proposer),
        "P": tasks[potion],
        "P_tasks": tasks,
        "oracle": int(idx[argmax_first(s.mean(axis=0))]),
    }


def exact_values(strangers, human, eat, proposer, potion: int, j: int, tasks=None) -> dict:
    """An action's exact turn value under each scorer. strangers_c10 is C10's stranger
    value: for P, the mean over the seven tasks of the held-out stranger's value of that
    task's pick; for a single pick, the mean over the seven."""
    mean7 = float(strangers[:, j].mean())
    c10 = mean7 if tasks is None else float(np.mean([strangers[t, k] for t, k in enumerate(tasks)]))
    return {
        "strangers": mean7,
        "strangers_c10": c10,
        "potion": float(strangers[potion, j]),
        "human": float(human[j]),
        "eat": float(eat[j]),
        "proposer": float(proposer[j]),
    }


def board_lists(board: Board, strict: bool) -> dict:
    """The arms of one board under one filter, with their exact values (needs vectors)."""
    from codenames.actions import proposer, shortlist
    from codenames.c10score import BoardData
    from codenames.human_pilot import clue_pool
    from codenames.legality import worst
    from codenames.listen import prefilter

    sp, tau = proposer()
    clues = shortlist(board, sp, tau, pool=clue_pool(board) if strict else None)
    d = BoardData(board, clues, prefilter(board, clues, K))
    p = pick_indices(d.strangers, d.proposer, d.kidx(K), d.potion)
    act = {a: "{}/{}".format(*d.acts[p[a]]) for a in ARMS}
    return {
        **act,
        "P_tasks": ["{}/{}".format(*d.acts[j]) for j in p["P_tasks"]],
        "values": {
            a: exact_values(
                d.strangers,
                d.human,
                d.eat,
                d.proposer,
                d.potion,
                p[a],
                p["P_tasks"] if a == "P" else None,
            )
            for a in ARMS
        },
        "level": {a: worst(act[a].split("/")[0], board.words) for a in ARMS},
    }


def check_picks(seeds=(PILOT, TEST), n: int = 3) -> dict:
    """lists' picks equal human_pilot.giver_picks on the first n boards of each seed, with
    and without the strict filter; and without it, C10's committed picks (human_pilot)."""
    from codenames.human_pilot import check_against_c10, giver_picks

    for seed in seeds:
        for b in c4.boards(seed, n):
            for strict, filt in ((True, "strict"), (False, "c10")):
                got = board_lists(b, strict)
                want = {a: f"{c}/{k}" for a, (c, k) in giver_picks(b, strict).items()}
                assert {a: got[a] for a in ARMS} == want, (seed, filt)
    check_against_c10(n)
    return {"giver_picks_boards_per_seed": n, "c10_boards": n, "passed": True}


def lists(seeds=(PILOT, TEST), ns=(N_PILOT, N_TEST)) -> tuple[str, str]:
    """The texts of c11-picks.txt and c11-lists.json."""
    from codenames.c10score import scorers

    assert list(scorers()[1]).index(HELD_OUT) is not None
    checks = check_picks(seeds)
    boards: list[dict] = []
    for seed, n in zip(seeds, ns, strict=True):
        for i, b in enumerate(c4.boards(seed, n)):
            boards.append(
                {"seed": seed, "board": i, **{f: board_lists(b, f == "strict") for f in FILTERS}}
            )
            if i % 100 == 0:
                print(seed, i, flush=True)
    return picks_text(boards), lists_text(boards, checks)


def picks_text(boards: list[dict]) -> str:
    """Clue lines (c4.clue_lines) for the six arms, seed by seed."""
    given: dict[int, dict[str, list[dict]]] = {}
    for row in boards:
        for f in FILTERS:
            for a in ARMS:
                clue, count = row[f][a].split("/")
                given.setdefault(row["seed"], {}).setdefault(giver(a, f), []).append(
                    {"clue": clue, "count": int(count)}
                )
    return c4.clue_lines(given)


def lists_text(boards: list[dict], checks: dict) -> str:
    """c11-lists.json: settings, then one board per line, values unrounded."""
    head = {
        "plan": PLAN,
        "seeds": sorted({r["seed"] for r in boards}),
        "K": K,
        "held_out": HELD_OUT,
        "filters": {
            "strict": "codenames.legality.strict_legal (amendment of 2026-11-20)",
            "c10": "codenames.words.legal alone, as in C10",
        },
        "measures": list(MEASURES),
        "checks": checks,
    }
    lines = [json.dumps(r, sort_keys=True) for r in boards]
    return (
        "{\n"
        + "".join(f'"{k}": {json.dumps(v, sort_keys=True)},\n' for k, v in head.items())
        + '"boards": [\n'
        + ",\n".join(lines)
        + "\n]}\n"
    )


# Guessing (in a job).


def parse_range(text: str) -> range:
    start, stop = text.split("-")
    return range(int(start), int(stop))


def status_of(rec: dict | None, words: tuple[str, ...]) -> str:
    """ok, cut (answered, but at the token cap), empty, failed (five failed tries),
    unpinned (every try served by another provider), noword (no board word named), or
    stopped (the cost cap did not allow the call)."""
    if rec is None:
        return "stopped"
    if "usage" not in rec:
        return "unpinned" if any(e.startswith("served by") for e in rec["errors"]) else "failed"
    if not rec["reply"].strip():
        return "unpinned" if rec.get("pin_held") is False else "empty"
    if c4.read_ranking(rec["reply"], words)["named"] == 0:
        return "noword"
    return "cut" if rec.get("finish") == "length" else "ok"


def cell(board: Board, count: int, rec: dict | None) -> str:
    """found/value (c if cut at the cap), or - and the status."""
    st = status_of(rec, board.unrevealed())
    if st not in ANSWERED:
        return f"-{st}"
    assert rec is not None
    ranking = c4.read_ranking(rec["reply"], board.unrevealed())["ranking"]
    found, value = c4.play_turn(board, ranking, count)
    return f"{found}/{value:g}" + ("c" if st == "cut" else "")


def read_cell(text: str) -> tuple[int, float, bool] | None:
    """(found, value, cut) of an answered cell, else None."""
    if text.startswith("-"):
        return None
    cut = text.endswith("c")
    found, value = text.rstrip("c").split("/")
    return int(found), float(value), cut


Key = tuple[int, int, str, int]  # seed, board index, clue, count


def targets(picks: str, seed: int, boards: range, arms) -> dict[str, list[tuple[Key, Board]]]:
    """{arm: [(key, board)] for each board in range}, from clue lines."""
    given = c4.read_clue_lines(picks)
    deals = c4.boards(seed, boards.stop)
    out: dict[str, list[tuple[Key, Board]]] = {}
    for a in arms:
        rows = given[(a, seed)]
        out[a] = []
        for i in boards:
            clue, count, _ = rows[i]
            assert clue is not None and count is not None, (a, seed, i)
            out[a].append(((seed, i, clue, count), deals[i]))
    return out


def item_targets(path: Path = ITEMS) -> dict[str, list[tuple[Key, Board]]]:
    """The 80 items of part B, in the order of boards.json: one arm named items."""
    data = json.loads(path.read_text())
    idx = {b["board"]: b["seed_index"] for b in data["boards"]}
    deals = c4.boards(data["board_seed"], 1 + max(idx.values()))
    out = []
    for x in data["items"]:
        board = deals[idx[x["board"]]]
        assert list(board.words) == data["boards"][x["board"]]["words"]
        out.append(((data["board_seed"], idx[x["board"]], x["clue"], x["count"]), board))
    return {"items": out}


def send_all(hosted, keys: dict[Key, Board], workers: int, part: str) -> dict[Key, dict | None]:
    """One reply per distinct key; failed, empty and unpinned replies sent once more at
    the end. Returns the final record of each key, with sends and first status added."""
    from codenames.llm import guesser_prompt

    order = list(keys)

    def jobs(ks):
        return [
            (
                guesser_prompt(k[2], k[3], keys[k].unrevealed()),
                {"part": part, "seed": k[0], "board": k[1], "clue": k[2], "count": k[3]},
            )
            for k in ks
        ]

    recs = dict(zip(order, c4.run_many(hosted.call, jobs(order), workers), strict=True))
    first = {k: status_of(recs[k], keys[k].unrevealed()) for k in order}
    again = [k for k in order if first[k] in RESENT]
    if again:
        for k, rec in zip(again, c4.run_many(hosted.call, jobs(again), workers), strict=True):
            recs[k] = rec
    out = {}
    for k in order:
        rec = None if recs[k] is None else dict(recs[k])
        if rec is not None:
            rec["first_status"], rec["sends"] = first[k], 1 + (k in again)
        out[k] = rec
    return out


def guess_lines(name: str, by_arm, recs: dict[Key, dict | None], send: int) -> list[str]:
    lines = []
    for a, rows in by_arm.items():
        seed = rows[0][0][0]
        cells = [cell(b, k[3], recs[k]) for k, b in rows]
        lines.append(" ".join([a, name, str(seed), str(send), *cells]))
    return lines


def reply_stats(recs: list[dict | None], keys: list[Key], boards: dict[Key, Board]) -> dict:
    """Counts of each status at the first send and at the end, sends, and latency."""
    done = [r for r in recs if r is not None]
    secs = [r["seconds"] for r in done if isinstance(r.get("seconds"), int | float)]
    final = Counter(status_of(r, boards[k].unrevealed()) for r, k in zip(recs, keys, strict=True))
    return {
        "prompts": len(recs),
        "first_status": dict(Counter(r["first_status"] for r in done)),
        "final_status": dict(final),
        "resent": sum(r["sends"] > 1 for r in done),
        "cut_share": (final["cut"] / len(recs)) if recs else None,
        "seconds_median": statistics.median(secs) if secs else None,
        "pin_held": dict(Counter(str(r.get("pin_held")) for r in done)),
        "served_providers": dict(
            Counter(str((r.get("served_by") or {}).get("provider")) for r in done)
        ),
        "served_models": dict(Counter(str(r.get("served")) for r in done)),
    }


def run(
    models,
    picks: str,
    seed: int,
    boards: range | None,
    repeat_boards: range | None,
    times: int,
    items: bool,
    meter,
    out: Path,
    workers: int = 8,
) -> dict:
    """The guessers, one after the other: every distinct prompt of the six arms on the
    boards (and the 80 items), then the strict arms' prompts on repeat_boards, times more
    times. Writes guesses.txt and returns the job's summary."""
    lines: list[str] = []
    summary: dict = {"seed": seed, "models": {}}
    for name in models:
        hosted = Hosted(
            c4.MODELS[name], meter, out / f"calls-{name}.jsonl", c4.tokens(name, "guess")
        )
        w = c4.WORKERS.get(name, workers)
        s: dict = {"max_tokens": c4.tokens(name, "guess"), **vars(c4.MODELS[name])}
        parts: list[tuple[str, dict]] = []
        if boards is not None:
            parts.append(
                (
                    "guess",
                    targets(picks, seed, boards, [giver(a, f) for f in FILTERS for a in ARMS]),
                )
            )
        if items:
            parts.append(("items", item_targets()))
        for part, by_arm in parts:
            keys = {k: b for rows in by_arm.values() for k, b in rows}
            recs = send_all(hosted, keys, w, part)
            lines += guess_lines(name, by_arm, recs, 0)
            s[part] = reply_stats(list(recs.values()), list(recs), keys)
            print(name, part, s[part]["final_status"], f"${meter.spent:.4f}", flush=True)
        if repeat_boards is not None and times:
            by_arm = targets(picks, seed, repeat_boards, ARMS)
            keys = {k: b for rows in by_arm.values() for k, b in rows}
            s["repeat"] = {}
            for r in range(1, times + 1):
                recs = send_all(hosted, keys, w, f"repeat-{r}")
                lines += guess_lines(name, by_arm, recs, r)
                s["repeat"][r] = reply_stats(list(recs.values()), list(recs), keys)
                print(name, "repeat", r, s["repeat"][r]["final_status"], flush=True)
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


def read_guesses(text: str) -> dict[tuple[str, str, int, int], list[str]]:
    """{(arm, guesser, seed, send): cells}."""
    out = {}
    for line in text.splitlines():
        arm, name, seed, send, *cells = line.split()
        key = (arm, name, int(seed), int(send))
        assert key not in out, key
        out[key] = cells
    return out


def column(cells: list[str], what: int = 1) -> np.ndarray:
    """found (what=0) or value (what=1) per board, nan where there is no answer."""
    vals = [read_cell(c) for c in cells]
    return np.array([np.nan if v is None else v[what] for v in vals])


def played(g, seed: int, send: int, filt: str, what: int = 1) -> dict:
    """{(arm, guesser): array over boards} of played-out values."""
    return {
        (a, m): column(g[(giver(a, filt), m, seed, send)], what)
        for a in ARMS
        for m in GUESSERS
        if (giver(a, filt), m, seed, send) in g
    }


def exact(rows: list[dict], filt: str, arm: str, measure: str) -> np.ndarray:
    return np.array([r[filt]["values"][arm][measure] for r in rows])


def primary_diffs(v: dict, pot_p: np.ndarray, pot_f: np.ndarray, models=GUESSERS):
    """Per board: H-A1's d (mean over the guessers of P - F, played out), H-A2's d minus
    potion's exact P - F, and the mask of boards where every needed reply was answered."""
    d = np.stack([v[("P", m)] - v[("F", m)] for m in models])
    keep = ~np.isnan(d).any(axis=0)
    h1 = d.mean(axis=0)
    return h1, h1 - (pot_p - pot_f), keep


def mde(d: np.ndarray) -> dict:
    sd = float(np.std(d, ddof=1))
    n = len(d)
    return {
        "sd": sd,
        "n": n,
        "at_0.05": MDE_Z * sd / np.sqrt(n),
        "at_holm_0.025": HOLM_Z * sd / np.sqrt(n),
    }


def assassin(cells: list[str]) -> np.ndarray:
    """1 where the played-out turn ended on the assassin (a value below -1), nan where
    there is no answer. An opponent word gives found - 1 >= -1; the assassin gives
    found - (team words left) <= 3 - 8."""
    v = column(cells)
    return np.where(np.isnan(v), np.nan, (v < -1).astype(float))


def nanboot(x: np.ndarray) -> dict:
    from codenames.c4score import bootstrap

    x = np.asarray(x, dtype=float)
    return bootstrap(x[~np.isnan(x)])


def reply_counts(cells: list[str]) -> dict:
    c = Counter(
        "cut" if x.endswith("c") else "ok" if not x.startswith("-") else x[1:] for x in cells
    )
    return dict(sorted(c.items()))


def score_filter(rows, g, filt: str, seed: int) -> dict:
    """Every quantity of the plan for one filter's arms on the test boards."""
    from codenames.c4score import bootstrap

    v = played(g, seed, 0, filt)
    f = played(g, seed, 0, filt, what=0)
    ex = {(a, m): exact(rows, filt, a, m) for a in ARMS for m in MEASURES}
    h1, h2, keep = primary_diffs(v, ex[("P", "potion")], ex[("F", "potion")])
    out: dict = {
        "boards": len(rows),
        "boards_kept": int(keep.sum()),
        "boards_dropped": int((~keep).sum()),
        "h_a1": bootstrap(h1[keep]),
        "h_a2": bootstrap(h2[keep]),
        "potion_p_minus_f": bootstrap((ex[("P", "potion")] - ex[("F", "potion")])[keep]),
        "mde": {"h_a1": mde(h1[keep]), "h_a2": mde(h2[keep])},
    }
    sec: dict = {}
    for m in GUESSERS:
        sec[m] = {
            "P_minus_F": nanboot(v[("P", m)] - v[("F", m)]),
            "P_minus_F_found": nanboot(f[("P", m)] - f[("F", m)]),
            "oracle_minus_F": nanboot(v[("oracle", m)] - v[("F", m)]),
            "oracle_minus_P": nanboot(v[("oracle", m)] - v[("P", m)]),
            "assassin_rate": {a: nanboot(assassin(g[(giver(a, filt), m, seed, 0)])) for a in ARMS},
            "found": {a: nanboot(f[(a, m)]) for a in ARMS},
        }
    both = {a: np.stack([v[(a, m)] for m in GUESSERS]).mean(axis=0) for a in ARMS}
    mask = {a: ~np.isnan(both[a]) for a in ARMS}
    k_of = mask["oracle"] & mask["F"]
    sec["mean_of_guessers"] = {
        "oracle_minus_F": bootstrap((both["oracle"] - both["F"])[k_of]),
        "oracle_minus_P": bootstrap((both["oracle"] - both["P"])[k_of & mask["P"]]),
    }
    kinds = {m: {a: nanboot(v[(a, m)]) for a in ARMS} for m in GUESSERS}
    for m in ("strangers", "potion", "human", "eat"):
        kinds[m] = {a: bootstrap(ex[(a, m)]) for a in ARMS}
    sec["arm_means"] = kinds
    sec["order"] = {k: " > ".join(sorted(ARMS, key=lambda a: -kinds[k][a]["mean"])) for k in kinds}
    sec["exact_P_minus_F"] = {
        m: bootstrap(ex[("P", m)] - ex[("F", m)])
        for m in ("strangers_c10", "strangers", "potion", "human", "eat")
    }
    sec["exact_oracle_minus_F"] = {
        m: bootstrap(ex[("oracle", m)] - ex[("F", m)]) for m in ("strangers", "human", "eat")
    }
    acts = {a: [r[filt][a] for r in rows] for a in ARMS}
    sec["same_action_P_F"] = float(
        np.mean([p == q for p, q in zip(acts["P"], acts["F"], strict=True)])
    )
    sec["same_clue_P_F"] = float(
        np.mean(
            [p.split("/")[0] == q.split("/")[0] for p, q in zip(acts["P"], acts["F"], strict=True)]
        )
    )
    sec["mean_count"] = {a: float(np.mean([int(x.split("/")[1]) for x in acts[a]])) for a in ARMS}
    sec["level"] = {a: dict(Counter(r[filt]["level"][a] for r in rows)) for a in ARMS}
    sec["replies"] = {
        m: reply_counts([c for a in ARMS for c in g[(giver(a, filt), m, seed, 0)]])
        for m in GUESSERS
    }
    out["secondary"] = sec
    out["per_board"] = [
        {
            "board": r["board"],
            "kept": bool(keep[i]),
            "h_a1": float(h1[i]) if keep[i] else None,
            "h_a2": float(h2[i]) if keep[i] else None,
            "played": {
                a: [
                    r[filt][a],
                    *(None if np.isnan(x) else float(x) for x in (v[(a, m)][i] for m in GUESSERS)),
                ]
                for a in ARMS
            },
        }
        for i, r in enumerate(rows)
    ]
    return out


def repeat_summary(rows, g, seed: int, models=GUESSERS) -> dict:
    """Prompts of the strict arms on the first boards whose played-out value changed in
    either repeat, and H-A1 recomputed from each send on those boards."""
    from codenames.c4score import bootstrap

    sends = sorted({s for (_, _, sd, s) in g if sd == seed and s > 0})
    if not sends:
        return {}
    n = len(g[("F", models[0], seed, sends[0])])
    out: dict = {"boards": n, "times": len(sends), "models": {}}
    for m in models:
        seen: dict[tuple, list] = {}
        for a in ARMS:
            for i in range(n):
                key = (i, rows[i]["strict"][a])
                cells = [g[(a, m, seed, s)][i] for s in [0, *sends]]
                seen.setdefault(key, cells)
        vals = [[read_cell(c) for c in cs] for cs in seen.values()]
        full = [vs for vs in vals if all(v is not None for v in vs)]
        changed = sum(len({v[1] for v in vs if v is not None}) > 1 for vs in full)
        out["models"][m] = {
            "prompts": len(vals),
            "answered_every_send": len(full),
            "value_changed": changed,
            "value_changed_share": changed / len(full) if full else None,
        }
    pot = exact(rows[:n], "strict", "P", "potion") - exact(rows[:n], "strict", "F", "potion")
    out["h_a1_by_send"] = {}
    for s in [0, *sends]:
        v = {(a, m): column(g[(a, m, seed, s)][:n]) for a in ("P", "F") for m in models}
        h1, _, keep = primary_diffs(v, pot, np.zeros(n), models)
        out["h_a1_by_send"][s] = bootstrap(h1[keep])
    return out


def score(lists_path: Path = LISTS, guesses: Path = GUESSES, runs: Path = RUNS) -> dict:
    from codenames.c4score import bootstrap, holm, rounded

    data = json.loads(lists_path.read_text())
    g = read_guesses(guesses.read_text())
    rows = [r for r in data["boards"] if r["seed"] == TEST]
    result: dict = {
        "plan": PLAN,
        "seed": TEST,
        "lists_sha256": sha256(lists_path.read_text()),
        "guesses_sha256": sha256(guesses.read_text()),
        "guessers": list(GUESSERS),
    }
    strict = score_filter(rows, g, "strict", TEST)
    ps = {"h_a1": strict["h_a1"]["p"], "h_a2": strict["h_a2"]["p"]}
    reject = holm(ps)
    for h, sign in (("h_a1", 1), ("h_a2", -1)):
        strict[h]["holm_rejects"] = bool(reject[h])
        strict[h]["holds"] = bool(reject[h] and sign * strict[h]["mean"] > 0)
    result["primary"] = {
        k: strict[k] for k in ("h_a1", "h_a2", "boards_kept", "boards_dropped", "mde")
    }
    result["strict"] = strict
    result["c10_arms"] = score_filter(rows, g, "c10", TEST)
    a, b = strict["per_board"], result["c10_arms"]["per_board"]
    both = [i for i in range(len(a)) if a[i]["kept"] and b[i]["kept"]]
    result["strict_minus_c10"] = {
        h: bootstrap([a[i][h] - b[i][h] for i in both]) for h in ("h_a1", "h_a2")
    }
    result["strict_minus_c10"]["boards"] = len(both)
    result["repeat"] = repeat_summary(rows, g, TEST)
    result["pilot_replies"] = {
        m: reply_counts(
            [c for (a, mm, sd, s), cs in g.items() if mm == m and sd == PILOT for c in cs]
        )
        for m in GUESSERS
    }
    result["items"] = score_items(g)
    if runs.exists():
        result["runs"] = json.loads(runs.read_text())
    return rounded(result, 4)


def score_items(g) -> dict:
    """P - F under each guesser on the 80 items of part B (secondary; no people yet)."""
    from codenames.c4score import bootstrap

    if not any(k[0] == "items" for k in g):
        return {}
    boards = json.loads(ITEMS.read_text())
    key = json.loads((ITEMS.parent / "key.json").read_text())
    out: dict = {}
    for m in GUESSERS:
        cells = g.get(("items", m, ITEM_SEED, 0))
        if cells is None:
            continue
        by = {}
        for x, c in zip(boards["items"], cells, strict=True):
            got = read_cell(c)
            by[(x["board"], key["items"][x["item"]])] = np.nan if got is None else got[1]
        d = np.array([by[(b, "P")] - by[(b, "F")] for b in range(len(boards["boards"]))])
        d = d[~np.isnan(d)]
        out[m] = {"P_minus_F": bootstrap(d), "replies": reply_counts(cells)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["lists", "guess", "repeat", "score"])
    parser.add_argument("--models", default=",".join(GUESSERS))
    parser.add_argument("--seed", type=int, default=TEST)
    parser.add_argument("--boards", type=parse_range, help="guess: boards, as 0-1000")
    parser.add_argument(
        "--repeat-boards", type=parse_range, help="boards whose strict prompts are sent again"
    )
    parser.add_argument("--times", type=int, default=2)
    parser.add_argument("--items", action="store_true", help="guess: also the 80 items of part B")
    parser.add_argument("--picks", type=Path, default=PICKS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "lists":
        picks, lists_ = lists()
        PICKS.write_text(picks)
        LISTS.write_text(lists_)
        print(PICKS, sha256(picks), LISTS, sha256(lists_))
        return
    if args.what == "score":
        result = score()
        (args.out or OUT).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result["primary"], indent=1))
        return
    models = args.models.split(",")
    assert all(m in GUESSERS for m in models), models
    args.out.mkdir(parents=True, exist_ok=True)
    c4.price_models(models, args.out)
    meter = Meter(args.cap)
    boards = args.boards if args.what == "guess" else None
    repeat = args.repeat_boards if args.what == "guess" else args.boards
    run(
        models,
        args.picks.read_text(),
        args.seed,
        boards,
        repeat,
        args.times,
        args.items,
        meter,
        args.out,
        args.workers,
    )


if __name__ == "__main__":
    main()
