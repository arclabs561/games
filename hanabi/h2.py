"""Experiment H2: hosted language models against the open-game ceiling.

    python -m hanabi.h2 play --model openai/gpt-oss-120b --provider crusoe --players 2 \
        --seeds 0-29 --deduction on --cap 1.0 --out OUT
    python -m hanabi.h2 report --runs RUNS --out hanabi/results/h2.json
    python -m hanabi.h2 harness --runs RUNS --out hanabi/results/h2-harness.json

`play` plays each seeded deal once, with the same model in every seat, and writes
`games.jsonl` (one line per game: score, lives lost, fallbacks, the moves, token counts and
cost), `scores.jsonl` (the same without the moves), `summary.json` (model, provider,
prices, spending, where it stopped) and `calls/seed-N.jsonl` (every prompt and reply).

`report` reads the `scores.jsonl` and `summary.json` of each run, computes the ceiling of each
deal with `upper_bound.verdict` and `upper_bound.best_score`, plays the information strategy
on the three-player deals, and writes the gaps with bootstrap intervals. The plan is
`experiments/plans/H2.md`.

`harness` replays each recorded game in `rules.py` and counts what the fallback moves did:
how many were plays and discards, and how many lives they lost, against the lives lost by
the model's own moves. It writes counts only.

The reply cap is `--max-tokens` (16,384 by default; the published runs used 4,096) and the
reasoning setting is `--reasoning-effort` (low, medium or high, for models that take one).
gpt-oss-120b counts its reasoning against the cap: at 4,096 tokens and medium effort, 24%
of its turns reached the cap before writing a move (results/h2-harness.md).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hanabi.information import play_info_game
from hanabi.llm import move_text, observation, parse_action
from hanabi.rules import Event, Rules, State, legal_actions, new_game, seeded_deck, step
from hanabi.upper_bound import best_score, verdict

# --- playing ---------------------------------------------------------------------------


def default_fallback(state: State):
    """As in `llm.play_text_game`: the first legal discard, else the first legal move."""
    moves = legal_actions(state)
    return next((a for a in moves if a.kind == "discard"), moves[0])


def play_game(
    rules: Rules, deck: list[int], complete: Callable[[str], str], deduction: bool
) -> tuple[State, dict]:
    """`llm.play_text_game`, keeping a record of each move and whether it was a fallback."""
    state = new_game(rules, deck)
    events: list[Event] = []
    moves, fell_back = [], []
    while not state.over:
        reply = complete(observation(state, state.current, events, deduction))
        action = parse_action(reply, state)
        fell_back.append(action is None)
        if action is None:
            action = default_fallback(state)
        moves.append(move_text(action, rules))
        events.append(step(state, action))
    record = {
        "score": state.score,
        "board": list(state.board),
        "lives_lost": rules.lives - state.lives,
        "turns": state.turn,
        "fallbacks": sum(fell_back),
        "moves": moves,
        "fallback_turns": [i for i, f in enumerate(fell_back) if f],
    }
    return state, record


def parse_seeds(text: str) -> list[int]:
    """'0-29' or '3,5,8' or a mix: '0-2,7'."""
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += range(int(a), int(b) + 1)
        else:
            seeds.append(int(part))
    return seeds


def complete_prefix(rows: list[dict]) -> list[dict]:
    """The games in seed order up to the first one that is not complete."""
    out = []
    for r in sorted(rows, key=lambda r: r["seed"]):
        if r["status"] != "complete":
            break
        out.append(r)
    return out


def run_games(args, models: list[dict] | None = None, client=None) -> int:
    """The `play` command. `models` (the router's model list) and `client` are fetched and
    made here unless given, as the tests give them."""
    import huggingface_hub

    from hanabi.hosted import CapError, Hosted, Meter, ProviderError, fetch_models, listed_price

    out = Path(args.out)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    rules = Rules(players=args.players)
    seeds = parse_seeds(args.seeds)
    deduction = args.deduction == "on"
    token = os.environ.get("HF_TOKEN")
    if models is None:
        models = fetch_models(token or "")
    price = listed_price(models, args.model, args.provider)
    # the longest prompt seen in stub games is under 1,300 tokens; allow 2,500
    max_call = price.cost(2500, args.max_tokens)
    meter = Meter(price, args.cap, max_call)
    if client is None:
        client = huggingface_hub.InferenceClient(provider=args.provider, token=token, timeout=300)
    deadline = time.time() + args.stop_after * 60  # no new game after this
    hard_stop = time.time() + args.hard_stop * 60  # games in progress stop here
    lock = threading.Lock()
    rows: list[dict] = []
    started = time.time()

    def one(seed: int) -> dict:
        base = {
            "seed": seed,
            "players": args.players,
            "deduction": deduction,
            "model": args.model,
            "provider": args.provider,
            "label": args.label,
        }
        if time.time() > deadline:
            return base | {"status": "not started", "reason": "time limit"}
        tries = []
        for attempt in range(2):  # a game lost to provider errors is replayed once
            calls: list[dict] = []
            player = Hosted(
                args.model,
                args.provider,
                meter,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                seed=0,
                client=client,
                log=calls.append,
                wait=args.retry_wait,
            )
            name = f"seed-{seed}" + (f"-replay{attempt}" if attempt else "")

            def timed(prompt: str, player=player, calls=calls) -> str:
                if time.time() > hard_stop:
                    raise TimeoutError("time limit")
                if calls and len(calls) % 20 == 0:
                    print(f"seed {seed}: turn {len(calls)}, total ${meter.spent:.4f}", flush=True)
                return player(prompt)

            try:
                _, record = play_game(rules, seeded_deck(rules, seed), timed, deduction)
                status, reason = "complete", None
            except CapError as e:
                record, status, reason = {}, "stopped", f"cap: {e}"
            except TimeoutError:
                record, status, reason = {}, "stopped", "time limit"
            except ProviderError as e:
                record, status, reason = {}, "provider error", str(e)[:500]
            with open(out / "calls" / f"{name}.jsonl", "w") as f:
                for c in calls:
                    f.write(json.dumps(c) + "\n")
            tries.append(reason)
            usage = {
                "calls": len(calls),
                "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                "completion_tokens": sum(c["completion_tokens"] for c in calls),
                "cost": sum(c["cost"] for c in calls),
                "truncated": sum(c["finish_reason"] == "length" for c in calls),
                "retried_calls": sum(c["attempt"] > 0 for c in calls),
                "served_models": sorted({str(c["served_model"]) for c in calls}),
            }
            row = base | {"status": status, "reason": reason, "attempts": attempt + 1}
            row |= record | usage
            if status != "provider error":
                break
        row["earlier_attempts"] = tries[:-1]
        if row["status"] == "provider error":
            row["status"] = "missing"
        with lock:
            print(
                f"seed {seed}: {row['status']} score {row.get('score')} "
                f"fallbacks {row.get('fallbacks')} cost ${row['cost']:.4f}; "
                f"total ${meter.spent:.4f}",
                flush=True,
            )
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, seeds))
    prefix = complete_prefix(rows)
    with open(out / "games.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "moves"}) + "\n")
    summary = {
        "experiment": "H2",
        "label": args.label,
        "model": args.model,
        "provider": args.provider,
        "players": args.players,
        "deduction": deduction,
        "seeds": seeds,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "temperature": 0,
        "seed": 0,
        "price_per_million": {"input": price.input, "output": price.output},
        "price_source": "router.huggingface.co/v1/models, fetched at the start of the run",
        "cap": args.cap,
        "spent": meter.spent,
        "calls": meter.calls,
        "games_complete": sum(r["status"] == "complete" for r in rows),
        "complete_prefix": len(prefix),
        "stopped_at_seed": None if len(prefix) == len(rows) else rows[len(prefix)]["seed"],
        "statuses": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seconds": round(time.time() - started, 1),
        "huggingface_hub": huggingface_hub.__version__,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


# --- what the fallbacks did -------------------------------------------------------------


def fallback_effect(rules: Rules, deck: list[int], moves: list[str], fallbacks) -> dict:
    """Replay one recorded game and split the lives lost between fallback turns and the
    model's own moves. Raises if a move is illegal in the replay."""
    state = new_game(rules, deck)
    fell_back = set(fallbacks)
    out = {"fallback_plays": 0, "fallback_discards": 0, "fallback_hints": 0}
    out |= {"lives_lost_by_fallbacks": 0, "lives_lost_by_model": 0, "third_life_by_fallback": 0}
    for i, text in enumerate(moves):
        action = parse_action(text, state)
        if action is None:
            raise ValueError(f"move {i} ({text}) is not legal in the replay")
        before = state.lives
        step(state, action)
        lost = before - state.lives
        if i in fell_back:
            out[f"fallback_{action.kind}s"] += 1
            out["lives_lost_by_fallbacks"] += lost
            out["third_life_by_fallback"] += int(lost > 0 and state.lives == 0)
        else:
            out["lives_lost_by_model"] += lost
    out["score"] = state.score
    return out


def harness(args) -> int:
    """The `harness` command: fallback counts per run, from each run's games.jsonl."""
    arms = []
    for path in sorted(Path(args.runs).rglob("games.jsonl")):
        summary = json.loads((path.parent / "summary.json").read_text())
        rules = Rules(players=summary["players"])
        totals: dict = {}
        games = 0
        for line in path.read_text().splitlines():
            g = json.loads(line)
            if g["status"] != "complete":
                continue
            e = fallback_effect(
                rules, seeded_deck(rules, g["seed"]), g["moves"], g["fallback_turns"]
            )
            if e["score"] != g["score"]:
                raise ValueError(f"{path}: seed {g['seed']} replays to {e['score']}")
            games += 1
            for k, v in e.items():
                totals[k] = totals.get(k, 0) + v
            totals["turns"] = totals.get("turns", 0) + g["turns"]
            totals["truncated"] = totals.get("truncated", 0) + g["truncated"]
            totals["games_with_a_fallback"] = totals.get("games_with_a_fallback", 0) + bool(
                g["fallback_turns"]
            )
        arms.append(
            {
                "run": path.parent.name,
                "model": summary["model"],
                "max_tokens": summary["max_tokens"],
                "reasoning_effort": summary["reasoning_effort"],
                "deduction": summary["deduction"],
                "games": games,
                **totals,
            }
        )
    result = {
        "what": "fallback moves of H2, replayed in rules.py; counts over the complete games",
        "how_known": "computed exactly by replaying the recorded moves on the seeded decks",
        "arms": arms,
    }
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result))
    return 0


# --- the ceiling and the information strategy ------------------------------------------


def ceiling(rules: Rules, deck: list[int], budgets=(20000, 300000)) -> tuple[int, str]:
    """The best open-game score of a deal and the solver's verdict. A won deal's ceiling is
    the maximum score; a lost deal's is `best_score`, an upper bound; an undecided deal
    counts as the maximum."""
    for b in budgets:
        v, _ = verdict(rules, deck, b)
        if v == "won":
            return rules.max_score, v
        if v == "lost":
            return best_score(rules, deck, b), v
    return rules.max_score, "undecided"


# --- statistics ------------------------------------------------------------------------


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def bootstrap(xs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, resampling the units in `xs`."""
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def describe(rows: list[dict], ceilings: dict[int, int]) -> dict:
    """The numbers the plan asks for, for one arm's complete prefix of games."""
    gaps = [ceilings[r["seed"]] - r["score"] for r in rows]
    scores = [r["score"] for r in rows]
    n = len(rows)
    perfect = sum(s == 25 for s in scores)
    zero = sum(r["lives_lost"] == 3 for r in rows)
    turns = sum(r["turns"] for r in rows)
    return {
        "games": n,
        "seeds": [r["seed"] for r in rows],
        "gap_mean": mean(gaps),
        "gap_interval": bootstrap(gaps),
        "score_mean": mean(scores),
        "score_interval": bootstrap(scores),
        "perfect": perfect,
        "perfect_interval": wilson(perfect, n),
        "all_lives_lost": zero,
        "all_lives_lost_interval": wilson(zero, n),
        "fallbacks_per_game": sum(r["fallbacks"] for r in rows) / n,
        "fallback_share_of_turns": sum(r["fallbacks"] for r in rows) / turns,
        "truncated_share_of_calls": sum(r["truncated"] for r in rows)
        / sum(r["calls"] for r in rows),
        "turns_per_game": turns / n,
        "cost": sum(r["cost"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
    }


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    """Mean of b - a over the seeds both have, with its bootstrap interval."""
    seeds = sorted(set(a) & set(b))
    d = [b[s] - a[s] for s in seeds]
    return {"deals": len(seeds), "mean": mean(d), "interval": bootstrap(d)}


# --- report ----------------------------------------------------------------------------


def load_runs(root: Path) -> list[tuple[dict, list[dict]]]:
    runs = []
    for s in sorted(root.rglob("summary.json")):
        summary = json.loads(s.read_text())
        rows = [json.loads(line) for line in (s.parent / "scores.jsonl").read_text().splitlines()]
        runs.append((summary | {"run": s.parent.name}, rows))
    return runs


def report(args) -> int:
    runs = load_runs(Path(args.runs))
    need: dict[int, set[int]] = {}
    for summary, rows in runs:
        need.setdefault(summary["players"], set()).update(r["seed"] for r in rows)
    deals = {}
    for p, seeds in sorted(need.items()):
        rules = Rules(players=p)
        for seed in sorted(seeds):
            c, v = ceiling(rules, seeded_deck(rules, seed))
            deals[(p, seed)] = {"players": p, "seed": seed, "ceiling": c, "verdict": v}
    info = {}
    for p, seeds in sorted(need.items()):
        if p >= 3:
            rules = Rules(players=p)
            for seed in sorted(seeds):
                g = play_info_game(rules, seeded_deck(rules, seed), hansim=False)
                info[(p, seed)] = g.score
                deals[(p, seed)]["information_strategy"] = g.score
    arms = []
    for summary, rows in runs:
        p = summary["players"]
        prefix = complete_prefix(rows)
        ceil = {s: int(deals[(p, s)]["ceiling"]) for s in need[p]}
        arm = {
            k: summary[k]
            for k in (
                "run",
                "label",
                "model",
                "provider",
                "players",
                "deduction",
                "max_tokens",
                "reasoning_effort",
                "price_per_million",
                "cap",
                "spent",
                "started",
                "statuses",
                "stopped_at_seed",
                "huggingface_hub",
            )
        }
        arm["served_models"] = sorted({m for r in rows for m in r.get("served_models", [])})
        arm["planned_seeds"] = summary["seeds"]
        if prefix:
            arm |= describe(prefix, ceil)
        if p >= 3 and prefix:
            arm["information_strategy"] = paired(
                {r["seed"]: r["score"] for r in prefix},
                {r["seed"]: info[(p, r["seed"])] for r in prefix},
            )
            arm["information_strategy"]["note"] = "information strategy minus model, by deal"
        arm["games_by_seed"] = [
            {
                k: r.get(k)
                for k in (
                    "seed",
                    "status",
                    "score",
                    "lives_lost",
                    "turns",
                    "fallbacks",
                    "truncated",
                    "calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "cost",
                    "attempts",
                )
            }
            for r in sorted(rows, key=lambda r: r["seed"])
        ]
        arms.append(arm)
    comparisons = []
    by_key = {
        (a["label"], a["model"], a["players"], a["max_tokens"], a["deduction"]): a for a in arms
    }
    for (label, model, p, cap, ded), on in sorted(by_key.items()):
        off = by_key.get((label, model, p, cap, False))
        shared = set(on.get("seeds", [])) & set(off.get("seeds", [])) if off else set()
        if ded and off and shared:
            gap = {
                g["seed"]: deals[(p, g["seed"])]["ceiling"] - g["score"]
                for g in on["games_by_seed"]
                if g["seed"] in on["seeds"]
            }
            gap_off = {
                g["seed"]: deals[(p, g["seed"])]["ceiling"] - g["score"]
                for g in off["games_by_seed"]
                if g["seed"] in off["seeds"]
            }
            comparisons.append(
                {
                    "label": label,
                    "model": model,
                    "players": p,
                    "max_tokens": cap,
                    "what": "gap with deduction off minus gap with deduction on, by deal",
                    **paired(gap, gap_off),
                }
            )
    ceilings = {}
    for p in sorted(need):
        rows = [d for (q, _), d in deals.items() if q == p]
        ceilings[p] = {
            "deals": len(rows),
            "lost": sum(d["verdict"] == "lost" for d in rows),
            "undecided": sum(d["verdict"] == "undecided" for d in rows),
            "mean": mean(d["ceiling"] for d in rows),
        }
        if p >= 3:
            ceilings[p]["information_strategy_mean"] = mean(d["information_strategy"] for d in rows)
    result = {
        "experiment": "H2",
        "plan": "experiments/plans/H2.md",
        "bootstrap": "95% percentile, 10,000 resamples over deals, seed 0",
        "ceiling_method": (
            "computed here: upper_bound.verdict with 20,000 then 300,000 nodes; lost deals "
            "take upper_bound.best_score; undecided deals count as 25"
        ),
        "information_strategy": "information.play_info_game, official rules (hansim=False)",
        "cost_method": "reported tokens times the provider's listed price; not a bill",
        "ceilings": {str(p): v for p, v in ceilings.items()},
        "deals": sorted(deals.values(), key=lambda d: (d["players"], d["seed"])),
        "arms": arms,
        "deduction_effect": comparisons,
        "total_cost": sum(a["spent"] for a in arms),
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("ceilings", "deduction_effect", "total_cost")}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("--model", required=True)
    p.add_argument("--provider", required=True)
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seeds", default="0-29")
    p.add_argument("--deduction", choices=["on", "off"], default="on")
    p.add_argument("--max-tokens", type=int, default=16384, help="reply cap; H2 used 4,096")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--cap", type=float, required=True, help="dollars this run may spend")
    p.add_argument("--workers", type=int, default=8, help="games played at once")
    p.add_argument("--stop-after", type=float, default=300, help="minutes; no new games after")
    p.add_argument("--hard-stop", type=float, default=330, help="minutes; games in progress stop")
    p.add_argument("--label", default="confirmatory")
    p.add_argument("--retry-wait", type=float, default=5, help="seconds before the first retry")
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--runs", required=True)
    r.add_argument("--out", required=True)
    h = sub.add_parser("harness")
    h.add_argument("--runs", required=True)
    h.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"play": run_games, "report": report, "harness": harness}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
