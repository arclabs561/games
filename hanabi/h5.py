"""Experiment H5 (experiments/plans/H5.md): H2's three-player part, gpt-oss-120b against the
information strategy and the open-game ceiling, with the reply cap at 16,384.

The games are played by `hanabi.h2 play` (see the plan for the two jobs). This module
reads their `summary.json`, `scores.jsonl` and `games.jsonl` and writes the report:

    uv run python -m hanabi.h5 report --runs RUNS --out hanabi/results/h5.json

Per deal: the ceiling (`h2.ceiling`: `upper_bound.verdict`, then `best_score` for a lost
deal), the information strategy's score (`information.play_info_game`, official rules),
and the model's score under both scorings: official (0 when the third life is lost) and
the fireworks on the table at that moment, as Ramesh et al. record it. The primary is the
mean over deals of the strategy's score minus the model's (official), with a 95% bootstrap
interval over deals (10,000 resamples, seed 0). Behavior comes from replaying every game in
`rules.py`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from hanabi import h2
from hanabi.information import play_info_game
from hanabi.llm import parse_action
from hanabi.rules import Rules, new_game, seeded_deck, step

PLAN = "experiments/plans/H5.md"
PLAYERS = 3
MDE_Z = 2.80


def behavior(rules: Rules, deck: list[int], moves: list[str], fallbacks) -> dict:
    """Replay one game: per seat, plays, misplays, discards and hints; hints whose target's
    next move plays a card the hint touched; lives lost to fallbacks and to the model."""
    state = new_game(rules, deck)
    seat = [Counter() for _ in range(rules.players)]
    pending: dict[int, tuple[int, ...]] = {}  # target -> touched positions of the last hint
    followed = hints = 0
    fell = set(fallbacks)
    lost_fb = lost_model = 0
    for i, text in enumerate(moves):
        a = parse_action(text, state)
        if a is None:
            raise ValueError(f"move {i} ({text}) is not legal in the replay")
        p = state.current
        if p in pending:
            followed += a.kind == "play" and a.index in pending.pop(p)
        before = state.lives
        ev = step(state, a)
        lost = before - state.lives
        if a.kind == "hint":
            seat[p]["hints"] += 1
            hints += 1
            pending[a.target] = ev.touched
        elif a.kind == "discard":
            seat[p]["discards"] += 1
        else:
            seat[p]["plays"] += 1
            seat[p]["misplays"] += not ev.success
        if i in fell:
            lost_fb += lost
        else:
            lost_model += lost
    return {
        "seats": [dict(c) for c in seat],
        "hints": hints,
        "hints_followed_by_a_touched_play": followed,
        "lives_lost_by_fallbacks": lost_fb,
        "lives_lost_by_model": lost_model,
        "fireworks": sum(state.board),
        "score": state.score,
    }


def report(runs: Path, label: str = "confirmatory") -> dict:
    rules = Rules(players=PLAYERS)
    arms = [
        (s, rows)
        for s, rows in h2.load_runs(runs)
        if s["players"] == PLAYERS and s["label"] == label
    ]
    assert len(arms) == 1, [s["run"] for s, _ in arms]
    summary, rows = arms[0]
    games = {}
    for path in sorted(runs.rglob("games.jsonl")):
        if path.parent.name == summary["run"]:
            for line in path.read_text().splitlines():
                g = json.loads(line)
                games[g["seed"]] = g
    prefix = h2.complete_prefix(rows)
    deals: list[dict[str, Any]] = []
    for r in prefix:
        s = r["seed"]
        deck = seeded_deck(rules, s)
        ceil, verdict = h2.ceiling(rules, deck)
        info = play_info_game(rules, deck, hansim=False).score
        g = games[s]
        b = behavior(rules, deck, g["moves"], g["fallback_turns"])
        assert b["score"] == r["score"], (s, b["score"], r["score"])
        deals.append(
            {
                "seed": s,
                "ceiling": ceil,
                "verdict": verdict,
                "information_strategy": info,
                "score": r["score"],
                "score_at_failure": b["fireworks"],
                "lives_lost": r["lives_lost"],
                "fallbacks": r["fallbacks"],
                "truncated": r["truncated"],
                "turns": r["turns"],
                "calls": r["calls"],
                "cost": r["cost"],
                "completion_tokens": r["completion_tokens"],
                "behavior": b,
            }
        )
    gap = [d["information_strategy"] - d["score"] for d in deals]
    sd = (
        (sum((x - h2.mean(gap)) ** 2 for x in gap) / (len(gap) - 1)) ** 0.5
        if len(gap) > 1
        else None
    )
    lo, hi = h2.bootstrap(gap)
    primary = {
        "deals": len(deals),
        "seeds": [d["seed"] for d in deals],
        "gap_to_information_strategy": h2.mean(gap),
        "interval": [lo, hi],
        "sd": sd,
        "detectable_effect": MDE_Z * sd / len(gap) ** 0.5 if sd else None,
        "decision": "within 3 points"
        if hi < 3
        else "far below (lower end above 10)"
        if lo > 10
        else "between: reported as it is",
    }

    def mean_of(k):
        return h2.mean(d[k] for d in deals)

    ceil_gap = [d["ceiling"] - d["score"] for d in deals]
    fail_gap = [d["information_strategy"] - d["score_at_failure"] for d in deals]
    tot = Counter()
    for d in deals:
        for c in d["behavior"]["seats"]:
            tot.update(c)
    hints = sum(d["behavior"]["hints"] for d in deals)
    sec = {
        "gap_to_ceiling": {"mean": h2.mean(ceil_gap), "interval": list(h2.bootstrap(ceil_gap))},
        "gap_to_information_strategy_at_failure": {
            "mean": h2.mean(fail_gap),
            "interval": list(h2.bootstrap(fail_gap)),
        },
        "score": mean_of("score"),
        "score_at_failure": mean_of("score_at_failure"),
        "information_strategy": mean_of("information_strategy"),
        "ceiling": mean_of("ceiling"),
        "perfect": sum(d["score"] == rules.max_score for d in deals),
        "perfect_wilson": list(
            h2.wilson(sum(d["score"] == rules.max_score for d in deals), len(deals))
        ),
        "all_lives_lost": sum(d["lives_lost"] == rules.lives for d in deals),
        "all_lives_lost_wilson": list(
            h2.wilson(sum(d["lives_lost"] == rules.lives for d in deals), len(deals))
        ),
        "turns_per_game": mean_of("turns"),
        "fallbacks_per_game": mean_of("fallbacks"),
        "truncated_share_of_calls": sum(d["truncated"] for d in deals)
        / sum(d["calls"] for d in deals),
        "completion_tokens_per_call": sum(d["completion_tokens"] for d in deals)
        / sum(d["calls"] for d in deals),
        "moves": dict(tot),
        "hint_followed_share": sum(d["behavior"]["hints_followed_by_a_touched_play"] for d in deals)
        / hints
        if hints
        else None,
        "lives_lost_by_fallbacks": sum(d["behavior"]["lives_lost_by_fallbacks"] for d in deals),
        "lives_lost_by_model": sum(d["behavior"]["lives_lost_by_model"] for d in deals),
        "cost": sum(d["cost"] for d in deals),
    }
    return {
        "plan": PLAN,
        "run": summary["run"],
        "model": summary["model"],
        "provider": summary["provider"],
        "max_tokens": summary["max_tokens"],
        "reasoning_effort": summary["reasoning_effort"],
        "spent": summary["spent"],
        "stopped_at_seed": summary.get("stopped_at_seed"),
        "bootstrap": "95% percentile, 10,000 resamples over deals, seed 0",
        "primary": primary,
        "secondary": sec,
        "deals": deals,
    }


def first_prompts(source: Path, n: int) -> list[str]:
    """The first n prompts of the lowest-numbered games' call logs under source."""
    out: list[str] = []
    logs = sorted(source.rglob("seed-*.jsonl"), key=lambda p: int(p.stem.split("-")[1]))
    for path in logs:
        for line in path.read_text().splitlines():
            if len(out) < n:
                out.append(json.loads(line)["prompt"])
    return out


def last_move(reply: str) -> str | None:
    """The last move a reply writes, as `llm.parse_action` reads it, without the state."""
    from hanabi.llm import MOVE

    ms = list(MOVE.finditer(reply))
    return " ".join(ms[-1].group(0).upper().split()) if ms else None


def repeat(args) -> dict:
    """Send the first prompts of a run again, `times` times each, at the run's settings;
    count the prompts whose move changed and whose reply text changed."""
    from concurrent.futures import ThreadPoolExecutor

    from hanabi.hosted import Hosted, Meter, RouterClient, fetch_models, listed_price

    source = Path(args.source)
    if args.source.startswith("hf:"):
        from huggingface_hub import snapshot_download

        sub = args.source.removeprefix("hf:")
        root = snapshot_download(
            "arclabs561/games-runs", repo_type="dataset", allow_patterns=[f"{sub}/calls/*"]
        )
        source = Path(root) / sub
    prompts = first_prompts(source, args.n)
    originals: dict[str, str] = {}
    for path in source.rglob("seed-*.jsonl"):
        for line in path.read_text().splitlines():
            c = json.loads(line)
            originals.setdefault(c["prompt"], c["reply"])
    token = os.environ.get("HF_TOKEN", "")
    price = listed_price(fetch_models(token), args.model, args.provider)
    meter = Meter(price, args.cap, price.cost(2500, args.max_tokens))
    calls: list[dict] = []
    player = Hosted(
        args.model, args.provider, meter, max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort, client=RouterClient(token), log=calls.append,
    )  # fmt: skip
    jobs = [p for p in prompts for _ in range(args.times)]
    with ThreadPoolExecutor(args.workers) as pool:
        replies = list(pool.map(player, jobs))
    moves_changed = text_changed = 0
    for i, p in enumerate(prompts):
        got = replies[i * args.times : (i + 1) * args.times]
        moves_changed += len({last_move(originals[p]), *(last_move(r) for r in got)}) > 1
        text_changed += len({originals[p], *got}) > 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "repeat-calls.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    result = {
        "source": args.source,
        "prompts": len(prompts),
        "times": args.times,
        "move_changed": moves_changed,
        "reply_changed": text_changed,
        "truncated": sum(c["finish_reason"] == "length" for c in calls),
        "served_providers": dict(Counter(str(c.get("served_provider")) for c in calls)),
        "spent": meter.spent,
        "cap": args.cap,
    }
    (out / "repeat.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("report")
    r.add_argument("--runs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="confirmatory")
    q = sub.add_parser("repeat")
    q.add_argument("--source", required=True, help="a folder of calls, or hf:runs/h5/<run>")
    q.add_argument("--n", type=int, default=20)
    q.add_argument("--times", type=int, default=2)
    q.add_argument("--model", default="openai/gpt-oss-120b")
    q.add_argument("--provider", default="novita")
    q.add_argument("--max-tokens", type=int, default=16384)
    q.add_argument("--reasoning-effort", default="medium")
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--cap", type=float, required=True)
    q.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "repeat":
        repeat(args)
        return 0
    result = report(Path(args.runs), args.label)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result["primary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
