"""How close do human teams get to the best score their deal allowed?

Vocabulary:

- The **ceiling** of a game in progress is the highest score the open game allows from it:
  the most that a team who could see every card, the deck included, could still reach. The
  ceiling of the deal is the ceiling before the first turn. No team scores above it, and no
  team that cannot see its own cards can be expected to reach it: the strongest published
  two-player agents average 24.1 to 24.6 in self-play (the README's table), while nearly
  every deal's ceiling is 25.
- The gap of a game, 25 minus its score, has two parts. 25 minus the deal's ceiling is
  **ruled out by the deal**. The ceiling minus the score is **below the ceiling**: points
  lost to hidden information or to play. Without a reference agent that plays the real
  game well on the same decks, the second part cannot be split further, and this module
  does not try. As a rough reference it scores the information strategy of
  `hanabi.information`, one fixed convention, on each deck.
- The ceiling never rises during a game: the moves the team made are one of the open
  game's continuations. A **drop** is a turn after which it is lower than before. The
  game's drops add up to the points below the ceiling, and each is charged to the action of
  its turn. A drop is not a judgment of that action: a player who cannot see their own hand
  may have had no better choice.
- A card is **critical** when it is still needed, its color can still reach it, and every
  other copy is gone. Discarding or misplaying it lowers the most that its color can reach.

`ceiling` finds the ceiling with the fast solver of `hanabi.solver`, trying targets from 25
down; `trajectory` finds every drop by binary search over the turns, one target at a time.
A search that runs out of its node budget leaves a range instead of a number, and the
records say so.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hanabi import dfs_solver
from hanabi.humans_data import FILES, HumanGame, decode, load
from hanabi.information import play_info_game
from hanabi.rules import (
    Rules,
    State,
    card_name,
    color_of,
    copies,
    legal,
    new_game,
    rank_of,
    step,
)
from hanabi.solver import SCHEDULE, copy_state, replay, root_certificates, winnable

BUDGETS = (20_000, 300_000)  # nodes per search, the first two budgets of plan H1
ATTEMPTS = ((SCHEDULE, True), ((10**9,), True), (SCHEDULE, False))  # as in upper_bound


def reachable(
    rules: Rules, deck, target: int, state: State | None = None, budgets=BUDGETS, effort=None
):
    """Can the open game reach `target` from `state` (by default the deal)? True, False, or
    None when every search ran out of budget. A True answer is replayed by the rules engine.

    Each budget of nodes is tried with the searches of plan H1; for a game in progress, the
    free-order game then gets its turn to prove the target out of reach. The
    searches find reachable targets quickly, and the free-order game is often quicker at
    proving the others lost. `effort`, a dict with "hard" and "cap", counts the questions
    that go past the first budget and answers None once there have been "cap" of them.

    The free-order game gets ten times each budget, at most 1,000,000 states."""
    if state is not None:
        if state.score >= target:
            return True
        if state.over:
            return False
    for k, budget in enumerate(budgets):
        if k == 1 and effort is not None:
            effort["hard"] += 1
            if effort["hard"] > effort["cap"]:
                return None
        for schedule, pass_first in ATTEMPTS:
            ok, line = winnable(
                rules,
                list(deck),
                budget,
                pass_first=pass_first,
                schedule=schedule,
                target=target,
                start=state,
            )
            if ok:
                assert replay(rules, list(deck), line, start=state) >= target, "bad line"
                return True
            if ok is False:
                return False
        nodes = min(10 * budget, 1_000_000)
        if state is not None and free_order_lost(rules, deck, target, state, nodes) is False:
            return False
    return None


def free_order_lost(rules: Rules, deck, target: int, state: State, nodes: int) -> bool | None:
    """False when the free-order game of `hanabi.dfs_solver` (unlimited tokens and lives)
    cannot reach `target` from `state`, which proves the real game cannot either; otherwise
    None, since a relaxation that reaches the target proves nothing."""
    g = dfs_solver.Game(rules, list(deck), goal=target)
    pile = tuple(state.board)
    hands = tuple(g.tidy(h, pile) for h in state.hands)
    budget = dfs_solver.Budget(nodes) if nodes else None  # 0: no limit
    ok = dfs_solver.free_order_winnable(
        g, pile, hands, state.pos, state.current, state.turns_left, {}, budget
    )
    return False if ok is False else None


@dataclass(frozen=True)
class Ceiling:
    """The ceiling lies in [low, high]; exact when they are equal. `certificates` are the
    bounds that prove at the deal that 25 is out of reach (empty when it is not, or when only
    the complete search proves it)."""

    low: int
    high: int
    certificates: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return self.low == self.high


def ceiling(rules: Rules, deck, state: State | None = None, budgets=BUDGETS) -> Ceiling:
    """The ceiling of the deal, or of a game in progress on it."""
    floor = state.score if state is not None else 0
    certs: list[str] = []
    high = rules.max_score
    if state is None:
        certs = root_certificates(rules, list(deck))
        if not certs:
            ok = reachable(rules, deck, high, None, budgets)
            if ok:
                return Ceiling(high, high)
            if ok is None:  # the tests of the second solver, as in plan H1's last steps
                more = dfs_solver.certificates(rules, list(deck), 1_000_000)
                certs = [{"schedule": "time", "one-hand": "pooled"}.get(c, c) for c in more]
            if ok is False or certs:
                high -= 1
        else:
            high -= 1
    top = high
    for target in range(top, floor, -1):
        ok = reachable(rules, deck, target, state, budgets)
        if ok:
            return Ceiling(target, high, tuple(sorted(set(certs))))
        if ok is False and high == target:
            high = target - 1
    return Ceiling(floor, high, tuple(sorted(set(certs))))


# --- one recorded game -------------------------------------------------------------------


def states_of(game: HumanGame) -> list[State]:
    """The game before each turn, and the final state last."""
    rules = Rules(players=game.players)
    s = new_game(rules, list(game.deck))
    out = [copy_state(s)]
    for a_id in game.actions:
        a = decode(a_id, s.current, rules)
        assert legal(s, a), f"game {game.game_id}: illegal action on turn {s.turn}"
        step(s, a)
        out.append(copy_state(s))
    return out


def is_critical(card: int, s: State) -> bool:
    """Is this card, about to leave a hand in state `s`, the last copy that can still count?"""
    rules = s.rules
    c, r = color_of(card, rules), rank_of(card, rules)
    if s.board[c] >= r:
        return False
    gone = Counter(s.discards)
    for lower in range(s.board[c] + 1, r):
        if gone[card - (r - lower)] >= copies(card - (r - lower), rules):
            return False  # the color can no longer reach this card
    return gone[card] == copies(card, rules) - 1


def action_kind(before: State, after: State, a) -> str:
    """'play', 'strike', 'critical discard', 'discard' or 'hint'."""
    if a.kind == "hint":
        return "hint"
    card = before.hands[before.current][a.index]
    if a.kind == "play":
        return "play" if after.board != before.board else "strike"
    return "critical discard" if is_critical(card, before) else "discard"


def first_below(rules, deck, states, target, lo, hi, budgets, effort=None):
    """The first turn t in (lo, hi] whose state cannot reach `target`, given that state lo
    can and state hi cannot. None if a search is undecided."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok = reachable(rules, deck, target, states[mid], budgets, effort)
        if ok is None:
            return None
        lo, hi = (mid, hi) if ok else (lo, mid)
    return hi


def trajectory(game: HumanGame, start: int, budgets=BUDGETS, hard: int = 0) -> list[int] | None:
    """For each target from the score plus one up to `start` (the deal's ceiling), the turn
    after which the ceiling fell below it: drop turns, latest first. None if a search is
    undecided, or if more than `hard` questions (0: no limit) needed the larger budgets."""
    effort = {"hard": 0, "cap": hard} if hard else None
    rules = Rules(players=game.players)
    states = states_of(game)
    last = len(states) - 1
    score = states[-1].score
    out = []
    hi = last
    for target in range(score + 1, start + 1):
        t = first_below(rules, game.deck, states, target, 0, hi, budgets, effort)
        if t is None:
            return None
        out.append(t)
        hi = t  # a higher target is lost no later
    return out


HARD = 8  # questions per game allowed past the first budget while finding its drops


def analyze(game: HumanGame, budgets=BUDGETS, hard: int = HARD) -> dict:
    """The record of one game: replay check, ceiling, drops, strikes, critical discards."""
    rules = Rules(players=game.players)
    t0 = time.time()
    states = states_of(game)
    final = states[-1]
    row: dict = {
        "game_id": game.game_id,
        "players": game.players,
        "file": game.source,
        "score": game.score,
        "replayed": final.score,
        "turns": len(game.actions),
        "ended": final.over,
    }
    c = ceiling(rules, game.deck, None, budgets)
    row.update(ceiling_low=c.low, ceiling_high=c.high, certificates=list(c.certificates))
    events = []
    for t, a_id in enumerate(game.actions):
        before, after = states[t], states[t + 1]
        a = decode(a_id, before.current, rules)
        kind = action_kind(before, after, a)
        if kind in ("strike", "critical discard"):
            card = before.hands[before.current][a.index]
            events.append({"turn": t + 1, "kind": kind, "card": card_name(card, rules)})
    row["events"] = events
    # the reference: one fixed convention on the same deck, with the official discard rule
    row["info_score"] = play_info_game(rules, list(game.deck), hansim=False).score
    # hanab.live ends a game early once no more points can be scored; check that this is
    # why a record stops before the end of the rules, so that its last state is final
    finished = final.over
    if not final.over:
        row["no_points_left"] = reachable(rules, game.deck, final.score + 1, final, budgets)
        finished = row["no_points_left"] is False
    drops = None
    if finished and c.exact and c.low > final.score:
        drops = trajectory(game, c.low, budgets, hard)
    elif finished and c.exact:
        drops = []
    if drops is not None:
        row["drops"] = [
            {
                "turn": t,
                "kind": action_kind(
                    states[t - 1],
                    states[t],
                    decode(game.actions[t - 1], states[t - 1].current, rules),
                ),
                "deck_left": states[t - 1].deck_left,
                "hints": states[t - 1].hints,
            }
            for t in sorted(drops)
        ]
    row["seconds"] = round(time.time() - t0, 2)
    return row


def run(cache, players: int, part: int, parts: int, out: Path, budgets=BUDGETS, limit=0):
    """Analyze every `parts`-th game, starting at `part`, and write the records as JSON,
    rewritten after each game so that a run stopped early keeps what it finished."""
    games = load(cache, players)[part::parts]
    if limit:
        games = games[:limit]
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    for g in games:
        rows.append(analyze(g, budgets))
        out.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    return rows


# --- the summary -------------------------------------------------------------------------


def bootstrap(xs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, resampling games."""
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def paired_bootstrap(pairs, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of sum(a) / sum(b), resampling games."""
    rng = random.Random(seed)
    k = len(pairs)
    out = []
    for _ in range(n):
        sample = rng.choices(pairs, k=k)
        den = sum(b for _, b in sample)
        out.append(sum(a for a, _ in sample) / den if den else 0.0)
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


PHASES = ((20, 99, "deck 20 or more"), (10, 19, "deck 10 to 19"), (1, 9, "deck 1 to 9"))
PHASES += ((0, 0, "final round"),)


def phase(deck_left: int) -> str:
    return next(name for lo, hi, name in PHASES if lo <= deck_left <= hi)


def summarize(rows: list[dict]) -> dict:
    """The numbers the README quotes, for one number of players."""
    n = len(rows)
    exact = [r for r in rows if r["ceiling_low"] == r["ceiling_high"]]
    open_rows = [r for r in rows if r["ceiling_low"] != r["ceiling_high"]]
    below = [r for r in exact if r["ceiling_low"] < 25]
    ceil = [r["ceiling_low"] for r in exact]
    score = [r["score"] for r in exact]
    forced = [25 - r["ceiling_low"] for r in exact]
    under = [r["ceiling_low"] - r["score"] for r in exact]
    gap = [25 - r["score"] for r in exact]
    info = [r["info_score"] for r in exact]
    info_under = [r["ceiling_low"] - r["info_score"] for r in exact]
    out: dict = {
        "games": n,
        "replay_matches": sum(r["score"] == r["replayed"] for r in rows),
        "games_that_end": sum(r["ended"] for r in rows),
        "ceiling_exact": len(exact),
        "ceiling_open": [
            {"game_id": r["game_id"], "low": r["ceiling_low"], "high": r["ceiling_high"]}
            for r in open_rows
        ],
        "deal_below_25": len(below),
        "deal_below_25_share": len(below) / len(exact),
        "deal_below_25_wilson": wilson(len(below), len(exact)),
        "ceilings": dict(sorted(Counter(ceil).items())),
        "certificates": dict(sorted(Counter(c for r in below for c in r["certificates"]).items())),
        "below_25_without_certificate": sum(not r["certificates"] for r in below),
        "mean_ceiling": mean(ceil),
        "mean_ceiling_ci": bootstrap(ceil),
        "mean_score": mean(score),
        "mean_score_ci": bootstrap(score),
        "mean_score_all_games": mean(r["score"] for r in rows),
        "mean_gap": mean(gap),
        "ruled_out_by_deal": mean(forced),
        "ruled_out_by_deal_ci": bootstrap(forced),
        "below_ceiling": mean(under),
        "below_ceiling_ci": bootstrap(under),
        "deal_share_of_gap": sum(forced) / sum(gap) if sum(gap) else 0.0,
        "deal_share_of_gap_ci": paired_bootstrap(list(zip(forced, gap, strict=True))),
        "perfect": sum(x == 25 for x in score),
        "winnable": sum(c == 25 for c in ceil),
        "perfect_of_winnable": sum(r["score"] == 25 for r in exact if r["ceiling_low"] == 25),
        "reached_ceiling": sum(x == c for x, c in zip(score, ceil, strict=True)),
        "score_above_ceiling": sum(r["score"] > r["ceiling_high"] for r in rows),
        "perfect_games_with_ceiling_below_25": sum(
            r["score"] == 25 and r["ceiling_high"] < 25 for r in rows
        ),
        "info_mean_score": mean(info),
        "info_mean_score_ci": bootstrap(info),
        "info_below_ceiling": mean(info_under),
        "info_perfect": sum(x == 25 for x in info),
        "human_minus_info": mean(h - i for h, i in zip(score, info, strict=True)),
        "human_minus_info_ci": bootstrap([h - i for h, i in zip(score, info, strict=True)]),
        "human_ahead_of_info": sum(h > i for h, i in zip(score, info, strict=True)),
        "info_ahead_of_human": sum(h < i for h, i in zip(score, info, strict=True)),
    }
    out["reached_ceiling_wilson"] = wilson(out["reached_ceiling"], len(exact))
    out["perfect_of_winnable_wilson"] = wilson(out["perfect_of_winnable"], out["winnable"])
    # where the ceiling falls
    traced = [r for r in exact if "drops" in r]
    drops = [d for r in traced for d in r["drops"]]
    out["traced_games"] = len(traced)
    out["untraced_games"] = [r["game_id"] for r in exact if "drops" not in r]
    out["drops"] = len(drops)
    out["drops_by_kind"] = dict(sorted(Counter(d["kind"] for d in drops).items()))
    out["drops_by_phase"] = {
        name: sum(phase(d["deck_left"]) == name for d in drops) for _, _, name in PHASES
    }
    out["drops_by_phase_and_kind"] = {
        name: dict(
            sorted(Counter(d["kind"] for d in drops if phase(d["deck_left"]) == name).items())
        )
        for _, _, name in PHASES
    }
    firsts = [min(d["turn"] for d in r["drops"]) for r in traced if r["drops"]]
    lens = [r["turns"] for r in traced if r["drops"]]
    out["games_with_a_drop"] = len(firsts)
    out["first_drop_turn_median"] = sorted(firsts)[len(firsts) // 2] if firsts else None
    out["first_drop_share_of_game_mean"] = (
        mean(f / n_ for f, n_ in zip(firsts, lens, strict=True)) if firsts else None
    )
    events = [e for r in rows for e in r["events"]]
    strikes = [sum(e["kind"] == "strike" for e in r["events"]) for r in rows]
    crit = [sum(e["kind"] == "critical discard" for e in r["events"]) for r in rows]
    out["strikes_per_game"] = mean(strikes)
    out["strikes_per_game_ci"] = bootstrap(strikes)
    out["games_by_strikes"] = dict(sorted(Counter(strikes).items()))
    out["critical_discards_per_game"] = mean(crit)
    out["critical_discards_per_game_ci"] = bootstrap(crit)
    out["games_with_critical_discard"] = sum(k > 0 for k in crit)
    out["critical_by_rank"] = dict(
        sorted(Counter(e["card"][1] for e in events if e["kind"] == "critical discard").items())
    )
    out["strikes_by_rank"] = dict(
        sorted(Counter(e["card"][1] for e in events if e["kind"] == "strike").items())
    )
    return out


SOURCE = (
    "AH2AC2 open human dataset (Dizdarevic et al., 2025), ah2ac2/datasets on Hugging Face; "
    "only numbers derived from each game are kept here"
)
GAMES_FILE = "humans-games.jsonl"


def merge(folder: Path) -> list[dict]:
    """The records of every block under `folder`, each game once, sorted."""
    rows, seen = [], set()
    for path in sorted(folder.rglob("p*-of-*.json")):
        for r in json.loads(path.read_text()):
            key = (r["players"], r["game_id"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    return sorted(rows, key=lambda r: (r["players"], r["game_id"]))


def summary_of(rows: list[dict]) -> dict:
    players = sorted({r["players"] for r in rows})
    return {
        "source": SOURCE,
        "budgets": list(BUDGETS),
        "players": {str(p): summarize([r for r in rows if r["players"] == p]) for p in players},
    }


def read_games(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def report(folder: Path, out: Path) -> dict:
    """Merge the blocks under `folder`; write the per-game records (one per line, next to
    `out`) and the summary computed from them."""
    rows = merge(folder)
    for r in rows:
        r.pop("file", None)
    (out.parent / GAMES_FILE).write_text(
        "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in rows)
    )
    summary = summary_of(read_games(out.parent / GAMES_FILE))
    out.write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# --- the picture -------------------------------------------------------------------------


def figure(rows: list[dict]) -> str:
    """An SVG: for each number of players, a grid of games counted by the deal's ceiling
    (rows) and the team's score (columns). Cells on the outlined diagonal reached the
    ceiling."""
    cell, left, top, gap_x = 25, 70, 64, 40
    panels = []
    for p in sorted({r["players"] for r in rows}):
        rs = [r for r in rows if r["players"] == p and r["ceiling_low"] == r["ceiling_high"]]
        counts = Counter((r["ceiling_low"], r["score"]) for r in rs)
        panels.append((p, rs, counts))
    lo_score = min(x for _, rs, _ in panels for x in [r["score"] for r in rs])
    lo_ceil = min(c for _, rs, _ in panels for c in [r["ceiling_low"] for r in rs])
    scores = list(range(lo_score, 26))
    ceils = list(range(25, lo_ceil - 1, -1))
    width_one = len(scores) * cell
    width = left + len(panels) * width_one + (len(panels) - 1) * gap_x + 20
    height = top + len(ceils) * cell + 70
    biggest = max(max(c.values()) for _, _, c in panels)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    for k, (p, rs, counts) in enumerate(panels):
        x0 = left + k * (width_one + gap_x)
        out.append(
            f'<text x="{x0}" y="22" font-size="14" font-weight="bold">{p} players, '
            f"{len(rs):,} games</text>"
        )
        out.append(
            f'<text x="{x0 + width_one / 2}" y="{top + len(ceils) * cell + 38}" '
            'text-anchor="middle">score of the team</text>'
        )
        for j, sc in enumerate(scores):
            if sc % 2 == 1 or sc == 25:
                out.append(
                    f'<text x="{x0 + j * cell + cell / 2}" y="{top + len(ceils) * cell + 16}" '
                    f'text-anchor="middle">{sc}</text>'
                )
        diagonal = []
        for i, c in enumerate(ceils):
            y = top + i * cell
            if k == 0:
                out.append(
                    f'<text x="{x0 - 8}" y="{y + cell / 2 + 4}" text-anchor="end">{c}</text>'
                )
            for j, sc in enumerate(scores):
                n = counts.get((c, sc), 0)
                x = x0 + j * cell
                shade = 0.0 if n == 0 else 0.12 + 0.88 * math.log1p(n) / math.log1p(biggest)
                fill = "#f4f4f4" if sc > c else (f"rgba(31,90,160,{shade:.3f})" if n else "white")
                out.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" '
                    'stroke="#ddd" stroke-width="0.6"/>'
                )
                if sc == c:
                    diagonal.append((x, y))
                if n:
                    color = "white" if shade > 0.55 else "#111"
                    out.append(
                        f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                        f'font-size="9" fill="{color}">{n}</text>'
                    )
        out += [
            f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="none" stroke="#222" '
            'stroke-width="1.6"/>'
            for x, y in diagonal
        ]
    out.append(
        f'<text x="14" y="{top + len(ceils) * cell / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {top + len(ceils) * cell / 2})">ceiling of the deal</text>'
    )
    out.append(
        f'<text x="{left}" y="44" fill="#444">Games counted by ceiling and score. Outlined '
        "cells: the team reached the ceiling. Gray: above the ceiling, impossible.</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="analyze one block of games")
    r.add_argument("--players", type=int, choices=sorted(FILES), required=True)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--parts", type=int, default=1)
    r.add_argument("--limit", type=int, default=0, help="only the first games of the block")
    r.add_argument("--cache", default="data/ah2ac2")
    r.add_argument("--fetch", action="store_true", help="download the dataset first")
    r.add_argument("--out", required=True, help="a folder")
    s = sub.add_parser("report", help="merge the blocks and write the summary")
    s.add_argument("--runs", required=True)
    s.add_argument("--out", default="hanabi/results/humans.json")
    f = sub.add_parser("figure", help="draw the picture from the per-game records")
    f.add_argument("--games", default=f"hanabi/results/{GAMES_FILE}")
    f.add_argument("--out", default="hanabi/results/humans-ceiling.svg")
    args = ap.parse_args(argv)
    if args.command == "figure":
        Path(args.out).write_text(figure(read_games(Path(args.games))))
        return
    if args.command == "run":
        if args.fetch:
            from hanabi.humans_data import download

            download(args.cache)
        path = Path(args.out) / f"p{args.players}-{args.part}-of-{args.parts}.json"
        t0 = time.time()
        rows = run(args.cache, args.players, args.part, args.parts, path, limit=args.limit)
        print(f"{len(rows)} games in {time.time() - t0:.0f} s -> {path}")
    else:
        summary = report(Path(args.runs), Path(args.out))
        for p, s_ in summary["players"].items():
            print(p, {k: v for k, v in s_.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
