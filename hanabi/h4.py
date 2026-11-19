"""Plan H4: ceilings, replay, behavior and the paired bootstrap.

    uv run python -m hanabi.h4 ceilings          # the open-game ceiling of each test deal
    uv run python -m hanabi.h4 check FILE SHA256 # a copied summary matches the job's digest

`hanabi.ippo evaluate` calls `write_summary` at the end of its job. It replays every game
in `rules.py`, which checks the score JaxMARL gave and gives the events (cards touched by
each hint) that the behavior counts need, then computes every number of the plan and
writes `h4.json`. The machine that writes the README cannot reach the Hub, so the summary
is copied from the private dataset and checked against the SHA-256 the job printed.

A **policy** is one training seed of one arm. In a **pairing** (i, j), policy i sits in
seat 0 and moves first, and policy j in seat 1. Cross-play within an arm uses its 56
ordered pairs of different seeds; self-play pairs a policy with itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from hanabi import dfs_solver
from hanabi.jax_bridge import H4_RULES, to_rules_action
from hanabi.rules import new_game, seeded_deck, step
from hanabi.upper_bound import best_score, verdict, wilson

RESULTS = Path(__file__).parent / "results"
CEILINGS = RESULTS / "h4-ceilings.json"
TEST_SEEDS = range(400_000, 402_000)
BOOT, BOOT_SEED = 10_000, 0

# Behavior counts per seat; the slot lists have one entry per hand slot.
FEATURES = (
    ["turns", "plays", "misplays", "discards", "color_hints", "rank_hints"]
    + ["color_hint_then_play", "rank_hint_then_play"]
    + [f"play_slot{i}" for i in range(5)]
    + [f"discard_slot{i}" for i in range(5)]
)
F = {name: i for i, name in enumerate(FEATURES)}


# ---------------------------------------------------------------------------- ceilings


def ceiling(seed: int, budget: int = 200_000) -> dict:
    """The highest score of the deal with every card in view, by Solver A (`verdict`,
    then `best_score` for a lost deal), with Solver B's verdict on reaching 10 as a check."""
    rules = H4_RULES
    deck = seeded_deck(rules, seed)
    v, certs = verdict(rules, deck, budget)
    b, _ = dfs_solver.solve(rules, deck, nodes=1_000_000)
    assert v != "undecided" and b is not None, seed
    assert (v == "won") == b, f"the two solvers disagree on deal {seed}"
    top = rules.max_score if v == "won" else best_score(rules, deck, budget)
    return {"seed": seed, "ceiling": top, "certificates": certs}


def compute_ceilings(seeds=TEST_SEEDS) -> dict:
    rows = [ceiling(s) for s in seeds]
    values = [r["ceiling"] for r in rows]
    return {
        "rules": "two players, two colors, ranks 1-5 with 3, 2, 2, 2, 1 copies, hands of 5, 8 tokens, 3 lives",
        "seeds": [seeds[0], seeds[-1]],
        "how": "Solver A (upper_bound.verdict, 200,000 nodes, then best_score) and Solver B "
        "(dfs_solver.solve) agree on whether each deal reaches 10",
        "winnable": sum(v == H4_RULES.max_score for v in values),
        "counts": {str(k): values.count(k) for k in sorted(set(values))},
        "mean": round(float(np.mean(values)), 4),
        "ceiling": values,
        "lost": {str(r["seed"]): r["certificates"] for r in rows if r["ceiling"] < 10},
    }


# ---------------------------------------------------------------------------- replay


def replay_counts(actions, deck: list[int], score: int | None = None) -> np.ndarray:
    """Replay one game's actions in `rules.py`. Returns its behavior counts per seat (2 x
    features). With JaxMARL's `score`, raises unless the game has ended with that score."""
    rules = H4_RULES
    state = new_game(rules, deck)
    counts = np.zeros((2, len(FEATURES)), dtype=np.int32)
    pending = None  # (receiver, attr, touched slots) of the last hint
    for a in actions:
        if a < 0 or state.over:
            break
        p = state.current
        action = to_rules_action(int(a), p, rules)
        if pending and pending[0] == p:
            if action.kind == "play" and action.index in pending[2]:
                counts[1 - p, F[f"{pending[1]}_hint_then_play"]] += 1
            pending = None
        ev = step(state, action)
        c = counts[p]
        c[F["turns"]] += 1
        if action.kind == "hint":
            c[F[f"{action.attr}_hints"]] += 1
            pending = (action.target, action.attr, ev.touched)
        elif action.kind == "discard":
            c[F["discards"]] += 1
            c[F[f"discard_slot{action.index}"]] += 1
        else:
            c[F["plays"]] += 1
            c[F["misplays"]] += not ev.success
            c[F[f"play_slot{action.index}"]] += 1
    if score is not None and (state.score != score or not state.over):
        raise AssertionError(f"rules.py scores {state.score}, JaxMARL {score} (over: {state.over})")
    return counts


def _replay_block(args):
    actions, scores, decks = args
    out = np.zeros(actions.shape[:-2] + (2, len(FEATURES)), dtype=np.int64)
    for idx in np.ndindex(actions.shape[:-2]):  # pairs of this block
        for d in range(actions.shape[-2]):
            out[idx] += replay_counts(actions[idx][d], decks[d], int(scores[idx][d]))
    return out


def replay_all(actions: np.ndarray, scores: np.ndarray, deal_seeds) -> np.ndarray:
    """Behavior counts per pairing and seat, summed over deals (n x n x 2 x features),
    replaying every game; uses every CPU."""
    decks = [seeded_deck(H4_RULES, int(s)) for s in deal_seeds]
    n = actions.shape[0]
    jobs = [(actions[i], scores[i], decks) for i in range(n)]
    with Pool(os.cpu_count() or 1) as pool:
        return np.stack(pool.map(_replay_block, jobs))


# ---------------------------------------------------------------------------- statistics


def iqm(values) -> float:
    """Interquartile mean: the mean of the middle half."""
    v = np.sort(np.asarray(values, dtype=float))
    k = len(v) // 4
    return float(v[k : len(v) - k].mean())


def cross_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def point(x: np.ndarray) -> dict:
    """x: one arm's matrix per deal (n x n x deals). Self-play and cross-play means."""
    n = x.shape[0]
    per_pair = x.mean(axis=2)
    return {
        "self": float(np.diag(per_pair).mean()),
        "cross": float(per_pair[cross_mask(n)].mean()),
    }


def two_level(arms: dict[str, np.ndarray], seed: int = BOOT_SEED, reps: int = BOOT) -> dict:
    """The plan's bootstrap: resample deals (the same deals for every arm) and, within each
    arm, its seeds; rebuild the pairs from the resampled seeds. A pair of two draws of the
    same seed is not a cross-play pair and is left out. Returns, per arm, the resampled
    self-play and cross-play means (reps each)."""
    rng = np.random.default_rng(seed)
    d = next(iter(arms.values())).shape[2]
    out = {k: {"self": np.empty(reps), "cross": np.empty(reps)} for k in arms}
    block = 500
    for b0 in range(0, reps, block):
        m = min(block, reps - b0)
        w = rng.multinomial(d, np.full(d, 1 / d), size=m) / d  # m x deals
        for name in sorted(arms):
            x = arms[name]
            n = x.shape[0]
            means = w @ x.reshape(n * n, d).T  # m x (n*n)
            r = rng.integers(0, n, size=(m, n))
            diag = np.take_along_axis(means, r * n + r, axis=1)
            out[name]["self"][b0 : b0 + m] = diag.mean(axis=1)
            a, c = np.nonzero(cross_mask(n))
            ra, rc = r[:, a], r[:, c]
            vals = np.take_along_axis(means, ra * n + rc, axis=1)
            valid = ra != rc
            with np.errstate(invalid="ignore"):
                out[name]["cross"][b0 : b0 + m] = (vals * valid).sum(axis=1) / valid.sum(axis=1)
    return out


def interval(samples: np.ndarray) -> list[float]:
    """The 95% percentile interval. A resample whose seeds are all one seed has no
    cross-play pair; such resamples (about 1 in 2 million with 8 seeds) are left out."""
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def estimate(value: float, samples: np.ndarray) -> dict:
    return {"value": round(value, 4), "interval": interval(samples)}


def arm_stats(arms: dict[str, np.ndarray], prefix: str = "") -> dict:
    """Self-play, cross-play and their gap per arm, with intervals; and the primary
    difference of cross-play between the arms, when there are two."""
    boot = two_level(arms)
    out = {}
    for name, x in arms.items():
        p, b = point(x), boot[name]
        out[name] = {
            f"{prefix}self": estimate(p["self"], b["self"]),
            f"{prefix}cross": estimate(p["cross"], b["cross"]),
            f"{prefix}self_minus_cross": estimate(p["self"] - p["cross"], b["self"] - b["cross"]),
        }
    names = sorted(arms)
    if len(names) == 2:
        lo, hi = names
        pl, ph = point(arms[lo])["cross"], point(arms[hi])["cross"]
        out[f"{prefix}cross {hi} minus {lo}"] = estimate(
            ph - pl, boot[hi]["cross"] - boot[lo]["cross"]
        )
        sl, sh = point(arms[lo])["self"], point(arms[hi])["self"]
        out[f"{prefix}self {hi} minus {lo}"] = estimate(
            sh - sl, boot[hi]["self"] - boot[lo]["self"]
        )
    return out


# ---------------------------------------------------------------------------- summary


def behavior_rates(c: np.ndarray, games: int) -> dict:
    """Rates from summed counts of one seat or of both (features,)."""
    hints = c[F["color_hints"]] + c[F["rank_hints"]]
    plays, discards = c[F["plays"]], c[F["discards"]]
    safe = lambda a, b: round(float(a) / b, 4) if b else None  # noqa: E731
    return {
        "per_game": {
            k: safe(c[F[k]], games)
            for k in ("turns", "plays", "misplays", "discards", "color_hints", "rank_hints")
        },
        "color_share_of_hints": safe(c[F["color_hints"]], hints),
        "play_after_color_hint": safe(c[F["color_hint_then_play"]], c[F["color_hints"]]),
        "play_after_rank_hint": safe(c[F["rank_hint_then_play"]], c[F["rank_hints"]]),
        "play_slots": [safe(c[F[f"play_slot{i}"]], plays) for i in range(5)],
        "discard_slots": [safe(c[F[f"discard_slot{i}"]], discards) for i in range(5)],
    }


def behavior(counts: np.ndarray, members: dict[str, list[int]], deals: int) -> dict:
    """Per policy, its own actions in self-play (both seats) and in cross-play within its
    arm (its seat in each pair), and per arm and seat in self-play and in cross-play."""
    per_policy, per_arm = {}, {}
    for arm, idx in members.items():
        cross_seat = np.zeros((2, len(FEATURES)), dtype=np.int64)
        self_seat = np.zeros((2, len(FEATURES)), dtype=np.int64)
        for i in idx:
            others = [j for j in idx if j != i]
            mine_cross = counts[i, others, 0].sum(axis=0) + counts[others, i, 1].sum(axis=0)
            per_policy[str(i)] = {
                "self": behavior_rates(counts[i, i].sum(axis=0), deals),
                "cross": behavior_rates(mine_cross, 2 * len(others) * deals),
            }
            self_seat += counts[i, i]
            for j in others:
                cross_seat += counts[i, j]
        n = len(idx)
        per_arm[arm] = {
            "self_seat0": behavior_rates(self_seat[0], n * deals),
            "self_seat1": behavior_rates(self_seat[1], n * deals),
            "cross_seat0": behavior_rates(cross_seat[0], n * (n - 1) * deals),
            "cross_seat1": behavior_rates(cross_seat[1], n * (n - 1) * deals),
        }
    return {"by_policy": per_policy, "by_arm": per_arm}


def curves(run_dirs: dict[str, Path], policies) -> dict:
    """Each policy's training curve: the mean score of the games that ended in each update,
    and the policy's entropy, at about 25 points."""
    out, cache = {}, {}
    for i, p in enumerate(policies):
        folder = run_dirs[p["run"]]
        if folder not in cache:
            cache[folder] = (
                json.loads((folder / "train.json").read_text()),
                np.load(folder / "curves.npz"),
            )
        rec, c = cache[folder]
        k = rec["seeds"].index(p["seed"])
        ret, ent = c["return"][k], c["entropy"][k]
        pts = np.unique(np.linspace(0, len(ret) - 1, min(25, len(ret))).round().astype(int))
        batch = rec["config"]["num_envs"] * rec["config"]["num_steps"]
        out[str(i)] = {
            "steps": [int((t + 1) * batch) for t in pts],
            "score": [round(float(ret[t]), 3) for t in pts],
            "entropy": [round(float(ent[t]), 3) for t in pts],
            "last_10_updates": round(float(ret[-10:].mean()), 4),
        }
    return out


def summarize(eval_dir: Path, ceilings: list[int] | None, run_dirs: dict | None = None) -> dict:
    """Every number of plan H4 from the evaluation's games."""
    rec = json.loads((eval_dir / "evaluate.json").read_text())
    games = np.load(eval_dir / "games.npz")
    policies = rec["policies"]
    deal_seeds = games["deal_seeds"]
    n_deals = len(deal_seeds)
    members: dict[str, list[int]] = {}
    for i, p in enumerate(policies):
        members.setdefault(f"{p['ent_coef']:g}", []).append(i)
    out: dict = {
        "policies": policies,
        "deal_seeds": [int(deal_seeds[0]), int(deal_seeds[-1])],
        "deals": n_deals,
        "arms": members,
        "bootstrap": {
            "resamples": BOOT,
            "seed": BOOT_SEED,
            "units": "deals, and seeds within each arm",
        },
    }
    for mode in ("greedy", "sampled"):
        s = games[f"{mode}_scores"].astype(np.int64)
        arms = {a: s[np.ix_(idx, idx)] for a, idx in members.items()}
        stats = arm_stats(arms)
        perfect = arm_stats(
            {a: (x == H4_RULES.max_score).astype(float) for a, x in arms.items()}, "perfect_"
        )
        block: dict = {"scores": stats, "perfect": perfect}
        if ceilings is not None:
            ceil = np.asarray(ceilings, dtype=np.int64)
            gap = arm_stats({a: ceil[None, None, :] - x for a, x in arms.items()}, "gap_")
            block["gap_to_ceiling"] = gap
        per_pair = s.mean(axis=2)
        block["matrix"] = [[round(float(v), 3) for v in row] for row in per_pair]
        block["iqm_cross"] = {
            a: round(iqm(per_pair[np.ix_(idx, idx)][cross_mask(len(idx))]), 4)
            for a, idx in members.items()
        }
        perfect_by_policy: dict[str, list[float]] = {}
        block["self_perfect_by_policy"] = perfect_by_policy
        for i in range(len(policies)):
            k = int((s[i, i] == H4_RULES.max_score).sum())
            lo, hi = wilson(k, n_deals)
            perfect_by_policy[str(i)] = [
                round(k / n_deals, 4),
                round(lo, 4),
                round(hi, 4),
            ]
        counts = replay_all(games[f"{mode}_actions"], games[f"{mode}_scores"], deal_seeds)
        block["replayed_in_rules_py"] = int(s.size)
        block["behavior"] = behavior(counts, members, n_deals)
        out[mode] = block
    out["primary"] = {
        k: v for k, v in out["greedy"]["scores"].items() if k.startswith("cross ") and "minus" in k
    }
    if ceilings is not None:
        out["ceiling_mean"] = round(float(np.mean(ceilings)), 4)
    if run_dirs:
        out["curves"] = curves(run_dirs, policies)
    return out


def write_summary(out: Path, runs) -> Path:
    """In the evaluation job: the summary next to the games, and its SHA-256."""
    rec = json.loads((out / "evaluate.json").read_text())
    ceil = None
    if CEILINGS.exists():
        table = json.loads(CEILINGS.read_text())
        by_seed = dict(
            zip(range(table["seeds"][0], table["seeds"][1] + 1), table["ceiling"], strict=True)
        )
        seeds = [int(x) for x in np.load(out / "games.npz")["deal_seeds"]]
        if all(x in by_seed for x in seeds):
            ceil = [by_seed[x] for x in seeds]
    run_dirs = {r: resolve_run(r) for r in sorted({p["run"] for p in rec["policies"]})}
    summary = summarize(out, ceil, run_dirs)
    text = json.dumps(summary, separators=(",", ":"), sort_keys=True) + "\n"
    path = out / "h4.json"
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (out / "h4.json.sha256").write_text(digest + "\n")
    print(f"h4.json: {len(text)} bytes, sha256 {digest}", flush=True)
    return path


def resolve_run(run: str) -> Path:
    """A run folder, or the local copy of an `hf:` run that `ippo.load_runs` downloaded."""
    if not run.startswith("hf:"):
        return Path(run)
    from huggingface_hub import snapshot_download

    sub = run[3:].strip("/")
    local = snapshot_download(
        "arclabs561/games-runs",
        repo_type="dataset",
        allow_patterns=[f"{sub}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(local) / sub


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ceilings")
    c = sub.add_parser("check")
    c.add_argument("file", type=Path)
    c.add_argument("sha256")
    args = parser.parse_args()
    if args.cmd == "ceilings":
        result = compute_ceilings()
        CEILINGS.write_text(json.dumps(result, indent=1) + "\n")
        print({k: v for k, v in result.items() if k not in ("ceiling", "lost")})
    else:
        digest = hashlib.sha256(args.file.read_bytes()).hexdigest()
        print("match" if digest == args.sha256 else f"MISMATCH: {digest}")
        raise SystemExit(digest != args.sha256)


if __name__ == "__main__":
    main()
