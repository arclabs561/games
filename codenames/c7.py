"""Plan C7: tables for the GRPO job, the feature policy (arm 2), and the analysis.

Every board gets C6's 120 actions: the 30 clues that rl.actions shortlists in the
EmbeddingGemma 2 space (bare template, tau from results/calibrate-game.json), from all
of words.clue_words(), times the counts 1 to 4. Every action's exact expected turn
value is computed under every guesser (rl.rewards). Boards:

    train   step s of training uses the 16 boards dealt from seed 7000 + s
    val     50 boards from seed 6005 (C6's validation boards; the pilot is scored here)
    test    200 boards from seed 6004 (C6's test boards)

Commands, all run on a CPU where the vectors are:

    python -m codenames.c7 table --steps 300 --out DIR      # DIR/table.npz for the job,
                                                            # DIR/local.npz for the rest
    python -m codenames.c7 feature --data DIR --seed 1 --out OUT      # arm 2, seconds
    python -m codenames.c7 score --data DIR --runs RUNS --out codenames/results/c7.json

table.npz holds the boards, the clues and the training guessers' values on the training
boards: everything the job needs. local.npz holds the features of rl.py and the values
under every guesser on the val and test boards.
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from codenames.actions import expand, read_action_lines, shortlist
from codenames.actions import proposer as c6_proposer
from codenames.evaluate import all_words, guess_scores
from codenames.game import Board, deal
from codenames.norms import eat_counts
from codenames.players import Space, Weigher, embedding_weigher, norm_weigher, space
from codenames.rl import NMAX, actions, fit, policy, rewards
from codenames.rlvr import BOARDS_PER_STEP, INVALID
from codenames.stats import DRAWS
from codenames.study import chosen_spaces, load
from codenames.train import adapted, items, train
from codenames.words import board_words, clue_words

GIVER = "google/embeddinggemma-2"
TRAINING = ("wordllama", "granite", "bge", "microsoft/harrier-oss-v1-270m")
HELD_OUT = ("potion", "Qwen/Qwen3-Embedding-0.6B", "microsoft/harrier-oss-v1-0.6b")
MAP_SEEDS = (1, 2, 3, 4, 5)
SHORTLIST = 30
TRAIN_SEED, VAL_SEED, TEST_SEED = 7000, 6005, 6004
EVAL_BOARDS = {"val": (VAL_SEED, 50), "test": (TEST_SEED, 200)}
C6_ACTIONS = Path(__file__).parent / "results" / "c6-actions.txt"
C6_PICKS = Path(__file__).parent / "results" / "c6-picks-6004.txt"


def deal_boards(seed: int, n: int) -> list[Board]:
    rng = random.Random(seed)
    return [deal(board_words(), rng) for _ in range(n)]


def train_boards(steps: int) -> list[Board]:
    return [b for s in range(steps) for b in deal_boards(TRAIN_SEED + s, BOARDS_PER_STEP)]


def candidates(board: Board, sp: Space, tau: float, pool=None) -> tuple[list[str], np.ndarray]:
    """C6's 30 clues (codenames.actions.shortlist) and the features (120, 7) of
    rl.actions, which computes the same list; action a is clue a // 4 with count
    a % 4 + 1. The pool is words.clue_words()."""
    acts, phi = actions(sp, tau, board, pool or clue_words(), SHORTLIST)
    clues = shortlist(board, sp, tau, SHORTLIST, pool)
    assert acts == expand(clues)
    return clues, phi


def c6_lists_agree(boards: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """For the val and test boards: [boards, boards whose clues equal C6's committed list
    in results/c6-actions.txt]."""
    lists = read_action_lines(C6_ACTIONS.read_text())
    out = {}
    for split, (seed, _) in EVAL_BOARDS.items():
        if f"{split}_clues" in boards:
            mine = [[str(c) for c in row] for row in boards[f"{split}_clues"]]
            out[split] = [len(mine), sum(a == b for a, b in zip(mine, lists[seed], strict=False))]
    return out


MAP_CHECK: dict[str, list[float]] = {}


def potion_maps() -> dict[str, Weigher]:
    """The potion guesser with the linear map that C1 trained on the human guesses of the
    train games, one per training seed, each with its fitted tau (results/train-potion-game.json).
    Each rebuilt map's top-1 on C1's test turns is kept in MAP_CHECK beside C1's."""
    fitted = load("train-potion-game")
    sp = space("potion", "bare", "bare", all_words())
    out = {}
    for seed in MAP_SEEDS:
        a, _ = train(sp, items(sp, seed, eat=False, split="train", scheme="game"), 3, seed=seed)
        run = fitted["runs"][str(seed)]["guess"]
        tuned = adapted(sp, a, f"potion-map-{seed}")
        top1 = guess_scores(tuned, "test", run["tau"], "game")["top1"]
        MAP_CHECK[str(seed)] = [round(top1, 6), round(run["top1"], 6)]
        out[f"potion-map/{seed}"] = embedding_weigher(tuned, run["tau"])
    return out


def guessers(maps: bool = True) -> dict[str, Weigher]:
    """Every scoring guesser: the eight calibrated embedding models, EAT, and the five
    potion maps."""
    cal = load("calibrate-game")
    out: dict[str, Weigher] = {
        m: embedding_weigher(sp, tau) for m, (sp, tau) in chosen_spaces(cal).items()
    }
    out["eat"] = norm_weigher(eat_counts(), cal["eat"]["eps"])
    return out | (potion_maps() if maps else {})


def build(boards: list[Board], giver: tuple[Space, float], scorers: dict[str, Weigher], pool=None):
    """Words, kinds, clues (B, 30), features (B, 120, 7) and values (B, 120, G)."""
    sp, tau = giver
    clues, phis, values = [], [], []
    for b in boards:
        c, phi = candidates(b, sp, tau, pool)
        acts = [(x, n) for x in c for n in range(1, NMAX + 1)]
        clues.append(c)
        phis.append(phi)
        values.append(np.stack([rewards(w, b, acts) for w in scorers.values()], axis=1))
    return {
        "words": np.array([b.words for b in boards]),
        "kinds": np.array([b.kinds for b in boards]),
        "clues": np.array(clues),
        "phi": np.array(phis, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(steps: int, out: Path, eval_boards: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    every = guessers()
    names = list(every)
    giver = c6_proposer()
    job: dict[str, Any] = {"training_guessers": np.array(TRAINING)}
    local: dict[str, Any] = {"guessers": np.array(names)}
    train = build(train_boards(steps), giver, {g: every[g] for g in TRAINING})
    for k in ("words", "kinds", "clues", "values"):
        job[f"train_{k}"] = train[k]
    local["train_phi"] = train["phi"]
    for split, (seed, n) in (eval_boards or EVAL_BOARDS).items():
        part = build(deal_boards(seed, n), giver, every)
        for k in ("words", "kinds", "clues"):
            job[f"{split}_{k}"] = part[k]
        local[f"{split}_phi"], local[f"{split}_values"] = part["phi"], part["values"]
    np.savez_compressed(out / "table.npz", **job)
    np.savez_compressed(out / "local.npz", **local)
    meta = {
        "steps": steps,
        "boards": {"train": [TRAIN_SEED, steps, BOARDS_PER_STEP], **(eval_boards or EVAL_BOARDS)},
        "giver": GIVER,
        "training": list(TRAINING),
        "guessers": names,
        "table_sha256": sha256(out / "table.npz"),
        "local_sha256": sha256(out / "local.npz"),
        "potion_map_top1_rebuilt_and_c1": MAP_CHECK,
        "clues_equal_c6_actions_txt": c6_lists_agree(job),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "table.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_data(folder: Path) -> dict:
    job, local = np.load(folder / "table.npz"), np.load(folder / "local.npz")
    out = {k: job[k] for k in job.files} | {k: local[k] for k in local.files}
    out["guessers"] = [str(g) for g in out["guessers"]]
    out["training_guessers"] = [str(g) for g in out["training_guessers"]]
    return out


# Arm 2: the seven-feature policy of rl.py on the same boards and candidates.
def feature_policy(data: dict, seed: int, steps: int = 400, lr: float = 0.05) -> dict:
    """rl.fit at its defaults against the training guessers, on the training boards of
    the table, and its greedy pick on every val and test board."""
    tr = [
        (phi, {g: data["train_values"][b, :, j] for j, g in enumerate(data["training_guessers"])})
        for b, phi in enumerate(data["train_phi"])
    ]
    t0 = time.perf_counter()
    params = fit(tr, data["training_guessers"], steps, lr, seed)
    f = data["train_phi"].shape[2]
    theta, mu, sd = params[:f], params[f : 2 * f], params[2 * f :]
    picks = {}
    for split in EVAL_BOARDS:
        if f"{split}_phi" in data:
            picks[split] = [
                int(np.argmax(policy(theta, (phi - mu) / sd))) for phi in data[f"{split}_phi"]
            ]
    return {
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "theta": theta.round(4).tolist(),
        "picks": picks,
        "seconds": round(time.perf_counter() - t0, 1),
    }


# The bootstrap over boards and seed pairs.
def two_way_draws(n_seeds: int, n_boards: int, draws: int = DRAWS, seed: int = 0):
    """(draws, n_seeds) and (draws, n_boards) multinomial counts: each replicate draws the
    seed pairs and, once for all of them, the boards, with replacement."""
    rng = np.random.default_rng(seed)
    cs = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=draws)
    cb = rng.multinomial(n_boards, np.full(n_boards, 1 / n_boards), size=draws)
    return cs, cb


def two_way(d: np.ndarray, cs: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Replicates of the mean of d (seeds, boards) under the draws."""
    s, b = d.shape
    return np.einsum("rs,sb,rb->r", cs, d, cb) / (s * b)


def two_way_exact(d: np.ndarray) -> dict[float, float]:
    """The exact distribution of the two-way bootstrap mean, for a few seeds and boards:
    every multiset of seeds and of boards, with its multinomial probability."""
    from collections import Counter
    from itertools import combinations_with_replacement
    from math import factorial

    def multisets(n: int):
        for combo in combinations_with_replacement(range(n), n):
            k = Counter(combo)
            ways = factorial(n)
            for c in k.values():
                ways //= factorial(c)
            yield np.bincount(combo, minlength=n), ways / n**n

    out: dict[float, float] = {}
    s, b = d.shape
    for cs, ps in multisets(s):
        for cb, pb in multisets(b):
            key = round(float(cs @ d @ cb) / (s * b), 12)
            out[key] = out.get(key, 0.0) + ps * pb
    return out


def interval(d: np.ndarray, draws: int = DRAWS, seed: int = 0) -> dict:
    d = np.asarray(d, dtype=float)
    cs, cb = two_way_draws(*d.shape, draws, seed)
    reps = two_way(d, cs, cb)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "mean": float(d.mean()),
        "ci": [float(lo), float(hi)],
        "per_seed": d.mean(axis=1).round(4).tolist(),
        "share_at_or_below_0": float((reps <= 0).mean()),
    }


def values_of(picks: list[int | None], values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """(boards, guessers) values of the picked actions; a reply with no listed clue gets
    the fallback action (C6's rule: the embedding giver's first choice)."""
    idx = np.array([fallback[b] if a is None else a for b, a in enumerate(picks)])
    return values[np.arange(len(idx)), idx]


def behavior(picks: list[int], clues: np.ndarray, first: np.ndarray) -> dict:
    """Share of boards where the pick is the first choice, mean count, mean rarity
    (max(0, 3.5 - Zipf), as in rl.py) and the number of distinct clues."""
    from codenames.words import zipf

    words = [str(clues[b, a // NMAX]) for b, a in enumerate(picks)]
    return {
        "first_choice": float(np.mean(np.array(picks) == first)),
        "mean_count": float(np.mean([a % NMAX + 1 for a in picks])),
        "rarity": float(np.mean([max(0.0, 3.5 - zipf(w)) for w in words])),
        "distinct_clues": len(set(words)),
    }


def windows(curve: list[list[float]], width: int = 20) -> list[dict]:
    """Means over steps [0, 20), [20, 40), ... of the reward, the share of -1 replies,
    the share of truncated replies and TRL's entropy; curve rows are
    [step, mean reward, share -1, share truncated, entropy]."""
    out = []
    for lo in range(0, len(curve), width):
        part = np.array(curve[lo : lo + width], dtype=float)
        row: dict = {"steps": [int(part[0, 0]), int(part[-1, 0])]}
        for j, k in enumerate(("mean_reward", "invalid", "truncated", "entropy"), 1):
            row[k] = round(float(part[:, j].mean()), 4)
        out.append(row)
    return out


def compact(run: Path) -> dict:
    """The summary of one job's output folder that is copied into results/c7-runs/:
    each board's action, the reason when there is none, the reply's length in tokens
    and the reply itself when it has no action; the curves; timings and settings."""

    def read(name: str):
        raw = (run / name).read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    picks, hp = read("picks.json")
    cfg, _ = read("config.json")
    tim, _ = read("timings.json")
    cur, hc = read("curves.json")
    out: dict = {"run": run.name, "picks_sha256": hp, "curves_sha256": hc}
    for k in ("untrained", "trained"):
        out[k] = [
            [r["action"], r["why"], r["tokens"], r["reply"] if r["action"] is None else ""]
            for r in picks[k]
        ]
    keys = ("seed", "steps", "eval", "batch", "steps_done", "gpu", "max_memory_gb", "fla")
    keys += ("trainable_parameters", "boards_seen", "versions", "stop_flat", "min_steps")
    out["config"] = {k: cfg.get(k) for k in (*keys, "stop_after")}
    out["since_container_start"] = tim.get("since_container_start")
    if (run / "manifest.json").exists():
        man, _ = read("manifest.json")
        out["manifest"] = {k: man.get(k) for k in ("commit", "job_id", "seconds", "returncode")}
    out["curve"] = [
        [
            c["step"],
            round(c["mean_reward"], 4),
            round(c["invalid"], 4),
            round(c.get("truncated", 0), 4),
            round(c.get("trl", {}).get("entropy", -1), 6),
        ]
        for c in cur
    ]
    return out


def read_runs(runs: Path, split: str) -> tuple[dict[int, dict], dict[int, dict]]:
    """Language-model runs (summaries written by compact) scored on split, and
    feature-policy runs, by seed, from the JSON files in runs."""
    lm, fp = {}, {}
    for path in sorted(runs.glob("*.json")):
        run = json.loads(path.read_text())
        if "picks" in run:
            fp[run["seed"]] = run
        elif run["config"]["eval"] == split:
            seed = run["config"]["seed"]
            assert seed not in lm, f"two runs of seed {seed}"
            for k in ("untrained", "trained"):
                run[k] = [{"action": a, "why": w, "tokens": t, "reply": r} for a, w, t, r in run[k]]
            lm[seed] = run
    return lm, fp


def score(data: dict, runs: Path, split: str = "test", c6_picks: Path | None = C6_PICKS) -> dict:
    """Arm 1 minus arm 2 under the held-out guessers, the potion maps and EAT, paired by
    board and by seed pair; with the secondary metrics of the plan."""
    names = data["guessers"]
    col = {g: i for i, g in enumerate(names)}
    values = data[f"{split}_values"]  # (B, 120, G)
    n_boards = values.shape[0]
    first = values[:, :, col[GIVER]].argmax(axis=1)
    groups = {
        "held_out": [col[g] for g in HELD_OUT],
        "training": [col[g] for g in TRAINING],
        "potion_map": [col[f"potion-map/{s}"] for s in MAP_SEEDS],
        "eat": [col["eat"]],
    }

    def summary(v: np.ndarray) -> dict[str, np.ndarray]:  # (..., B, G) -> (..., B)
        return {k: v[..., idx].mean(axis=-1) for k, idx in groups.items()}

    lm, fp = read_runs(runs, split)
    seeds = sorted(set(lm) & set(fp))
    out: dict = {"split": split, "boards": int(n_boards), "seeds": seeds, "guessers": names}
    if not seeds:
        return out
    acts1 = {s: [r["action"] for r in lm[s]["trained"]] for s in seeds}
    arm1 = np.stack([values_of(acts1[s], values, first) for s in seeds])
    arm2 = np.stack([values[np.arange(n_boards), fp[s]["picks"][split]] for s in seeds])
    s1, s2 = summary(arm1), summary(arm2)
    out["difference"] = {k: interval(s1[k] - s2[k]) for k in groups}
    out["arm1"] = {k: interval(s1[k]) for k in groups}
    out["arm2"] = {k: interval(s2[k]) for k in groups}
    out["gap_training_minus_held_out"] = {
        "arm1": interval(s1["training"] - s1["held_out"]),
        "arm2": interval(s2["training"] - s2["held_out"]),
    }
    # A reply with no listed clue valued at -1 under every guesser, instead of the fallback.
    minus = arm1.copy()
    for i, s in enumerate(seeds):
        minus[i, [b for b, a in enumerate(acts1[s]) if a is None]] = INVALID
    out["difference_invalid_as_minus_1"] = interval(summary(minus)["held_out"] - s2["held_out"])

    untrained = values_of([r["action"] for r in lm[seeds[0]]["untrained"]], values, first)
    base = {
        "first_choice": values[np.arange(n_boards), first],
        "uniform": values.mean(axis=1),
        "oracle_held_out": values[
            np.arange(n_boards), values[:, :, groups["held_out"]].mean(axis=2).argmax(axis=1)
        ],
        "untrained": untrained,
    }
    if split == "test" and c6_picks is not None:  # C6's chosen model, fallback included
        clues = data["test_clues"]
        for line in c6_picks.read_text().splitlines():
            name, seed, *picks = line.split()
            idx = []
            for b, p in enumerate(picks):
                c, n = p.split("/")
                idx.append([str(x) for x in clues[b]].index(c) * NMAX + int(n) - 1)
            base[f"c6/{name}"] = values[np.arange(n_boards), idx]
    out["baselines"] = {}
    for name, x in base.items():
        bs = summary(x)
        out["baselines"][name] = {k: float(v.mean()) for k, v in bs.items()}
        if name in ("first_choice", "untrained") or name.startswith("c6/"):
            out["baselines"][name]["arm1_minus_this"] = {
                k: interval(s1[k] - bs[k][None, :]) for k in groups
            }
    out["arm2_all_seeds"] = {
        str(s): {
            k: float(v.mean())
            for k, v in summary(values[np.arange(n_boards), r["picks"][split]]).items()
        }
        for s, r in sorted(fp.items())
    }

    def why(rows: list[dict]) -> dict:
        return dict(Counter(r.get("why") or "read" for r in rows))

    out["replies"] = {str(s): why(lm[s]["trained"]) for s in seeds}
    out["replies"]["untrained"] = why(lm[seeds[0]]["untrained"])
    clues = data[f"{split}_clues"]
    filled = {s: [first[b] if a is None else a for b, a in enumerate(acts1[s])] for s in seeds}
    out["behavior"] = {
        "arm1": {str(s): behavior(filled[s], clues, first) for s in seeds},
        "arm2": {str(s): behavior(r["picks"][split], clues, first) for s, r in sorted(fp.items())},
        "untrained": behavior(
            [
                first[b] if r["action"] is None else r["action"]
                for b, r in enumerate(lm[seeds[0]]["untrained"])
            ],
            clues,
            first,
        ),
    }
    out["per_board"] = {
        "arm1": {
            str(s): {"picks": acts1[s], "values": arm1[i].round(4).tolist()}
            for i, s in enumerate(seeds)
        },
        "arm2": {
            str(s): {
                "picks": r["picks"][split],
                "values": values[np.arange(n_boards), r["picks"][split]].round(4).tolist(),
            }
            for s, r in sorted(fp.items())
        },
    }
    out["runs"] = {
        str(seed): {
            "curves_every_20_steps": run.get("curves_every_20_steps") or windows(run["curve"]),
            "since_container_start": run["since_container_start"],
            "config": run["config"],
            "picks_sha256": run["picks_sha256"],
        }
        for seed, run in sorted(lm.items())
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--eval-boards", type=int, default=None, help="first n of val and test")
    t.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("feature")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--runs", type=Path, required=True)
    s.add_argument("--split", default="test", choices=list(EVAL_BOARDS))
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "table":
        n = args.eval_boards
        sets = {k: (seed, n or m) for k, (seed, m) in EVAL_BOARDS.items()}
        print(json.dumps(table(args.steps, args.out, sets), indent=1))
    elif args.cmd == "feature":
        args.out.mkdir(parents=True, exist_ok=True)
        result = feature_policy(read_data(args.data), args.seed)
        (args.out / f"feature-{args.seed}.json").write_text(json.dumps(result) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "picks"}))
    else:
        result = score(read_data(args.data), args.runs, args.split)
        boards = result.pop("per_board", None)
        text = json.dumps(result, indent=1)  # the per-board lists last, one line each
        if boards is not None:
            rows = ",\n".join(f' "{k}": {json.dumps(v)}' for k, v in boards.items())
            text = text[:-2] + ',\n "per_board": {\n' + rows + "\n }\n}"
        json.loads(text)
        args.out.write_text(text + "\n")
        print(json.dumps(result["difference"], indent=1))


if __name__ == "__main__":
    main()
