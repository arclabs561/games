"""Exploratory: this package's givers and guessers against the framework's word-vector bots.

No plan was written before these runs (experiments/METHOD.md); they are exploratory and
meant to motivate a planned comparison. Three parts:

1. **Reproduce.** The 900 published single-team games of the framework's word-vector
   bots (Stephenson et al. 2025, Table I; 9 pairs, seeds 0 to 99, the framework's word
   pool), replayed with the port in codenames.bench and compared move by move with the
   published logs. The workflow on the compute branch also runs the framework's own
   code, at the pinned commit, on a subset of the same games; `original` compares its
   logs with the published ones and with the port.
2. **Cross-play.** Every giver against every guesser on 200 boards of codenames/words.txt
   (the framework's word pool is not used here; see codenames.bench_data), dealt as the
   framework deals them, seeds 0 to 199. Scored by the framework's single-team rule.
3. **Analyze.** Mean scores with 95% bootstrap intervals over boards (10,000 draws, seed
   0), and paired differences.

Givers: the framework's three codemasters (w2v, glove, w2vglove); the expected-value
giver of codenames.players on EmbeddingGemma 2 and on potion, which gives its own first
choice; and the potion givers of plan C3 trained by self-play and against a population
(the task with EmbeddingGemma 2 held out, training seed 1), rebuilt by refitting.
Guessers: the framework's three guessers; greedy EmbeddingGemma 2 and potion guessers;
and the LoRA encoder of plan C8 (arm B, seed 1), when its weights can be read.

Our givers choose clues only from words that both GloVe and word2vec have, so that the
framework's guessers can read every clue.

    uv run python -m codenames.bench_play reproduce --out OUT          # needs the vectors
    uv run python -m codenames.bench_play crossplay --giver glove --out OUT
    uv run python -m codenames.bench_play collect --runs DIR --lora DIR  # results/bench-runs.json
    uv run python -m codenames.bench_play analyze                      # results/bench.{json,md}
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

from codenames import bench, bench_data, rl
from codenames.bench import Vectors, framework_deal, score
from codenames.game import Board, Clue, deal, play
from codenames.luce import ASSASSIN, OPPONENT, TEAM, luce_weights, turn_odds
from codenames.norms import eat_counts
from codenames.players import (
    ClueTable,
    Space,
    embedding_guesser,
    embedding_weigher,
    expected_giver,
    shortlist_rows,
    space,
)
from codenames.stats import cluster_draws, replicates
from codenames.study import chosen_spaces, load
from codenames.words import board_words, clue_words, vocabulary, zipf

RESULTS = Path(__file__).parent / "results"
FRAMEWORK = ("w2v", "glove", "w2vglove")
OURS_GIVERS = ("gemma2-first", "potion-first", "potion-c3-self", "potion-c3-population")
GIVERS = (*FRAMEWORK, *OURS_GIVERS)
GUESSERS = (*FRAMEWORK, "gemma2", "potion", "lora-c8")
N_BOARDS = 200
GEMMA = "google/embeddinggemma-2"
C3_TASK = ("potion", GEMMA, 1)  # giver, held-out model, training seed
LORA_RUN = "runs/c8/b-s1"
DRAWS, BOOT_SEED = 10_000, 0

# Table I of Stephenson et al. (2025), arXiv 2412.11373v2, the eight word-vector rows:
# (codemaster, guesser): mean score, standard deviation, loss percentage.
TABLE_I = {
    ("w2v", "w2v"): (6.81, 0.97, 0),
    ("w2v", "glove"): (12.35, 7.14, 22),
    ("w2v", "w2vglove"): (10.64, 6.72, 16),
    ("glove", "w2v"): (12.21, 7.52, 24),
    ("glove", "glove"): (5.24, 0.79, 0),
    ("glove", "w2vglove"): (5.42, 0.89, 0),
    ("w2vglove", "w2v"): (11.18, 7.37, 21),
    ("w2vglove", "glove"): (5.86, 2.18, 1),
}


# Part 1: the published games.


def framework_players(strict: bool = False):
    tabs = bench_data.tables()
    return bench.framework_bots(tabs, list(bench_data.cm_words()), bench_data.forms(), strict)


def game_record(board: Board, result) -> dict:
    counts = [c.count for c, _ in result.clues]
    return {
        "score": score(result, board),
        "score_table": score(result, board, "table"),
        "won": result.won,
        "turns": result.turns,
        "loss": bench.loss_kind(result, board),
        "mean_count": float(np.mean(counts)) if counts else 0.0,
        "empty_turns": sum(1 for _, made in result.clues if not made),
        **dict(
            zip(("flagged_turns", "illegal_turns"), bench.flagged_turns(result, board), strict=True)
        ),
    }


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


def published_flags(board: Board, published) -> int:
    """Turns of a published game whose clue codenames.legality flags against an
    unrevealed word."""
    left, n = set(board.words), 0
    for clue, _, picks in published:
        n += bench.flagged(clue, sorted(left))
        left -= {p.lower() for p in picks}
    return n


def reproduce() -> dict:
    """Replay every published game with the port; compare transcripts and scores.
    Only match flags, scores and digests are kept, never the board words."""
    givers, guessers = framework_players()
    pool = bench_data.framework_pool()
    games = []
    for g in bench_data.published_games():
        cm, gu, seed = g["codemaster_red"], g["guesser_red"], int(g["seed"])
        board = framework_deal(pool, seed)
        published = bench_data.published_transcript(g["moves"])
        try:
            result = play(board, givers[cm], guessers[gu])
            ours = [list(t) for t in bench.transcript(result)]
            rec = game_record(board, result)
        except (RuntimeError, KeyError) as e:
            ours, rec = [], {"error": str(e)}
        pub = [list(t) for t in published]
        same = ours == pub
        turn = next(
            (i for i in range(max(len(ours), len(pub))) if ours[i : i + 1] != pub[i : i + 1]), None
        )
        games.append(
            {
                "codemaster": cm,
                "guesser": gu,
                "seed": seed,
                "published": {
                    "turns": g["total_turns"],
                    "winner": g["winner"],
                    "A": g["A"],
                    "B": g["B"],
                    "C": g["C"],
                },
                "port": rec,
                "same_transcript": same,
                "first_different_turn": turn if not same else None,
                "published_flagged_turns": published_flags(board, published),
                "digest_published": digest([list(t) for t in published]),
                "digest_port": digest(ours),
            }
        )
    return {"games": games}


def compare_original(logs: Path) -> dict:
    """Compare the move logs that the framework's own code wrote (stepmat/Codenames_GPT at
    bench_data.COMMIT, run by the workflow on the compute branch) with the published logs
    and with the port's games."""
    published = {
        (g["codemaster_red"], g["guesser_red"], int(g["seed"])): g
        for g in bench_data.published_games()
    }
    givers, guessers = framework_players()
    pool = bench_data.framework_pool()
    out = []
    for results in sorted(logs.glob("*/bot_results_new_style.txt")):
        rows = [json.loads(x) for x in results.read_text().splitlines()]
        moves = [json.loads(x) for x in (results.parent / "move_log.txt").read_text().splitlines()]
        for row, mv in zip(rows, moves, strict=True):
            cm, gu, seed = row["codemaster_red"], row["guesser_red"], int(row["seed"])
            theirs = [list(x) for x in bench_data.published_transcript(mv)]
            pub = [
                list(x) for x in bench_data.published_transcript(published[(cm, gu, seed)]["moves"])
            ]
            board = framework_deal(pool, seed)
            ours = [list(x) for x in bench.transcript(play(board, givers[cm], guessers[gu]))]
            out.append(
                {
                    "codemaster": cm,
                    "guesser": gu,
                    "seed": seed,
                    "turns": row["total_turns"],
                    "winner": row["winner"],
                    "same_as_published": theirs == pub,
                    "same_as_port": theirs == ours,
                    "seconds": round(row["time_s"], 1),
                }
            )
    return {"games": out}


# Part 2: cross-play on this package's words.


def readable(word: str, tabs: dict[str, Vectors]) -> bool:
    return word in tabs["glove"] and word in tabs["w2v"]


def our_boards(tabs: dict[str, Vectors], n: int = N_BOARDS) -> tuple[list[Board], list[str]]:
    """Boards of words.txt (the words both vector tables have), framework-dealt."""
    pool = [w for w in board_words() if readable(w, tabs)]
    return [framework_deal(pool, s) for s in range(n)], pool


_RELATED: dict = {}


def strict_pool(pool: tuple[str, ...], board: Board) -> tuple[str, ...]:
    """The clues that codenames.legality does not flag against any unrevealed word. The
    relation of every clue to every board word is computed once per board."""
    key = (id(pool), board.words)
    if key not in _RELATED:
        if len(_RELATED) > 400:
            _RELATED.clear()
        _RELATED[key] = np.array([[bench.related(c, w) for w in board.words] for c in pool])
    live = [i for i, w in enumerate(board.words) if w not in board.revealed]
    keep = ~_RELATED[key][:, live].any(axis=1)
    return tuple(c for c, k in zip(pool, keep, strict=True) if k)


class StrictGiver:
    """An expected-value giver whose candidates exclude flagged clues on each board, so
    that it gives its best remaining choice."""

    def __init__(self, giver):
        self.giver, self.pool = giver, giver.pool

    def __call__(self, board: Board) -> Clue:
        self.giver.pool = strict_pool(self.pool, board)
        return self.giver(board)


class PolicyGiver:
    """A plan-C3 policy as a giver: the action with the largest theta . phi."""

    def __init__(
        self, params: np.ndarray, sp: Space, tau: float, pool: tuple[str, ...], strict=False
    ):
        f = len(rl.FEATURES)
        self.theta, self.mu, self.sd = params[:f], params[f : 2 * f], params[2 * f :]
        self.sp, self.tau, self.pool, self.strict = sp, tau, pool, strict

    def __call__(self, board: Board) -> Clue:
        pool = strict_pool(self.pool, board) if self.strict else self.pool
        clues, phi = policy_actions(self.sp, self.tau, board, pool, 30)
        i = int(np.argmax(((phi - self.mu) / self.sd) @ self.theta))
        return Clue(*clues[i])


def policy_actions(sp: Space, tau: float, board: Board, pool, k: int):
    """codenames.rl.actions, with counts up to the number of team words left (rl.actions
    was written for fresh boards, which always have 9); the same on a fresh board."""
    words = board.unrevealed()
    kinds = tuple(board.kind(w) for w in words)
    nmax = min(rl.NMAX, kinds.count(TEAM))
    tab = ClueTable.build(sp, words, pool)
    rows = shortlist_rows(tab.sims, kinds, k)
    rows = sorted(rows, key=lambda r: -tab.sims[r][[x == TEAM for x in kinds]].max())[:k]
    sims = tab.sims[rows]
    odds = turn_odds(luce_weights(sims, tau), kinds, nmax)
    team = np.array([x == TEAM for x in kinds])
    team_sorted = -np.sort(-sims[:, team], axis=1)
    best_bad = sims[:, ~team].max(axis=1)
    feats, clues = [], []
    for i, r in enumerate(rows):
        for n in range(1, nmax + 1):
            feats.append(
                [
                    odds.expected_team(n)[i],
                    odds.stopped_by(ASSASSIN, n)[i],
                    odds.stopped_by(OPPONENT, n)[i],
                    n,
                    max(0.0, 3.5 - zipf(tab.clues[r])),
                    team_sorted[i, n - 1] - best_bad[i],
                    1.0,
                ]
            )
            clues.append((tab.clues[r], n))
    return clues, np.array(feats)


@cache
def c3_policies() -> dict:
    """Refit the self-play and population policies of one C3 task, as codenames.rl.run
    does (same boards, data and seed), and check theta against results/c3.json."""
    giver, held, seed = C3_TASK
    spaces = chosen_spaces(load("calibrate"))
    weighers = {m: embedding_weigher(sp, tau) for m, (sp, tau) in spaces.items()}
    rng = random.Random(seed)
    train = [deal(board_words(), rng) for _ in range(300)]
    data, _ = rl.build(giver, weighers, train, 30)
    population = [g for g in weighers if g not in (giver, held)]
    params = {
        "self": rl.fit(data, [giver], 400, 0.05, seed),
        "population": rl.fit(data, population, 400, 0.05, seed),
    }
    run = next(
        r for r in load("c3")["theta"]["runs"] if (r["giver"], r["held_out"], r["seed"]) == C3_TASK
    )
    f = len(rl.FEATURES)
    check = {k: float(np.abs(params[k][:f].round(4) - np.array(run[k])).max()) for k in params}
    return {"params": {k: v.tolist() for k, v in params.items()}, "theta_max_difference": check}


def cpu_quota() -> int:
    """CPUs this container may use (cgroup v2 cpu.max), not the host's count, so that
    torch does not start more threads than it has CPUs."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


@cache
def lora_space(folder: Path, extra: tuple[str, ...]) -> tuple[Space, dict]:
    """The C8 encoder's vectors (clue side in its chosen template, word side bare), with
    the extra words encoded on this machine by its adapter. Reports how closely this
    machine's encoding matches the stored vectors on 200 stored words."""
    summary = json.loads((folder / "summary.json").read_text())
    template = summary["after"]["chosen"]["clue"]
    name = "Qwen--Qwen3-Embedding-0.6B-lora"
    stored = {t: np.load(folder / f"{name}__{t}.npz") for t in sorted({template, "bare"})}
    words = [str(w) for w in stored["bare"]["words"]]
    have = set(words)
    missing = sorted(w for w in extra if w not in have)
    st = importlib.import_module("sentence_transformers")
    torch = importlib.import_module("torch")
    torch.set_num_threads(cpu_quota())
    t0 = time.perf_counter()
    model = st.SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B", device="cpu", model_kwargs={"torch_dtype": torch.float32}
    )
    model.load_adapter(str(folder / "adapter"))
    from codenames.embed import TEMPLATES, normalize

    def enc(t: str, ws: list[str]) -> np.ndarray:
        texts = [TEMPLATES[t].format(w) for w in ws]
        return normalize(np.asarray(model.encode(texts, batch_size=128, prompt=""))).astype(
            np.float16
        )

    probe = words[:: max(1, len(words) // 200)][:200]
    pos = {w: i for i, w in enumerate(words)}
    info = {"template": template, "extra_words": len(missing), "threads": torch.get_num_threads()}
    print(
        "C8 encoder: encoding",
        len(missing),
        "words with",
        torch.get_num_threads(),
        "threads",
        flush=True,
    )
    mats = {}
    for t, z in stored.items():
        v = z["vectors"]
        info[f"max_abs_difference_{t}"] = float(
            np.abs(
                enc(t, probe).astype(np.float32) - v[[pos[w] for w in probe]].astype(np.float32)
            ).max()
        )
        mats[t] = (
            np.concatenate([v, enc(t, missing)]).astype(np.float32)
            if missing
            else v.astype(np.float32)
        )
    sp = Space(f"lora-c8:{template}/bare", mats[template], mats["bare"], words + missing)
    info["seconds"] = round(time.perf_counter() - t0)
    print("C8 encoder:", json.dumps(info), flush=True)
    return sp, info


def our_guesser_spaces(names, extra: list[str], lora_dir: Path | None) -> tuple[dict, dict]:
    words = list(dict.fromkeys([*vocabulary(), *extra]))
    out, info = {}, {}
    cal = load("calibrate")
    for name in names:
        if name == "gemma2":
            b = cal[GEMMA]["best"]
            out[name] = space(GEMMA, b["clue"], b["word"], words)
        elif name == "potion":
            b = cal["potion"]["best"]
            out[name] = space("potion", b["clue"], b["word"], words)
        elif name == "lora-c8":
            if lora_dir is None:
                continue
            out[name], info[name] = lora_space(lora_dir, tuple(words))
    return out, info


def make_giver(name: str, tabs, pool_ours: tuple[str, ...], fw_givers, info: dict, strict=False):
    if name in FRAMEWORK:
        return fw_givers[name]
    cal = load("calibrate")
    wrap = StrictGiver if strict else (lambda g: g)
    if name == "gemma2-first":
        b = cal[GEMMA]["best"]
        return wrap(
            expected_giver(space(GEMMA, b["clue"], b["word"]), b["tau"], candidates=pool_ours)
        )
    if name == "potion-first":
        b = cal["potion"]["best"]
        return wrap(
            expected_giver(space("potion", b["clue"], b["word"]), b["tau"], candidates=pool_ours)
        )
    c3 = c3_policies()
    info["c3"] = {"task": C3_TASK, "theta_max_difference": c3["theta_max_difference"]}
    sp, tau = chosen_spaces(cal)["potion"]
    counts = eat_counts()
    pool = tuple(c for c in pool_ours if c in counts)
    kind = name.removeprefix("potion-c3-")
    return PolicyGiver(np.array(c3["params"][kind]), sp, tau, pool, strict)


def crossplay_many(
    givers, guessers, n_boards: int = N_BOARDS, lora_dir: Path | None = None, strict: bool = False
) -> list[dict]:
    """Every giver with every guesser on the same boards; one result per giver. With
    strict, every giver skips clues that share a form with an unrevealed board word."""
    t0 = time.perf_counter()
    tabs = bench_data.tables()
    fw_givers, fw_guessers = framework_players(strict)
    boards, pool = our_boards(tabs, n_boards)
    pool_ours = tuple(c for c in clue_words() if readable(c, tabs))
    common: dict = {
        "board_words": len(pool),
        "board_words_dropped": sorted(set(board_words()) - set(pool)),
        "our_clue_pool": len(pool_ours),
        "clue_words_dropped": len(clue_words()) - len(pool_ours),
    }
    ours = [g for g in guessers if g not in FRAMEWORK]
    spaces, sinfo = our_guesser_spaces(ours, list(bench_data.cm_words()), lora_dir)
    common |= sinfo
    players = {g: fw_guessers[g] for g in guessers if g in FRAMEWORK}
    players |= {g: embedding_guesser(sp) for g, sp in spaces.items()}
    out = []
    for giver in givers:
        t1 = time.perf_counter()
        info = dict(common)
        give = make_giver(giver, tabs, pool_ours, fw_givers, info, strict)
        games = []
        for gname, guess in players.items():
            for seed, b in enumerate(boards):
                board = Board(b.words, b.kinds)
                try:
                    rec = game_record(board, play(board, give, guess))
                except RuntimeError as e:  # a framework codemaster with no clue
                    rec = {"error": str(e), "score": bench.LOSS_SCORE, "won": False}
                games.append({"giver": giver, "guesser": gname, "seed": seed} | rec)
            print(giver, gname, f"{time.perf_counter() - t0:.0f} s", flush=True)
        out.append(
            {
                "giver": giver,
                "strict": strict,
                "info": info,
                "games": games,
                "seconds": round(time.perf_counter() - t1),
            }
        )
    return out


def crossplay(
    giver: str,
    guessers,
    n_boards: int = N_BOARDS,
    lora_dir: Path | None = None,
    strict: bool = False,
) -> dict:
    return crossplay_many([giver], guessers, n_boards, lora_dir, strict)[0]


def lora_run(out: Path, n_boards: int = N_BOARDS, variants=("all", "strict")) -> None:
    """The column of the C8 encoder, inside a Hugging Face Job (its weights are in the
    private dataset). Builds the same inputs as the workflow's data job, from the same
    sources, and records their digests so that the two can be compared."""
    import shutil
    import subprocess
    import urllib.request

    from codenames import embed

    hub = importlib.import_module("huggingface_hub")
    tmp = Path("/tmp/bench")
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch", "compute/codenames-embed",
         "https://github.com/arclabs561/games.git", str(tmp / "embed")],
        check=True,
    )  # fmt: skip
    embed.COMPUTED.mkdir(parents=True, exist_ok=True)
    for f in (tmp / "embed" / "codenames" / "vectors").glob("*.npz"):
        shutil.copy(f, embed.COMPUTED / f.name)
    for name in bench_data.FILES:
        bench_data.fetch_file(name)
    glove = Path(hub.hf_hub_download("stanfordnlp/glove", "glove.6B.zip", cache_dir=str(tmp)))
    w2v = tmp / "w2v.gz"
    with urllib.request.urlopen(bench_data.W2V_URL) as r, open(w2v, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 22)
    info = bench_data.extract(glove, w2v)
    print("vectors:", json.dumps(info), flush=True)
    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", sys.executable, "nltk==3.10.4"], check=True
    )
    forms = bench_data.compute_forms(list(bench_data.cm_words()))
    text = json.dumps(forms, sort_keys=True) + "\n"
    (bench_data.BENCH / "forms.json").write_text(text)
    info["forms_sha256"] = bench_data.sha256(text.encode())
    info["nltk"] = forms["nltk"]
    c8 = hub.snapshot_download(
        "arclabs561/games-runs", repo_type="dataset", allow_patterns=[f"{LORA_RUN}/*"],
        local_dir=str(tmp / "c8"),
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(info, indent=1) + "\n")
    for strict in [v == "strict" for v in variants]:
        for giver in GIVERS:
            res = crossplay_many([giver], ["lora-c8"], n_boards, Path(c8) / LORA_RUN, strict)[0]
            tag = "strict-" if strict else ""
            (out / f"crossplay-lora-{tag}{giver}.json").write_text(json.dumps(res) + "\n")


# Part 3: analysis.


def boot(values: np.ndarray, draws: int = DRAWS, seed: int = BOOT_SEED) -> list[float]:
    """95% percentile interval of the mean over boards (one board per cluster)."""
    index, counts = cluster_draws([f"{i:04d}" for i in range(len(values))], draws, seed)
    lo, hi = np.percentile(replicates(values, index, counts), [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def summarize(xs: list[float]) -> dict:
    v = np.asarray(xs, dtype=float)
    return {"mean": round(float(v.mean()), 3), "ci": boot(v), "n": len(v)}


def published_table() -> dict:
    """Table I's word-vector rows recomputed from the published logs, by both rules."""
    out = {}
    by = {}
    for g in bench_data.published_games():
        by.setdefault((g["codemaster_red"], g["guesser_red"]), []).append(g)
    for (cm, gu), gs in by.items():
        rule = [25 if g["winner"] == "B" else g["total_turns"] for g in gs]
        table = [25 if g["A"] == 1 else g["total_turns"] for g in gs]
        out[f"{cm}/{gu}"] = {
            "games": len(gs),
            "seeds": [min(int(g["seed"]) for g in gs), max(int(g["seed"]) for g in gs)],
            "mean_framework_rule": round(float(np.mean(rule)), 2),
            "sd_framework_rule": round(float(np.std(rule, ddof=1)), 2),
            "mean_table_rule": round(float(np.mean(table)), 2),
            "sd_table_rule": round(float(np.std(table, ddof=1)), 2),
            "losses": sum(g["winner"] == "B" for g in gs),
            "assassin_losses": sum(g["A"] for g in gs),
            "table_i": dict(
                zip(("mean", "sd", "loss_percent"), TABLE_I.get((cm, gu), (None,) * 3), strict=True)
            ),
        }
    return out


# The committed results: bench-runs.json holds every game's outcome (no board words),
# collected from the runs; bench.json and bench.md are computed from it alone.

RUNS_FILE = RESULTS / "bench-runs.json"
SUMMARY_FILE = RESULTS / "bench.json"
REPORT_FILE = RESULTS / "bench.md"
LOSS_CODE = {None: "-", "A": "A", "O": "O", "turns": "T"}


def latest(runs: list[Path], name: str) -> Path | None:
    found = [r / name for r in runs if (r / name).exists()]
    return found[-1] if found else None


def read_latest(runs: list[Path], name: str):
    path = latest(runs, name)
    if path is None:
        raise FileNotFoundError(f"{name} is in none of {runs}")
    return json.loads(path.read_text())


def collect(runs: list[Path], lora_compact: Path | None = None) -> dict:
    """Compact the raw outputs of the workflow runs (later runs replace earlier ones) and
    the C8 column (compact_lora's file) into one file."""
    rep = read_latest(runs, "reproduce.json")["games"]
    out: dict = {
        "status": "exploratory: no plan was written before these runs",
        "source": {"runs": [json.loads((r / "source.json").read_text()) for r in runs]},
        "data": {
            "vectors": read_latest(runs, "vectors.json"),
            "forms": read_latest(runs, "forms.json"),
        },
        "published": published_table(),
        "reproduce": [
            [
                g["codemaster"],
                g["guesser"],
                g["seed"],
                g["same_transcript"],
                g["first_different_turn"],
                g["port"].get("score"),
                g["port"].get("score_table"),
                g["published"]["turns"],
                g["published"]["winner"],
                g["published"]["A"],
                g.get("published_flagged_turns"),
            ]
            for g in rep
        ],
    }
    orig = latest(runs, "original.json")
    if orig:
        og = json.loads(orig.read_text())["games"]
        out["original"] = [
            [
                g["codemaster"],
                g["guesser"],
                g["seed"],
                g["same_as_published"],
                g["same_as_port"],
                g["seconds"],
            ]
            for g in og
        ]
        versions = sorted(orig.parent.glob("original/*/versions.txt"))
        if versions:
            out["data"]["original_versions"] = versions[0].read_text().split()
    slices: dict[tuple[bool, str], dict] = {}
    for r in runs:
        for p in sorted(r.glob("crossplay-*.json")):
            sl = json.loads(p.read_text())
            strict = sl.get("strict", p.name.startswith("crossplay-strict-"))
            earlier = slices.get((strict, sl["giver"]))
            if earlier is not None and not strict:  # a later run replaces it: do they agree?
                same = [g["score"] for g in earlier["games"]] == [g["score"] for g in sl["games"]]
                rerun = out["data"].setdefault(
                    "reruns", {"slices": 0, "same_scores": 0, "games": 0}
                )
                rerun["slices"] += 1
                rerun["same_scores"] += same
                rerun["games"] += len(sl["games"])
            slices[(strict, sl["giver"])] = sl  # later runs replace
    cells, info = slice_cells(slices.values(), [p for (p, _) in slices])
    if lora_compact:
        extra = json.loads(lora_compact.read_text())
        for variant, more in extra["cells"].items():
            cells[variant] |= decode_cells(more)
        for giver, i in extra["info"].items():
            info.setdefault(giver, {}).update(i)
        out["data"]["lora_jobs"] = extra["data"]
        out["source"]["lora_jobs"] = extra["source"]
        rerun = out["data"].setdefault("reruns", {"slices": 0, "same_scores": 0, "games": 0})
        for k in rerun:
            rerun[k] += extra["reruns"][k]
    out["crossplay"] = {"info": info}
    for variant in ("all", "strict"):
        out["crossplay"][variant] = encode_cells(dict(sorted(cells[variant].items())))
    return out


def slice_cells(slices, stricts) -> tuple[dict[str, dict], dict[str, dict]]:
    """Per variant, per "giver/guesser": every game's score, loss, mean count, turns and
    flagged turns, in board order."""
    variants: dict[str, dict] = {"all": {}, "strict": {}}
    info: dict[str, dict] = {}
    for sl, strict in zip(slices, stricts, strict=True):
        info.setdefault(sl["giver"], {}).update(sl["info"])
        cells = variants["strict" if strict else "all"]
        for g in sl["games"]:
            c = cells.setdefault(
                f"{g['giver']}/{g['guesser']}",
                {
                    "seeds": [],
                    "score": [],
                    "loss": "",
                    "count": [],
                    "turns": [],
                    "flagged": [],
                    "illegal": [],
                },
            )
            c["seeds"].append(g["seed"])
            c["score"].append(g["score"])
            c["loss"] += LOSS_CODE[g.get("loss")] if "error" not in g else "E"
            c["count"].append(round(g.get("mean_count", 0.0), 2))
            c["turns"].append(g.get("turns", 0))
            c["flagged"].append(g.get("flagged_turns") or 0)
            c["illegal"].append(g.get("illegal_turns") or 0)
    for cells in variants.values():
        for c in cells.values():
            assert c["seeds"] == list(range(len(c["seeds"])))
            del c["seeds"]
    return variants, info


# Small whole numbers (0 to 51) as one letter each, so a cell of 200 games is short.
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode_cells(cells: dict) -> dict:
    return {
        k: {
            "score": "".join(LETTERS[x] for x in c["score"]),
            "turns": "".join(LETTERS[x] for x in c["turns"]),
            "flagged": "".join(LETTERS[x] for x in c["flagged"]),
            "illegal": "".join(LETTERS[x] for x in c["illegal"]),
            "loss": c["loss"],
            "count": c["count"],
        }
        for k, c in cells.items()
    }


def decode_cells(cells: dict) -> dict:
    return {
        k: {
            "score": [LETTERS.index(x) for x in c["score"]],
            "turns": [LETTERS.index(x) for x in c["turns"]],
            "flagged": [LETTERS.index(x) for x in c["flagged"]],
            "illegal": [LETTERS.index(x) for x in c["illegal"]],
            "loss": c["loss"],
            "count": c["count"],
        }
        for k, c in cells.items()
    }


def compact_lora(folders: list[Path]) -> dict:
    """The C8 jobs' outputs in the encoding of bench-runs.json, to copy out of the
    private dataset (the games' outcomes only). Later folders replace earlier ones; a
    repeated giver's scores are compared."""
    slices: dict[tuple[bool, str], dict] = {}
    data, source = [], []
    reruns = {"slices": 0, "same_scores": 0, "games": 0}
    for folder in folders:
        for p in sorted(folder.glob("crossplay-lora-*.json")):
            sl = json.loads(p.read_text())
            key = (sl.get("strict", p.name.startswith("crossplay-lora-strict-")), sl["giver"])
            if key in slices and not key[0]:
                same = [g["score"] for g in slices[key]["games"]] == [
                    g["score"] for g in sl["games"]
                ]
                reruns["slices"] += 1
                reruns["same_scores"] += same
                reruns["games"] += len(sl["games"])
            slices[key] = sl
        data.append(json.loads((folder / "data.json").read_text()))
        if (folder / "manifest.json").exists():
            source.append(json.loads((folder / "manifest.json").read_text()))
    cells, info = slice_cells(slices.values(), [strict for strict, _ in slices])
    return {
        "cells": {v: encode_cells(c) for v, c in cells.items()},
        "info": info,
        "data": data,
        "source": source,
        "reruns": reruns,
    }


def vectors_of(player: str) -> set[str]:
    """The vector tables a giver or guesser uses; w2vglove uses both of the framework's."""
    if player == "w2vglove":
        return {"w2v", "glove"}
    return {{"gemma2-first": "gemma2"}.get(player, player.split("-")[0])}


def stranger(giver: str, guesser: str) -> bool:
    """True when the two share no vectors."""
    return not vectors_of(giver) & vectors_of(guesser)


def analyze(runs: dict) -> dict:
    out: dict = {"status": runs["status"], "source": runs["source"], "data": runs["data"]}
    # Reproduction.
    by: dict[str, list] = {}
    for cm, gu, _seed, same, first, sc, sct, pturns, pwin, pa, _flags in runs["reproduce"]:
        by.setdefault(f"{cm}/{gu}", []).append((same, first, sc, sct, pturns, pwin, pa))
    rep = {}
    for k, gs in by.items():
        pub_rule = [25 if w == "B" else t for *_, t, w, a in gs]
        rep[k] = {
            "games": len(gs),
            "same_transcript": sum(g[0] for g in gs),
            "same_score": sum(g[2] == p for g, p in zip(gs, pub_rule, strict=True)),
            "port_mean": round(float(np.mean([g[2] for g in gs if g[2] is not None])), 2),
            "port_mean_table_rule": round(
                float(np.mean([g[3] for g in gs if g[3] is not None])), 2
            ),
            "first_different_turns": sorted(Counter(g[1] for g in gs if not g[0]).items()),
        } | runs["published"][k]
    out["reproduction"] = rep
    if "original" in runs:
        og = runs["original"]
        out["original_code"] = {
            "games": len(og),
            "same_as_published": sum(g[3] for g in og),
            "same_as_port": sum(g[4] for g in og),
            "pairs": len({(g[0], g[1]) for g in og}),
            "seeds": sorted({g[2] for g in og}),
            "seconds_per_game": round(float(np.mean([g[5] for g in og])), 1),
        }
    # Cross-play, with every clue and with flagged clues replaced by the next choice.
    out["crossplay"] = {"info": runs["crossplay"]["info"]}
    cells = {v: decode_cells(runs["crossplay"].get(v, {})) for v in ("all", "strict")}
    for variant in ("all", "strict"):
        if cells[variant]:
            out["crossplay"][variant] = crossplay_summary(cells[variant])
    if cells["strict"]:
        out["crossplay"]["strict_minus_all"] = strict_effect(cells["all"], cells["strict"])
    flags = [r[10] for r in runs["reproduce"] if r[10] is not None]
    if flags:
        turns = [r[7] for r in runs["reproduce"] if r[10] is not None]
        out["reproduction_flags"] = {"flagged_turns": sum(flags), "turns": sum(turns)}
    return out


GROUPS = {"framework guessers": FRAMEWORK, "our guessers": ("gemma2", "potion", "lora-c8")}


def crossplay_summary(cells: dict) -> dict:
    scores = {k: np.array(v["score"], dtype=float) for k, v in cells.items()}
    matrix = {
        k: summarize(v["score"])
        | {
            "wins": v["loss"].count("-"),
            "assassin": v["loss"].count("A"),
            "opponent": v["loss"].count("O"),
            "other_losses": len(v["loss"]) - sum(v["loss"].count(x) for x in "-AO"),
            "mean_count": round(float(np.mean(v["count"])), 2),
            "turns_in_wins": round(
                float(
                    np.mean(
                        [t for t, c in zip(v["turns"], v["loss"], strict=True) if c == "-"] or [0]
                    )
                ),
                2,
            ),
            "flagged_turns": sum(v["flagged"]),
            "turns": sum(v["turns"]),
        }
        for k, v in cells.items()
    }
    givers = [g for g in GIVERS if any(k.startswith(g + "/") for k in cells)]
    guessers = [g for g in GUESSERS if any(k.endswith("/" + g) for k in cells)]
    means = {}
    for gv in givers:
        for label, group in GROUPS.items():
            members = [gu for gu in group if f"{gv}/{gu}" in cells and stranger(gv, gu)]
            if members:
                m = np.mean([scores[f"{gv}/{gu}"] for gu in members], axis=0)
                means[f"{gv} with {label}"] = summarize(m.tolist()) | {"guessers": members}
    paired = {}
    for a, b in PAIRED:
        for gu in guessers:
            ka, kb = f"{a}/{gu}", f"{b}/{gu}"
            if ka in cells and kb in cells and stranger(a, gu) and stranger(b, gu):
                paired[f"{a} minus {b} / {gu}"] = summarize((scores[ka] - scores[kb]).tolist())
        for label, group in GROUPS.items():
            members = [
                gu for gu in group
                if f"{a}/{gu}" in cells and f"{b}/{gu}" in cells and stranger(a, gu) and stranger(b, gu)
            ]  # fmt: skip
            if members:
                d = np.mean([scores[f"{a}/{gu}"] - scores[f"{b}/{gu}"] for gu in members], axis=0)
                paired[f"{a} minus {b} / {label}"] = summarize(d.tolist()) | {"guessers": members}
    return {
        "givers": givers,
        "guessers": guessers,
        "cells": matrix,
        "giver_means": means,
        "paired": paired,
    }


def strict_effect(all_cells: dict, strict_cells: dict) -> dict:
    """Per giver: the share of its turns with a flagged clue (every guesser pooled), and
    the change in its mean score over the guessers that share no vectors with it when
    flagged clues are replaced, paired by board."""
    out = {}
    for gv in GIVERS:
        keys = [k for k in all_cells if k.startswith(gv + "/") and k in strict_cells]
        if not keys:
            continue
        flagged = sum(x for k in keys for x in all_cells[k]["flagged"])
        illegal = sum(x for k in keys for x in all_cells[k]["illegal"])
        turns = sum(t for k in keys for t in all_cells[k]["turns"])
        others = [k for k in keys if stranger(gv, k.split("/")[1])]
        d = np.mean(
            [
                np.array(strict_cells[k]["score"], float) - np.array(all_cells[k]["score"], float)
                for k in others
            ],
            axis=0,
        )
        out[gv] = {
            "flagged_turns": flagged,
            "illegal_turns": illegal,
            "turns": turns,
            "share": round(flagged / turns, 4) if turns else None,
            "strict_flagged_turns": sum(x for k in keys for x in strict_cells[k]["flagged"]),
            "score_change": summarize(d.tolist()) | {"guessers": [k.split("/")[1] for k in others]},
        }
    return out


PAIRED = (
    ("potion-c3-population", "potion-c3-self"),
    ("potion-c3-population", "potion-first"),
    ("gemma2-first", "potion-first"),
    ("gemma2-first", "glove"),
    ("gemma2-first", "w2v"),
    ("potion-c3-population", "glove"),
)


NAMES = {
    "w2v": "framework w2v",
    "glove": "framework GloVe",
    "w2vglove": "framework w2v+GloVe",
    "gemma2-first": "EmbeddingGemma 2, first choice",
    "potion-first": "potion, first choice",
    "potion-c3-self": "potion, C3 self-play",
    "potion-c3-population": "potion, C3 population",
    "gemma2": "EmbeddingGemma 2",
    "potion": "potion",
    "lora-c8": "C8 LoRA encoder",
}


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "\u2212")


def fmt(s: dict, digits: int = 2) -> str:
    lo, hi = s["ci"]
    return f"{num(s['mean'], digits)} ({num(lo, digits)} to {num(hi, digits)})"


def table_i_differences(rep: dict) -> list[str]:
    """Where Table I differs from the published logs it was computed from."""
    out = []
    for k, r in rep.items():
        t = r["table_i"]
        if t["mean"] is None:
            out.append(
                f"{k} is in the logs ({r['games']} games, mean {r['mean_table_rule']:.2f}) but not in Table I"
            )
            continue
        if abs(t["mean"] - r["mean_table_rule"]) > 0.005:
            out.append(
                f"{k}: Table I gives mean {t['mean']:.2f}, the logs {r['mean_table_rule']:.2f}"
            )
        if abs(t["sd"] - r["sd_table_rule"]) > 0.015:
            out.append(
                f"{k}: Table I gives standard deviation {t['sd']:.2f}, the logs {r['sd_table_rule']:.2f}"
            )
        if t["loss_percent"] != r["assassin_losses"]:
            out.append(
                f"{k}: Table I gives {t['loss_percent']}% losses, the logs {r['assassin_losses']} assassin losses"
            )
    return out


def range_of(means: dict, givers, label: str) -> str:
    vals = [means[f"{g} with {label}"]["mean"] for g in givers if f"{g} with {label}" in means]
    return f"{min(vals):.1f} to {max(vals):.1f}"


def findings(summary: dict) -> list[str]:
    rep, cp = summary["reproduction"], summary["crossplay"]
    games = sum(r["games"] for r in rep.values())
    same = sum(r["same_transcript"] for r in rep.values())
    lines = [
        f"- **The port reproduces the published games.** `bench.py` replays {same} of the {games}"
        " published single-team games of the framework's word-vector bots move for move: every"
        " clue, count and pick (computed exactly, on the framework's boards for seeds 0 to 99).",
    ]
    oc = summary.get("original_code")
    if oc:
        lines[-1] += (
            f" The framework's own code, rerun here on {oc['games']} of them, gives the published"
            f" moves in {oc['same_as_published']} and the port's in {oc['same_as_port']}."
        )
    diffs = table_i_differences(rep)
    blue = [k for k, r in rep.items() if r["losses"] != r["assassin_losses"]]
    lines.append(
        "- **Table I against its own logs.** Table I scores a loss on the last blue word by its"
        " turns, not 25, though the paper's rule says 25"
        + (
            f" ({', '.join(blue)}: {sum(rep[k]['losses'] - rep[k]['assassin_losses'] for k in blue)} such games)"
            if blue
            else ""
        )
        + ". Other differences: "
        + "; ".join(diffs)
        + "."
    )
    a = cp["all"]
    m = a["giver_means"]
    lines.append(
        "- **Strangers cost every giver.** With the framework's guessers that share no vectors"
        f" with them, the framework's codemasters take {range_of(m, FRAMEWORK, 'framework guessers')}"
        f" turns on average, and this package's givers {range_of(m, OURS_GIVERS, 'framework guessers')}."
        f" With this package's guessers, the framework's codemasters take {range_of(m, FRAMEWORK, 'our guessers')}"
        f" and this package's givers {range_of(m, OURS_GIVERS, 'our guessers')} (other models only). Most of"
        " the gap is assassin losses, which score 25. Simulated: 200 boards of words.txt, seeds 0 to"
        " 199, 95% bootstrap intervals in the tables."
    )
    key = "potion-c3-population minus potion-c3-self / framework guessers"
    if key in a["paired"]:
        cells = a["cells"]

        def pooled(gv: str, field: str) -> float:
            return float(np.mean([cells[f"{gv}/{gu}"][field] for gu in FRAMEWORK]))

        wins = {
            g: sum(cells[f"{g}/{gu}"]["wins"] for gu in FRAMEWORK)
            for g in ("potion-c3-population", "potion-c3-self")
        }
        lines.append(
            "- **Population training does not carry over to this score.** With the framework's"
            f" three guessers, the C3 population giver wins {wins['potion-c3-population']} of 600"
            f" games against {wins['potion-c3-self']} for the C3 self-play giver, but it asks for"
            f" {pooled('potion-c3-population', 'mean_count'):.2f} words per clue against"
            f" {pooled('potion-c3-self', 'mean_count'):.2f}, so its wins take"
            f" {pooled('potion-c3-population', 'turns_in_wins'):.1f} turns against"
            f" {pooled('potion-c3-self', 'turns_in_wins'):.1f}. Its mean score is worse by"
            f" {fmt(a['paired'][key])} turns. C3 ranked the two the other way by expected words"
            " per turn; the single-team score charges for every extra turn."
        )
    if "strict_minus_all" in cp:
        e = cp["strict_minus_all"]
        shares = ", ".join(f"{NAMES[g]} {100 * v['share']:.1f}%" for g, v in e.items())
        lines.append(
            "- **Clues that share a form with a board word are rare, and replacing them changes little.**"
            f" Share of turns flagged by `codenames.legality` (illegal or borderline): {shares}. Replacing them with each"
            " giver's next legal choice changes no giver's mean score over its strangers by more"
            f" than {max(abs(v['score_change']['mean']) for v in e.values()):.2f} turns; the widest"
            f" interval reaches {max(abs(x) for v in e.values() for x in v['score_change']['ci']):.2f}"
            " (table at the end of section 2)."
        )
    return lines


def report(summary: dict) -> str:
    """bench.md, from bench.json alone."""
    rep = summary["reproduction"]
    lines = [
        "# Codenames: the AI competition framework's word-vector bots",
        "",
        "Exploratory. No plan was written before these runs (experiments/METHOD.md); the numbers",
        "motivate a planned comparison and are not its result. Written by",
        "`python -m codenames.bench_play analyze` from `bench-runs.json`.",
        "",
        *findings(summary),
        "",
        "## 1. The published games, replayed",
        "",
        "The 900 single-team games of Stephenson et al. (2025, Table I), replayed with",
        "`codenames/bench.py` on the framework's boards (seeds 0 to 99) and compared move by",
        "move with the published logs. A score is the number of turns to find the 9 red words,",
        'or 25 for a loss. "Table I" is copied from arXiv 2412.11373v2. "Logs" is recomputed',
        "here from the published logs, once by Table I's apparent rule (25 only for the",
        "assassin) and once by the framework's rule (25 also for the last blue word).",
        "",
        "| Codemaster / guesser | Table I mean (loss %) | Logs, Table I's rule | Logs, framework rule"
        " | Port: same moves | Port: same score | Port mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, r in rep.items():
        t = r["table_i"]
        ti = (
            f"{t['mean']:.2f} ({t['loss_percent']}%)"
            if t["mean"] is not None
            else "not in the table"
        )
        lines.append(
            f"| {k} | {ti} | {r['mean_table_rule']:.2f} | {r['mean_framework_rule']:.2f}"
            f" ({r['losses']} losses) | {r['same_transcript']} of {r['games']}"
            f" | {r['same_score']} of {r['games']} | {r['port_mean']:.2f} |"
        )
    oc = summary.get("original_code")
    if oc:
        lines += [
            "",
            f"The framework's own code, at the pinned commit, played {oc['games']} of these games"
            f" ({oc['pairs']} pairs, seeds {oc['seeds'][0]} to {oc['seeds'][-1]}) on a GitHub-hosted"
            f" runner: {oc['same_as_published']} have the published moves and {oc['same_as_port']}"
            f" the port's ({oc['seconds_per_game']} s a game).",
        ]
    cp = summary["crossplay"]
    lines += [
        "",
        "## 2. Cross-play on this package's words",
        "",
        "Mean score over 200 boards of `words.txt`, dealt as the framework deals them (seeds 0"
        " to 199), with 95% bootstrap intervals over boards (10,000 draws, seed 0). Lower is"
        " better; 25 is a loss. Rows are givers, columns guessers; a giver with its own model's"
        " guesser is in bold.",
    ]
    lines += crossplay_tables(cp["all"], "Every clue that the giver's own rule allows")
    if "strict" in cp:
        lines += crossplay_tables(
            cp["strict"],
            "Clues that share a form with an unrevealed board word replaced by the giver's next choice",
        )
        lines += [
            "",
            "### Clues that share a form with a board word",
            "",
            "A clue is flagged when `codenames.legality` finds an illegal or borderline relation"
            " between it and an unrevealed board word (a form of the word, a part of a compound,"
            " a shared stem; see that module). Turns flagged with every clue"
            " allowed, pooled over guessers, and the change in mean score over the guessers that"
            " share no vectors with the giver when flagged clues are replaced (paired by board):",
            "",
            "| Giver | Flagged turns | Of them illegal | Share | Flagged after replacing | Change in mean score (95% interval) |",
            "|---|---|---|---|---|---|",
        ]
        for gv, e in cp["strict_minus_all"].items():
            lines.append(
                f"| {NAMES[gv]} | {e['flagged_turns']} of {e['turns']} | {e['illegal_turns']} | {100 * e['share']:.1f}%"
                f" | {e['strict_flagged_turns']} | {fmt(e['score_change'])} |"
            )
    flags = summary.get("reproduction_flags")
    if flags:
        lines += [
            "",
            f"In the 900 published games, {flags['flagged_turns']} of {flags['turns']} clues are flagged"
            " the same way, against the board words still unrevealed.",
        ]
    lines += static_sections(summary)
    return "\n".join(lines) + "\n"


def rerun_note(d: dict) -> str:
    r = d.get("reruns")
    if not r:
        return ""
    return (
        f" Of {r['slices']} givers' runs with every clue allowed that were repeated on other"
        f" runners ({r['games']} games), {r['same_scores']} gave the same score in every game."
    )


def static_sections(summary: dict) -> list[str]:
    d = summary["data"]
    info = next(iter(summary["crossplay"]["info"].values()))
    lora = next(
        (v["lora-c8"] for v in summary["crossplay"]["info"].values() if "lora-c8" in v), None
    )
    c3 = next((v["c3"] for v in summary["crossplay"]["info"].values() if "c3" in v), None)
    lines = [
        "",
        "## The rule each giver enforces",
        "",
        "- **The framework's codemasters** (players/codemaster_*.py): a clue is not a board word,"
        " not a part of one and does not contain one, and no board word equals the clue's"
        " WordNet lemma or Lancaster stem (NLTK"
        f" {d['forms']['nltk']}). Clues come from the framework's list of {d['forms']['words']} words."
        " The rule is checked against the words still on the board.",
        "- **This package's givers** (`codenames.words.legal`): a clue is not an unrevealed"
        " board word, not a part of one, and does not contain one. Clues come from the"
        f" {info['our_clue_pool']} of the 12,000 clue words that both GloVe and word2vec have, so"
        f" that the framework's guessers can read them ({info['clue_words_dropped']} dropped).",
        "- **The flag** (`codenames.legality`, for the second set of tables): any relation that"
        " this module calls illegal (a form of the word, a part of a compound, a spelling variant)"
        " or borderline (a shared stem, affix stem, three-letter compound part or long prefix)."
        " It is stricter than both rules: it catches broken for BREAK and teaching for TEACHER,"
        " which both rules allow.",
        "",
        "## How each number is known",
        "",
        "- Table I values: taken from Stephenson et al. (2025), arXiv 2412.11373v2, Table I.",
        "- Values from the published logs: computed exactly from bot_results_new_style.txt and"
        " move_log.txt in stepmat/Codenames_GPT at commit fc28bc9.",
        "- The port's games: computed exactly; compared move by move with the logs. The"
        " framework's own code was rerun on 90 games as a second method.",
        "- Cross-play: simulated games, 200 boards, board seeds 0 to 199; every bot is"
        " deterministic, so each cell is one game per board."
        + rerun_note(d)
        + " Intervals are 95% percentile"
        " bootstrap intervals over boards, 10,000 draws, seed 0; paired differences resample"
        " boards once for both givers.",
    ]
    if c3:
        lines.append(
            "- The C3 givers were refit with codenames.rl on the task with the potion giver,"
            " EmbeddingGemma 2 held out and training seed 1. The largest difference from the θ in results/c3.json is"
            f" {c3['theta_max_difference']['self']:.4f} (self-play) and"
            f" {c3['theta_max_difference']['population']:.4f} (population)."
        )
    if lora:
        lines.append(
            f"- The C8 encoder (arm B, seed 1, clue template {lora.get('template', '?')}) encoded"
            f" {lora.get('extra_words', '?')} words that its stored vectors lack, on a CPU in float32"
            f" ({lora.get('seconds', '?')} s);"
            " on 200 stored words the largest difference from the stored float16 vectors is"
            f" {lora.get('max_abs_difference_bare', float('nan')):.4f} (bare) and"
            f" {lora.get('max_abs_difference_' + str(lora.get('template')), float('nan')):.4f} ({lora.get('template')})."
        )
    lines += [
        "",
        "## Sources and licenses",
        "",
        "| Source | What | License | Committed |",
        "|---|---|---|---|",
        "| stepmat/Codenames_GPT, commit fc28bc9 (Stephenson et al. 2025) | the framework's code,"
        " clue list and published logs | MIT | no; downloaded and pinned by SHA-256 |",
        "| CodenamesAICompetition/Game (Summerville et al. 2019) | the original framework; the same"
        " bots and word pool | MIT | no |",
        "| The framework's board words | the 394 words of the published game | the rights to the"
        " game's words belong to Czech Games Edition; whether the MIT license of the file covers"
        " them is unclear | no; used only to rebuild the published boards |",
        "| GloVe 6B 300d (Pennington et al. 2014) | word vectors | Public Domain Dedication and"
        " License v1.0, per the GloVe page | no |",
        "| GoogleNews word2vec 300d (Mikolov et al. 2013), via gensim-data | word vectors | the"
        " word2vec project page lists Apache 2.0 for the project; no separate license for the"
        " vectors is stated | no |",
        "",
        "## Compute",
        "",
        "The vectors, the replays and the cross-play ran on GitHub-hosted runners (CPU)"
        " through a workflow on the branch `compute/codenames-bench`, which commits the raw"
        " outputs to that branch under `runs/bench/<run id>/`. The C8 column ran in Hugging Face"
        " Jobs (`cpu-upgrade`), since its weights are in the private dataset; its outputs are"
        " in that dataset under `runs/bench/`. Both built the same vector subsets (SHA-256"
        f" {d['vectors']['glove300_sha256'][:12]} for GloVe and {d['vectors']['w2v_npz_sha256'][:12]}"
        " for word2vec).",
        "",
        "## Limits",
        "",
        "- Exploratory, one deterministic game per board and pair, one training seed for the"
        " C3 givers, one seed for the C8 encoder. The comparison worth planning is named in the"
        " findings; a plan should fix its metric, guessers and board seeds before running.",
        "- The framework's guessers pick exactly the count; they never stop early or take the"
        " extra pick that the printed rules allow. This package's guessers do the same here.",
        "- Our givers were built for expected turn value, not for the single-team score, which"
        " charges 25 for a loss; they take more risk than that score rewards.",
        "",
        "## Usage",
        "",
        "```sh",
        "uv run python -m codenames.bench_data fetch        # the framework's files",
        "uv run python -m codenames.bench_play analyze      # bench.json and bench.md from bench-runs.json",
        "```",
        "",
        "The other commands need the vector subsets and run in the workflow; see the module docstrings.",
    ]
    return lines


def crossplay_tables(cp: dict, title: str) -> list[str]:
    lines = [
        "",
        f"### {title}",
        "",
        "| Giver \\ guesser | " + " | ".join(NAMES[g] for g in cp["guessers"]) + " |",
        "|---|" + "---|" * len(cp["guessers"]),
    ]
    for gv in cp["givers"]:
        row = []
        for gu in cp["guessers"]:
            c = cp["cells"].get(f"{gv}/{gu}")
            cell = "" if c is None else fmt(c, 1)
            row.append(f"**{cell}**" if c is not None and not stranger(gv, gu) else cell)
        lines.append(f"| {NAMES[gv]} | " + " | ".join(row) + " |")
    lines += [
        "",
        "Games won of 200 (losses on the assassin / on the last opponent word):",
        "",
        "| Giver \\ guesser | " + " | ".join(NAMES[g] for g in cp["guessers"]) + " |",
        "|---|" + "---|" * len(cp["guessers"]),
    ]
    for gv in cp["givers"]:
        row = []
        for gu in cp["guessers"]:
            c = cp["cells"].get(f"{gv}/{gu}")
            row.append("" if c is None else f"{c['wins']} ({c['assassin']} / {c['opponent']})")
        lines.append(f"| {NAMES[gv]} | " + " | ".join(row) + " |")
    lines += [
        "",
        "Each giver averaged, board by board, over the guessers of a group that share no vectors with it:",
        "",
        "| Giver | Group | Guessers | Mean score (95% interval) |",
        "|---|---|---|---|",
    ]
    for k, m in cp["giver_means"].items():
        gv, label = k.split(" with ")
        lines.append(
            f"| {NAMES[gv]} | {label} | {', '.join(NAMES[g] for g in m['guessers'])} | {fmt(m)} |"
        )
    lines += [
        "",
        "Paired differences in score, board by board (negative: the first giver finishes sooner):",
        "",
        "| Difference | Guesser | Mean (95% interval) |",
        "|---|---|---|",
    ]
    for k, d in cp["paired"].items():
        pair, gu = k.split(" / ")
        a, b = pair.split(" minus ")
        who = gu if gu in GROUPS else NAMES[gu]
        lines.append(f"| {NAMES[a]} minus {NAMES[b]} | {who} | {fmt(d)} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce")
    r.add_argument("--out", type=Path, required=True)
    o = sub.add_parser("original", help="compare the framework's own logs (one folder per job)")
    o.add_argument("--logs", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("crossplay")
    c.add_argument("--giver", choices=GIVERS, required=True)
    c.add_argument("--guessers", nargs="+", default=[g for g in GUESSERS if g != "lora-c8"])
    c.add_argument("--boards", type=int, default=N_BOARDS)
    c.add_argument("--lora", type=Path, default=None, help="folder of the C8 run (private)")
    c.add_argument(
        "--strict", action="store_true", help="skip clues that share a form with a board word"
    )
    c.add_argument("--out", type=Path, required=True)
    lr = sub.add_parser("lora-run", help="the C8 column, inside a Hugging Face Job")
    lr.add_argument("--boards", type=int, default=N_BOARDS)
    lr.add_argument("--variants", nargs="+", default=["all", "strict"], choices=["all", "strict"])
    lr.add_argument("--out", type=Path, required=True)
    co = sub.add_parser("collect", help="compact a run's outputs into results/bench-runs.json")
    co.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="later runs replace earlier"
    )
    co.add_argument("--lora", type=Path, default=None, help="compact_lora's file for the C8 column")
    cl = sub.add_parser("compact-lora", help="the C8 jobs' outputs, compacted (in a job)")
    cl.add_argument("--from-dataset", nargs="+", required=True, help="runs/bench/<run> folders")
    cl.add_argument("--out", type=Path, required=True)
    sub.add_parser("analyze", help="results/bench.json and bench.md from bench-runs.json")
    args = parser.parse_args()
    if args.cmd == "reproduce":
        res = reproduce()
        args.out.write_text(json.dumps(res) + "\n")
        same = sum(g["same_transcript"] for g in res["games"])
        print(f"{same} of {len(res['games'])} games have the published transcript")
    elif args.cmd == "original":
        res = compare_original(args.logs)
        args.out.write_text(json.dumps(res) + "\n")
    elif args.cmd == "crossplay":
        res = crossplay(args.giver, args.guessers, args.boards, args.lora, args.strict)
        args.out.write_text(json.dumps(res) + "\n")
    elif args.cmd == "lora-run":
        lora_run(args.out, args.boards, args.variants)
    elif args.cmd == "compact-lora":
        hub = importlib.import_module("huggingface_hub")
        local = Path(
            hub.snapshot_download(
                "arclabs561/games-runs", repo_type="dataset",
                allow_patterns=[f"{f}/*" for f in args.from_dataset],
            )
        )  # fmt: skip
        text = json.dumps(compact_lora([local / f for f in args.from_dataset]), sort_keys=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "lora-compact.json").write_text(text + "\n")
        print("lora-compact.json sha256", bench_data.sha256((text + "\n").encode()), flush=True)
    elif args.cmd == "collect":
        RUNS_FILE.write_text(json.dumps(collect(args.runs, args.lora), sort_keys=True) + "\n")
    else:
        summary = analyze(json.loads(RUNS_FILE.read_text()))
        SUMMARY_FILE.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        REPORT_FILE.write_text(report(summary))


if __name__ == "__main__":
    main()
