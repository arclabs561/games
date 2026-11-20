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


def framework_players():
    tabs = bench_data.tables()
    return bench.framework_bots(tabs, list(bench_data.cm_words()), bench_data.forms())


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
    }


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


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


class PolicyGiver:
    """A plan-C3 policy as a giver: the action with the largest theta . phi."""

    def __init__(self, params: np.ndarray, sp: Space, tau: float, pool: tuple[str, ...]):
        f = len(rl.FEATURES)
        self.theta, self.mu, self.sd = params[:f], params[f : 2 * f], params[2 * f :]
        self.sp, self.tau, self.pool = sp, tau, pool

    def __call__(self, board: Board) -> Clue:
        clues, phi = policy_actions(self.sp, self.tau, board, self.pool, 30)
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
    model = st.SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    model.load_adapter(str(folder / "adapter"))
    from codenames.embed import TEMPLATES, normalize

    def enc(t: str, ws: list[str]) -> np.ndarray:
        texts = [TEMPLATES[t].format(w) for w in ws]
        return normalize(np.asarray(model.encode(texts, batch_size=128, prompt=""))).astype(
            np.float16
        )

    probe = words[:: max(1, len(words) // 200)][:200]
    pos = {w: i for i, w in enumerate(words)}
    info = {"template": template, "extra_words": len(missing)}
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


def make_giver(name: str, tabs, pool_ours: tuple[str, ...], fw_givers, info: dict):
    if name in FRAMEWORK:
        return fw_givers[name]
    cal = load("calibrate")
    if name == "gemma2-first":
        b = cal[GEMMA]["best"]
        return expected_giver(space(GEMMA, b["clue"], b["word"]), b["tau"], candidates=pool_ours)
    if name == "potion-first":
        b = cal["potion"]["best"]
        return expected_giver(space("potion", b["clue"], b["word"]), b["tau"], candidates=pool_ours)
    c3 = c3_policies()
    info["c3"] = {"task": C3_TASK, "theta_max_difference": c3["theta_max_difference"]}
    sp, tau = chosen_spaces(cal)["potion"]
    counts = eat_counts()
    pool = tuple(c for c in pool_ours if c in counts)
    kind = name.removeprefix("potion-c3-")
    return PolicyGiver(np.array(c3["params"][kind]), sp, tau, pool)


def crossplay_many(
    givers, guessers, n_boards: int = N_BOARDS, lora_dir: Path | None = None
) -> list[dict]:
    """Every giver with every guesser on the same boards; one result per giver."""
    t0 = time.perf_counter()
    tabs = bench_data.tables()
    fw_givers, fw_guessers = framework_players()
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
        give = make_giver(giver, tabs, pool_ours, fw_givers, info)
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
                "info": info,
                "games": games,
                "seconds": round(time.perf_counter() - t1),
            }
        )
    return out


def crossplay(giver: str, guessers, n_boards: int = N_BOARDS, lora_dir: Path | None = None) -> dict:
    return crossplay_many([giver], guessers, n_boards, lora_dir)[0]


def lora_run(out: Path, n_boards: int = N_BOARDS) -> None:
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
    for giver in GIVERS:
        res = crossplay_many([giver], ["lora-c8"], n_boards, Path(c8) / LORA_RUN)[0]
        (out / f"crossplay-lora-{giver}.json").write_text(json.dumps(res) + "\n")


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


def collect(runs: Path, lora: Path | None) -> dict:
    """Compact the raw outputs of the workflow run (and of the C8 job) into one file."""
    rep = json.loads((runs / "reproduce.json").read_text())["games"]
    out: dict = {
        "status": "exploratory: no plan was written before these runs",
        "source": json.loads((runs / "source.json").read_text()),
        "data": {
            "vectors": json.loads((runs / "vectors.json").read_text()),
            "forms": json.loads((runs / "forms.json").read_text()),
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
            ]
            for g in rep
        ],
    }
    if (runs / "original.json").exists():
        og = json.loads((runs / "original.json").read_text())["games"]
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
        versions = sorted(runs.glob("original/*/versions.txt"))
        if versions:
            out["data"]["original_versions"] = versions[0].read_text().split()
    slices = [json.loads(p.read_text()) for p in sorted(runs.glob("crossplay-*.json"))]
    if lora:
        slices += [json.loads(p.read_text()) for p in sorted(lora.glob("crossplay-lora-*.json"))]
        out["data"]["lora_job"] = json.loads((lora / "data.json").read_text())
        if (lora / "manifest.json").exists():
            out["source"]["lora_job"] = json.loads((lora / "manifest.json").read_text())
    cells: dict[str, dict] = {}
    info: dict[str, dict] = {}
    for sl in slices:
        info.setdefault(sl["giver"], {}).update(sl["info"])
        for g in sl["games"]:
            c = cells.setdefault(
                f"{g['giver']}/{g['guesser']}", {"seeds": [], "score": [], "loss": "", "count": []}
            )
            c["seeds"].append(g["seed"])
            c["score"].append(g["score"])
            c["loss"] += LOSS_CODE[g.get("loss")] if "error" not in g else "E"
            c["count"].append(round(g.get("mean_count", 0.0), 3))
    for c in cells.values():
        assert c["seeds"] == list(range(len(c["seeds"])))
        del c["seeds"]
    out["crossplay"] = {"info": info, "cells": dict(sorted(cells.items()))}
    return out


def model_of(player: str) -> str:
    return {"gemma2-first": "gemma2"}.get(player, player.split("-")[0])


def analyze(runs: dict) -> dict:
    out: dict = {"status": runs["status"], "source": runs["source"], "data": runs["data"]}
    # Reproduction.
    by: dict[str, list] = {}
    for cm, gu, _seed, same, first, sc, sct, pturns, pwin, pa in runs["reproduce"]:
        by.setdefault(f"{cm}/{gu}", []).append((same, first, sc, sct, pturns, pwin, pa))
    rep = {}
    for k, gs in by.items():
        pub_rule = [25 if w == "B" else t for *_, t, w, a in gs]
        rep[k] = {
            "games": len(gs),
            "same_transcript": sum(g[0] for g in gs),
            "same_score": sum(g[2] == p for g, p in zip(gs, pub_rule, strict=True)),
            "port_mean": round(float(np.mean([g[2] for g in gs])), 2),
            "port_mean_table_rule": round(float(np.mean([g[3] for g in gs])), 2),
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
    # Cross-play.
    cells = runs["crossplay"]["cells"]
    scores = {k: np.array(v["score"], dtype=float) for k, v in cells.items()}
    matrix = {
        k: summarize(v["score"])
        | {
            "wins": v["loss"].count("-"),
            "assassin": v["loss"].count("A"),
            "opponent": v["loss"].count("O"),
            "other_losses": len(v["loss"]) - sum(v["loss"].count(x) for x in "-AO"),
            "mean_count": round(float(np.mean(v["count"])), 2),
        }
        for k, v in cells.items()
    }
    givers = [g for g in GIVERS if any(k.startswith(g + "/") for k in cells)]
    guessers = [g for g in GUESSERS if any(k.endswith("/" + g) for k in cells)]
    groups = {"framework guessers": FRAMEWORK, "our guessers": ("gemma2", "potion", "lora-c8")}
    means = {}
    for gv in givers:
        for label, group in groups.items():
            members = [gu for gu in group if f"{gv}/{gu}" in cells and model_of(gv) != gu]
            if members:
                m = np.mean([scores[f"{gv}/{gu}"] for gu in members], axis=0)
                means[f"{gv} with {label}"] = summarize(m.tolist()) | {"guessers": members}
    paired = {}
    for a, b in PAIRED:
        for gu in guessers:
            ka, kb = f"{a}/{gu}", f"{b}/{gu}"
            if ka in cells and kb in cells and gu not in (model_of(a), model_of(b)):
                paired[f"{a} minus {b} / {gu}"] = summarize((scores[ka] - scores[kb]).tolist())
        for label, group in groups.items():
            members = [
                gu for gu in group
                if f"{a}/{gu}" in cells and f"{b}/{gu}" in cells and gu not in (model_of(a), model_of(b))
            ]  # fmt: skip
            if members:
                d = np.mean([scores[f"{a}/{gu}"] - scores[f"{b}/{gu}"] for gu in members], axis=0)
                paired[f"{a} minus {b} / {label}"] = summarize(d.tolist()) | {"guessers": members}
    out["crossplay"] = {
        "givers": givers,
        "guessers": guessers,
        "cells": matrix,
        "giver_means": means,
        "paired": paired,
        "info": runs["crossplay"]["info"],
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


def report(summary: dict) -> str:
    """bench.md, from bench.json."""
    return json.dumps(summary["status"]) + "\n"


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
    c.add_argument("--out", type=Path, required=True)
    lr = sub.add_parser("lora-run", help="the C8 column, inside a Hugging Face Job")
    lr.add_argument("--boards", type=int, default=N_BOARDS)
    lr.add_argument("--out", type=Path, required=True)
    co = sub.add_parser("collect", help="compact a run's outputs into results/bench-runs.json")
    co.add_argument("--runs", type=Path, required=True)
    co.add_argument("--lora", type=Path, default=None, help="outputs of the C8 job")
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
        res = crossplay(args.giver, args.guessers, args.boards, args.lora)
        args.out.write_text(json.dumps(res) + "\n")
    elif args.cmd == "lora-run":
        lora_run(args.out, args.boards)
    elif args.cmd == "collect":
        RUNS_FILE.write_text(json.dumps(collect(args.runs, args.lora), sort_keys=True) + "\n")
    else:
        summary = analyze(json.loads(RUNS_FILE.read_text()))
        SUMMARY_FILE.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        REPORT_FILE.write_text(report(summary))


if __name__ == "__main__":
    main()
