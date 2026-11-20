"""Calibration of the guessers on human turns, and how far people agree with each other.

Section 1 of the README scores a guesser by whether its top word is the word the person
picked. A giver uses the guesser's whole distribution, so this module scores the
distribution too. For each guesser (a Luce guesser with the temperature and template of
results/calibrate-game.json, the EAT counts with their fitted epsilon, and C1's linear
maps, retrained here with seeds 1 to 5) on the test turns of C1's split by game:

- **log-likelihood per turn**: the mean of log q(person's word);
- **expected calibration error** (ECE) of the top word: turns are put in 10 equal bins by
  the probability q(top) that the guesser gives its own top word; ECE is the sum over
  bins of (turns in bin / turns) times |share of hits in bin - mean q(top) in bin|;
- **Brier score on the chosen word**: the mean of (q(top) - hit)^2, where hit is 1 when
  the top word is the person's word; and the multiclass Brier score, the mean over turns
  of the sum over words of (q(w) - [w is the person's word])^2.

Intervals resample test games 10,000 times (seed 0), as `stats.py` does. These metrics
were added after the test turns had been scored for top-1, so they are exploratory.

**Turns with several picks.** 19% of the rows record several words for one clue, in the
order the file lists them, which is taken to be the order of picking (the data do not
say). They are left out of every other result. Here they give the share of turns whose
first pick is the guesser's top word, the share whose top word is any of the picks, and
the Plackett-Luce log-likelihood of the whole sequence.

**Agreement between people.** No clue and board was answered by several people, except
where this module finds exact repeats. Under Luce's choice axiom (a person picks w from a
set S with probability p(w) / p(S), the same p for everyone), two turns with the same clue
whose picks each lie on the other's board are two draws from p restricted to the shared
words C. The share of such pairs that agree estimates the mean of sum over w in C of
(p(w)/p(C))^2. For one item, sum q^2 <= max q <= sqrt(sum q^2), so the best possible
top-1 on these restricted items lies between that share a and sqrt(a) (the upper bound by
Jensen's inequality). The guessers' agreement with each person on the same pairs (their
top word among C) is reported beside it. Pairs share clues, so intervals resample clues.

    uv run python -m codenames.calibration            # writes results/guesser-calibration.json

Only aggregates are written: no clue, board or pick from Cultural Codes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from codenames.embed import COMPUTED, cache_path
from codenames.evaluate import TAUS, all_words, guess_loglik
from codenames.fetch import fetch
from codenames.luce import luce_weights
from codenames.norms import eat_counts
from codenames.players import Space, norm_weigher, space
from codenames.split import PUBLISHED, digest, game_keys, game_split, player_id, read_rows
from codenames.stats import DRAWS, cluster_draws, replicates
from codenames.study import LOCAL, load, models, short
from codenames.train import adapted, items, train

RESULTS = Path(__file__).parent / "results"
BINS = 10
SEEDS = (1, 2, 3, 4, 5)
MAPS = (("potion", "bare"), ("Qwen/Qwen3-Embedding-0.6B", "query"))

# --- turns, with every pick ------------------------------------------------------------


@dataclass(frozen=True)
class Picks:
    """One row of Cultural Codes: the clue, the unrevealed words and every word picked."""

    clue: str
    words: tuple[str, ...]
    picks: tuple[str, ...]  # in the order the file lists them
    game: str
    split: str  # C1's split by game
    guesser: str


def parse_picks(output: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in output.split(",") if p.strip())


@cache
def pick_turns() -> tuple[Picks, ...]:
    """Every row whose clue is alphabetic and whose picks are distinct board words. The
    rows with one pick are the turns of codenames.split."""
    folder = fetch("cultural")
    rows = read_rows({s: folder / f"guess_{s}.csv" for s in PUBLISHED})
    out = []
    for r, key in zip(rows, game_keys(rows), strict=True):
        picks = parse_picks(r.output)
        ok = picks and len(set(picks)) == len(picks) and all(p in r.words for p in picks)
        if ok and r.clue.isalpha():
            out.append(
                Picks(r.clue, r.words, picks, digest(key), game_split(key), player_id(r.guesser))
            )
    return tuple(out)


# --- scoring one distribution ----------------------------------------------------------


def probabilities(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def plackett_luce(weights: np.ndarray, order: list[int]) -> float:
    """log P(the picks come in this order), each pick from the words not yet picked."""
    w = np.asarray(weights, dtype=float)
    left = np.ones(len(w), bool)
    total = 0.0
    for i in order:
        total += math.log(w[i] / w[left].sum())
        left[i] = False
    return total


def turn_scores(q: np.ndarray, person: int) -> dict[str, float]:
    """Per-turn scores of a distribution q over the words, against the person's word."""
    top = int(np.argmax(q))
    hit = float(top == person)
    onehot = np.zeros(len(q))
    onehot[person] = 1.0
    return {
        "loglik": math.log(q[person]),
        "confidence": float(q[top]),
        "hit": hit,
        "brier_top": (float(q[top]) - hit) ** 2,
        "brier_all": float(((q - onehot) ** 2).sum()),
    }


def bin_of(confidence: np.ndarray, bins: int = BINS) -> np.ndarray:
    return np.minimum((np.asarray(confidence) * bins).astype(int), bins - 1)


def ece(confidence, hit, bins: int = BINS, weights=None) -> float:
    """Expected calibration error with equal-width bins; weights repeat turns."""
    c, h = np.asarray(confidence, float), np.asarray(hit, float)
    w = np.ones(len(c)) if weights is None else np.asarray(weights, float)
    b = bin_of(c, bins)
    gap = np.bincount(b, weights=w * (h - c), minlength=bins)
    return float(np.abs(gap).sum() / w.sum())


def ece_replicates(confidence, hit, index, counts, bins: int = BINS) -> np.ndarray:
    """ECE in each bootstrap replicate of the clusters (the rows of counts)."""
    c, h = np.asarray(confidence, float), np.asarray(hit, float)
    n = counts.shape[1]
    b = bin_of(c, bins)
    gap = np.zeros((n, bins))
    np.add.at(gap, (index, b), h - c)
    size = np.bincount(index, minlength=n).astype(float)
    return np.abs(counts @ gap).sum(axis=1) / (counts @ size)


def reliability(confidence, hit, bins: int = BINS) -> list[dict]:
    """Turns, mean confidence and share of hits in each bin that has turns."""
    c, h = np.asarray(confidence, float), np.asarray(hit, float)
    b = bin_of(c, bins)
    out = []
    for k in range(bins):
        m = b == k
        if m.any():
            out.append(
                {
                    "bin": [k / bins, (k + 1) / bins],
                    "turns": int(m.sum()),
                    "confidence": float(c[m].mean()),
                    "hits": float(h[m].mean()),
                }
            )
    return out


def interval(reps: np.ndarray) -> list[float]:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return [float(lo), float(hi)]


# --- people against each other ---------------------------------------------------------


def exact_repeats(turns: tuple[Picks, ...]) -> dict:
    """Rows of different guessers that share a clue and the same unrevealed words, and
    how often their first picks agree."""
    groups: dict[tuple, list[Picks]] = defaultdict(list)
    for t in turns:
        groups[(t.clue, frozenset(t.words))].append(t)
    pairs = agree = 0
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if g[i].guesser == g[j].guesser:
                    continue
                pairs += 1
                agree += g[i].picks[0] == g[j].picks[0]
    return {
        "items_with_two_or_more_rows": sum(len(g) > 1 for g in groups.values()),
        "pairs": pairs,
        "pairs_agreeing": agree,
    }


def shared_pairs(turns: tuple[Picks, ...]) -> list[tuple[int, int]]:
    """Index pairs (i, j), i < j: the same clue, different games and different guessers,
    and each first pick on the other's board."""
    by_clue: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(turns):
        by_clue[t.clue].append(i)
    out = []
    for ids in by_clue.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                s, t = turns[ids[a]], turns[ids[b]]
                if s.game == t.game or s.guesser == t.guesser:
                    continue
                if s.picks[0] in t.words and t.picks[0] in s.words:
                    out.append((ids[a], ids[b]))
    return out


def shared_words(s: Picks, t: Picks) -> tuple[str, ...]:
    other = set(t.words)
    return tuple(w for w in s.words if w in other)


def agreement_bounds(a: float) -> dict:
    """The best possible top-1 on the restricted items lies in [a, sqrt(a)]."""
    return {"lower": a, "upper": math.sqrt(a)}


# --- the guessers ----------------------------------------------------------------------


def cached_words(model: str, template: str) -> set[str] | None:
    """The words a remote model has vectors for (None for a model that runs here)."""
    if model in LOCAL:
        return None
    path = cache_path(model, template)
    return set(np.load(path)["words"]) if path.exists() else set()


def guesser_spaces(needed: list[str]) -> dict[str, tuple[Space, float, set[str] | None]]:
    """Each untrained model at the template and temperature of calibrate-game.json, with
    the words it has vectors for."""
    cal = load("calibrate-game")
    out = {}
    for m in models():
        b = cal[m]["best"]
        have = None
        for t in {b["clue"], b["word"]}:
            ws = cached_words(m, t)
            if ws is not None:
                have = ws if have is None else have & ws
        words = needed if have is None else [w for w in needed if w in have]
        out[short(m)] = (space(m, b["clue"], b["word"], words), b["tau"], have)
    return out


def score_turns(weigh, turns, person_of) -> dict[str, np.ndarray]:
    """Per-turn scores, as arrays, of a weight function on turns."""
    rows = [turn_scores(probabilities(weigh(t)), person_of(t)) for t in turns]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def summary(scores: list[dict[str, np.ndarray]], index, counts) -> dict:
    """Metrics averaged over the runs (one for an untrained model, five seeds for a map),
    with intervals from the same replicates for every run."""
    out: dict = {"turns": len(scores[0]["hit"]), "runs": len(scores)}
    for key in ("hit", "loglik", "confidence", "brier_top", "brier_all"):
        reps = np.mean([replicates(s[key], index, counts) for s in scores], axis=0)
        mean = float(np.mean([s[key].mean() for s in scores]))
        out[key] = {"mean": mean, "ci": interval(reps)}
    reps = np.mean(
        [ece_replicates(s["confidence"], s["hit"], index, counts) for s in scores], axis=0
    )
    out["ece"] = {
        "mean": float(np.mean([ece(s["confidence"], s["hit"]) for s in scores])),
        "ci": interval(reps),
    }
    out["overconfidence"] = out["confidence"]["mean"] - out["hit"]["mean"]
    if len(scores) == 1:
        out["reliability"] = reliability(scores[0]["confidence"], scores[0]["hit"])
    return out


def multi_pick_scores(weigh, turns: list[Picks]) -> dict:
    """On turns with several picks: top word = first pick, = second pick, in the picks;
    the Plackett-Luce log-likelihood per pick; and 1/k, the top-1 an oracle would get
    that knew the set of picks but drew their order at random."""
    first = second = anyp = 0
    pl, picks = 0.0, 0
    for t in turns:
        w = np.asarray(weigh(t), float)
        top = t.words[int(np.argmax(w))]
        first += top == t.picks[0]
        second += top == t.picks[1]
        anyp += top in t.picks
        pl += plackett_luce(w, [t.words.index(p) for p in t.picks])
        picks += len(t.picks)
    n = len(turns)
    return {
        "turns": n,
        "top_is_first_pick": first / n,
        "top_is_second_pick": second / n,
        "top_is_any_pick": anyp / n,
        "plackett_luce_per_pick": pl / picks,
        "oracle_set_random_order": sum(1 / len(t.picks) for t in turns) / n,
    }


def pair_scores(pick_of_model, turns, pairs) -> np.ndarray:
    """Per pair: the mean over its two people of [the model's top word among the shared
    words is that person's first pick]."""
    out = []
    for i, j in pairs:
        s, t = turns[i], turns[j]
        top = pick_of_model(s.clue, shared_words(s, t))
        out.append(((top == s.picks[0]) + (top == t.picks[0])) / 2)
    return np.array(out, float)


def clue_draws(turns, pairs, draws: int = DRAWS):
    return cluster_draws([turns[i].clue for i, _ in pairs], draws)


def run() -> dict:
    every = pick_turns()
    # The words of section 1, and no others: granite and bge are int8 with dynamic
    # quantization, so a word's vector depends on the other words in its batch, and
    # this list reproduces C1's numbers exactly. Rows with other words are left out.
    needed = all_words()
    spaces = guesser_spaces(needed)
    covered = set(needed)
    for _, _, have in spaces.values():
        if have is not None:
            covered &= have

    def ok(t: Picks) -> bool:
        return t.clue in covered and all(w in covered for w in t.words)

    single = [t for t in every if len(t.picks) == 1 and t.split == "test"]
    multi = [t for t in every if len(t.picks) > 1 and t.split == "test" and ok(t)]
    index, counts = cluster_draws([t.game for t in single])
    person = [t.words.index(t.picks[0]) for t in single]
    pos = {id(t): k for k, t in enumerate(single)}

    def first(t):
        return person[pos[id(t)]]

    def luce(sp: Space, tau: float):
        return lambda t: luce_weights(sp.sims(t.clue, t.words), tau)[0]

    guessers: dict = {}
    multis: dict = {}
    top_pickers: dict = {}
    for name, (sp, tau, _) in spaces.items():
        guessers[name] = summary([score_turns(luce(sp, tau), single, first)], index, counts)
        guessers[name] |= {"tau": tau, "template": sp.name.split(":")[1]}
        multis[name] = multi_pick_scores(luce(sp, tau), multi)
        top_pickers[name] = lambda clue, ws, sp=sp: ws[int(np.argmax(sp.sims(clue, ws)))]
        print(name, json.dumps({k: guessers[name][k] for k in ("hit", "ece")}), flush=True)
    eps = load("calibrate-game")["eat"]["eps"]
    eat = norm_weigher(eat_counts(), eps)

    def eat_weigh(t):
        return eat(t.clue, t.words)

    guessers["EAT counts"] = summary([score_turns(eat_weigh, single, first)], index, counts)
    guessers["EAT counts"]["eps"] = eps
    multis["EAT counts"] = multi_pick_scores(eat_weigh, multi)
    for model, template in MAPS:
        sp = space(model, template, "bare", all_words())
        runs, multi_runs = [], []
        for seed in SEEDS:
            a, _ = train(sp, items(sp, seed, eat=False, split="train", scheme="game"), 3, seed=seed)
            tuned = adapted(sp, a, f"{model}+map")
            tau = max(TAUS, key=lambda t, tuned=tuned: guess_loglik(tuned, "train", t, "game"))
            runs.append(score_turns(luce(tuned, tau), single, first))
            multi_runs.append(multi_pick_scores(luce(tuned, tau), multi))
            print(short(model), "map", seed, tau, round(float(runs[-1]["hit"].mean()), 3))
        name = f"{short(model)} + map"
        guessers[name] = summary(runs, index, counts) | {"seeds": list(SEEDS)}
        multis[name] = {
            k: float(np.mean([r[k] for r in multi_runs])) for k in multi_runs[0] if k != "turns"
        } | {"turns": len(multi), "seeds": list(SEEDS)}
    pairs_all = shared_pairs(every)
    pairs_cov = [(i, j) for i, j in pairs_all if ok(every[i]) and ok(every[j])]
    agree = np.array([every[i].picks[0] == every[j].picks[0] for i, j in pairs_cov], float)
    pidx, pcounts = clue_draws(every, pairs_cov)
    a = float(agree.mean())
    people = {
        "pairs": len(pairs_cov),
        "pairs_before_vector_coverage": len(pairs_all),
        "clues": len({every[i].clue for i, _ in pairs_cov}),
        "rows_in_pairs": len({k for p in pairs_cov for k in p}),
        "mean_shared_words": float(
            np.mean([len(shared_words(every[i], every[j])) for i, j in pairs_cov])
        ),
        "agreement": {"mean": a, "ci": interval(replicates(agree, pidx, pcounts))},
        "best_top1_bounds": agreement_bounds(a),
        "chance": float(np.mean([1 / len(shared_words(every[i], every[j])) for i, j in pairs_cov])),
        "guessers": {},
    }
    for name, pick in top_pickers.items():
        m = pair_scores(pick, every, pairs_cov)
        people["guessers"][name] = {
            "mean": float(m.mean()),
            "ci": interval(replicates(m, pidx, pcounts)),
            "minus_people": {
                "mean": float((m - agree).mean()),
                "ci": interval(replicates(m - agree, pidx, pcounts)),
            },
        }
    multi_all = [t for t in every if len(t.picks) > 1]
    return {
        "what": "calibration of guessers on C1's test turns, and agreement between people",
        "label": "exploratory: added after the test turns were scored for top-1",
        "split": "game (experiments/plans/C1.md)",
        "bootstrap": f"95% percentile, {DRAWS:,} resamples of test games (pairs: of clues), seed 0",
        "bins": BINS,
        "rows": {
            "all_kept": len(every),
            "one_pick": sum(len(t.picks) == 1 for t in every),
            "several_picks": len(multi_all),
            "several_picks_by_count": {
                str(k): sum(len(t.picks) == k for t in multi_all)
                for k in sorted({len(t.picks) for t in multi_all})
            },
            "test_one_pick": len(single),
            "test_several_picks_with_vectors": len(multi),
            "test_several_picks": sum(t.split == "test" for t in multi_all),
        },
        "guessers": guessers,
        "several_picks": multis,
        "exact_repeats": exact_repeats(every),
        "people": people,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "guesser-calibration.json")
    args = parser.parse_args(argv)
    if not (COMPUTED / "google--embeddinggemma-2__bare.npz").exists():
        print("the remote models' vectors are missing; see the README's Models section")
    out = run()
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
