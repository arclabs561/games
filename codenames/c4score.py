"""Scoring for C4's giving part: language-model clues heard by embedding guessers.

Each giver names one clue per board. Each of the seven calibrated embedding guessers
other than EmbeddingGemma 2 hears it, and the clue is worth the exact expected value of
the turn under that guesser's Luce model (codenames.luce), with the costs of
codenames.players.costs_for. A board's score is the mean over the seven. The baseline is
the EmbeddingGemma 2 expected-value giver with the whole clue-word list.

A reply with no legal clue gets the fallback: the clue the EmbeddingGemma 2 expected
giver names when it may only ask for one word (ExpectedGiver with nmax 1).

The comparison is paired by board. Intervals are 95% percentile bootstrap intervals over
boards, 10,000 resamples, seed 0 (experiments/METHOD.md).
"""

import json
from pathlib import Path

import numpy as np

from codenames.c4 import boards, read_clue_lines
from codenames.evaluate import all_words
from codenames.game import Board, Clue
from codenames.norms import eat_counts
from codenames.players import (
    ExpectedGiver,
    Space,
    embedding_weigher,
    expected_found,
    expected_value,
    norm_weigher,
    space,
)
from codenames.study import load, models
from codenames.words import legal

BASE = "google/embeddinggemma-2"
RESAMPLES = 10_000


def bootstrap(d, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """Mean of d, its 95% percentile interval over resampled units, and the bootstrap
    p-value: twice the share of resampled means on the other side of 0 from the mean."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (resamples, len(d)))].mean(axis=1)
    m = float(d.mean())
    wrong = float((means <= 0).mean() if m > 0 else (means >= 0).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": m, "ci": [float(lo), float(hi)], "p": min(1.0, 2 * wrong), "n": len(d)}


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down test: which hypotheses are rejected at family level alpha."""
    out, k = {}, len(ps)
    for i, (name, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
        if p > alpha / (k - i):
            out.update({n: False for n in ps if n not in out})
            break
        out[name] = True
    return out


def new_words(text: str) -> list[str]:
    """Legal clue words in a clues file that have no vectors yet."""
    have = set(all_words())
    clues = {c for rows in read_clue_lines(text).values() for c, _, _ in rows if c}
    return sorted(clues - have)


def spaces_with(extra: list[str]) -> dict[str, tuple[Space, float]]:
    """Each model's calibrated space and temperature, with vectors for the extra words."""
    cal = load("calibrate-game")
    words = all_words()
    words += [w for w in extra if w not in set(words)]
    return {
        m: (space(m, cal[m]["best"]["clue"], cal[m]["best"]["word"], words), cal[m]["best"]["tau"])
        for m in models()
    }


def score_clue(weighers, board: Board, clue: Clue) -> dict:
    values = {g: expected_value(w, board, clue) for g, w in weighers.items()}
    found = {g: expected_found(w, board, clue) for g, w in weighers.items()}
    return {"value": values, "found": found}


def score(clues_text: str, val_seed: int = 4005, test_seed: int = 4004) -> dict:
    """Every number for the giving part, from the clues file of the give job."""
    given = read_clue_lines(clues_text)
    llms = sorted({m for m, _ in given})
    seeds = sorted({s for _, s in given})
    extra = new_words(clues_text)
    spaces = spaces_with(extra)
    assert BASE in spaces and len(spaces) == 8, sorted(spaces)
    strangers = [m for m in spaces if m != BASE]
    weighers = {m: embedding_weigher(*spaces[m]) for m in strangers}
    weighers["eat"] = norm_weigher(eat_counts(), load("calibrate-game")["eat"]["eps"])
    base_sp, base_tau = spaces[BASE]
    baseline = ExpectedGiver(base_sp, base_tau)
    fallback = ExpectedGiver(base_sp, base_tau, nmax=1)
    vocab = set(all_words())
    out: dict = {"llms": llms, "strangers": strangers, "new_words": extra, "seeds": {}}
    for seed in seeds:
        n = len(given[(llms[0], seed)])
        bs = boards(seed, n)
        per: dict = {}
        base_clues = [baseline(b) for b in bs]
        fall_clues = [fallback(b) for b in bs]
        per["baseline"] = [
            {"clue": c.word, "count": c.count, **score_clue(weighers, b, c)}
            for b, c in zip(bs, base_clues, strict=True)
        ]
        for m in llms:
            rows = []
            for b, (word, count, why), fb in zip(bs, given[(m, seed)], fall_clues, strict=True):
                if word is not None and count is not None:
                    assert legal(word, b.unrevealed())
                    clue, used = Clue(word, count), False
                else:
                    clue, used = fb, True
                row = {"clue": clue.word, "count": clue.count, "fallback": why if used else None}
                row["new_word"] = (not used) and word not in vocab
                rows.append({**row, **score_clue(weighers, b, clue)})
            per[m] = rows
        out["seeds"][seed] = {"n": n, "boards": per, "summary": summarize(per, strangers)}
    if val_seed in out["seeds"] and test_seed in out["seeds"]:
        val = out["seeds"][val_seed]["summary"]["givers"]
        best = max(llms, key=lambda m: val[m]["value"]["mean"])
        test = out["seeds"][test_seed]["boards"]
        d = np.array(mean_value(test[best], strangers)) - np.array(
            mean_value(test["baseline"], strangers)
        )
        out["h_give"] = {
            "selected_on": val_seed,
            "best": best,
            "tested_on": test_seed,
            **bootstrap(d),
        }
    return out


def mean_value(rows: list[dict], strangers: list[str], key: str = "value") -> list[float]:
    """Per board: the mean over the stranger guessers."""
    return [float(np.mean([r[key][g] for g in strangers])) for r in rows]


def summarize(per: dict, strangers: list[str]) -> dict:
    base_v = np.array(mean_value(per["baseline"], strangers))
    givers = {}
    for name, rows in per.items():
        v = np.array(mean_value(rows, strangers))
        f = np.array(mean_value(rows, strangers, "found"))
        g: dict = {
            "value": bootstrap(v),
            "found": bootstrap(f),
            "eat_value": bootstrap([r["value"]["eat"] for r in rows]),
            "by_stranger": {s: float(np.mean([r["value"][s] for r in rows])) for s in strangers},
            "mean_count": float(np.mean([r["count"] for r in rows])),
        }
        if name != "baseline":
            used = [r["fallback"] for r in rows if r["fallback"]]
            ok = np.array([not r["fallback"] for r in rows])
            g["minus_baseline"] = bootstrap(v - base_v)
            g["fallbacks"] = {w: used.count(w) for w in sorted(set(used))}
            g["fallback_rate"] = len(used) / len(rows)
            legal_rows = [r["new_word"] for r in rows if not r["fallback"]]
            g["new_word_rate"] = float(np.mean(legal_rows)) if legal_rows else None
            # Two sensitivity checks on the fallback: a failed reply worth 0, and only the
            # boards where the reply was legal.
            g["minus_baseline_fallback_zero"] = bootstrap(np.where(ok, v, 0.0) - base_v)
            if ok.sum() > 1:
                g["minus_baseline_legal_only"] = bootstrap((v - base_v)[ok])
        givers[name] = g
    return {"givers": givers}


def main(clues: Path | None, guesses: Path | None, out: Path, cross: Path | None = None) -> None:
    """Write every number of C4 to out: the giving part from clues.txt and the guessing
    part from guesses.txt, each as the job wrote it."""
    result: dict = {"plan": "experiments/plans/C4.md"}
    if guesses is not None:
        result["guess"] = score_guess(guesses.read_text())
        result["guess"]["per_turn"] = guesses.read_text().splitlines()
    if clues is not None:
        result["give"] = score(clues.read_text())
    if cross is not None:
        result["crossguess"] = score_crossguess(cross.read_text())
    out.write_text(json.dumps(rounded(result), indent=1, sort_keys=True) + "\n")


def rounded(x, digits: int = 5):
    """The same structure with every float rounded, to keep the file small."""
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, digits) for v in x]
    return x


def read_guess_lines(text: str) -> dict[tuple[str, str], str]:
    """{(model, split): one character per turn} from guesses.txt (codenames.c4)."""
    out = {}
    for line in text.splitlines():
        name, split, chars = line.split()
        out[(name, split)] = chars
    return out


def clustered_p(d: np.ndarray, index: np.ndarray, counts: np.ndarray) -> float:
    """Bootstrap p-value of a paired difference: twice the share of game resamples on
    the other side of 0 from the mean."""
    from codenames.stats import replicates

    reps = replicates(d, index, counts)
    wrong = (reps <= 0).mean() if d.mean() > 0 else (reps >= 0).mean()
    return min(1.0, 2 * float(wrong))


def score_guess(text: str) -> dict:
    """Part 1: top-1 agreement with the person on C1's game split, for each model and
    for EmbeddingGemma 2 (bare template, as in C1), with game-bootstrap intervals; the
    model chosen on val; and H-guess on test, paired by turn."""
    from codenames.heldout import hits, sim_rows
    from codenames.split import turns
    from codenames.stats import cluster_draws, clustered

    lines = read_guess_lines(text)
    sp = space(BASE, "bare", "bare", all_words())
    out: dict = {"splits": {}}
    for split in sorted({s for _, s in lines}):
        data = turns(f"game:{split}")
        index, counts = cluster_draws([t.game for t in data])
        base = hits(sim_rows(sp, data))
        rows: dict = {"turns": len(data), "games": int(counts.shape[1])}
        rows["embeddinggemma-2"] = {"top1": clustered(base, index, counts)}
        for (name, s), chars in sorted(lines.items()):
            if s != split:
                continue
            assert len(chars) == len(data) and "-" not in chars, (name, split)
            h = np.array([c in "1b" for c in chars], dtype=float)
            d = h - base
            rows[name] = {
                "top1": clustered(h, index, counts),
                "minus_embeddinggemma": {
                    **clustered(d, index, counts),
                    "p": clustered_p(d, index, counts),
                },
                "named_no_board_word": sum(c in "ab" for c in chars),
            }
        out["splits"][split] = rows
    if "val" in out["splits"]:
        val = out["splits"]["val"]
        models_ = [k for k in val if isinstance(val[k], dict) and k != "embeddinggemma-2"]
        out["selected"] = max(models_, key=lambda m: val[m]["top1"]["mean"])
    if "test" in out["splits"] and "selected" in out:
        test = out["splits"]["test"][out["selected"]]["minus_embeddinggemma"]
        out["h_guess"] = {"model": out["selected"], **test}
    return out


def score_crossguess(text: str) -> dict:
    """Language-model guessers on other givers' clues: per giver and guesser, the found
    words and the turn value of one sampled ranking per board, with intervals over
    boards; and per giver, the mean over its guessers, paired with the embedding giver
    on the boards that both have."""
    by_pair: dict = {}
    for line in text.splitlines():
        giver, guesser, _seed, *cells = line.split()
        assert "-" not in cells, (giver, guesser)
        found, value = zip(*(c.split("/") for c in cells), strict=True)
        by_pair[(giver, guesser)] = (
            np.array(found, dtype=float),
            np.array(value, dtype=float),
        )
    out: dict = {"pairs": {}, "givers": {}}
    for (giver, guesser), (f, v) in sorted(by_pair.items()):
        out["pairs"][f"{giver} -> {guesser}"] = {"found": bootstrap(f), "value": bootstrap(v)}
    givers = sorted({g for g, _ in by_pair})
    for giver in givers:
        mine = [k for k in by_pair if k[0] == giver]
        v = np.mean([by_pair[k][1] for k in mine], axis=0)
        f = np.mean([by_pair[k][0] for k in mine], axis=0)
        row = {"guessers": [k[1] for k in mine], "found": bootstrap(f), "value": bootstrap(v)}
        if giver != "embeddinggemma-2":
            # The embedding giver's clues heard by the same guessers.
            base = np.mean([by_pair[("embeddinggemma-2", k[1])][1] for k in mine], axis=0)
            row["minus_embeddinggemma_same_guessers"] = bootstrap(v - base)
        out["givers"][giver] = row
    return out
