"""Scoring for C10 (experiments/plans/C10.md): the validation choices, the clues for the
language-model guessers, and every number in results/c10.json.

Every action of a board's 30-clue list (codenames.actions) is scored as in C6
(codenames.c6score): by its exact expected turn value under the seven strangers, the five
potion maps of C1 (the human-fitted guesser, their mean), the EAT guesser and the
proposer. The listener's value of the 4K kept actions comes from its rankings
(codenames.listen). On each board:

- F, the first choice: the action the proposer values most, over the 120 (as in C6);
- L: the kept action the listener values most;
- P and LP, in task s (one of the seven strangers held out): the kept action with the
  highest mean value over the other six, or w times the listener's value plus 1 - w
  times that mean;
- C: Qwen3.8-27B's choice among the 120 (C6's chooser), and S: gpt-oss-120b's choice among
  the K kept clues, each the first choice when the reply named no action;
- the oracle at K (the kept action best for the strangers' mean) and uniform (the mean
  over the kept actions).

A single pick's stranger value is the mean over the seven. For P and LP it is the mean
over the seven tasks of the held-out stranger's value for that task's pick; their other
values are those of the task that holds out potion. Argmaxes take the first action in
prefilter order, clue by clue, counts 1 to 4.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from codenames.actions import COUNTS, expand, read_action_lines, sha256
from codenames.c4 import boards, clue_lines
from codenames.c4score import bootstrap, holm, rounded, score_crossguess
from codenames.c6score import ratio_bootstrap
from codenames.listen import (
    ACTIONS,
    FAILED,
    IN_ORDER,
    KS,
    LISTS,
    NAMED_NONE,
    STOPPED,
    TEST,
    VAL,
    WS,
    argmax_first,
    kept_index,
    listener_values,
    load_temperature,
    pick_mixture,
    pick_population,
    read_cell,
)
from codenames.rerank import action_index, read_choice_lines

RESULTS = Path(__file__).parent / "results"
CHOOSER = "qwen3.8-27b"
SPEAKER = "speaker"
HELD_OUT_HUMAN = "potion"  # the task whose picks the human-fitted guesser scores
REPEAT_BOARDS = 10
# Names of the arms in the clue lines for the language-model guessers.
GIVERS = {
    "F": "first-choice",
    "L": "listener",
    "P": "population",
    "LP": "listener-population",
    "C": CHOOSER,
    "S": SPEAKER,
}


def read_rankings(text: str) -> dict[tuple[int, int, int], list[str]]:
    """{(seed, board, repeat): cells} from rankings.txt (codenames.listen)."""
    out = {}
    for line in text.splitlines():
        seed, board, rep, *cells = line.split()
        key = (int(seed), int(board), int(rep))
        assert key not in out, key
        out[key] = cells
    return out


@cache
def scorers():
    from codenames.actions import PROPOSER
    from codenames.c6score import guessers

    weighers, strangers, maps, spaces = guessers()
    return weighers, strangers, maps, spaces, PROPOSER


class BoardData:
    """One board: its 30 clues, its kept clues, and every guesser's value of the 120."""

    def __init__(self, board, clues: list[str], kept: list[str]):
        from codenames.actions import check
        from codenames.actions import values as action_values

        weighers, strangers, maps, spaces, proposer = scorers()
        acts = expand(clues)
        check(board, acts, spaces)
        assert set(kept) <= set(clues), "a kept clue is not in the 30"
        self.board, self.clues, self.kept, self.acts = board, clues, kept, acts
        v = {g: action_values(w, board, acts) for g, w in weighers.items()}
        self.proposer = v[proposer]
        self.strangers = np.stack([v[g] for g in strangers])  # (7, 120)
        self.human = np.mean([v[g] for g in maps], axis=0)
        self.eat = v["eat"]
        self.potion = strangers.index(HELD_OUT_HUMAN)
        self.first = argmax_first(self.proposer)

    def kidx(self, k: int) -> list[int]:
        return kept_index(self.clues, self.kept[:k])

    def listener(self, cells: list[str], k: int, temp: float) -> np.ndarray:
        """The listener's value of each kept action, in prefilter order."""
        words = self.board.unrevealed()
        prefixes = {c: read_cell(x, words) for c, x in zip(self.kept[:k], cells[:k], strict=True)}
        return listener_values(self.board, self.kept[:k], prefixes, temp)

    def picks(self, lv: np.ndarray, k: int, w: float) -> dict:
        """Index in the 120 of each arm's pick; P and LP are lists over the seven tasks."""
        idx = np.array(self.kidx(k))
        s = self.strangers[:, idx]
        tasks = range(len(self.strangers))
        return {
            "F": self.first,
            "L": int(idx[argmax_first(lv)]),
            "P": [int(idx[pick_population(s, t)]) for t in tasks],
            "LP": [int(idx[pick_mixture(lv, s, t, w)]) for t in tasks],
            "oracle": int(idx[argmax_first(s.mean(axis=0))]),
        }

    def stranger_value(self, pick) -> float:
        if isinstance(pick, list):
            return float(np.mean([self.strangers[t, j] for t, j in enumerate(pick)]))
        return float(self.strangers[:, pick].mean())

    def human_pick(self, pick) -> int:
        return pick[self.potion] if isinstance(pick, list) else pick

    def values(self, pick) -> dict:
        j = self.human_pick(pick)
        return {
            "strangers": self.stranger_value(pick),
            "human": float(self.human[j]),
            "eat": float(self.eat[j]),
            "proposer": float(self.proposer[j]),
        }

    def uniform(self, k: int) -> dict:
        idx = self.kidx(k)
        return {
            "strangers": float(self.strangers[:, idx].mean()),
            "human": float(self.human[idx].mean()),
            "eat": float(self.eat[idx].mean()),
            "proposer": float(self.proposer[idx].mean()),
        }

    def rank(self, j: int) -> int:
        """1 + the number of the 120 actions the proposer values strictly above j."""
        return int((self.proposer > self.proposer[j]).sum()) + 1


def board_data(seed: int, n: int) -> list[BoardData]:
    clues = read_action_lines(ACTIONS.read_text())[seed]
    kept = read_action_lines(LISTS.read_text())[seed]
    return [BoardData(b, clues[i], kept[i]) for i, b in enumerate(boards(seed, n))]


def cell_counts(cells: list[str]) -> dict:
    marks = {IN_ORDER: "in_prompt_order", NAMED_NONE: "named_no_board_word"}
    marks |= {FAILED: "failed", STOPPED: "stopped"}
    out: dict = {v: sum(c == m for c in cells) for m, v in marks.items()}
    out["prefix"] = sum(c not in marks for c in cells)
    out["replies"] = len(cells)
    prefix = [0 if c in marks else len(c) for c in cells]
    out["prefix_length_mean"] = float(np.mean(prefix))
    out["prefix_length_median"] = float(np.median(prefix))
    return out


# Validation: K, then w.


def select(runs: Path, out: Path) -> dict:
    """Choose K and w on the validation boards, as the plan orders, and write them."""
    text = (runs / "c10-rankings.txt").read_text()
    rk = read_rankings(text)
    n = 1 + max(b for s, b, r in rk if s == VAL and r == 0)
    temp = load_temperature()
    data = board_data(VAL, n)
    cells = [rk[(VAL, i, 0)] for i in range(n)]
    assert all(len(c) == 20 for c in cells)
    curves: dict = {"L": {}, "F": {}, "oracle": {}, "uniform": {}, "P": {}}
    lv = {k: [d.listener(c, k, temp) for d, c in zip(data, cells, strict=True)] for k in KS}
    by_k = {k: [d.picks(lv[k][i], k, WS[0]) for i, d in enumerate(data)] for k in KS}
    for k in KS:
        for arm in ("L", "F", "oracle", "P"):
            x = [d.stranger_value(p[arm]) for d, p in zip(data, by_k[k], strict=True)]
            curves[arm][k] = bootstrap(x)
        curves["uniform"][k] = bootstrap([d.uniform(k)["strangers"] for d in data])
    best_k = max(KS, key=lambda k: (curves["L"][k]["mean"], -k))  # ties to the smaller K
    lp = {}
    for w in WS:
        ps = [d.picks(lv[best_k][i], best_k, w) for i, d in enumerate(data)]
        lp[w] = bootstrap([d.stranger_value(p["LP"]) for d, p in zip(data, ps, strict=True)])
    best_w = max(WS, key=lambda w: (lp[w]["mean"], -w))  # ties to the smaller w
    all_cells = [c for i in range(n) for c in rk[(VAL, i, 0)]]
    counts = cell_counts(all_cells)
    result = {
        "plan": "experiments/plans/C10.md",
        "seed": VAL,
        "boards": n,
        "T": temp,
        "rankings_sha256": sha256(text),
        "lists_sha256": sha256(LISTS.read_text()),
        "K": best_k,
        "w": best_w,
        "w_name": {WS[0]: "1/7", WS[1]: "1/4", WS[2]: "1/2", WS[3]: "3/4"}[best_w],
        "rule": "K: highest mean stranger value of L, ties to the smaller K; then w: highest "
        "mean stranger value of LP at that K, ties to the smaller w",
        "strangers": {
            "L_by_K": curves["L"],
            "F_by_K": curves["F"],
            "oracle_by_K": curves["oracle"],
            "uniform_by_K": curves["uniform"],
            "P_by_K": curves["P"],
            "LP_by_w_at_K": {str(round(w, 4)): lp[w] for w in WS},
            "gap_oracle_minus_F_by_K": {
                k: curves["oracle"][k]["mean"] - curves["F"][k]["mean"] for k in KS
            },
        },
        "replies": counts,
        "stop_rule_named_none_share": counts["named_no_board_word"] / counts["replies"],
        "stop_rule_holds": counts["named_no_board_word"] / counts["replies"] <= 0.05,
        "repeat": repeat_check(rk, data, temp),
    }
    out.write_text(json.dumps(rounded(result, 5), indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("K", "w", "T", "stop_rule_holds")}))
    return result


def repeat_check(rk, data, temp) -> dict:
    """The first boards' 20 prompts, sent twice more: prompts whose prefix changed, and
    boards whose L at K = 20 changed, in either repeat."""
    boards_ = [i for i in range(REPEAT_BOARDS) if (VAL, i, 1) in rk]
    if not boards_:
        return {}
    changed_prefix = changed_pick = 0
    for i in boards_:
        first = rk[(VAL, i, 0)]
        reps = [rk[(VAL, i, r)] for r in (1, 2)]
        for j in range(len(first)):
            changed_prefix += any(rep[j] != first[j] for rep in reps)
        pick = argmax_first(data[i].listener(first, 20, temp))
        changed_pick += any(argmax_first(data[i].listener(rep, 20, temp)) != pick for rep in reps)
    return {
        "boards": len(boards_),
        "prompts": 20 * len(boards_),
        "times": 2,
        "prefix_changed": changed_prefix,
        "pick_changed_at_K20": changed_pick,
    }


# The test.


def test_rows(runs: Path) -> tuple[list[dict], list[BoardData], dict]:
    """Per board of the test: every arm's pick (index in the 120) and the listener's values."""
    val = json.loads((RESULTS / "c10-val.json").read_text())
    k, w, temp = val["K"], val["w"], load_temperature()
    rk = read_rankings((runs / "c10-rankings.txt").read_text())
    n = 1 + max(b for s, b, r in rk if s == TEST)
    chosen = read_choice_lines((runs / "c10-choices.txt").read_text())
    data = board_data(TEST, n)
    rows = []
    for i, d in enumerate(data):
        cells = rk[(TEST, i, 0)]
        assert len(cells) == k, (i, len(cells), k)
        lv = d.listener(cells, k, temp)
        p = d.picks(lv, k, w)
        why = {}
        for arm, name, clues in (("C", CHOOSER, d.clues), ("S", SPEAKER, d.kept[:k])):
            if (name, TEST) not in chosen:
                continue
            row = chosen[(name, TEST)][i]
            j = action_index(row, clues)
            if j is not None and arm == "S":
                j = d.clues.index(row["clue"]) * len(COUNTS) + row["count"] - 1
            p[arm] = d.first if j is None else j
            if j is None:
                why[arm] = row["why"]
        sens = {}
        for label, t in (("half", temp / 2), ("double", temp * 2)):
            sens[label] = int(np.array(d.kidx(k))[argmax_first(d.listener(cells, k, t))])
        rows.append(
            {"board": i, "cells": cells, "lv": lv, "picks": p, "fallback": why, "sens": sens}
        )
    return rows, data, {"K": k, "w": w, "T": temp, "n": n}


def arm_values(rows, data, arm: str, measure: str) -> np.ndarray:
    if arm == "uniform":
        return np.array(
            [d.uniform(len(r["cells"]))[measure] for r, d in zip(rows, data, strict=True)]
        )
    return np.array([d.values(r["picks"][arm])[measure] for r, d in zip(rows, data, strict=True)])


def listener_self(r, d, pick) -> float | None:
    """The listener's own value of a pick, or None for a pick outside the kept clues."""
    j = d.human_pick(pick)
    idx = d.kidx(len(r["cells"]))
    return float(r["lv"][idx.index(j)]) if j in idx else None


def behavior(rows, data, arm: str) -> dict:
    picks = [d.human_pick(r["picks"][arm]) for r, d in zip(rows, data, strict=True)]
    acts = [d.acts[j] for j, d in zip(picks, data, strict=True)]
    own = [listener_self(r, d, r["picks"][arm]) for r, d in zip(rows, data, strict=True)]
    known = [x for x in own if x is not None]
    out = {
        "is_first": float(np.mean([j == d.first for j, d in zip(picks, data, strict=True)])),
        "mean_count": float(np.mean([n for _, n in acts])),
        "rank_median": float(np.median([d.rank(j) for j, d in zip(picks, data, strict=True)])),
        "distinct_clues": len({c for c, _ in acts}),
        "listener_value": float(np.mean(known)) if known else None,
        "listener_value_boards": len(known),
    }
    if arm in ("P", "LP"):
        out["note"] = "the pick of the task that holds out potion"
    return out


def score(runs: Path, out: Path) -> dict:
    rows, data, cfg = test_rows(runs)
    arms = ["F", "L", "P", "LP", "oracle"] + [a for a in ("C", "S") if a in rows[0]["picks"]]
    measures = ("strangers", "human", "eat")
    col = {(a, m): arm_values(rows, data, a, m) for a in [*arms, "uniform"] for m in measures}
    tests = {
        "h_listen_strangers": col[("L", "strangers")] - col[("F", "strangers")],
        "h_listen_human": col[("L", "human")] - col[("F", "human")],
        "h_add_strangers": col[("LP", "strangers")] - col[("P", "strangers")],
        "h_add_human": col[("LP", "human")] - col[("P", "human")],
    }
    primary = {k: bootstrap(d) for k, d in tests.items()}
    reject = holm({k: float(t["p"]) for k, t in primary.items()})
    for k, t in primary.items():
        t["holm_rejects"] = bool(reject[k])
        t["holds"] = bool(reject[k] and t["mean"] > 0)
    pairs = [("L", "C"), ("L", "S"), ("LP", "F"), ("P", "F"), ("LP", "C"), ("C", "F")]
    pairs += [("S", "F"), ("oracle", "F"), ("uniform", "F")]
    secondary: dict = {}
    for a, b in pairs:
        if (a, "strangers") in col and (b, "strangers") in col:
            secondary[f"{a}_minus_{b}"] = {
                m: bootstrap(col[(a, m)] - col[(b, m)]) for m in measures
            }
    secondary["room"] = ratio_bootstrap(
        col[("L", "strangers")] - col[("F", "strangers")],
        col[("oracle", "strangers")] - col[("F", "strangers")],
    )
    sens = {}
    for label in ("half", "double"):
        for m in ("strangers", "human"):
            v = np.array([d.values(r["sens"][label])[m] for r, d in zip(rows, data, strict=True)])
            sens[f"{label}_T_{m}_minus_F"] = bootstrap(v - col[("F", m)])
            sens[f"{label}_T_same_pick_as_L"] = float(
                np.mean([r["sens"][label] == r["picks"]["L"] for r in rows])
            )
    secondary["T_sensitivity"] = sens
    summary = {a: {m: bootstrap(col[(a, m)]) for m in measures} for a in [*arms, "uniform"]}
    result: dict = {
        "plan": "experiments/plans/C10.md",
        "seed": TEST,
        **cfg,
        "rankings_sha256": sha256((runs / "c10-rankings.txt").read_text()),
        "choices_sha256": sha256((runs / "c10-choices.txt").read_text()),
        "lists_sha256": sha256(LISTS.read_text()),
        "actions_sha256": sha256(ACTIONS.read_text()),
        "primary": primary,
        "arms": summary,
        "secondary": secondary,
        "behavior": {a: behavior(rows, data, a) for a in arms},
        "replies": cell_counts([c for r in rows for c in r["cells"]]),
        "chooser_fallbacks": {
            a: dict(sorted(_count(r["fallback"].get(a) for r in rows).items()))
            for a in ("C", "S")
            if a in rows[0]["picks"]
        },
        "listener_fit": json.loads((RESULTS / "c10-listener.json").read_text()),
        "validation": json.loads((RESULTS / "c10-val.json").read_text()),
    }
    cross = runs / "c10-crossguess.txt"
    if cross.exists():
        result["crossguess"] = score_crossguess(cross.read_text(), base=GIVERS["F"])
        result["crossguess"]["per_board"] = cross.read_text().splitlines()
    result["boards"] = [
        {
            "board": r["board"],
            "cells": " ".join(r["cells"]),
            "picks": {a: _act(d, r["picks"][a]) for a in arms},
            "value": {a: [d.values(r["picks"][a])[m] for m in measures] for a in arms},
            "uniform": [d.uniform(cfg["K"])[m] for m in measures],
            "fallback": r["fallback"],
        }
        for r, d in zip(rows, data, strict=True)
    ]
    out.write_text(json.dumps(rounded(result, 4), indent=1, sort_keys=True) + "\n")
    print(json.dumps(rounded(primary, 4), indent=1))
    return result


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        if x:
            out[x] = out.get(x, 0) + 1
    return out


def _act(d: BoardData, pick):
    if isinstance(pick, list):
        return [f"{c}/{n}" for c, n in (d.acts[j] for j in pick)]
    c, n = d.acts[pick]
    return f"{c}/{n}"


def picks(runs: Path, out: Path) -> str:
    """Clue lines (codenames.c4 format) for the language-model guessers: the six arms'
    picks on each test board, P and LP from the task that holds out potion."""
    rows, data, _ = test_rows(runs)
    given = {
        TEST: {
            name: [
                dict(zip(("clue", "count"), d.acts[d.human_pick(r["picks"][arm])], strict=True))
                for r, d in zip(rows, data, strict=True)
            ]
            for arm, name in GIVERS.items()
            if arm in rows[0]["picks"]
        }
    }
    text = clue_lines(given)
    out.write_text(text)
    print(out, sha256(text))
    return text
