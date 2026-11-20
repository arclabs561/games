"""Diagnosis of single problems, board by board: clue legality, case files, slices and
disagreements between kinds of guesser. Everything here is exploratory (METHOD.md): it
reads committed results and does not change them.

    uv run python -m codenames.diagnose legality             # legality.json and .md
    uv run python -m codenames.diagnose refilter              # needs vectors, ~10 min
    uv run python -m codenames.diagnose case C10 17           # needs vectors
    uv run python -m codenames.diagnose case C11-pilot 4      # needs vectors
    uv run python -m codenames.diagnose slices                # needs vectors
    uv run python -m codenames.diagnose disagree              # needs vectors

Outputs go to codenames/results/diagnostics/. Intervals are 95% percentile bootstrap
intervals over boards, 10,000 resamples, seed 0 (`c4score.bootstrap`); a difference
between two groups of boards resamples each group on its own (`two_group`).
"""

import argparse
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from codenames import diagnose_records as records
from codenames import legality as lg
from codenames.c4score import RESAMPLES, bootstrap, holm, rounded

HERE = Path(__file__).parent
OUT = HERE / "results" / "diagnostics"
LABELS = OUT / "legality-labels.tsv"
PILOT_BEFORE = OUT / "pilot-before-key.json"
PILOT_NOW = HERE / "results" / "human-pilot" / "key.json"
LEVELS = ("illegal", "borderline", "legal")
FAMILIES = ("strangers", "human", "eat", "llm")
FAMILY_NAMES = {
    "strangers": "seven embedding strangers (C10's mean over tasks)",
    "human": "human-fitted guesser (five potion maps)",
    "eat": "EAT association guesser",
    "llm": "Llama 3.3 70B and DeepSeek-V3.2, one played-out reply each",
}


# Statistics.


def two_group(a, b, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """mean(a) - mean(b) for two independent groups of units, with a percentile interval
    from resampling each group on its own, and the bootstrap p-value of c4score."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), (resamples, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), (resamples, len(b)))].mean(axis=1)
    reps = ma - mb
    m = float(a.mean() - b.mean())
    wrong = float((reps <= 0).mean() if m > 0 else (reps >= 0).mean())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "mean": m,
        "ci": [float(lo), float(hi)],
        "p": min(1.0, 2 * wrong),
        "n": [len(a), len(b)],
    }


def share_by_board(rows: list[dict], test) -> dict:
    """Share of rows passing test, with the interval over boards: each board counts once,
    with the share of its rows (several runs or tasks per board) that pass."""
    by: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by[r["unit"]].append(bool(test(r)))
    x = [float(np.mean(v)) for _, v in sorted(by.items())]
    out = bootstrap(x)
    out.pop("p")
    out["rows"] = len(rows)
    out["count"] = int(sum(test(r) for r in rows))
    return out


# Legality: the rulebook, the hand labels and the rates by arm.


def rulebook_check() -> list[dict]:
    out = []
    for clue, word, ok in lg.RULEBOOK:
        r = lg.relation(clue, word)
        strict = lg.level(r) == "legal"
        out.append(
            {
                "clue": clue,
                "word": word,
                "rulebook_legal": ok,
                "relation": r,
                "strict_legal": strict,
                "old_rule_legal": records_legal(clue, word),
                "agrees": strict == ok,
            }
        )
    return out


def records_legal(clue: str, word: str) -> bool:
    from codenames.words import legal

    return legal(clue, (word,))


def read_labels(path: Path = LABELS) -> dict[tuple[str, str], tuple[str, str]]:
    """{(clue, board word): (label, note)} from the hand-labeled file."""
    out = {}
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == ["clue", "word", "label", "note"], lines[0]
    for line in lines[1:]:
        clue, word, label, note = line.split("\t")
        assert label in LEVELS, line
        out[(clue, word)] = (label, note)
    return out


def all_rows() -> list[dict]:
    rows = records.committed()
    rows += records.pilot_units(PILOT_BEFORE, "C11-pilot-before")
    if PILOT_NOW.exists():
        rows += records.pilot_units(PILOT_NOW, "C11-pilot")
    return records.with_legality(rows)


def flagged_pairs(rows) -> dict[tuple[str, str], str]:
    """{(clue, board word): relation} over every flag in the rows."""
    out = {}
    for r in rows:
        for f in r["flags"] or []:
            if f["relation"] not in ("same", "substring"):
                out[(r["clue"], f["word"])] = f["relation"]
    return out


def label_summary(rows, labels) -> dict:
    """Hand labels of the flagged pairs, by relation and level, and the unflagged pairs
    that were labeled (those sharing three or more first letters)."""
    pairs = flagged_pairs(rows)
    missing = sorted(p for p in pairs if p not in labels)
    by_relation: dict = {}
    for rel in lg.RELATIONS[2:]:
        mine = [labels[p][0] for p, r in pairs.items() if r == rel and p in labels]
        by_relation[rel] = {x: mine.count(x) for x in LEVELS}
    by_level: dict = {}
    for lev in ("illegal", "borderline"):
        mine = [labels[p][0] for p, r in pairs.items() if lg.level(r) == lev and p in labels]
        c = Counter(mine)
        n = len(mine)
        by_level[lev] = {
            "pairs": n,
            **{x: c[x] for x in LEVELS},
            "precision_illegal": c["illegal"] / n if n else None,
            "precision_illegal_or_borderline": (c["illegal"] + c["borderline"]) / n if n else None,
        }
    unflagged = {p: v for p, v in labels.items() if p not in pairs}
    for p in unflagged:
        assert lg.relation(*p) is None, p
    misses = {x: sorted(f"{c}/{w}" for (c, w), v in unflagged.items() if v[0] == x) for x in LEVELS}
    return {
        "flagged_pairs": len(pairs),
        "unlabeled_flagged_pairs": [f"{c}/{w}" for c, w in missing],
        "by_relation": by_relation,
        "by_level": by_level,
        "unflagged_labeled": len(unflagged),
        "unflagged_misses": {x: misses[x] for x in ("illegal", "borderline")},
    }


def rates(rows) -> dict:
    """Share of each arm's picks that the strict check flags, by experiment and arm."""
    groups: dict = defaultdict(list)
    for r in rows:
        if r["clue"] is None:
            continue
        groups[(r["experiment"], r["arm"])].append(r)
        if r["experiment"] in ("C10",) and r["arm"] in ("P", "LP") and r["task"] == "potion":
            groups[(r["experiment"], r["arm"] + " (potion task)")].append(r)
    out: dict = {}
    for (exp, arm), rs in sorted(groups.items()):
        out.setdefault(exp, {})[arm] = {
            "illegal": share_by_board(rs, lambda r: r["legality"] == "illegal"),
            "borderline": share_by_board(rs, lambda r: r["legality"] == "borderline"),
            "boards": len({r["unit"] for r in rs}),
        }
    return out


# The C10 re-analysis: P - F by kind of guesser, on all boards and on clean ones.


def c10_table(rows) -> dict:
    """Per C10 test board: P - F under each family, and the levels of F's clue and of
    P's clues (worst over the seven tasks, and the potion task's alone)."""
    by = defaultdict(dict)
    for r in rows:
        if r["experiment"] != "C10":
            continue
        by[r["board_index"]].setdefault(r["arm"], []).append(r)
    n = len(by)
    d = {f: np.zeros(n) for f in FAMILIES}
    lev = {k: [] for k in ("F", "P_any", "P_potion")}
    rank = {x: i for i, x in enumerate(LEVELS)}
    for i in range(n):
        f = by[i]["F"][0]
        ps = by[i]["P"]
        p = next(r for r in ps if r["task"] == "potion")
        vf = {v["scorer"]: v["value"] for v in f["values"]}
        vp = {v["scorer"]: v["value"] for v in p["values"]}
        for fam in ("strangers", "human", "eat"):
            d[fam][i] = vp[fam] - vf[fam]
        d["llm"][i] = np.mean([vp[g] - vf[g] for g in records.C10_LLM])
        lev["F"].append(f["legality"])
        lev["P_any"].append(min((r["legality"] for r in ps), key=lambda x: rank[x]))
        lev["P_potion"].append(p["legality"])
    return {"d": d, "levels": lev, "n": n}


def clean_mask(levels: dict, allow: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [f in allow and p in allow for f, p in zip(levels["F"], levels["P_any"], strict=True)]
    )


def reanalysis(rows) -> dict:
    t = c10_table(rows)
    d = t["d"]
    clean = clean_mask(t["levels"], ("legal", "borderline"))
    strict = clean_mask(t["levels"], ("legal",))
    subsets = {"all": np.ones(t["n"], bool), "clean": clean, "strict_clean": strict}
    out: dict = {
        "boards": t["n"],
        "boards_clean": int(clean.sum()),
        "boards_strict_clean": int(strict.sum()),
        "flagged_by_arm": {
            "F_illegal": sum(x == "illegal" for x in t["levels"]["F"]),
            "P_any_task_illegal": sum(x == "illegal" for x in t["levels"]["P_any"]),
            "P_potion_task_illegal": sum(x == "illegal" for x in t["levels"]["P_potion"]),
        },
        "p_minus_f": {
            fam: {name: bootstrap(d[fam][m]) for name, m in subsets.items()} for fam in FAMILIES
        },
        "flagged_minus_clean": {fam: two_group(d[fam][~clean], d[fam][clean]) for fam in FAMILIES},
    }
    gap = {}
    for other in ("human", "eat", "llm"):
        g = d["strangers"] - d[other]
        gap[f"strangers_minus_{other}"] = {
            **{name: bootstrap(g[m]) for name, m in subsets.items()},
            "flagged_minus_clean": two_group(g[~clean], g[clean]),
        }
    out["embedding_advantage"] = gap
    return out


def pilot_flags(rows, experiment: str) -> dict:
    mine = [r for r in rows if r["experiment"] == experiment]
    out = {
        "items": len(mine),
        **{x: sum(r["legality"] == x for r in mine) for x in LEVELS},
        "boards_with_a_flag": len(
            {r["unit"] for r in mine if r["legality"] in ("illegal", "borderline")}
        ),
        "flagged": [
            {
                "board": r["unit"],
                "arm": r["arm"],
                "clue": r["clue"],
                "level": r["legality"],
                "flags": [f"{f['word']} ({f['relation']})" for f in r["flags"]],
            }
            for r in mine
            if r["legality"] != "legal"
        ],
    }
    return out


def legality_report() -> dict:
    rows = all_rows()
    labels = read_labels()
    result = {
        "rule": "codenames.legality.relation; levels illegal and borderline",
        "rulebook": rulebook_check(),
        "labels": label_summary(rows, labels),
        "rates": rates(rows),
        "pilot_before": pilot_flags(rows, "C11-pilot-before"),
        "pilot_now": pilot_flags(rows, "C11-pilot"),
        "reanalysis_c10": reanalysis(rows),
    }
    c3 = OUT / "c3-picks.json"
    if c3.exists():
        result["c3_regeneration"] = json.loads(c3.read_text())
    refilter = OUT / "refilter.json"
    if refilter.exists():
        rf = json.loads(refilter.read_text())
        rf["embedding_advantage"] = refilter_gaps(rf)
        rf.pop("per_board")
        result["refilter_c10"] = rf
    result["eat_coverage_c10"] = eat_coverage(rows)
    return rounded(result, 4)


def refilter_gaps(rf: dict) -> dict:
    """Per board, the strangers' P - F minus the human-fitted or EAT guesser's, after the
    strict filter and in C10, and the change."""
    out = {}
    for j, other in ((1, "human"), (2, "eat")):
        new = np.array(
            [
                (b["value"]["P"][0] - b["value"]["F"][0])
                - (b["value"]["P"][j] - b["value"]["F"][j])
                for b in rf["per_board"]
            ]
        )
        old = np.array(
            [
                (b["old"]["P"][0] - b["old"]["F"][0]) - (b["old"]["P"][j] - b["old"]["F"][j])
                for b in rf["per_board"]
            ]
        )
        out[f"strangers_minus_{other}"] = {
            "strict": bootstrap(new),
            "c10": bootstrap(old),
            "change": bootstrap(new - old),
        }
    return out


def eat_coverage(rows) -> dict:
    """Share of C10's F and P (potion task) clues that are EAT cues, and their mean EAT
    value, by level: a clue that is not a cue gets EAT's uniform guess."""
    from codenames.norms import eat_counts

    eat = eat_counts()
    out: dict = {}
    for arm in ("F", "P"):
        mine = [
            r
            for r in rows
            if r["experiment"] == "C10" and r["arm"] == arm and r["task"] in (None, "potion")
        ]
        for lev in ("illegal", "legal"):
            x = [r for r in mine if r["legality"] == lev]
            v = [next(a["value"] for a in r["values"] if a["scorer"] == "eat") for r in x]
            out[f"{arm}_{lev}"] = {
                "picks": len(x),
                "eat_cue": float(np.mean([r["clue"] in eat for r in x])),
                "eat_value": float(np.mean(v)),
            }
    return out


# The report.


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(b: dict, digits: int = 3) -> str:
    lo, hi = b["ci"]
    return f"{b['mean']:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def ci_pct(b: dict) -> str:
    lo, hi = b["ci"]
    return f"{pct(b['mean'])} ({pct(lo)} to {pct(hi)})"


ARM_NOTES = {
    ("C3", "potion/population"): "potion giver trained against the population",
    ("C3", "potion/self"): "potion giver trained against itself",
    ("C10", "F"): "first choice of EmbeddingGemma 2",
    ("C10", "P"): "population pick, all seven tasks",
    ("C10", "P (potion task)"): "population pick sent to the other guessers",
    ("C10", "oracle"): "best kept action for the strangers' mean",
    ("C6", "first"): "first choice of EmbeddingGemma 2",
    ("C6", "oracle_s"): "best of the 120 for the strangers",
    ("C6", "oracle_h"): "best of the 120 for the human-fitted guesser",
    ("C4", "embeddinggemma-2"): "the embedding giver",
}


def render_legality(d: dict) -> str:
    r10 = d["rates"]["C10"]
    ra = d["reanalysis_c10"]
    lab = d["labels"]["by_level"]
    pb, pn = d["pilot_before"], d["pilot_now"]
    out = []
    out += ["# Clues that are forms of a board word", ""]
    out += [
        "Exploratory. Written by `uv run python -m codenames.diagnose legality` from "
        "committed results; nothing earlier is changed.",
        "",
    ]
    out += ["## Findings", ""]
    out += [
        "- Every experiment here filtered clues with `words.legal`, which forbids a board word "
        "and a clue that contains one or is contained in one. That lets through forms of a "
        "board word (residency with RESIDENCE, teaching with TEACHER) and shared compound "
        "parts (railway with RAILROAD), which the rulebook forbids.",
        f"- Under a stricter check (`codenames/legality.py`), C10's population pick P is "
        "illegal (a form of a board word or a shared compound part) on "
        f"{ci_pct(r10['P (potion task)']['illegal'])} of test boards, "
        f"against {ci_pct(r10['F']['illegal'])} for the first choice F and "
        f"{ci_pct(r10['oracle']['illegal'])} for the oracle. Picks chosen by their value to "
        "embedding guessers break the rule most often.",
        f"- On the {ra['boards_clean']} of {ra['boards']} C10 boards where neither F nor any of "
        "P's seven picks is flagged illegal, P − F under the embedding strangers is "
        f"{ci(ra['p_minus_f']['strangers']['clean'])}, against "
        f"{ci(ra['p_minus_f']['strangers']['all'])} on all boards; under the two language "
        f"models it is {ci(ra['p_minus_f']['llm']['clean'])}, against "
        f"{ci(ra['p_minus_f']['llm']['all'])}. The embedding-only advantage (strangers minus "
        f"language models) is {ci(ra['embedding_advantage']['strangers_minus_llm']['clean'])} "
        f"on clean boards and {ci(ra['embedding_advantage']['strangers_minus_llm']['all'])} on "
        "all. Leaving those boards out does not shrink it beyond its interval.",
    ]
    if "refilter_c10" in d:
        rf = d["refilter_c10"]
        out += [
            "- With every arm choosing among strictly legal clues, P − F under the strangers "
            f"is {ci(rf['p_minus_f']['strangers'])} on all {rf['boards']} boards (C10: "
            f"{ci(rf['p_minus_f_c10']['strangers'])}); under the human-fitted guesser "
            f"{ci(rf['p_minus_f']['human'])} (C10: {ci(rf['p_minus_f_c10']['human'])}); under "
            f"EAT {ci(rf['p_minus_f']['eat'])} (C10: {ci(rf['p_minus_f_c10']['eat'])}). These "
            "are exact values; the language models were not asked about the new picks. So "
            "with legal clues the population's lead holds under the embedding and "
            "human-fitted guessers and appears under EAT: the advantage of the strangers over "
            f"EAT falls from {ci(rf['embedding_advantage']['strangers_minus_eat']['c10'])} to "
            f"{ci(rf['embedding_advantage']['strangers_minus_eat']['strict'])}. Part of EAT's "
            "change is coverage: in C10, "
            f"{pct(d['eat_coverage_c10']['P_illegal']['eat_cue'])} of P's illegal clues are EAT "
            f"cues, against {pct(d['eat_coverage_c10']['P_legal']['eat_cue'])} of its legal "
            "ones, and a clue that is not a cue gets EAT's uniform guess.",
        ]
    out += [
        f"- The C11 pilot's first materials had {pb['illegal']} illegal and {pb['borderline']} "
        f"borderline items of {pb['items']}, on {pb['boards_with_a_flag']} of 40 boards. They "
        f"were made again with the strict filter before anyone played; the new 80 items have "
        f"{pn['illegal'] + pn['borderline']} flags (experiments/plans/C11.md, amendment of "
        "2026-11-19).",
        "",
    ]
    out += ["## The rule", ""]
    out += [
        "From the rulebook (Czech Games Edition, English rules of July 2015, page 8, "
        "https://filemanager.czechgames.com/storage/files/codenames-2015/rules/codenames-rules-en.pdf, "
        "read 2026-11-19):",
        "",
        "> You can't say any form of a visible word on the table. Until BREAK is covered up "
        "by a card, you can't say break, broken, breakage, or breakdown.",
        ">",
        "> You can't say part of a compound word on the table. Until HORSESHOE is covered up, "
        "you can't say horse, shoe, unhorsed, or snowshoe.",
        "",
        'Under "Don\'t be too strict" it adds that island and even land are valid clues for '
        'ENGLAND, that sparrow is fine with ROW on the table, and that "if the opposing '
        'spymaster allows it, the clue is valid." Theatre and theater "are different forms '
        'of the same word."',
        "",
    ]
    out += ["## The check", ""]
    out += [
        "`legality.relation(clue, word)` names the strongest relation between a clue and a "
        "board word. Its levels and resources are in the module's docstring: WordNet 3.0 "
        "morphology and derivational pointers (WordNet 3.0 license), the Snowball English "
        "stemmer (BSD-3-Clause), compound splits into two WordNet words, and spelling rules. "
        "A clue is strictly legal when `words.legal` allows it and no board word has an "
        "illegal or borderline relation to it. The check keeps `words.legal`, so it is "
        "stricter than the rulebook in two of its examples: land for ENGLAND and sparrow for "
        "ROW are substrings and stay forbidden.",
        "",
    ]
    out += ["### The rulebook's examples", ""]
    out += [
        "| Clue | Board word | Rulebook | Relation | Strict check | Agrees |",
        "|---|---|---|---|---|---|",
    ]
    for x in d["rulebook"]:
        out.append(
            f"| {x['clue']} | {x['word'].upper()} | {'legal' if x['rulebook_legal'] else 'illegal'} "
            f"| {x['relation'] or '-'} | {'legal' if x['strict_legal'] else 'illegal'} "
            f"| {'yes' if x['agrees'] else 'no'} |"
        )
    out += [""]
    out += ["### Hand labels", ""]
    lab_all = d["labels"]
    out += [
        f"Every distinct flagged pair among the picks of C3, C4, C6, C7, C10 and the pilot "
        f"({lab_all['flagged_pairs']} pairs, relations other than same word and substring) "
        "was labeled by hand as illegal (a form of the board word, or a compound part, as the "
        "rule's examples read), borderline (a shared root or combining form, not a form of the "
        "same English word), or legal. The labels are in `legality-labels.tsv`, with a note "
        "on the hard cases. One reading choice matters: railway with RAILROAD, and restroom "
        "with BATHROOM, are labeled illegal, since they share a compound part as snowshoe "
        "shares one with HORSESHOE. A group that reads the rule more loosely would call them "
        "borderline.",
        "",
        "| Level the check gives | Pairs | Labeled illegal | Borderline | Legal | Precision (illegal) | Precision (illegal or borderline) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lev in ("illegal", "borderline"):
        x = lab[lev]
        out.append(
            f"| {lev} | {x['pairs']} | {x['illegal']} | {x['borderline']} | {x['legal']} "
            f"| {pct(x['precision_illegal'])} | {pct(x['precision_illegal_or_borderline'])} |"
        )
    out += [
        "",
        "By relation:",
        "",
        "| Relation | Illegal | Borderline | Legal |",
        "|---|---:|---:|---:|",
    ]
    for rel, x in lab_all["by_relation"].items():
        if sum(x.values()):
            out.append(f"| {rel} | {x['illegal']} | {x['borderline']} | {x['legal']} |")
    miss = lab_all["unflagged_misses"]
    out += [
        "",
        f"Misses. The {lab_all['unflagged_labeled']} unflagged pairs that share three or more "
        "first letters were labeled too. Among them the check missed "
        f"{len(miss['illegal'])} illegal pairs ({', '.join(miss['illegal'])}) and "
        f"{len(miss['borderline'])} borderline ones ({', '.join(miss['borderline'])}). Pairs "
        "that share fewer letters were not labeled, so recall is measured on that stratum "
        "only. Pertainyms such as lunar for MOON and episcopal for BISHOP are not flagged; "
        "the rule concerns forms of a word, not words of the same meaning.",
        "",
    ]
    out += ["## How often each arm's clue is flagged", ""]
    out += [
        "Share of picks at each level, with 95% intervals over boards (each board counts "
        "once, with the share of its runs or tasks). All of these clues pass `words.legal`.",
        "",
        "| Experiment | Arm | Boards | Picks | Illegal | Borderline | Note |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for exp in ("C3", "C4", "C6", "C7", "C10", "C11-pilot-before", "C11-pilot"):
        for arm, x in d["rates"].get(exp, {}).items():
            out.append(
                f"| {exp} | {arm} | {x['boards']} | {x['illegal']['rows']} | "
                f"{ci_pct(x['illegal'])} | {pct(x['borderline']['mean'])} | "
                f"{ARM_NOTES.get((exp, arm), '')} |"
            )
    if "c3_regeneration" in d:
        c3 = d["c3_regeneration"]
        out += [
            "",
            f"C3's clues are not committed; `codenames.diagnose_c3` regenerated them from the "
            f"committed weights. {c3['held_out_value_matches_c3']} of {c3['picks']} picks "
            f"have the held-out value C3 recorded (within {c3['tolerance']:g}).",
        ]
    out += ["", "## Exploratory re-analysis of C10: P − F by kind of guesser", ""]
    fb = ra["flagged_by_arm"]
    out += [
        f"C10's 300 test boards (seed 9004). A board is flagged when F's clue or one of P's "
        f"seven task picks is illegal: F on {fb['F_illegal']} boards, P (any task) on "
        f"{fb['P_any_task_illegal']}, P (potion task) on {fb['P_potion_task_illegal']}. "
        f"'Clean' boards have no such flag ({ra['boards_clean']}); 'strict clean' boards "
        f"have no borderline flag either ({ra['boards_strict_clean']}). The subsets were "
        "chosen after C10's results were known, and the boards are not a random half, so "
        "the comparison is descriptive. Words per turn, 95% bootstrap intervals over boards.",
        "",
        "| Scored by | All boards | Clean | Strict clean | Flagged − clean |",
        "|---|---|---|---|---|",
    ]
    for fam in FAMILIES:
        x = ra["p_minus_f"][fam]
        out.append(
            f"| {FAMILY_NAMES[fam]} | {ci(x['all'])} | {ci(x['clean'])} | "
            f"{ci(x['strict_clean'])} | {ci(ra['flagged_minus_clean'][fam])} |"
        )
    out += [
        "",
        "The embedding-only advantage, per board, strangers' P − F minus another family's:",
        "",
    ]
    out += [
        "| Difference | All boards | Clean | Strict clean | Flagged − clean |",
        "|---|---|---|---|---|",
    ]
    for k, x in ra["embedding_advantage"].items():
        name = (
            "strangers − "
            + {
                "strangers_minus_human": "human-fitted",
                "strangers_minus_eat": "EAT",
                "strangers_minus_llm": "language models",
            }[k]
        )
        out.append(
            f"| {name} | {ci(x['all'])} | {ci(x['clean'])} | {ci(x['strict_clean'])} "
            f"| {ci(x['flagged_minus_clean'])} |"
        )
    out += [
        "",
        "The language-model column is one sampled reply per clue, so its per-board values are "
        "noisy (C11's plan puts the standard deviation of a board's P − F near 1.1).",
        "",
    ]
    if "refilter_c10" in d:
        rf = d["refilter_c10"]
        out += ["## C10 re-picked with the strict filter", ""]
        out += [
            "`diagnose refilter` runs C10's giver again on its 300 test boards, with every arm "
            "choosing among strictly legal clues, and scores the new picks exactly. "
            f"F is unchanged on {rf['same_F_as_c10']} boards and P (first task) on "
            f"{rf['same_P_potion_as_c10']}.",
            "",
            "| Scored by | P − F, strict | P − F, C10 | Change | Oracle − F, strict |",
            "|---|---|---|---|---|",
        ]
        for m in ("strangers", "human", "eat"):
            out.append(
                f"| {FAMILY_NAMES[m]} | {ci(rf['p_minus_f'][m])} | {ci(rf['p_minus_f_c10'][m])} "
                f"| {ci(rf['change_in_p_minus_f'][m])} | {ci(rf['oracle_minus_f'][m])} |"
            )
        out += ["", "| Difference | Strict | C10 | Change |", "|---|---|---|---|"]
        for k, x in rf["embedding_advantage"].items():
            name = "strangers − " + ("human-fitted" if k.endswith("human") else "EAT")
            out.append(f"| {name} | {ci(x['strict'])} | {ci(x['c10'])} | {ci(x['change'])} |")
        cov = d["eat_coverage_c10"]
        out += [
            "",
            "EAT knows fewer of the illegal clues. In C10 (picks before the filter):",
            "",
            "| Picks | Number | EAT cues | Mean EAT value |",
            "|---|---:|---:|---:|",
        ]
        for k, x in cov.items():
            arm, lev = k.split("_")
            out.append(
                f"| {arm}, {lev} | {x['picks']} | {pct(x['eat_cue'])} | {x['eat_value']:.3f} |"
            )
        out += [""]
    out += ["## The C11 pilot", ""]
    out += [
        f"First materials ({pb['items']} items): {pb['illegal']} illegal, {pb['borderline']} "
        "borderline.",
        "",
        "| Board | Arm | Clue | Level | Related board words |",
        "|---|---|---|---|---|",
    ]
    for x in pb["flagged"]:
        out.append(
            f"| {x['board']} | {x['arm']} | {x['clue']} | {x['level']} | {', '.join(x['flags'])} |"
        )
    out += [
        "",
        f"Materials now committed ({pn['items']} items): {pn['illegal']} illegal, "
        f"{pn['borderline']} borderline.",
        "",
    ]
    out += ["## Rerunning", ""]
    out += [
        "```sh",
        "uv run python -m codenames.legality residency residence railway railroad",
        "uv run python -m codenames.diagnose_c3 --out codenames/results/diagnostics/c3-picks.txt  # vectors, ~10 min",
        "uv run python -m codenames.diagnose refilter                                          # vectors, ~15 min",
        "uv run python -m codenames.diagnose legality",
        "```",
    ]
    return "\n".join(out) + "\n"


def write_legality(out: Path = OUT) -> dict:
    result = legality_report()
    (out / "legality.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "legality.md").write_text(render_legality(result))
    print(out / "legality.json", out / "legality.md")
    return result


# Turn outcomes under one guesser: the exact value split into its parts.


def outcome(weights: np.ndarray, kinds: tuple[str, ...], count: int, costs: dict) -> dict:
    """Chances of each way a turn of up to count picks ends, the expected team words
    found and the value, which equals found - sum of cost(kind) * P(ends on kind)."""
    from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM, turn_odds

    n = min(count, kinds.count(TEAM))
    odds = turn_odds(weights, kinds, n)
    out = {
        "all_hit": float(odds.all_team[n][0]),
        "neutral": float(odds.stopped_by(NEUTRAL, n)[0]),
        "opponent": float(odds.stopped_by(OPPONENT, n)[0]),
        "assassin": float(odds.stopped_by(ASSASSIN, n)[0]),
        "found": float(odds.expected_team(n)[0]),
        "value": float(odds.value(n, costs)[0]),
    }
    hits = [float(x[0]) for x in odds.all_team[: n + 1]]
    out["hits_at_least"] = hits[1:]
    return out


def first_pick(weights: np.ndarray, words, top: int = 5) -> list[tuple[str, float]]:
    p = np.asarray(weights, dtype=float) / np.sum(weights)
    order = np.argsort(-p, kind="stable")[:top]
    return [(words[j], float(p[j])) for j in order]


# Case files.


@cache
def guessers():
    """{name: weigher} for the proposer, the seven strangers, the human-fitted maps and
    EAT (C10's scorers, codenames.c6score.guessers)."""
    from codenames.c10score import scorers

    weighers, strangers, maps, _, proposer = scorers()
    return weighers, list(strangers), list(maps), proposer


def short(name: str) -> str:
    return name.split("/")[-1]


def arms_of(experiment: str, board: int) -> tuple[list[dict], dict]:
    """The rows of one board of an experiment, and extra values (the language models')."""
    if experiment == "C10":
        rows = records.with_legality([r for r in records.c10_units() if r["board_index"] == board])
    elif experiment == "C11-pilot":
        rows = records.with_legality(
            [r for r in records.pilot_units(PILOT_NOW) if r["board_index"] == board]
        )
    else:
        raise SystemExit(f"case files are written for C10 and C11-pilot, not {experiment}")
    if not rows:
        raise SystemExit(f"{experiment} has no board {board}")
    return rows, {}


def check_case(rows, split_rows, strangers) -> None:
    """The values computed here equal those committed (C10's stranger mean of a single
    pick, its human-fitted and EAT values), to the 4 decimals they were rounded to."""
    got = {(g, a): o["value"] for g, a, o in split_rows}
    for r in rows:
        if r["task"] not in (None, "potion") or not r["values"]:
            continue
        want = {v["scorer"]: v["value"] for v in r["values"]}
        a = r["arm"]
        assert abs(got[("human-fitted (mean of maps)", a)] - want["human"]) < 2e-4, (a, "human")
        assert abs(got[("eat", a)] - want["eat"]) < 2e-4, (a, "eat")
        if r["task"] is None:
            mean = np.mean([got[(g, a)] for g in strangers])
            assert abs(mean - want["strangers"]) < 2e-4, (a, "strangers")


def case_text(experiment: str, board: int) -> str:
    from codenames.game import Board
    from codenames.players import costs_for

    rows, _ = arms_of(experiment, board)
    r0 = rows[0]
    b = Board(tuple(r0["words"]), tuple(r0["kinds"]))
    words = b.unrevealed()
    kinds = tuple(b.kind(w) for w in words)
    costs = costs_for(b)
    weighers, strangers, maps, proposer = guessers()
    order = [proposer, *strangers, "eat"]
    lines = [f"# {experiment}, board {r0['unit']}", ""]
    lines += ["Exploratory case file, written by `codenames.diagnose case`.", ""]
    lines += ["## Board", "", "T team, O opponent, N neutral, A assassin.", ""]
    grid = [f"{w} ({k})" for w, k in zip(b.words, b.kinds, strict=True)]
    lines += ["| | | | | |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(grid[i : i + 5]) + " |" for i in range(0, 25, 5)]
    lines += ["", "## Arms", ""]
    lines += ["| Arm | Task | Clue | Legality | Related board words |", "|---|---|---|---|---|"]
    acts = {}
    for r in rows:
        name = r["arm"] if r["task"] in (None, "potion") else None
        if name is not None:
            acts[name] = (r["clue"], r["count"])
        flags = ", ".join(f"{f['word']} ({f['relation']})" for f in r["flags"] or []) or "-"
        lines.append(
            f"| {r['arm']} | {short(r['task']) if r['task'] else '-'} | {r['clue']}/{r['count']} "
            f"| {r['legality']} | {flags} |"
        )
    lines += ["", "P and LP are the potion task's picks in the tables below.", ""]
    lines += ["## Exact turn value by guesser", ""]
    lines += [
        "Each cell is value = found − P(opponent) − "
        f"{costs['A']:g} × P(assassin); P(all) is the chance of finding all count words.",
        "",
    ]
    arm_names = sorted(acts)
    weights = {}
    for g in [*order, *maps]:
        weights[g] = {c: weighers[g](c, words) for c, _ in acts.values()}
    lines += ["| Guesser | " + " | ".join(arm_names) + " |", "|---|" + "---:|" * len(arm_names)]
    split_rows = []
    for g in [*order, "human-fitted (mean of maps)"]:
        cells = []
        for a in arm_names:
            c, n = acts[a]
            if g.startswith("human"):
                parts = [outcome(weights[m][c], kinds, n, costs) for m in maps]
                o = {
                    k: float(np.mean([p[k] for p in parts]))
                    for k in parts[0]
                    if k != "hits_at_least"
                }
            else:
                o = outcome(weights[g][c], kinds, n, costs)
            cells.append(f"{o['value']:.3f}")
            split_rows.append((g, a, o))
        lines.append(f"| {short(g)} | " + " | ".join(cells) + " |")
    check_case(rows, split_rows, strangers)
    if experiment == "C10":
        lines += [
            "",
            "Played out by language models (one reply per clue, from c10-crossguess.txt):",
            "",
        ]
        llm_rows = [r for r in rows if r["task"] in (None, "potion") and r["values"]]
        for r in llm_rows:
            vals = {v["scorer"]: v["value"] for v in r["values"]}
            have = [f"{g} {vals[g]:g}" for g in records.C10_LLM if g in vals]
            if have:
                lines.append(f"- {r['arm']}: " + ", ".join(have))
    lines += ["", "## How each turn ends", ""]
    lines += [
        "| Guesser | Arm | Clue | P(all) | P(neutral) | P(opponent) | P(assassin) | Found | Value |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g, a, o in split_rows:
        c, n = acts[a]
        lines.append(
            f"| {short(g)} | {a} | {c}/{n} | {o['all_hit']:.3f} | {o['neutral']:.3f} | "
            f"{o['opponent']:.3f} | {o['assassin']:.3f} | {o['found']:.3f} | {o['value']:.3f} |"
        )
    lines += ["", "## First pick of each guesser", ""]
    lines += ["The five most likely first picks, with their kinds.", ""]
    for a in arm_names:
        c, n = acts[a]
        lines += [f"**{a}: {c}/{n}**", ""]
        for g in order:
            top = first_pick(weights[g][c], words)
            kind = {w: b.kind(w) for w in words}
            lines.append(f"- {short(g)}: " + ", ".join(f"{w} ({kind[w]}) {p:.2f}" for w, p in top))
        lines.append("")
    if "F" in acts and "P" in acts:
        lines += ["## Where the guessers disagree about P and F", ""]
        better = {"P": [], "F": [], "tie": []}
        for g, a, o in split_rows:
            if a != "P":
                continue
            f = next(x for gg, aa, x in split_rows if gg == g and aa == "F")
            diff = o["value"] - f["value"]
            better["P" if diff > 1e-9 else "F" if diff < -1e-9 else "tie"].append(
                f"{short(g)} ({diff:+.3f})"
            )
        for k, v in better.items():
            lines.append(f"- {k} better: " + (", ".join(v) if v else "none"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# C10 again, with every arm choosing among strictly legal clues (exact scorers only).


def strict_picks(board, k: int = 10) -> tuple[dict, Any]:
    """F, P (one pick per task) and the oracle at K, as C10 picks them, from the 30 clues
    the proposer shortlists out of the strict pool; and the board's BoardData."""
    from codenames.actions import proposer, shortlist
    from codenames.c10score import BoardData
    from codenames.human_pilot import clue_pool
    from codenames.listen import argmax_first, pick_population, prefilter

    sp, tau = proposer()
    clues = shortlist(board, sp, tau, pool=clue_pool(board))
    data = BoardData(board, clues, prefilter(board, clues, k))
    idx = np.array(data.kidx(k))
    st = data.strangers[:, idx]
    picks = {
        "F": data.first,
        "P": [int(idx[pick_population(st, t)]) for t in range(len(st))],
        "oracle": int(idx[argmax_first(st.mean(axis=0))]),
    }
    return picks, data


def refilter(n: int = 300, out: Path = OUT) -> dict:
    """C10's test boards with the strict filter: per board the new picks and their exact
    values, and P - F against C10's committed values on the same boards."""
    from codenames.c4 import boards
    from codenames.listen import TEST

    committed = json.loads((HERE / "results" / "c10.json").read_text())
    k = committed["K"]
    measures = ("strangers", "human", "eat")
    rows: list[dict] = []
    for i, b in enumerate(boards(TEST, n)):
        picks, data = strict_picks(b, k)
        old = committed["boards"][i]
        rows.append(
            {
                "board": i,
                "picks": {
                    a: (
                        [f"{c}/{m}" for c, m in (data.acts[j] for j in p)]
                        if isinstance(p, list)
                        else "{}/{}".format(*data.acts[p])
                    )
                    for a, p in picks.items()
                },
                "value": {a: [data.values(p)[m] for m in measures] for a, p in picks.items()},
                "old": {a: old["value"][a] for a in ("F", "P", "oracle")},
            }
        )
        if i % 25 == 0:
            print(i, flush=True)
    col = {
        (a, m, w): np.array([r[w][a][j] for r in rows])
        for a in ("F", "P", "oracle")
        for j, m in enumerate(measures)
        for w in ("value", "old")
    }
    summary = {
        "boards": n,
        "K": k,
        "rule": "every arm chooses among codenames.legality.strict_legal clues",
        "p_minus_f": {
            m: bootstrap(col[("P", m, "value")] - col[("F", m, "value")]) for m in measures
        },
        "p_minus_f_c10": {
            m: bootstrap(col[("P", m, "old")] - col[("F", m, "old")]) for m in measures
        },
        "change_in_p_minus_f": {
            m: bootstrap(
                (col[("P", m, "value")] - col[("F", m, "value")])
                - (col[("P", m, "old")] - col[("F", m, "old")])
            )
            for m in measures
        },
        "oracle_minus_f": {
            m: bootstrap(col[("oracle", m, "value")] - col[("F", m, "value")]) for m in measures
        },
        "arm_means": {
            a: {m: float(col[(a, m, "value")].mean()) for m in measures}
            for a in ("F", "P", "oracle")
        },
        "arm_means_c10": {
            a: {m: float(col[(a, m, "old")].mean()) for m in measures} for a in ("F", "P", "oracle")
        },
        "same_F_as_c10": sum(
            r["picks"]["F"] == committed["boards"][r["board"]]["picks"]["F"] for r in rows
        ),
        "same_P_potion_as_c10": sum(
            r["picks"]["P"][0] == committed["boards"][r["board"]]["picks"]["P"][0] for r in rows
        ),
    }
    result = rounded({**summary, "per_board": rows}, 4)
    (out / "refilter.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rounded(summary, 4), indent=1))
    return result


# Slices: P - F on C10's test boards, split by properties fixed before they were computed.

SLICES = {
    "flagged": "F's clue or one of P's seven clues is illegal by the strict check",
    "p_rarer": "P's clue is rarer than F's (lower wordfreq Zipf frequency)",
    "p_lower_count": "P's count is below F's",
    "p_nearer_assassin": "P's clue is closer to the assassin than F's, by the proposer's cosine",
    "embedding_families_agree": "the strangers and the human-fitted guesser give P - F the same sign",
}
# Families tested in each slice: a slice defined by some families' values is tested only
# on the others.
SLICE_FAMILIES: dict[str, tuple[str, ...]] = {name: FAMILIES for name in SLICES}
SLICE_FAMILIES["embedding_families_agree"] = ("eat", "llm")


def slice_masks(rows) -> tuple[dict, dict]:
    """{slice: boolean mask over C10's test boards} and the per-board P - F by family."""
    from codenames.actions import proposer
    from codenames.words import zipf

    t = c10_table(rows)
    by = defaultdict(dict)
    for r in rows:
        if r["experiment"] == "C10" and r["task"] in (None, "potion") and r["arm"] in ("F", "P"):
            by[r["board_index"]][r["arm"]] = r
    sp, _ = proposer()
    near, rarer, lower = [], [], []
    for i in range(t["n"]):
        f, p = by[i]["F"], by[i]["P"]
        assassin = f["words"][f["kinds"].index("A")]
        sim = {a: float(sp.sims(r["clue"], (assassin,))[0]) for a, r in (("F", f), ("P", p))}
        near.append(sim["P"] > sim["F"])
        rarer.append(zipf(p["clue"]) < zipf(f["clue"]))
        lower.append(p["count"] < f["count"])
    d = t["d"]
    masks = {
        "flagged": ~clean_mask(t["levels"], ("legal", "borderline")),
        "p_rarer": np.array(rarer),
        "p_lower_count": np.array(lower),
        "p_nearer_assassin": np.array(near),
        "embedding_families_agree": np.sign(d["strangers"]) == np.sign(d["human"]),
    }
    return masks, d


def slices(out: Path = OUT) -> dict:
    rows = records.with_legality(records.c10_units())
    masks, d = slice_masks(rows)
    tests: dict[str, dict[str, Any]] = {}
    for name, m in masks.items():
        for fam in SLICE_FAMILIES[name]:
            tests[f"{name} | {fam}"] = {
                "in": bootstrap(d[fam][m]),
                "out": bootstrap(d[fam][~m]),
                "in_minus_out": two_group(d[fam][m], d[fam][~m]),
            }
    reject = holm({k: v["in_minus_out"]["p"] for k, v in tests.items()})
    for k, v in tests.items():
        v["holm_rejects"] = bool(reject[k])
    result = rounded(
        {
            "label": "exploratory",
            "boards": len(d["strangers"]),
            "slices": SLICES,
            "sizes": {k: int(m.sum()) for k, m in masks.items()},
            "tests": tests,
            "holm_family": len(tests),
        },
        4,
    )
    (out / "slices.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "slices.md").write_text(render_slices(result))
    print(out / "slices.json", out / "slices.md")
    return result


def render_slices(d: dict) -> str:
    out = ["# C10's P − F by slice", ""]
    out += [
        "Exploratory. Written by `uv run python -m codenames.diagnose slices`. The slices "
        "were fixed in `diagnose.SLICES` before they were computed, but after C10's results "
        f"were known. {d['boards']} test boards of seed 9004; words per turn. 'In' is the "
        "mean P − F on the boards in the slice, 'out' on the others; the difference has a "
        "95% interval from resampling each group, 10,000 resamples, seed 0. Holm's "
        f"correction is over all {d['holm_family']} tests at a family level of 0.05. The "
        "language-model values are one reply per clue.",
        "",
        "| Slice | Boards in it |",
        "|---|---:|",
    ]
    for k, text in sorted(d["slices"].items()):
        out.append(f"| {k}: {text} | {d['sizes'][k]} |")
    out += [
        "",
        "| Slice | Scored by | In | Out | In − out | p | Holm rejects |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for k, v in sorted(d["tests"].items()):
        name, fam = k.split(" | ")
        out.append(
            f"| {name} | {fam} | {v['in']['mean']:.3f} | {v['out']['mean']:.3f} | "
            f"{ci(v['in_minus_out'])} | {v['in_minus_out']['p']:.4f} | "
            f"{'yes' if v['holm_rejects'] else 'no'} |"
        )
    return "\n".join(out) + "\n"


def disagree(top: int = 10, out: Path = OUT) -> dict:
    """The C10 boards where the strangers and the language models differ most about
    P - F, with a case file for each of the first `top`."""
    rows = records.with_legality(records.c10_units())
    t = c10_table(rows)
    gap = t["d"]["strangers"] - t["d"]["llm"]
    order = sorted(range(t["n"]), key=lambda i: (-abs(gap[i]), i))
    listing = [
        {
            "board": i,
            "strangers": float(t["d"]["strangers"][i]),
            "human": float(t["d"]["human"][i]),
            "eat": float(t["d"]["eat"][i]),
            "llm": float(t["d"]["llm"][i]),
            "gap": float(gap[i]),
            "F": t["levels"]["F"][i],
            "P": t["levels"]["P_any"][i],
        }
        for i in order[: 2 * top]
    ]
    cases = out / "cases"
    cases.mkdir(exist_ok=True)
    for x in listing[:top]:
        (cases / f"c10-{x['board']:03d}.md").write_text(case_text("C10", int(x["board"])))
    result = rounded({"label": "exploratory", "top": listing, "cases": top}, 4)
    (out / "disagree.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "disagree.md").write_text(render_disagree(result))
    print(out / "disagree.json", out / "disagree.md")
    return result


def render_disagree(d: dict) -> str:
    out = ["# C10 boards where the strangers and the language models disagree most", ""]
    out += [
        "Exploratory. Written by `uv run python -m codenames.diagnose disagree`. Each row is "
        "a test board of seed 9004 and P − F under each kind of guesser, in words per turn; "
        "'gap' is the strangers' P − F minus the language models'. The language-model value "
        "is the mean of two single replies, so a large gap is partly the draw of a reply. "
        f"Case files for the first {d['cases']} are in `cases/`. The level is the worst over "
        "F's clue and P's seven.",
        "",
        "| Board | Strangers | Human-fitted | EAT | Language models | Gap | F | P | Case |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for i, x in enumerate(d["top"]):
        link = f"[c10-{x['board']:03d}](cases/c10-{x['board']:03d}.md)" if i < d["cases"] else ""
        out.append(
            f"| {x['board']} | {x['strangers']:.3f} | {x['human']:.3f} | {x['eat']:.3f} | "
            f"{x['llm']:.3f} | {x['gap']:.3f} | {x['F']} | {x['P']} | {link} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("legality")
    r = sub.add_parser("refilter")
    r.add_argument("--n", type=int, default=300)
    sub.add_parser("slices")
    d = sub.add_parser("disagree")
    d.add_argument("--top", type=int, default=10)
    c = sub.add_parser("case")
    c.add_argument("experiment")
    c.add_argument("board", type=int)
    c.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.what == "legality":
        write_legality()
    elif args.what == "refilter":
        refilter(args.n)
    elif args.what == "slices":
        slices()
    elif args.what == "disagree":
        disagree(args.top)
    elif args.what == "case":
        text = case_text(args.experiment, args.board)
        if args.out:
            args.out.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
