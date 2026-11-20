"""C4's giving part rescored under the rulebook's clue rule (plan C4, amendment of
2026-11-20). Exploratory: written after C4's results were read, and it makes no model call.

    python -m codenames.c4 strict --runs hf --out OUT      # in a cpu-basic job
    python -m codenames.c4 strict --runs DIR --out OUT     # DIR holds give-<model>-<seed>.json
    python -m codenames.c4 strict --runs played --out codenames/results   # committed files only

C4 judged a clue with `words.legal` (not a board word, not containing one, not contained
in one). This reads every raw giver reply on the test boards (seed 4004) and the
validation boards (seed 4005), parses it as C4 parsed it (`c4.read_clue`), and classifies
the word it names with `legality.worst` into legal, borderline and illegal. Then H-give,
DeepSeek-V3.2 (C4's selected giver) minus the embedding giver over the seven strangers, is
recomputed with a clue accepted only when `legality.strict_legal` allows it, the fallback
being EmbeddingGemma 2's best strictly legal clue for one word, and the embedding giver
restricted to strictly legal clues. The same difference is given for every model.

`legality.relation` counts a substring as illegal, so a reply that `words.legal` rejects
is always illegal under `legality.worst`. The rulebook allows some substrings (land for
ENGLAND, sparrow for ROW), so `beyond_substring` also gives each word's level with the
substring relation set aside, which is where such replies would show.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from codenames import c4
from codenames.game import Board, Clue

TEST, VAL = 4004, 4005
N = {TEST: 200, VAL: 50}
SELECTED = "deepseek-v3.2"  # C4's giver, chosen on its validation boards
DATASET = "arclabs561/games-runs"
LEVELS = ("legal", "borderline", "illegal")


def named_word(reply: str, board: Board) -> tuple[str | None, dict]:
    """C4's reading of a reply, and the word it names (the parsed clue, or the word it
    rejected), or None when the reply names none (empty or unparsed)."""
    read = c4.read_clue(reply, board)
    if read["clue"] is not None:
        return read["clue"], read
    if "said" in read:
        return read["said"][0], read
    return None, read


def level_beyond_substring(word: str, words) -> str:
    """The worst level of the word's relations to the board other than being the same
    word or a substring (legality.relation stops at the first relation it finds)."""
    from codenames import legality as lg

    worst = "legal"
    for w in words:
        if word == w:
            return "illegal"
        rel = (
            relation_without_substring(word, w) if word in w or w in word else lg.relation(word, w)
        )
        lv = lg.level(rel)
        if lv == "illegal":
            return "illegal"
        if lv == "borderline":
            worst = "borderline"
    return worst


def relation_without_substring(clue: str, word: str) -> str | None:
    """legality.relation with the substring test left out."""
    from codenames import legality as lg

    clue, word = clue.lower(), word.lower()
    if lg.bases(clue) & lg.bases(word):
        return "inflection"
    if (
        lg.family(clue) & lg.family(word)
        and lg.common_prefix(clue, word) >= lg.MIN_DERIVATION_PREFIX
    ):
        return "derivation"
    if lg.american(clue) == lg.american(word):
        return "variant"
    if lg.unprefixed(clue) & lg.parts(word) or lg.unprefixed(word) & lg.parts(clue):
        return "compound"
    part = lg.shared_part(clue, word, same_place=False)
    if part is not None and len(part) >= 4:
        return "compound"
    if lg.stem(clue) == lg.stem(word):
        return "stem"
    if lg.affix_stems(clue) & lg.affix_stems(word):
        return "affix"
    if part is not None:
        return "compound3"
    n = lg.common_prefix(clue, word)
    if n >= lg.MIN_PREFIX and n > min(len(clue), len(word)) / 2:
        return "prefix"
    return None


def classify(reply: str, board: Board) -> dict:
    """One reply: C4's verdict, the strict level and the level beyond substrings."""
    from codenames.legality import strict_legal, worst

    word, read = named_word(reply, board)
    if word is None:
        return {"c4": read["why"], "level": "no word", "beyond_substring": "no word", "word": None}
    return {
        "c4": "accepted" if read["clue"] is not None else read["why"],
        "level": worst(word, board.words),
        "beyond_substring": level_beyond_substring(word, board.words),
        "word": word,
        "count": read["clue"] and read["count"],
        "strict_legal": strict_legal(word, board.words) and read["clue"] is not None,
    }


def read_runs(folder: Path) -> dict[tuple[str, int], list[dict]]:
    """{(model, seed): rows of give-<model>-<seed>.json}, for C4's four models."""
    out = {}
    for m in c4.PLAN:
        for seed in (TEST, VAL):
            data = json.loads((folder / f"give-{m}-{seed}.json").read_text())
            assert data["seed"] == seed and data["n"] == N[seed], (m, seed)
            out[(m, seed)] = data["rows"]
    return out


def download() -> Path:
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        DATASET,
        repo_type="dataset",
        allow_patterns=[f"runs/c4/give/give-*-{s}.json" for s in (TEST, VAL)],
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(root) / "runs" / "c4" / "give"


def tables(runs: dict, deals: dict[int, list[Board]]) -> tuple[dict, dict]:
    """Per model and seed: shares at each level, and the cross-table of C4's verdict
    against the strict level; and every reply's classification."""
    counts: dict = {}
    per: dict = {}
    for (m, seed), rows in runs.items():
        cls = []
        for r in rows:
            assert "stopped" not in r, (m, seed, r["board"])
            cls.append(classify(r["reply"], deals[seed][r["board"]]))
        per[(m, seed)] = cls
        n = len(cls)
        level = Counter(c["level"] for c in cls)
        cross = Counter(f"{c['c4']} | {c['level']}" for c in cls)
        beyond = Counter(f"{c['c4']} | {c['beyond_substring']}" for c in cls)
        counts.setdefault(m, {})[str(seed)] = {
            "replies": n,
            "level_share": {k: level[k] / n for k in (*LEVELS, "no word")},
            "c4_by_level": dict(sorted(cross.items())),
            "c4_by_level_beyond_substring": dict(sorted(beyond.items())),
            "accepted_by_c4": sum(c["c4"] == "accepted" for c in cls),
            "accepted_strictly": sum(bool(c.get("strict_legal")) for c in cls),
        }
    return counts, per


def h_give(per: dict, deals: dict[int, list[Board]]) -> dict:
    """Every model minus the strict embedding giver on the test boards, mean exact value
    over the seven strangers, with strictly legal clues and the strict fallback."""
    from codenames.c4score import BASE, bootstrap, mean_value, score_clue, spaces_with
    from codenames.human_pilot import clue_pool
    from codenames.players import ExpectedGiver, embedding_weigher

    spaces = spaces_with([])
    strangers = [m for m in spaces if m != BASE]
    weighers = {m: embedding_weigher(*spaces[m]) for m in strangers}
    sp, tau = spaces[BASE]
    bs = deals[TEST]
    base_rows, fall = [], []
    for b in bs:
        pool = clue_pool(b)
        e = ExpectedGiver(sp, tau, candidates=pool)(b)
        f = ExpectedGiver(sp, tau, nmax=1, candidates=pool)(b)
        base_rows.append({"clue": e.word, "count": e.count, **score_clue(weighers, b, e)})
        fall.append(f)
    base_v = np.array(mean_value(base_rows, strangers))
    out: dict = {
        "strangers": strangers,
        "embedding_giver": {
            "value": bootstrap(base_v),
            "mean_count": float(np.mean([r["count"] for r in base_rows])),
        },
        "models": {},
    }
    for m in c4.PLAN:
        rows, used = [], 0
        for b, c, fb in zip(bs, per[(m, TEST)], fall, strict=True):
            if c.get("strict_legal"):
                clue = Clue(c["word"], c["count"])
            else:
                clue, used = fb, used + 1
            rows.append(score_clue(weighers, b, clue))
        v = np.array(mean_value(rows, strangers))
        out["models"][m] = {
            "fallback_rate": used / len(bs),
            "value": bootstrap(v),
            "minus_embedding_giver": bootstrap(v - base_v),
        }
    out["h_give_strict"] = {"model": SELECTED, **out["models"][SELECTED]["minus_embedding_giver"]}
    return out


def ensure_vectors() -> None:
    """In a job: copy the word vectors from the branch outputs/codenames-embed."""
    import shutil
    import subprocess
    import tempfile

    from codenames import embed

    if any(embed.COMPUTED.glob("*.npz")):
        return
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch", "outputs/codenames-embed",
         "https://github.com/arclabs561/games.git", str(tmp / "embed")],
        check=True,
    )  # fmt: skip
    embed.COMPUTED.mkdir(parents=True, exist_ok=True)
    for f in (tmp / "embed" / "codenames" / "vectors").glob("*.npz"):
        shutil.copy(f, embed.COMPUTED / f.name)


PLAYED = Path(__file__).parent / "results" / "c4-strict-played.json"


def played() -> dict:
    """The clue each model's test boards were played with in C4 (`results/c4.json`: the
    model's own accepted clue, or the fallback), by strict level and by whether it was the
    fallback. Reads only committed files."""
    from codenames.legality import worst

    give = json.loads((Path(__file__).parent / "results" / "c4.json").read_text())["give"]
    rows = give["seeds"][str(TEST)]["boards"]
    bs = c4.boards(TEST, N[TEST])
    out = {}
    for m in c4.PLAN:
        c = Counter(
            f"{'fallback' if r['fallback'] else 'own'} | {worst(r['clue'], b.words)}"
            for r, b in zip(rows[m], bs, strict=True)
        )
        out[m] = dict(sorted(c.items()))
    return {"seed": TEST, "boards": N[TEST], "played_clue_by_level": out}


def run(runs_dir: str, out: Path) -> dict:
    from codenames.c4score import rounded

    if runs_dir == "played":
        text = json.dumps(played(), indent=1, sort_keys=True) + "\n"
        out.mkdir(parents=True, exist_ok=True)
        (out / PLAYED.name).write_text(text)
        print(text)
        return json.loads(text)
    folder = download() if runs_dir == "hf" else Path(runs_dir)
    ensure_vectors()
    runs = read_runs(folder)
    deals = {s: c4.boards(s, N[s]) for s in (TEST, VAL)}
    counts, per = tables(runs, deals)
    result = {
        "plan": "experiments/plans/C4.md (amendment of 2026-11-20, exploratory)",
        "seeds": {"test": TEST, "val": VAL},
        "levels": counts,
        "h_give": h_give(per, deals),
        "bootstrap": {"resamples": 10_000, "seed": 0},
    }
    result = rounded(result)
    out.mkdir(parents=True, exist_ok=True)
    (out / "c4-strict.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result["h_give"]["h_give_strict"], indent=1))
    return result
