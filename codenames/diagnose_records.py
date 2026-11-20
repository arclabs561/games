"""Per-unit records: one row per evaluation unit, so that a single board can be examined.

A **unit** is what an experiment samples and pairs its arms on: in this folder, a board
(or a C1 turn). A **record** is one arm on one unit: the action it chose, the candidates
it chose from, every scorer's exact value of the action, each guesser's first-pick
distribution, and, for a language-model call, the hash of the raw reply, the provider,
the latency and the cost. Future runs write records as Parquet to the private dataset
(`runs/<experiment>/records/*.parquet`, see experiments/README.md); `load` reads those
files and also the results already committed here, which hold a subset of the fields.

    uv run python -m codenames.diagnose_records --out records.parquet   # committed results

The schema is `SCHEMA`; a field that a source does not hold is null. Values are in
words per turn, as everywhere in this folder.
"""

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DIAGNOSTICS = RESULTS / "diagnostics"

SCORE = "STRUCT(scorer VARCHAR, value DOUBLE)"
CANDIDATE = "STRUCT(clue VARCHAR, count INTEGER, score DOUBLE)"
GUESS = "STRUCT(scorer VARCHAR, word VARCHAR, p DOUBLE)"
FLAG = "STRUCT(word VARCHAR, relation VARCHAR)"
# (field, DuckDB type, meaning)
SCHEMA = (
    ("experiment", "VARCHAR", "plan id, such as C10"),
    ("split", "VARCHAR", "test, val, pilot or train"),
    ("arm", "VARCHAR", "the arm's name in its plan, such as F or P"),
    ("task", "VARCHAR", "the held-out guesser or task the pick belongs to, if any"),
    ("unit", "VARCHAR", "unit id: board seed and index as 'seed:index'"),
    ("board_seed", "INTEGER", "seed that dealt the board (c4.boards)"),
    ("board_index", "INTEGER", "the board's index among those of its seed"),
    ("train_seed", "INTEGER", "training seed of a learned arm"),
    ("call_seed", "INTEGER", "seed sent with a model call"),
    ("input_sha256", "VARCHAR", "SHA-256 of the input: the board text, or the prompt"),
    ("words", "VARCHAR[]", "the 25 board words"),
    ("kinds", "VARCHAR", "their kinds, one letter each (T, O, N, A)"),
    ("clue", "VARCHAR", "the chosen clue, null if none"),
    ("count", "INTEGER", "the chosen count"),
    ("why_none", "VARCHAR", "why there is no clue (empty, unparsed, illegal, ...)"),
    ("candidates", f"{CANDIDATE}[]", "top-k candidate actions with the chooser's score"),
    ("values", f"{SCORE}[]", "each scorer's exact (or played-out) value of the action"),
    ("guesses", f"{GUESS}[]", "each guesser's chance of picking each word first"),
    ("legality", "VARCHAR", "codenames.legality level: legal, borderline or illegal"),
    ("flags", f"{FLAG}[]", "board words related to the clue, with the relation"),
    ("reply_sha256", "VARCHAR", "SHA-256 of the raw model reply"),
    ("provider", "VARCHAR", "inference provider of a model call"),
    ("model", "VARCHAR", "model id of a model call"),
    ("latency_s", "DOUBLE", "seconds from request to reply"),
    ("tokens_in", "INTEGER", "prompt tokens reported by the provider"),
    ("tokens_out", "INTEGER", "reply tokens reported by the provider"),
    ("cost_usd", "DOUBLE", "tokens times the listed price (codenames.hosted meter)"),
    ("commit", "VARCHAR", "full SHA of the code that produced the row"),
    ("source", "VARCHAR", "file the row was read from"),
)
FIELDS = tuple(name for name, _, _ in SCHEMA)
# C10's and C6's seven strangers, in task order: study.models() less the proposer.
STRANGERS = (
    "potion",
    "wordllama",
    "granite",
    "bge",
    "Qwen/Qwen3-Embedding-0.6B",
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-0.6b",
)
C10_LLM = ("llama-3.3-70b", "deepseek-v3.2")
C10_GIVERS = {"F": "first-choice", "L": "listener", "P": "population"}
C10_GIVERS |= {"LP": "listener-population", "C": "qwen3.8-27b", "S": "speaker"}


def record(**fields) -> dict:
    """A row with every field of SCHEMA, null where not given."""
    unknown = set(fields) - set(FIELDS)
    assert not unknown, unknown
    return {f: fields.get(f) for f in FIELDS}


def board_text(words, kinds) -> str:
    return " ".join(words) + "\n" + "".join(kinds) + "\n"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@cache
def dealt(seed: int, n: int):
    from codenames.c4 import boards

    return tuple(boards(seed, n))


def board_fields(seed: int, i: int, n: int) -> dict:
    b = dealt(seed, n)[i]
    kinds = "".join(b.kinds)
    return {
        "unit": f"{seed}:{i}",
        "board_seed": seed,
        "board_index": i,
        "words": list(b.words),
        "kinds": kinds,
        "input_sha256": sha256(board_text(b.words, kinds)),
    }


def action(text: str) -> tuple[str | None, int | None]:
    clue, count = text.split("/")
    return (None, None) if clue == "-" else (clue, int(count))


# Readers of committed results.


def c10_units() -> list[dict]:
    """C10's test boards (seed 9004): every arm's pick, its exact values under the
    strangers (C10's mean), the human-fitted guesser and EAT, and the language-model
    guessers' played-out values. P and LP have one row per task (held-out stranger);
    the values sit on the potion task's row, the pick C10 sent to the other guessers."""
    src = "codenames/results/c10.json"
    d = json.loads((RESULTS / "c10.json").read_text())
    seed, n = d["seed"], d["n"]
    llm = {}
    for line in d["crossguess"]["per_board"]:
        giver, guesser, _s, *cells = line.split()
        llm[(giver, guesser)] = [float(c.split("/")[1]) for c in cells]
    rows = []
    for b in d["boards"]:
        i = b["board"]
        for arm, pick in sorted(b["picks"].items()):
            picks = pick if isinstance(pick, list) else [pick]
            for t, p in enumerate(picks):
                task = STRANGERS[t] if isinstance(pick, list) else None
                carries = task in (None, "potion")
                values = None
                if carries:
                    v = b["value"][arm]
                    values = [
                        {"scorer": s, "value": x}
                        for s, x in zip(("strangers", "human", "eat"), v, strict=True)
                    ]
                    if arm in C10_GIVERS:
                        values += [
                            {"scorer": g, "value": llm[(C10_GIVERS[arm], g)][i]}
                            for g in C10_LLM
                            if (C10_GIVERS[arm], g) in llm
                        ]
                clue, count = action(p)
                rows.append(
                    record(
                        experiment="C10",
                        split="test",
                        arm=arm,
                        task=task,
                        clue=clue,
                        count=count,
                        values=values,
                        source=src,
                        **board_fields(seed, i, n),
                    )
                )
    return rows


def c6_units() -> list[dict]:
    """C6's boards (test 6004, validation 6005): each chooser's action among the 120."""
    src = "codenames/results/c6-boards.json"
    d = json.loads((RESULTS / "c6-boards.json").read_text())
    rows = []
    for seed_text, boards_ in d.items():
        seed = int(seed_text)
        for b in boards_:
            i = b["board"]
            for arm, (clue, count) in sorted(b["actions"].items()):
                v = b["value"].get(arm)
                values = None
                if v is not None:
                    values = [
                        {"scorer": s, "value": x}
                        for s, x in zip(("strangers", "human", "eat"), v, strict=True)
                    ]
                rows.append(
                    record(
                        experiment="C6",
                        split="test" if seed == 6004 else "val",
                        arm=arm,
                        clue=clue,
                        count=count,
                        values=values,
                        source=src,
                        **board_fields(seed, i, len(boards_)),
                    )
                )
    return rows


def c7_units() -> list[dict]:
    """C7's test boards (6004): each trained picker's action, an index into C6's 120."""
    from codenames.actions import expand, read_action_lines

    src = "codenames/results/c7.json"
    d = json.loads((RESULTS / "c7.json").read_text())
    lists = read_action_lines((RESULTS / "c6-actions.txt").read_text())[6004]
    n = d["boards"]
    rows = []
    for arm, by_seed in d["per_board"].items():
        for s, run in by_seed.items():
            for i, j in enumerate(run["picks"]):
                clue, count = expand(lists[i])[j]
                rows.append(
                    record(
                        experiment="C7",
                        split="test",
                        arm=arm,
                        train_seed=int(s),
                        clue=clue,
                        count=count,
                        source=src,
                        **board_fields(6004, i, n),
                    )
                )
    return rows


def c4_units() -> list[dict]:
    """C4's test boards (4004): the embedding giver's clue and four language models'."""
    from codenames.c4 import read_clue_lines

    src = "codenames/results/c4-clues-4004.txt"
    given = read_clue_lines((RESULTS / "c4-clues-4004.txt").read_text())
    rows = []
    for (model, seed), cells in sorted(given.items()):
        for i, (clue, count, why) in enumerate(cells):
            rows.append(
                record(
                    experiment="C4",
                    split="test",
                    arm=model,
                    clue=clue,
                    count=count,
                    why_none=why,
                    source=src,
                    **board_fields(seed, i, len(cells)),
                )
            )
    return rows


def c3_units() -> list[dict]:
    """C3's test boards (3003): each policy's pick per task and training seed, as
    regenerated by codenames.diagnose_c3 from the committed weights."""
    from codenames.diagnose_c3 import read_picks

    path = DIAGNOSTICS / "c3-picks.txt"
    if not path.exists():
        return []
    rows = []
    for (giver, held, s, policy), cells in sorted(read_picks(path.read_text()).items()):
        for i, (clue, count) in enumerate(cells):
            rows.append(
                record(
                    experiment="C3",
                    split="test",
                    arm=f"{giver}/{policy}",
                    task=held,
                    train_seed=s,
                    clue=clue,
                    count=count,
                    source="codenames/results/diagnostics/c3-picks.txt",
                    **board_fields(3003, i, len(cells)),
                )
            )
    return rows


def pilot_units(key_path: Path, label: str = "C11-pilot") -> list[dict]:
    """The C11 pilot's 40 boards (seed 11010): F and P on each, from a key.json, or from a
    boards.json, whose key human_pilot.key rebuilds."""
    key = json.loads(key_path.read_text())
    if "key_sha256" in key:  # a boards.json
        from codenames.human_pilot import key as pilot_key

        key = pilot_key(key)
    seed = key["board_seed"]
    n = 1 + max(r["seed_index"] for r in key["boards"])
    rows = []
    for r in key["boards"]:
        for arm in ("F", "P"):
            clue, count = action(r[arm])
            rows.append(
                record(
                    experiment=label,
                    split="pilot",
                    arm=arm,
                    clue=clue,
                    count=count,
                    source=str(key_path.resolve().relative_to(HERE.parent.resolve())),
                    **board_fields(seed, r["seed_index"], n),
                )
            )
    return rows


READERS = {"C3": c3_units, "C4": c4_units, "C6": c6_units, "C7": c7_units, "C10": c10_units}


def committed(experiments=tuple(READERS)) -> list[dict]:
    return [r for e in experiments for r in READERS[e]()]


def with_legality(rows: list[dict]) -> list[dict]:
    """The rows with legality and flags filled in by codenames.legality."""
    from codenames.legality import flags, level, worst
    from codenames.words import legal

    out = []
    for r in rows:
        r = dict(r)
        if r["clue"] is not None:
            f = flags(r["clue"], r["words"])
            r["flags"] = [{"word": w, "relation": x} for w, x in f]
            r["legality"] = worst(r["clue"], r["words"])
            if not legal(r["clue"], tuple(r["words"])):
                assert any(level(x) == "illegal" for _, x in f)
        out.append(r)
    return out


# Parquet, through DuckDB.


def ddl(table: str = "units") -> str:
    cols = ", ".join(f'"{name}" {kind}' for name, kind, _ in SCHEMA)
    return f"CREATE TABLE {table} ({cols})"


def write_parquet(rows: list[dict], path: Path) -> Path:
    """The rows as one Parquet file with the types of SCHEMA."""
    import duckdb

    con = duckdb.connect()
    con.execute(ddl())
    marks = ", ".join("?" for _ in FIELDS)
    con.executemany(f"INSERT INTO units VALUES ({marks})", [[r[f] for f in FIELDS] for r in rows])
    con.execute(f"COPY units TO '{path}' (FORMAT parquet)")
    return path


def load(paths=(), experiments=tuple(READERS)):
    """A DuckDB relation over Parquet record files and the committed results."""
    import duckdb

    con = duckdb.connect()
    con.execute(ddl())
    rows = committed(experiments)
    if rows:
        marks = ", ".join("?" for _ in FIELDS)
        con.executemany(
            f"INSERT INTO units VALUES ({marks})", [[r[f] for f in FIELDS] for r in rows]
        )
    for p in paths:
        cols = ", ".join(f'"{f}"' for f in FIELDS)
        con.execute(
            f"INSERT INTO units SELECT {cols} FROM read_parquet(?, union_by_name = true)",
            [str(p)],
        )
    return con.table("units")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=list(READERS))
    parser.add_argument("--legality", action="store_true", help="fill in the legality fields")
    args = parser.parse_args()
    rows = committed(args.experiments)
    if args.legality:
        rows = with_legality(rows)
    write_parquet(rows, args.out)
    print(args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
