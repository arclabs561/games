"""Experiment C11, part B (experiments/plans/C11.md): the materials of a human pilot.

Forty boards, each with two clues: the embedding giver's first choice (F) and the
population's pick (P), both as C10 defines them (codenames.listen, codenames.c10score):

- EmbeddingGemma 2 proposes 30 legal clues, each with counts 1 to 4 (codenames.actions),
  from the clue words that `legality.strict_legal` allows on the board (the amendment of
  2026-11-19 to plan C11: no form of a board word and no shared compound part);
- F is the action of the 120 that the proposer values most;
- P is the action, among the 40 of the K = 10 clues the prefilter keeps, with the highest
  mean exact value over six embedding guessers: the seven strangers less potion, the task
  of C10 whose pick the human-fitted guesser and the language-model guessers scored.

Boards are dealt by `c4.boards` from seed 11010, in order. A board on which P and F name
the same clue word is skipped, since a person would see the same clue twice.

A person plays 80 items, one per board and clue, in two blocks of 40. Each block holds
every board once, and block 2 gives each board the clue it did not get in block 1. In
block 1 of version A, 20 boards show P and 20 show F; version B swaps the two clues at
every position. Item ids are random, so nothing a player sees names an arm.

Files, in codenames/results/human-pilot/:

- boards.json: the boards (words and team map), the items (board, clue, count), the order
  of each version, and the SHA-256 of key.json;
- key.json: the arm of each item, and each board's F and P actions;
- play.html: the phone page. It holds the words, clues, counts and orders, and no team map
  and no arm.

    uv run python -m codenames.human_pilot build    # needs the word vectors (6 min, then under 1)
    uv run python -m codenames.human_pilot check    # rebuilds the files from key.json

A person's turn is scored as every other guesser in this folder is (`c4.play_turn`): the
picks in order, up to the count, ending at the first word that is not the team's.
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

from codenames import c4
from codenames.game import Board

HERE = Path(__file__).parent
OUT = HERE / "results" / "human-pilot"
PLAN = "experiments/plans/C11.md"
BOARD_SEED = 11010
ORDER_SEED = 11011  # historical order used by the model comparison
PLAY_ORDER_SEED = 11012
N_BOARDS = 40
K = 10  # C10's K, chosen on its validation boards
HELD_OUT = "potion"  # C10's task for the human-fitted and language-model guessers
ARMS = ("F", "P")
VERSIONS = ("A", "B")

Action = tuple[str, int]


# The two clues of a board, from the existing giver code (needs the word vectors).


def clue_pool(board: Board) -> tuple[str, ...]:
    """The clue words that the strict check allows on the board (codenames.legality)."""
    from codenames.legality import strict_legal
    from codenames.words import clue_words

    return tuple(c for c in clue_words() if strict_legal(c, board.words))


def giver_picks(board: Board, strict: bool = True) -> dict[str, Action]:
    """F, P and the oracle at K on one board, as C10 computes them; with strict, every
    arm chooses among clues of the strict pool, as the amendment of 2026-11-19 says."""
    import numpy as np

    from codenames.actions import proposer, shortlist
    from codenames.c10score import BoardData
    from codenames.listen import argmax_first, pick_population, prefilter

    sp, tau = proposer()
    clues = shortlist(board, sp, tau, pool=clue_pool(board) if strict else None)
    data = BoardData(board, clues, prefilter(board, clues, K))
    assert data.potion == scorer_names().index(HELD_OUT)
    idx = np.array(data.kidx(K))
    s = data.strangers[:, idx]
    picks = {
        "F": data.first,
        "P": int(idx[pick_population(s, data.potion)]),
        "oracle": int(idx[argmax_first(s.mean(axis=0))]),
    }
    return {arm: data.acts[j] for arm, j in picks.items()}


def scorer_names() -> list[str]:
    from codenames.c10score import scorers

    return list(scorers()[1])


def check_against_c10(n: int = 3) -> None:
    """giver_picks, without the strict filter, gives C10's committed F, P and oracle on the
    first n test boards."""
    from codenames.listen import TEST

    committed = json.loads((HERE / "results" / "c10.json").read_text())["boards"]
    potion = scorer_names().index(HELD_OUT)
    for i, board in enumerate(c4.boards(TEST, n)):
        got = {arm: f"{c}/{k}" for arm, (c, k) in giver_picks(board, strict=False).items()}
        want = committed[i]["picks"]
        assert got == {"F": want["F"], "P": want["P"][potion], "oracle": want["oracle"]}, i


def choose_boards(
    seed: int, n: int, picks_of=giver_picks, most: int = 1000
) -> tuple[list[dict], list[int]]:
    """The first n boards of the seed on which P and F differ in clue word, with their
    picks, and the indices of the boards skipped before the nth."""
    deals = c4.boards(seed, most)
    chosen, skipped, i = [], [], 0
    while len(chosen) < n:
        board = deals[i]
        picks = picks_of(board)
        if picks["P"][0] == picks["F"][0]:
            skipped.append(i)
        else:
            chosen.append({"seed_index": i, "board": board, "F": picks["F"], "P": picks["P"]})
        i += 1
    return chosen, skipped


# The items and their orders (no vectors needed).


def item_ids(rng: random.Random, n: int) -> list[str]:
    """n distinct ids of six hex digits."""
    ids: list[str] = []
    while len(ids) < n:
        x = f"{rng.getrandbits(24):06x}"
        if x not in ids:
            ids.append(x)
    return ids


def other(arm: str) -> str:
    return ARMS[1 - ARMS.index(arm)]


def orders(n: int, order_seed: int) -> dict:
    """Ids, block-1 arms of version A (half P, half F) and the two block orders.
    Block 2 never starts with the board that ends block 1."""
    rng = random.Random(order_seed)
    ids = item_ids(rng, 2 * n)
    first = [ARMS[j % 2] for j in range(n)]
    rng.shuffle(first)
    block1, block2 = list(range(n)), list(range(n))
    rng.shuffle(block1)
    rng.shuffle(block2)
    if block2[0] == block1[-1]:
        block2 = block2[1:] + block2[:1]
    ids_of = {(b, arm): ids[2 * b + j] for b in range(n) for j, arm in enumerate(ARMS)}
    return {"ids": ids_of, "first": first, "block1": block1, "block2": block2}


def version_order(o: dict, version: str) -> list[str]:
    """The 80 item ids of one version, in the order a player sees them."""
    swap = version == "B"

    def arm(b: int, block: int) -> str:
        a = o["first"][b] if block == 1 else other(o["first"][b])
        return other(a) if swap else a

    return [o["ids"][(b, arm(b, 1))] for b in o["block1"]] + [
        o["ids"][(b, arm(b, 2))] for b in o["block2"]
    ]


def dumps(x) -> str:
    return json.dumps(x, indent=1, sort_keys=True) + "\n"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def materials(chosen: list[dict], skipped: list[int], order_seed: int = ORDER_SEED):
    """The texts of boards.json, key.json and play.html."""
    n = len(chosen)
    o = orders(n, order_seed)
    key = {
        "plan": PLAN,
        "board_seed": BOARD_SEED,
        "order_seed": order_seed,
        "items": {o["ids"][(b, arm)]: arm for b in range(n) for arm in ARMS},
        "boards": [
            {
                "board": b,
                "seed_index": e["seed_index"],
                **{arm: f"{e[arm][0]}/{e[arm][1]}" for arm in ARMS},
            }
            for b, e in enumerate(chosen)
        ],
        "skipped": skipped,
        "giver": {
            "proposer": "google/embeddinggemma-2",
            "k": K,
            "held_out": HELD_OUT,
            "code": "codenames.human_pilot.giver_picks (codenames.listen, codenames.c10score)",
            "clue_filter": "codenames.legality.strict_legal (amendment of 2026-11-19)",
        },
    }
    key_text = dumps(key)
    items = sorted(
        (
            {"item": o["ids"][(b, arm)], "board": b, "clue": e[arm][0], "count": e[arm][1]}
            for b, e in enumerate(chosen)
            for arm in ARMS
        ),
        key=lambda x: x["item"],
    )
    boards = {
        "plan": PLAN,
        "board_seed": BOARD_SEED,
        "order_seed": order_seed,
        "boards": [
            {
                "board": b,
                "seed_index": e["seed_index"],
                "words": list(e["board"].words),
                "kinds": list(e["board"].kinds),
            }
            for b, e in enumerate(chosen)
        ],
        "items": items,
        "versions": {v: version_order(o, v) for v in VERSIONS},
        "block": n,
        "key_sha256": sha256(key_text),
    }
    boards_text = dumps(boards)
    return boards_text, key_text, page(boards)


def player_data(boards: dict) -> dict:
    """What the page needs: words, clues, counts and orders. No team map, no arm."""
    return {
        "plan": "C11",
        "block": boards["block"],
        "boards": [b["words"] for b in boards["boards"]],
        "items": {x["item"]: [x["board"], x["clue"], x["count"]] for x in boards["items"]},
        "versions": boards["versions"],
    }


def page(boards: dict) -> str:
    data = json.dumps(player_data(boards), sort_keys=True, separators=(",", ":"))
    return (
        PAGE.replace("/*DATA*/", data)
        .replace("/*EDITION*/", json.dumps(boards["key_sha256"]))
        .replace("/*LIVE*/", json.dumps(boards["order_seed"] == PLAY_ORDER_SEED))
    )


def rebuild(chosen_from_key: dict, words_seed: int = BOARD_SEED):
    """chosen and skipped from a key.json, with the boards dealt again (no vectors)."""
    rows = chosen_from_key["boards"]
    deals = c4.boards(words_seed, 1 + max(r["seed_index"] for r in rows))

    def action(text: str) -> Action:
        clue, count = text.split("/")
        return clue, int(count)

    chosen = [
        {"seed_index": r["seed_index"], "board": deals[r["seed_index"]]}
        | {arm: action(r[arm]) for arm in ARMS}
        for r in rows
    ]
    return chosen, chosen_from_key["skipped"]


# Scoring a person's picks.


def turn_value(words: list[str], kinds: list[str], picks: list[str], count: int):
    """Team words found and the turn's value, by c4.play_turn."""
    board = Board(tuple(words), tuple(kinds))
    assert len(set(picks)) == len(picks) and set(picks) <= set(words), picks
    return c4.play_turn(board, picks, count)


def score_response(boards: dict, key: dict, response: dict) -> list[dict]:
    """One row per item a person answered: its board, arm, block, picks and value."""
    items = {x["item"]: x for x in boards["items"]}
    order = boards["versions"][response["version"]]
    rows = []
    for position, r in enumerate(response["responses"]):
        if position >= len(order) or r["item"] != order[position]:
            raise ValueError("session items must follow the assigned order exactly once")
        x = items[r["item"]]
        b = boards["boards"][x["board"]]
        found, value = turn_value(b["words"], b["kinds"], r["picks"], x["count"])
        if len(r["picks"]) > x["count"] or r["t_done"] < 0:
            raise ValueError("invalid pick count or time")
        rows.append(
            {
                "item": r["item"],
                "board": x["board"],
                "arm": key["items"][r["item"]],
                "block": 1 + position // boards["block"],
                "position": position,
                "count": x["count"],
                "picks": len(r["picks"]),
                "words": list(r["picks"]),
                "assassin": value < -1,
                "found": found,
                "value": value,
                "stopped_early": len(r["picks"]) < x["count"],
                "seconds": r["t_done"] / 1000,
            }
        )
    return rows


def write(texts, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in zip(("boards.json", "key.json", "play.html"), texts, strict=True):
        (out / name).write_text(text)
        print(out / name, sha256(text))


def matched_benchmark() -> dict:
    """Score the 40 pilot boards under the held-out potion model and the two saved LMs."""
    from codenames import c11
    from codenames.c4score import bootstrap
    from codenames.game import Clue
    from codenames.players import embedding_weigher, expected_value, space
    from codenames.study import load

    boards = json.loads((OUT / "boards.json").read_text())
    key = json.loads((OUT / "key.json").read_text())
    words = {w for b in boards["boards"] for w in b["words"]}
    words.update(x["clue"] for x in boards["items"])
    cfg = load("calibrate-game")["potion"]["best"]
    weigh = embedding_weigher(space("potion", cfg["clue"], cfg["word"], sorted(words)), cfg["tau"])
    guesses = c11.read_guesses(c11.GUESSES.read_text())
    values: dict = {m: {} for m in ("potion", *c11.GUESSERS)}
    for j, item in enumerate(boards["items"]):
        i = item["board"]
        b = boards["boards"][i]
        board = Board(tuple(b["words"]), tuple(b["kinds"]))
        arm = key["items"][item["item"]]
        values["potion"][i, arm] = expected_value(weigh, board, Clue(item["clue"], item["count"]))
        for model in c11.GUESSERS:
            cell = c11.read_cell(guesses["items", model, c11.ITEM_SEED, 0][j])
            if cell is None:
                raise ValueError("pilot model reply missing")
            values[model][i, arm] = cell[1]
    out = {}
    for model, table in values.items():
        diffs = [table[i, "P"] - table[i, "F"] for i in range(len(boards["boards"]))]
        out[model] = {"board_differences": diffs, "estimate": bootstrap(diffs)}
    return out


def summarize(boards: dict, key: dict, exports: list[dict], draws: int = 10_000) -> dict:
    """The preregistered board-weighted contrasts; only consented, de-identified rows leave here."""
    import numpy as np

    retained: dict = {}
    for session in exports:
        code = session.get("session", "")
        if (
            not re.fullmatch(r"[a-f0-9]{8}", code)
            or session.get("consent") is not True
            or session.get("edition") != boards["key_sha256"]
            or session.get("plan") != "C11"
        ):
            raise ValueError("invalid session code, consent or pilot edition")
        old = retained.get(code)
        if old:
            shorter, longer = sorted((old, session), key=lambda s: len(s["responses"]))
            if (
                shorter["version"] != longer["version"]
                or shorter["responses"] != longer["responses"][: len(shorter["responses"])]
            ):
                raise ValueError("conflicting exports for one session")
            session = longer
        retained[code] = session
    if not retained:
        raise ValueError("no response files to score")
    n = boards["block"]
    rows = []
    for participant, session in enumerate(retained.values(), 1):
        scored = score_response(boards, key, session)
        rows.extend(r | {"participant": participant} for r in scored)
    lengths = {p: len(s["responses"]) for p, s in enumerate(retained.values(), 1)}
    complete = [p for p, count in lengths.items() if count == 2 * n]
    one_block = [p for p, count in lengths.items() if count >= n]
    eligible = [
        r
        for r in rows
        if r["participant"] in one_block and (r["block"] == 1 or r["participant"] in complete)
    ]
    result: dict = {
        "sessions": len(retained),
        "complete": len(complete),
        "completed_first_block": len(one_block),
        "primary": None,
        "first_block": None,
        "second_block": None,
        "block_by_arm": None,
        "items": rows,
    }
    rng = np.random.default_rng(0)
    board_draws = rng.integers(n, size=(draws, n))

    def estimate(d):
        return {
            "mean": float(d.mean()),
            "ci": np.percentile(d[board_draws].mean(axis=1), [2.5, 97.5]).tolist(),
        }

    if complete:
        values = {(r["participant"], r["board"], r["arm"]): r["value"] for r in eligible}
        pairs = np.array(
            [[values[p, b, "P"] - values[p, b, "F"] for b in range(n)] for p in complete]
        )
        diffs = pairs.mean(axis=0)
        result["primary"] = estimate(diffs)
        result["primary"]["board_differences"] = diffs.tolist()
        people_draws = rng.integers(len(complete), size=(draws, len(complete)))
        two_way = np.array(
            [
                pairs[people][:, bs].mean()
                for people, bs in zip(people_draws, board_draws, strict=True)
            ]
        )
        result["primary"]["two_way_ci"] = np.percentile(two_way, [2.5, 97.5]).tolist()
        result["primary"]["planned_detectable_effect"] = float(
            2.8 * np.sqrt((0.09 + 2.25 / len(complete)) / n)
        )
        benchmark_path = OUT / "benchmark.json"
        if n == N_BOARDS and boards["board_seed"] == BOARD_SEED:
            benchmarks = json.loads(benchmark_path.read_text())
            result["minus_matched_model"] = {
                m: estimate(diffs - v["board_differences"]) for m, v in benchmarks.items()
            }

    block_diffs = {}
    for block, label in ((1, "first_block"), (2, "second_block")):
        cells = {
            (b, a): [
                r["value"]
                for r in eligible
                if r["block"] == block and r["board"] == b and r["arm"] == a
            ]
            for b in range(n)
            for a in ARMS
        }
        if all(cells.values()):
            d = np.array([np.mean(cells[b, "P"]) - np.mean(cells[b, "F"]) for b in range(n)])
            result[label] = estimate(d)
            block_diffs[block] = d
    if len(block_diffs) == 2:
        result["block_by_arm"] = estimate(block_diffs[2] - block_diffs[1])
    result["by_arm"] = {
        a: {
            k: float(np.mean([r[k] for r in eligible if r["arm"] == a]))
            for k in ("found", "stopped_early", "assassin", "seconds")
        }
        for a in ARMS
        if any(r["arm"] == a for r in eligible)
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["build", "check", "prepare", "score", "benchmark"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--key", type=Path, help="local private key for prepare or score")
    parser.add_argument("--responses", type=Path, help="directory of consented response JSON files")
    parser.add_argument("--materials", type=Path, help="directory of the active boards.json")
    args = parser.parse_args()
    if args.what == "benchmark":
        print(dumps(matched_benchmark()))
        return
    if args.what == "prepare":
        if not args.key or args.out == OUT:
            parser.error("prepare requires --out PUBLIC_DIR and --key PRIVATE_FILE")
        if args.key.resolve().is_relative_to(
            HERE.parent.resolve()
        ) or args.key.resolve().is_relative_to(args.out.resolve()):
            parser.error("keep the private key outside the repository and the delivery directory")
        original = json.loads((OUT / "key.json").read_text())
        boards, key, html = materials(*rebuild(original), order_seed=PLAY_ORDER_SEED)
        args.key.parent.mkdir(parents=True, exist_ok=True)
        if args.key.exists() and args.key.read_text() != key:
            parser.error("the private key path already holds another key")
        if not args.key.exists():
            with open(args.key, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as f:
                f.write(key)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "boards.json").write_text(boards)
        (args.out / "play.html").write_text(html)
        return
    if args.what == "score":
        if not args.key or not args.materials or not args.responses or args.out == OUT:
            parser.error("score requires --materials DIR --key FILE --responses DIR --out FILE")
        boards = json.loads((args.materials / "boards.json").read_text())
        raw_key = args.key.read_text()
        if args.out.resolve() == args.key.resolve() or args.out.resolve().is_relative_to(
            args.responses.resolve()
        ):
            parser.error("write the de-identified summary separately from private inputs")
        if boards["key_sha256"] != sha256(raw_key) or boards["order_seed"] != PLAY_ORDER_SEED:
            parser.error("wrong key or materials edition")
        files = sorted(args.responses.glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
        sessions = [json.loads(p.read_text()) for p in files]
        report = summarize(boards, json.loads(raw_key), sessions)
        args.out.write_text(dumps(report))
        return
    if args.what == "build":
        check_against_c10()
        write(materials(*choose_boards(BOARD_SEED, N_BOARDS)), args.out)
        return
    key = json.loads((args.out / "key.json").read_text())
    texts = materials(*rebuild(key), order_seed=key["order_seed"])
    names = ("boards.json", "key.json", "play.html")
    same = [(args.out / n).read_text() == t for n, t in zip(names, texts, strict=True)]
    print(dict(zip(names, same, strict=True)))
    if not all(same):
        raise SystemExit(1)


# The page. It makes no network request; a person's answers stay on the phone until they
# share the result text at the end.

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Codenames guesses</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 32rem; padding: 0.8rem;
  color: #111; background: #fff; }
h1 { font-size: 1.2rem; }
p, li { line-height: 1.4; }
button { font: inherit; padding: 0.6rem 1rem; margin: 0.2rem; border: 1px solid #555;
  border-radius: 0.4rem; background: #f4f4f4; color: #111; }
button:disabled { opacity: 0.4; }
#clue { font-size: 1.5rem; font-weight: 600; text-align: center; margin: 0.6rem 0; }
#grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.25rem; }
#grid button { margin: 0; padding: 0.7rem 0.1rem; font-size: 3vw; text-transform: uppercase;
  position: relative; overflow-wrap: anywhere; }
@media (min-width: 32rem) { #grid button { font-size: 0.8rem; } }
#grid button.picked { background: #1f4e79; color: #fff; }
#grid button span { position: absolute; top: 0.05rem; right: 0.2rem; font-size: 0.7rem; }
#bar { display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem; }
textarea { width: 100%; min-height: 6rem; font: inherit; box-sizing: border-box; }
#result { width: 100%; min-height: 8rem; font-family: monospace; font-size: 0.7rem; }
.small { color: #555; font-size: 0.85rem; }
</style>
</head>
<body>
<div id="app"></div>
<script>
const DATA = /*DATA*/;
const STORE = "c11-pilot-" + /*EDITION*/;
const ACTIVE = /*LIVE*/;
let S = null;
try { S = JSON.parse(localStorage.getItem(STORE) || "null"); } catch (e) { S = null; }
let shown = 0, picks = [], times = [], undos = 0;
const app = document.getElementById("app");
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} };
const el = (tag, attrs, ...kids) => {
  const e = document.createElement(tag);
  Object.assign(e, attrs || {});
  kids.forEach(k => e.append(k));
  return e;
};

function intro() {
  if (!ACTIVE) {
    app.replaceChildren(el("h1", {}, "Codenames pilot materials"),
      el("p", {}, "This is the archived edition used for the model comparison. Volunteers receive a separate local page."));
    return;
  }
  const v = new URLSearchParams(location.search).get("v");
  app.replaceChildren(
    el("h1", {}, "Codenames: you are the guesser"),
    el("p", {}, "Each screen shows 25 words and a one-word clue with a number. " +
      "Nine of the words belong to your team; you cannot see which. The clue is meant " +
      "to point to as many of your team's words as its number."),
    el("p", {}, "Tap the words you think the clue means, in the order you would guess " +
      "them, up to the number. Tap Done when you would stop, even with fewer picks or " +
      "none. You will not be told which picks were right."),
    el("p", {}, "There are 80 screens in two halves of 40, and each board appears once " +
      "in each half with a different clue. About 20 to 30 minutes. You can stop and come " +
      "back on the same phone and browser."),
    el("p", {className: "small"}, "What is kept: your picks, how long each took, and an " +
      "optional comment at the end. No name or contact detail is asked; please do not " +
      "write any in the comment. Nothing is sent anywhere: at the end you copy or share " +
      "a block of text with the person who gave you this page."),
    el("p", {}, "This small study is for adults aged 18 or older who have never seen the keyed pilot boards. We study how people read one-word clues. By starting, you consent to take part. You may stop at any time. You can send your first half after 40 screens."),
    el("p", {}, "If you send your answers, your picks and times may be published in a public code repository. Your session code and comment stay private. The coordinator keeps your sent answers in a private Hugging Face dataset and deletes the original message. Keep your code to withdraw through the coordinator. Withdrawal removes your private record and excludes your answers from future reports. Copies of a public release may persist."),
    ...(v === "A" || v === "B" ? [el("button", {onclick: () => start(v)}, "Start")] :
      [el("p", {}, "Version (as you were told):"),
       el("button", {onclick: () => start("A")}, "A"),
       el("button", {onclick: () => start("B")}, "B")]));
}

function start(version) {
  const r = new Uint8Array(4);
  crypto.getRandomValues(r);
  S = {plan: DATA.plan, version, session: Array.from(r, x => x.toString(16).padStart(2, "0")).join(""),
       consent: true, edition: /*EDITION*/, responses: [], comment: ""};
  save();
  next();
}

function next() {
  const order = DATA.versions[S.version];
  const k = S.responses.length;
  if (k === order.length) return finish();
  if (k === DATA.block && !S.halfway) return halfway();
  const [b, clue, count] = DATA.items[order[k]];
  picks = []; times = []; undos = 0;
  const words = DATA.boards[b];
  const grid = el("div", {id: "grid"});
  const done = el("button", {onclick: () => answer(order[k])}, "Done");
  const undo = el("button", {onclick: () => { if (picks.length) { picks.pop(); times.pop(); undos++; draw(); } }}, "Undo");
  const counter = el("span", {className: "small"});
  function draw() {
    grid.replaceChildren(...words.map(w => {
      const i = picks.indexOf(w);
      const btn = el("button", {className: i >= 0 ? "picked" : "",
        onclick: () => { if (i < 0 && picks.length < count) { picks.push(w); times.push(Math.round(performance.now() - shown)); draw(); } }}, w);
      if (i >= 0) btn.append(el("span", {}, String(i + 1)));
      return btn;
    }));
    undo.disabled = !picks.length;
    counter.textContent = (k + 1) + " of " + order.length + " \\u00b7 " + picks.length + " of " + count + " picked";
  }
  app.replaceChildren(el("div", {id: "clue"}, clue.toUpperCase() + " \\u00b7 " + count), grid,
    el("div", {id: "bar"}, counter, el("span", {}, undo, done)),
    ...(k >= DATA.block ? [el("button", {onclick: result}, "Export completed turns")] : []));
  draw();
  shown = performance.now();
}

function answer(item) {
  S.responses.push({item, picks: picks.slice(), t_picks: times.slice(),
    t_done: Math.round(performance.now() - shown), undos});
  save();
  window.scrollTo(0, 0);
  next();
}

function halfway() {
  app.replaceChildren(el("h1", {}, "Halfway"),
    el("p", {}, "The second half shows the same 40 boards again, each with a different clue. " +
      "Guess each one on its own, as if you had not seen the board before."),
    el("button", {onclick: result}, "Export this half"),
    el("button", {onclick: () => { S.halfway = true; save(); next(); }}, "Continue"));
}

function finish() {
  if (S.finished) return result();
  const box = el("textarea", {placeholder: "Optional: anything about the clues or the page"});
  app.replaceChildren(el("h1", {}, "Done, thank you"),
    el("p", {}, "An optional comment (no names, please):"), box,
    el("button", {onclick: () => { S.comment = box.value; S.finished = true; save(); result(); }}, "Finish"));
}

function clearPhone() {
  if (!window.confirm("Clear all saved answers on this phone?")) return;
  try { localStorage.removeItem(STORE); } catch (e) { /* private browsing */ }
  S = null;
  intro();
}

function result() {
  const text = JSON.stringify(S);
  const area = el("textarea", {id: "result", readOnly: true, value: text});
  const share = el("button", {onclick: () => navigator.share ? navigator.share({text}) :
    navigator.clipboard.writeText(text)}, "Share");
  const copy = el("button", {onclick: () => navigator.clipboard.writeText(text)}, "Copy");
  const file = el("button", {onclick: () => {
    const a = el("a", {href: URL.createObjectURL(new Blob([text], {type: "application/json"})),
      download: "c11-" + S.session + ".json"});
    a.click(); }}, "Download");
  app.replaceChildren(el("h1", {}, "Your answers"),
    el("p", {}, "Send this text to the person who gave you the page, by any of the buttons."),
    el("p", {}, "Keep this code to withdraw: " + S.session),
    area, el("div", {}, share, copy, file),
    el("button", {onclick: clearPhone}, "Clear this phone's answers"),
    ...(!S.finished ? [el("button", {onclick: next}, "Return to the pilot")] : []),
    el("p", {className: "small"}, "It holds the version, a random session code, " +
      "your picks and times, and your comment."));
}

if (S) {
  app.replaceChildren(el("h1", {}, "Saved answers on this phone"),
    el("button", {onclick: () => S.finished ? result() : next()}, "Continue saved session"),
    el("button", {onclick: clearPhone}, "Clear and start a new session"));
} else { intro(); }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
