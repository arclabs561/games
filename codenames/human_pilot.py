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
- play.html: the phone page. It holds the words, clues, counts and orders, and no team map
  and no arm.

key.json, the arm of each item and each board's F and P actions, is written to
codenames/.cache/human-pilot/key.json, which git ignores, and is not committed, so that a
volunteer who looks at the folder beside the page does not see the arms. It is not a
secret from someone who runs the code: the item ids and the arm of each id follow from the
order seed (`orders`), so `key` rebuilds key.json from boards.json alone and checks it
against the SHA-256 that boards.json holds.

    uv run python -m codenames.human_pilot build    # needs the word vectors (6 min, then under 1)
    uv run python -m codenames.human_pilot check    # rebuilds the files from boards.json

A person's turn is scored as every other guesser in this folder is (`c4.play_turn`): the
picks in order, up to the count, ending at the first word that is not the team's.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from codenames import c4
from codenames.game import Board

HERE = Path(__file__).parent
OUT = HERE / "results" / "human-pilot"
KEY = HERE / ".cache" / "human-pilot" / "key.json"  # ignored by git
PLAN = "experiments/plans/C11.md"
BOARD_SEED = 11010
ORDER_SEED = 11011
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
    return PAGE.replace("/*DATA*/", data)


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
    for r in response["responses"]:
        x = items[r["item"]]
        b = boards["boards"][x["board"]]
        found, value = turn_value(b["words"], b["kinds"], r["picks"], x["count"])
        position = order.index(r["item"])
        rows.append(
            {
                "item": r["item"],
                "board": x["board"],
                "arm": key["items"][r["item"]],
                "block": 1 + position // boards["block"],
                "position": position,
                "count": x["count"],
                "picks": len(r["picks"]),
                "found": found,
                "value": value,
                "stopped_early": len(r["picks"]) < x["count"],
                "seconds": r["t_done"] / 1000,
            }
        )
    return rows


def key_rows(boards: dict) -> dict:
    """The parts of key.json that rebuild needs, from boards.json and the order seed: the
    arm of each item id comes from `orders`, and its action from the item's clue."""
    n = len(boards["boards"])
    arm_of = {i: arm for (_, arm), i in orders(n, boards["order_seed"])["ids"].items()}
    action = {(x["board"], arm_of[x["item"]]): f"{x['clue']}/{x['count']}" for x in boards["items"]}
    used = [b["seed_index"] for b in boards["boards"]]
    rows = [
        {"seed_index": b["seed_index"], **{arm: action[(i, arm)] for arm in ARMS}}
        for i, b in enumerate(boards["boards"])
    ]
    return {"boards": rows, "skipped": sorted(set(range(1 + max(used))) - set(used))}


def texts_from_boards(boards: dict) -> tuple[str, str, str]:
    """boards.json, key.json and play.html, rebuilt from boards.json (no vectors)."""
    return materials(*rebuild(key_rows(boards)), order_seed=boards["order_seed"])


def key(boards: dict | None = None) -> dict:
    """key.json, rebuilt from boards.json; it must match the SHA-256 that boards.json holds."""
    boards = boards or json.loads((OUT / "boards.json").read_text())
    key_text = texts_from_boards(boards)[1]
    if sha256(key_text) != boards["key_sha256"]:
        raise ValueError("the key rebuilt from boards.json does not match key_sha256")
    return json.loads(key_text)


def write(texts, out: Path = OUT, key_path: Path = KEY) -> None:
    boards_text, key_text, page_text = texts
    out.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((out / "boards.json", boards_text), (key_path, key_text),
                       (out / "play.html", page_text)):  # fmt: skip
        path.write_text(text)
        print(path, sha256(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["build", "check"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--key", type=Path, default=KEY, help="where key.json goes (not in git)")
    args = parser.parse_args()
    if args.what == "build":
        check_against_c10()
        write(materials(*choose_boards(BOARD_SEED, N_BOARDS)), args.out, args.key)
        return
    boards_text = (args.out / "boards.json").read_text()
    boards = json.loads(boards_text)
    new_boards, key_text, page_text = texts_from_boards(boards)
    same = {
        "boards.json": new_boards == boards_text,
        "key_sha256": sha256(key_text) == boards["key_sha256"],
        "play.html": page_text == (args.out / "play.html").read_text(),
    }
    if args.key.exists():
        same["key.json"] = args.key.read_text() == key_text
    print(same)
    if not all(same.values()):
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
body { font-family: system-ui, sans-serif; font-size: 1rem; margin: 0 auto; max-width: 32rem;
  padding: 0.8rem; color: #111; background: #fff; }
@media (max-width: 24rem) { body { padding: 0.8rem 0.25rem; } }
h1 { font-size: 1.25rem; }
h2 { font-size: 1.1rem; }
[tabindex="-1"]:focus { outline: none; }
p, li, label { line-height: 1.4; }
button { font: inherit; padding: 0.6rem 1rem; margin: 0.2rem; border: 1px solid #555;
  border-radius: 0.4rem; background: #f4f4f4; color: #111; }
button[aria-disabled="true"] { opacity: 0.45; }
:focus-visible { outline: 3px solid #b35900; outline-offset: 2px; }
#clue { font-size: 1.5rem; font-weight: 600; text-align: center; margin: 0.6rem 0; }
#grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.2rem; }
#grid button { margin: 0; padding: 0.6rem 0.05rem; font-size: 1rem; line-height: 1.2;
  overflow-wrap: anywhere; hyphens: auto; }
#grid button.picked { background: #1f4e79; color: #fff; }
#grid button .order { display: block; font-size: 1rem; }
#bar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
  margin-top: 0.6rem; }
textarea { width: 100%; min-height: 6rem; font: inherit; box-sizing: border-box; }
#result { min-height: 8rem; font-family: monospace; font-size: 1rem; }
.small { color: #444; }
</style>
</head>
<body>
<main id="app"></main>
<script>
const DATA = /*DATA*/;
const STORE = "c11-pilot";
const load = () => { try { return JSON.parse(localStorage.getItem(STORE) || "null"); } catch (e) { return null; } };
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} };
const forget = () => { try { localStorage.removeItem(STORE); } catch (e) {} };
let S = load();
let shown = 0, picks = [], times = [], undos = 0;
const app = document.getElementById("app");
const el = (tag, attrs, ...kids) => {
  const e = document.createElement(tag);
  Object.assign(e, attrs || {});
  kids.forEach(k => e.append(k));
  return e;
};
// Show a screen and, after the first, move the keyboard focus to its heading, so that
// focus never drops to the page and a screen reader reads the new screen from its top.
let loaded = false;
function show(...kids) {
  app.replaceChildren(...kids);
  const h = app.querySelector("h1, h2");
  if (h && loaded) { h.tabIndex = -1; h.focus(); }
  loaded = true;
}

function intro() {
  const v = new URLSearchParams(location.search).get("v");
  show(
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
    el("h2", {}, "What is kept, and where"),
    el("p", {className: "small"}, "This is a small study of how people read one-word " +
      "clues. What is kept: your picks, how long each took, and an optional comment at " +
      "the end. No name or contact detail is asked; please do not write any in the " +
      "comment. This page sends nothing: at the end you copy or share a block of text " +
      "with the person who gave you this page."),
    el("p", {className: "small"}, "That text is kept unchanged in a private dataset on " +
      "Hugging Face. Your scored answers will be published in a public code repository: " +
      "for each screen, how many words you picked, how many were your team's, the " +
      "score and the time, under a session number, without your words, the session " +
      "code, the date or your comment. Comments are not published; the results may say " +
      "in a sentence what they were about."),
    el("p", {className: "small"}, "To withdraw, now or later, send the session code " +
      "shown at the end to the person who gave you this page. Your text is then deleted " +
      "and your answers are left out of the results. If they were already published, a " +
      "new version removes them, though earlier versions of the repository keep them."),
    el("p", {className: "small"}, "Please take part only if you are 18 or older."),
    ...(v === "A" || v === "B" ? [el("button", {onclick: () => start(v)}, "Start")] :
      [el("p", {}, "Version (as you were told):"),
       el("button", {onclick: () => start("A")}, "A"),
       el("button", {onclick: () => start("B")}, "B")]));
}

function resume() {
  const n = S.responses.length;
  show(el("h1", {}, "This phone holds a session"),
    el("p", {}, S.finished ?
      "A finished session from " + S.date + " is stored in this browser." :
      "An unfinished session from " + S.date + " is stored in this browser (" + n +
      " of " + DATA.versions[S.version].length + " screens done)."),
    el("p", {}, "If it is yours, go on with it. If it is someone else's, start fresh: " +
      "that deletes it from this phone without showing it."),
    el("button", {onclick: () => S.finished ? result() : next()},
      S.finished ? "Show my answers" : "Continue my session"),
    el("button", {onclick: fresh}, "Start fresh"));
}

function fresh() {
  forget();
  S = null;
  intro();
}

function start(version) {
  const r = new Uint8Array(4);
  crypto.getRandomValues(r);
  S = {plan: DATA.plan, version, session: Array.from(r, x => x.toString(16).padStart(2, "0")).join(""),
       date: new Date().toISOString().slice(0, 10), responses: [], comment: ""};
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
  const counter = el("p", {className: "small"});
  counter.setAttribute("role", "status");
  const undo = el("button", {onclick: () => {
    if (picks.length) { picks.pop(); times.pop(); undos++; draw(); } }}, "Undo");
  const done = el("button", {onclick: () => answer(order[k])}, "Done");
  // The 25 buttons are made once per screen and only changed after, so the focus stays
  // on the word that was just picked.
  const buttons = words.map(w => el("button", {onclick: () => {
    if (!picks.includes(w) && picks.length < count) {
      picks.push(w); times.push(Math.round(performance.now() - shown)); draw(); } }}));
  function draw() {
    buttons.forEach((btn, j) => {
      const w = words[j], i = picks.indexOf(w);
      btn.className = i >= 0 ? "picked" : "";
      btn.setAttribute("aria-pressed", String(i >= 0));
      btn.setAttribute("aria-label", i >= 0 ? w + ", pick " + (i + 1) : w);
      const badge = el("span", {className: "order"}, "#" + (i + 1));
      badge.setAttribute("aria-hidden", "true");
      btn.replaceChildren(w, ...(i >= 0 ? [badge] : []));
    });
    undo.setAttribute("aria-disabled", String(!picks.length));
    counter.textContent = "Screen " + (k + 1) + " of " + order.length + ": " +
      picks.length + " of " + count + " picked" +
      (picks.length === count ? " (the most for this clue; Undo to change)" : "");
  }
  const grid = el("div", {id: "grid"}, ...buttons);
  grid.setAttribute("role", "group");
  grid.setAttribute("aria-label", "The 25 words");
  show(el("h2", {id: "clue"}, clue.toUpperCase() + " \\u00b7 " + count), grid,
    el("div", {id: "bar"}, counter, el("span", {}, undo, done)));
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
  show(el("h1", {}, "Halfway"),
    el("p", {}, "The second half shows the same 40 boards again, each with a different clue. " +
      "Guess each one on its own, as if you had not seen the board before."),
    el("button", {onclick: () => { S.halfway = true; save(); next(); }}, "Continue"));
}

function finish() {
  if (S.finished) return result();
  const box = el("textarea", {id: "comment", placeholder: "Anything about the clues or the page"});
  show(el("h1", {}, "Done, thank you"),
    el("label", {htmlFor: "comment"}, "An optional comment (no names, please):"), box,
    el("button", {onclick: () => { S.comment = box.value; S.finished = true; save(); result(); }}, "Finish"));
}

function result() {
  const text = JSON.stringify(S);
  const note = el("p", {});
  note.setAttribute("role", "status");
  // Once the text has left by a button, it is deleted from the browser's storage, so the
  // next person on this phone cannot see it. It stays on this screen until the page closes.
  const sent = how => { forget();
    note.textContent = how + ". The answers are deleted from this phone's storage and stay " +
      "on this screen until you close the page."; };
  const area = el("textarea", {id: "result", readOnly: true, value: text});
  const share = el("button", {onclick: () => (navigator.share ? navigator.share({text}) :
    navigator.clipboard.writeText(text)).then(() => sent("Shared"), () => {})}, "Share");
  const copy = el("button", {onclick: () =>
    navigator.clipboard.writeText(text).then(() => sent("Copied"), () => {})}, "Copy");
  const file = el("button", {onclick: () => {
    const a = el("a", {href: URL.createObjectURL(new Blob([text], {type: "application/json"})),
      download: "c11-" + S.session + ".json"});
    a.click(); sent("Downloaded"); }}, "Download");
  show(el("h1", {}, "Your answers"),
    el("p", {}, "Send this text to the person who gave you the page, by any of the buttons."),
    el("p", {}, "Your session code is " + S.session + ". Keep it if you may want to " +
      "withdraw later."),
    el("label", {htmlFor: "result"}, "Your answers, as text to send:"),
    area, el("div", {}, share, copy, file), note,
    el("p", {className: "small"}, "It holds the version, the session code, the date, " +
      "your picks and times, and your comment."),
    el("p", {}, "When you have sent it, or to hand the phone to someone else:"),
    el("button", {onclick: fresh}, "Start fresh (deletes these answers from this phone)"));
}

S ? resume() : intro();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
