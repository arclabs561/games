"""Cultural Codes split by game and by player, instead of by half-game as published.

The published files (train, val, test) carry no game or player ids. Both can be
recovered from the rows:

- **Games.** In Codenames Duet both players see the same 25 words, and each gives clues
  from their own side of the key. The files list each side of a game as its own run of
  turns, each starting with all 25 words unrevealed. A turn belongs to the game whose
  25 words contain its unrevealed words and whose two players are the same pair.
- **Players.** Each row lists the giver's and the guesser's survey answers
  (demographics, moral foundations, Big Five) and four fields typed in the game's
  interface (age, gender, country, native speaker). The survey part identifies a
  player; the typed age changes between games. About a fifth of player-turns have no
  survey answers, only the typed fields. Those profiles are taken as players of their
  own, which may split one person into two ids.

The published split puts each side of a game in one file, but the two sides of a game
often in different files. All 7,703 rows land in exactly one of 794 games; 165 games
have rows in more than one file, and 320 of the 373 test rows and 423 of the 455
validation rows are in games that also have training rows. 52 of the 54 test players
(by profile string) also play in training games.

Splits (``scheme``):

- ``published``: the files as published.
- ``game``: the split of experiments/plans/C1.md. A game goes to train, val or test by
  ``zlib.crc32`` of its key modulo 10: 0 to 7 train, 8 val, 9 test. The key is the
  board's 25 words, sorted and joined by "|", then "#", then the two players' profile
  strings, sorted and joined by "#". There is no seed.
- ``shuffled`` (exploratory, written before the plan): game ids shuffled with seed 0
  and cut 80/10/10.
- ``player`` (exploratory): guessers shuffled with seed 0 and added to test, then val,
  until each holds a tenth of the turns. A turn is in test if its guesser is a test
  player, in val if its guesser is a val player and its giver is not a test player,
  and in train if neither player is held out. Other turns (a held-out giver with a
  training guesser) are dropped. This follows the description of White et al. (2024),
  whose validation and test players do not appear in training.

In memory, game ids are hashes of the game key, and player ids hashes of the survey
answers (or of the typed fields). These ids are not written out. A typed-fields-only id
hashes a profile drawn from a small space (an age, a gender, a country, yes or no), so it
can be inverted by enumeration without the data; an earlier ``splits.json`` that listed
such ids could be read that way. ``splits.json`` now names each game, and each held-out
player, by the position of a turn in the published files, written "file:row": a game by
its first turn, a player by the first turn they guessed. These positions are resolved to
ids against the download. The file holds no ids, hashes or text from the rows, and says
nothing about a player to someone who does not have the files.

    uv run python -m codenames.split --check   # rebuild, compare with splits.json, print counts
    uv run python -m codenames.split --write   # rewrite splits.json
"""

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from codenames.fetch import fetch
from codenames.norms import PATTERN, Guess

SPLITS_FILE = Path(__file__).parent / "splits.json"
PUBLISHED = ("train", "val", "test")
SCHEMES = ("published", "game", "shuffled", "player")
PROFILES = re.compile(r"GIVER: \[(.*?)\], GUESSER: \[(.*?)\], remaining")
UI_FIELDS = ("gender", "country", "native")


@dataclass(frozen=True)
class Row:
    """One row of a published file: who played, what was unrevealed, the clue, the output."""

    source: str  # published split
    row: int  # position in that file
    words: tuple[str, ...]  # unrevealed words, in the file's order
    clue: str
    output: str
    giver: str  # profile strings as written in the row
    guesser: str


@dataclass(frozen=True)
class Turn:
    source: str
    row: int
    game: str  # digest of the game key
    giver: str  # player ids
    guesser: str
    guess: Guess
    key: str = ""  # the game key, see game_key
    profiles: tuple[str, str] = ("", "")  # giver's and guesser's profile strings


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def player_id(profile: str) -> str:
    """The survey answers if the row has them, else the whole profile as typed.

    The interface fields (a bare-number age, gender, country, native) are dropped from
    profiles with survey answers, since a player types them anew in each game.
    """
    survey = []
    for f in (f.strip() for f in profile.split(", ")):
        key, _, value = f.partition(":")
        typed_age = key == "age" and (value.strip().isdigit() or not value.strip())
        if f != "None" and key not in UI_FIELDS and not typed_age:
            survey.append(f)
    if not survey:  # no survey answers, only the typed fields
        return "p" + digest(profile)
    return "s" + digest(", ".join(survey))


def parse_row(source: str, i: int, rec: dict[str, str]) -> Row | None:
    m = PATTERN.search(rec["base_text"])
    p = PROFILES.search(rec["all_text"])
    if not m or not p:
        return None
    words = tuple(w.lower() for w in ast.literal_eval(m[1]))
    return Row(source, i, words, m[2].lower(), rec["output"].strip().lower(), p[1], p[2])


def read_rows(paths: dict[str, Path]) -> list[Row]:
    csv.field_size_limit(10**9)
    out = []
    for source, path in paths.items():
        with path.open() as f:
            for i, rec in enumerate(csv.DictReader(f)):
                r = parse_row(source, i, rec)
                if r is not None:
                    out.append(r)
    return out


def games(rows: list[Row]) -> list[str]:
    """The game id of each row: a digest of its game key."""
    return [digest(k) for k in game_keys(rows)]


def game_keys(rows: list[Row]) -> list[str]:
    """The game key of each row: its board's 25 words and its two players' profiles.

    A board is the unrevealed set of a turn with 25 words. A row belongs to the board of
    the same pair of players that contains its unrevealed words; the tests and
    ``check`` require that this board is unique.
    """
    pair = [tuple(sorted({r.giver, r.guesser})) for r in rows]
    boards: dict[tuple, set[frozenset[str]]] = defaultdict(set)
    for r, p in zip(rows, pair, strict=True):
        if len(r.words) == 25:
            boards[p].add(frozenset(r.words))
    out = []
    for r, p in zip(rows, pair, strict=True):
        found = [b for b in boards[p] if set(r.words) <= b]
        if len(found) != 1:
            raise ValueError(f"{r.source} row {r.row}: {len(found)} boards contain its words")
        out.append("|".join(sorted(found[0])) + "#" + "#".join(p))
    return out


def guess_of(r: Row) -> Guess | None:
    """The turn as a Guess, under the rule of norms.cultural_guesses: one picked word on
    the board and an alphabetic clue. Turns where the person picked several words are
    left out, as there."""
    if r.output in r.words and r.clue.isalpha():
        return Guess(r.clue, r.words, r.output)
    return None


def build_turns(rows: list[Row]) -> list[Turn]:
    out = []
    for r, k in zip(rows, game_keys(rows), strict=True):
        guess = guess_of(r)
        if guess is not None:
            ids = player_id(r.giver), player_id(r.guesser)
            out.append(Turn(r.source, r.row, digest(k), *ids, guess, k, (r.giver, r.guesser)))
    return out


def game_split(key: str) -> str:
    """The plan's split of one game, by crc32 of its key modulo 10."""
    b = zlib.crc32(key.encode()) % 10
    return "train" if b < 8 else "val" if b == 8 else "test"


def half(t: Turn) -> str:
    """'held' or 'sibling', by crc32 of the turn's text (clue, words, pick) modulo 2."""
    text = f"{t.guess.clue}|{' '.join(t.guess.words)}|{t.guess.picked}"
    return "held" if zlib.crc32(text.encode()) % 2 == 0 else "sibling"


def shuffled_split(turns: list[Turn], seed: int = 0, frac=(0.8, 0.1, 0.1)) -> dict[str, list[str]]:
    """Shuffle the game ids with the seed and cut them by the fractions."""
    ids = sorted({t.game for t in turns})
    random.Random(seed).shuffle(ids)
    a = round(frac[0] * len(ids))
    b = a + round(frac[1] * len(ids))
    return {"train": sorted(ids[:a]), "val": sorted(ids[a:b]), "test": sorted(ids[b:])}


def player_split(turns: list[Turn], seed: int = 0, frac: float = 0.1) -> dict[str, list[str]]:
    """Shuffle the guessers with the seed; fill test, then val, to a fraction of the turns."""
    by_guesser = Counter(t.guesser for t in turns)
    ids = sorted(by_guesser)
    random.Random(seed).shuffle(ids)
    out: dict[str, list[str]] = {"test": [], "val": []}
    target = frac * len(turns)
    it = iter(ids)
    for name in ("test", "val"):
        total = 0
        while total < target:
            p = next(it)
            out[name].append(p)
            total += by_guesser[p]
    return {k: sorted(v) for k, v in out.items()}


def ref(t: Turn) -> str:
    """The position of a turn in the published files, as "file:row"."""
    return f"{t.source}:{t.row}"


def ref_order(r: str) -> tuple[int, int]:
    source, row = r.split(":")
    return PUBLISHED.index(source), int(row)


def first_refs(turns: Sequence[Turn]) -> tuple[dict[str, str], dict[str, str]]:
    """The first turn of each game, and the first turn each player guessed, by position.

    Players who never guessed have no position; only guessers are ever held out.
    """
    game: dict[str, str] = {}
    guesser: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: ref_order(ref(t))):
        game.setdefault(t.game, ref(t))
        guesser.setdefault(t.guesser, ref(t))
    return game, guesser


def to_refs(turns: Sequence[Turn], scheme: str, spec: dict[str, list[str]]) -> dict[str, list[str]]:
    """A split's id lists as lists of positions, for splits.json."""
    game, guesser = first_refs(turns)
    where = guesser if scheme == "player" else game
    return {name: sorted((where[i] for i in ids), key=ref_order) for name, ids in spec.items()}


def from_refs(
    turns: Sequence[Turn], scheme: str, spec: dict[str, list[str]]
) -> dict[str, list[str]]:
    """A split's lists of positions, from splits.json, as sorted id lists."""
    by_ref = {ref(t): t for t in turns}
    pick = (lambda t: t.guesser) if scheme == "player" else (lambda t: t.game)
    return {name: sorted(pick(by_ref[r]) for r in refs) for name, refs in spec.items()}


def assign(turns: list[Turn], scheme: str, spec: dict) -> list[str | None]:
    """The split of each turn under a scheme, None for a dropped turn."""
    if scheme == "published":
        return [t.source for t in turns]
    if scheme == "game":
        return [game_split(t.key) for t in turns]
    if scheme == "shuffled":
        where = {g: name for name, ids in spec.items() for g in ids}
        return [where[t.game] for t in turns]
    test, val = set(spec["test"]), set(spec["val"])
    out: list[str | None] = []
    for t in turns:
        if t.guesser in test:
            out.append("test")
        elif t.guesser in val and t.giver not in test:
            out.append("val")
        elif not {t.giver, t.guesser} & (test | val):
            out.append("train")
        else:
            out.append(None)
    return out


def build(seed: int = 0) -> dict:
    """The contents of splits.json, from the downloaded files."""
    folder = fetch("cultural")
    rows = read_rows({s: folder / f"guess_{s}.csv" for s in PUBLISHED})
    turns = build_turns(rows)
    spec = {"shuffled": shuffled_split(turns, seed), "player": player_split(turns, seed)}
    counts = {}
    for scheme in SCHEMES:
        labels = assign(turns, scheme, spec.get(scheme, {}))
        counts[scheme] = dict(sorted(Counter(lab or "dropped" for lab in labels).items()))
    keys = sorted({t.key for t in turns})
    spec["game"] = {
        name: sorted(digest(k) for k in keys if game_split(k) == name)
        for name in ("train", "val", "test")
    }
    spec = {scheme: to_refs(turns, scheme, ids) for scheme, ids in spec.items()}
    return {
        "seed": seed,
        "turns": len(turns),
        "rows": len(rows),
        "games": len({t.game for t in turns}),
        "players": len({p for t in turns for p in (t.giver, t.guesser)}),
        "counts": counts,
        **spec,
    }


@cache
def all_turns() -> tuple[Turn, ...]:
    folder = fetch("cultural")
    return tuple(build_turns(read_rows({s: folder / f"guess_{s}.csv" for s in PUBLISHED})))


@cache
def spec() -> dict:
    """The id lists of each scheme in splits.json, resolved against the download."""
    saved = json.loads(SPLITS_FILE.read_text())
    every = all_turns()
    return {scheme: from_refs(every, scheme, saved[scheme]) for scheme in ("shuffled", "player")}


@cache
def turns(name: str) -> tuple[Turn, ...]:
    """The turns of a split named "<scheme>:<split>", such as "game:test", or "all"."""
    every = all_turns()
    if name == "all":
        return every
    scheme, split = name.split(":")
    labels = assign(list(every), scheme, spec().get(scheme, {}))
    return tuple(t for t, lab in zip(every, labels, strict=True) if lab == split)


def leaks(train: tuple[Turn, ...], test: tuple[Turn, ...]) -> dict[str, int]:
    """How many test turns share a game, or a guesser, with some training turn."""
    games_seen = {t.game for t in train}
    people = {p for t in train for p in (t.giver, t.guesser)}
    return {
        "test_turns": len(test),
        "game_in_train": sum(t.game in games_seen for t in test),
        "guesser_in_train": sum(t.guesser in people for t in test),
    }


def published_overlap() -> dict:
    """How the published files share games and players, counted on rows (before the
    filter of norms.cultural_guesses)."""
    folder = fetch("cultural")
    rows = read_rows({s: folder / f"guess_{s}.csv" for s in PUBLISHED})
    keys = game_keys(rows)  # raises if a row is on no board or on two
    files: dict[str, set[str]] = defaultdict(set)
    for r, k in zip(rows, keys, strict=True):
        files[k].add(r.source)
    boards: dict[str, set[str]] = defaultdict(set)
    for k in keys:
        boards[k.split("#")[0]].add(k)
    train_people = {p for r in rows if r.source == "train" for p in (r.giver, r.guesser)}
    out: dict = {
        "rows": len(rows),
        "games": len(files),
        "games_in_more_than_one_file": sum(len(v) > 1 for v in files.values()),
        "boards_played_by_more_than_one_pair": sum(len(v) > 1 for v in boards.values()),
    }
    for s in ("val", "test"):
        mine = [(r, k) for r, k in zip(rows, keys, strict=True) if r.source == s]
        people = {p for r, _ in mine for p in (r.giver, r.guesser)}
        out[s] = {
            "rows": len(mine),
            "rows_in_games_with_train_rows": sum("train" in files[k] for _, k in mine),
            "players": len(people),
            "players_also_in_train": len(people & train_people),
            "rows_whose_guesser_plays_in_train": sum(r.guesser in train_people for r, _ in mine),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="overwrite splits.json")
    parser.add_argument("--check", action="store_true", help="compare with splits.json")
    args = parser.parse_args()
    new = build(args.seed)
    if args.write:
        SPLITS_FILE.write_text(json.dumps(new, indent=1) + "\n")
    elif json.loads(SPLITS_FILE.read_text()) != new:
        raise SystemExit("splits.json differs from the split rebuilt from the download")
    print(json.dumps(published_overlap(), indent=1))
    print(json.dumps({k: new[k] for k in ("turns", "rows", "games", "players", "counts")}))
    for scheme in SCHEMES:
        print(scheme, leaks(turns(f"{scheme}:train"), turns(f"{scheme}:test")))


if __name__ == "__main__":
    main()
