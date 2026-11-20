"""The open human dataset of the AH2AC2 challenge, read with the standard library.

Dizdarević et al. (2025) released 1,858 two-player and 1,221 three-player games played on
hanab.live, each with its deck and its actions. The dataset card of `ah2ac2/datasets` on
Hugging Face gives the license as `apache-2.0` and says nothing else; hanab.live publishes no
terms for its game records. So this repository keeps only numbers derived from each game,
never the decks or the actions (see the README). The games come as four `.safetensors`
files: a training and a validation file for each number of players. This module downloads
them, reads them without numpy, and turns each game into this package's vocabulary.

The files store, for each game:

- `decks`: 50 (color, rank) pairs, both counted from 0, top of the deck first. The hands
  are dealt in blocks from the top, five cards to player 0 first: the same convention as
  `hanabi.rules`, so a deck needs no reordering.
- `actions`: one action id per player per turn, in the encoding of the JaxMARL Hanabi
  environment, which the challenge's code replays. Only the current player's entry counts;
  the others hold the no-op id. With hand size h and P players: 0 to h - 1 discard slot i,
  h to 2h - 1 play slot i, then a color hint for each other player (5 ids each, the next
  seat first), then a rank hint for each other player. Slots count from the oldest card,
  and a drawn card goes to the end, as in `hanabi.rules`.
- `num_actions`, `scores` and `game_ids`.

`replay` plays a game's actions with `hanabi.rules` and checks every action is legal; the
score it reaches must equal the recorded one.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from hanabi.rules import Action, Rules, State, discard, hint, legal, make_card, new_game, play, step

REPO_ID = "ah2ac2/datasets"
REVISION = "main"
FILES = {
    2: ("2_player_games_train_1k.safetensors", "2_player_games_val.safetensors"),
    3: ("3_player_games_train_1k.safetensors", "3_player_games_val.safetensors"),
}
DTYPES = {"I8": "b", "U8": "B", "I16": "h", "I32": "i", "I64": "q", "BOOL": "?"}


def read_safetensors(path: str | Path) -> dict[str, tuple[list[int], list[int]]]:
    """Tensor name -> (shape, flat values). Only integer and boolean tensors are read."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    body = raw[8 + n :]
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        code = DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        size = struct.calcsize(code)
        values = list(struct.unpack(f"<{(end - start) // size}{code}", body[start:end]))
        out[name] = (info["shape"], values)
    return out


@dataclass(frozen=True)
class HumanGame:
    """One recorded game: its id on hanab.live, the deck as `hanabi.rules` cards, and the
    current player's action id on each turn."""

    game_id: int
    players: int
    deck: tuple[int, ...]
    actions: tuple[int, ...]
    score: int
    source: str  # the file it came from


def games_from_tensors(tensors, source: str = "") -> list[HumanGame]:
    """Split the flat tensors of one file into games."""
    (num_players,) = tensors["num_players"][1] if tensors["num_players"][1] else (0,)
    rules = Rules(players=num_players)
    ids, scores = tensors["game_ids"][1], tensors["scores"][1]
    n = len(ids)
    deck_shape, decks = tensors["decks"]
    act_shape, acts = tensors["actions"]
    lengths = tensors["num_actions"][1]
    assert deck_shape == [n, 50, 2], deck_shape
    _, turns, p = act_shape
    assert p == num_players
    out = []
    for g in range(n):
        deck = tuple(
            make_card(decks[(g * 50 + i) * 2], decks[(g * 50 + i) * 2 + 1] + 1, rules)
            for i in range(50)
        )
        seq = []
        for t in range(lengths[g]):
            cur = t % p
            seq.append(acts[(g * turns + t) * p + cur])
        out.append(HumanGame(ids[g], num_players, deck, tuple(seq), scores[g], source))
    return out


def decode(action: int, player: int, rules: Rules) -> Action:
    """A JaxMARL action id for `player`, as an action of `hanabi.rules`."""
    h, p, nc, nr = rules.hand, rules.players, rules.colors, rules.ranks
    if action < h:
        return discard(action)
    if action < 2 * h:
        return play(action - h)
    a = action - 2 * h
    if a < (p - 1) * nc:
        return hint((player + 1 + a // nc) % p, "color", a % nc)
    a -= (p - 1) * nc
    if a < (p - 1) * nr:
        return hint((player + 1 + a // nr) % p, "rank", a % nr + 1)
    raise ValueError(f"action id {action} is a no-op or out of range")


def encode(a: Action, player: int, rules: Rules) -> int:
    """The JaxMARL action id of an action of `hanabi.rules`: the inverse of `decode`."""
    h, p, nc, nr = rules.hand, rules.players, rules.colors, rules.ranks
    if a.kind == "discard":
        return a.index
    if a.kind == "play":
        return h + a.index
    offset = (a.target - player - 1) % p
    if a.attr == "color":
        return 2 * h + offset * nc + a.value
    return 2 * h + (p - 1) * nc + offset * nr + a.value - 1


def replay(game: HumanGame) -> State:
    """Play the recorded actions with the rules engine, checking that each is legal and
    that the game does not end early, and return the final state."""
    rules = Rules(players=game.players)
    s = new_game(rules, list(game.deck))
    for a_id in game.actions:
        if s.over:
            raise ValueError(f"game {game.game_id}: actions after the end")
        a = decode(a_id, s.current, rules)
        if not legal(s, a):
            raise ValueError(f"game {game.game_id}: illegal action {a} on turn {s.turn + 1}")
        step(s, a)
    return s


def download(cache: str | Path) -> list[Path]:
    """Fetch the four files from Hugging Face into `cache` (needs the hanabi group)."""
    from huggingface_hub import hf_hub_download

    names = [f for pair in FILES.values() for f in pair]
    return [
        Path(hf_hub_download(REPO_ID, f, repo_type="dataset", revision=REVISION, local_dir=cache))
        for f in names
    ]


def load(cache: str | Path, players: int) -> list[HumanGame]:
    """Every game for this number of players, training file first, in file order."""
    out = []
    for name in FILES[players]:
        out += games_from_tensors(read_safetensors(Path(cache) / name), name)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["fetch", "check"])
    ap.add_argument("--cache", default="data/ah2ac2", help="folder for the four files")
    args = ap.parse_args(argv)
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        download(args.cache)
    for players in FILES:
        games = load(args.cache, players)
        finals = [replay(g) for g in games]
        bad = sum(s.score != g.score for s, g in zip(finals, games, strict=True))
        over = sum(s.over for s in finals)
        print(f"{players} players: {len(games)} games, {bad} replay to another score, {over} end")


if __name__ == "__main__":
    main()
