"""The best score a Hanabi deal allows, and why it is not 25, for one deck or one game.

    uv run python -m hanabi.ceiling --players 2 --deck "R1 Y3 G4 ..."   # 50 cards, top first
    uv run python -m hanabi.ceiling --json game.json    # a hanab.live export (/export/<id>)
    uv run python -m hanabi.ceiling --players 2 --seed 161                # a seeded deal here

A deck lists the cards from the top. The hands are dealt in blocks, as hanab.live does: the
first five cards (four with four or five players) to the first player, the next block to
the second, and so on. Colors are R, Y, G, B and a fifth color, written W or P.

The ceiling is the open game's: the best score a team could reach if everyone saw every
card, the deck included. When it is below 25 the program prints the certificates that prove
it. With a hanab.live export that has actions, it also replays the game and prints each
turn after which the ceiling fell, with the action that cost the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hanabi.humans import BUDGETS, Ceiling, action_kind, ceiling, first_below
from hanabi.rules import (
    Action,
    Rules,
    State,
    color_of,
    discard,
    full_deck,
    hint,
    legal,
    make_card,
    new_game,
    play,
    rank_of,
    seeded_deck,
    step,
)
from hanabi.solver import (
    TRASH,
    Deal,
    chain_heights,
    copy_state,
    storage_violation,
    time_bound,
    winnable,
)

COLORS = "RYGBW"
LIVE_COLORS = "RYGBP"  # hanab.live's No Variant: red, yellow, green, blue, purple
UNSUPPORTED_OPTIONS = (
    "deckPlays",
    "oneExtraCard",
    "oneLessCard",
    "allOrNothing",
    "detrimentalCharacters",
)


def name(card: int, rules: Rules, colors: str = COLORS) -> str:
    return f"{colors[color_of(card, rules)]}{rank_of(card, rules)}"


def parse_deck(text: str, rules: Rules) -> list[int]:
    """'R1 Y3 ...' (commas allowed, W or P for the fifth color) as cards, checked to be a
    full deck."""
    cards = []
    for word in text.replace(",", " ").split():
        letter, rank = word[0].upper(), int(word[1:])
        color = "RYGBWP".index(letter) if letter in "RYGBWP" else -1
        if color < 0 or not 1 <= rank <= rules.ranks:
            raise ValueError(f"not a card: {word}")
        cards.append(make_card(min(color, 4), rank, rules))
    if sorted(cards) != full_deck(rules):
        have, want = Counter(cards), Counter(full_deck(rules))
        extra = sorted(name(c, rules) for c in (have - want).elements())
        short = sorted(name(c, rules) for c in (want - have).elements())
        raise ValueError(f"not a full deck: extra {extra}, missing {short}")
    return cards


# --- hanab.live exports ------------------------------------------------------------------


@dataclass
class LiveGame:
    """A hanab.live game in this package's terms. hanab.live numbers the cards by their
    place in the deck ("order"), and plays and discards name a card by that number."""

    rules: Rules
    deck: list[int]
    actions: list[tuple[str, int, int]]  # (kind, card order or target player, value)
    names: list[str]
    game_id: int | None = None


def parse_live(data: dict) -> LiveGame:
    """Read a hanab.live JSON export: format 4 (actions named "play", "discard", "clue")
    or the older numbered actions (0 play, 1 discard, 2 color clue, 3 rank clue, 4 end).

    The format's own description (misc/example_game_with_comments.jsonc in the site's
    repository) fixes the conventions: the deck is listed from the top, the hands are dealt
    in blocks to the first player, then the second, and the first player moves first. The
    site's changelog records that the old starting-player option was removed, together with
    the games from before April 2020 that did not start with the first player, so an export
    that names another starting player is refused rather than guessed at."""
    opts = data.get("options") or {}
    variant = opts.get("variantName", opts.get("variant", "No Variant"))
    if variant != "No Variant":
        raise ValueError(f"only No Variant is supported, not {variant!r}")
    bad = [o for o in UNSUPPORTED_OPTIONS if opts.get(o)]
    if bad:
        raise ValueError(f"options not supported: {bad}")
    if int(opts.get("startingPlayer", data.get("startingPlayer", 0)) or 0):
        raise ValueError("games that do not start with the first player are not supported")
    names = list(data.get("playerNames") or data.get("players") or [])
    if not names and data.get("seed", "").startswith("p"):
        names = [f"player {i + 1}" for i in range(int(data["seed"][1:].split("v")[0]))]
    rules = Rules(players=len(names))
    if not 2 <= rules.players <= 5:
        raise ValueError(f"{rules.players} players: only 2 to 5 are supported")
    deck = [make_card(c["suitIndex"], c["rank"], rules) for c in data["deck"]]
    if sorted(deck) != full_deck(rules):
        raise ValueError("the deck is not a full No Variant deck")
    actions = []
    for a in data.get("actions", []):
        kind = a["type"]
        if kind in (0, "play"):
            actions.append(("play", a.get("order", a.get("target")), 0))
        elif kind in (1, "discard"):
            actions.append(("discard", a.get("order", a.get("target")), 0))
        elif kind in (2, 3):
            actions.append(("color" if kind == 2 else "rank", a["target"], a["value"]))
        elif kind == "clue":
            actions.append((a["clue"]["type"], a["target"], a["clue"]["value"]))
        elif kind in (4, "gameOver"):
            break
        else:
            raise ValueError(f"unknown action {a}")
    return LiveGame(rules, deck, actions, names, data.get("gameId", data.get("id")))


def live_states(game: LiveGame) -> tuple[list[State], list[Action]]:
    """Replay the export with the rules engine: the state before each turn (and the final
    state last) and the actions. The replay keeps track of which card number sits in which
    slot, since `hanabi.rules` names a card by its slot."""
    rules = game.rules
    s = new_game(rules, game.deck)
    p, h = rules.players, rules.hand
    held = [list(range(q * h, (q + 1) * h)) for q in range(p)]
    nxt = p * h
    states, acts = [copy_state(s)], []
    for kind, x, value in game.actions:
        cur = s.current
        if kind in ("play", "discard"):
            if x not in held[cur]:
                raise ValueError(f"turn {s.turn + 1}: card {x} is not in the hand of player {cur}")
            i = held[cur].index(x)
            a = play(i) if kind == "play" else discard(i)
            held[cur].pop(i)
            if nxt < len(game.deck):
                held[cur].append(nxt)
                nxt += 1
        else:
            a = hint(x, kind, value)
        if not legal(s, a):
            raise ValueError(f"illegal action on turn {s.turn + 1}: {kind} {x} {value}")
        step(s, a)
        states.append(copy_state(s))
        acts.append(a)
        if s.over:
            break
    return states, acts


# --- certificates in words -----------------------------------------------------------------


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def time_witness(rules: Rules, deck: list[int]) -> tuple[int, int, int]:
    """(bound, i, g) where the time bound is reached: before the i-th draw at most g cards
    are playable, and bound = g + (draws left after it) + final turns."""
    d = Deal(rules, deck)
    nr, n0 = rules.ranks, rules.players * rules.hand
    nd = len(deck) - n0
    have = set(deck[:n0])
    heights = chain_heights((0,) * rules.colors, have, nr)
    best = (nd + rules.players, 0, 0)
    for i in range(1, nd + 1):
        g = sum(heights)
        best = min(best, (g + nd - i + rules.players, i, g))
        card = deck[n0 + i - 1]
        have.add(card)
        c = card // nr
        while heights[c] < nr and c * nr + heights[c] in have:
            heights[c] += 1
    hands = tuple(tuple(deck[i * rules.hand : (i + 1) * rules.hand]) for i in range(rules.players))
    bound = time_bound(d, (0,) * rules.colors, hands, n0, rules.players)
    assert min(best[0], rules.max_score) == bound, (best, bound)  # the same bound as the solver's
    return best


def explain(rules: Rules, deck: list[int], c: Ceiling, colors: str = COLORS) -> list[str]:
    """One sentence for each certificate that rules out 25."""
    n0, h, p = rules.players * rules.hand, rules.hand, rules.players
    out = []
    if "time" in c.certificates:
        bound, i, g = time_witness(rules, deck)
        if bound < 25:
            card = name(deck[n0 + i - 1], rules, colors)
            out.append(
                f"time: the turns that draw the first {i} cards of the deck can play only "
                f"cards dealt or drawn before the {ordinal(i)} draw ({card}), and those allow "
                f"{g} plays; then come {50 - n0 - i} more draws and {p} final turns, so at "
                f"most {g} + {50 - n0 - i} + {p} = {bound} cards are played."
            )
        else:
            out.append(
                "time: the plays cannot all fit into the turns, counting the cards that are "
                "drawn too late."
            )
    if "storage" in c.certificates:
        d = Deal(rules, deck)
        hands = tuple(tuple(deck[q * h : (q + 1) * h]) for q in range(p))
        i = storage_violation(d, (0,) * rules.colors, hands, n0)
        out.append(
            f"storage: after the {i}th draw, more cards that are still needed and have no copy "
            f"left in the deck must sit in the hands at once than the {p * h} slots can hold."
        )
    if "pooled" in c.certificates:
        out.append(
            "pooled game: even one player holding every hand at once, with no turn order and "
            "no hint tokens, cannot reach 25 with these draws."
        )
    if "last card" in c.certificates:
        out.append(
            f"last card: the last card, {name(deck[-1], rules, colors)}, is not a 5, so a "
            "won game never plays it; with it removed the deal fails the tests above."
        )
    if "free-order" in c.certificates:
        out.append(
            "free-order game: even with unlimited hint tokens and lives, so that turns may be "
            "passed freely, the hands cannot play all 25 cards in time."
        )
    if not out and c.high < 25:
        out.append("complete search: no line of the open game reaches 25.")
    return out


def describe_line(rules: Rules, deck: list[int], target: int, colors: str, names) -> list[str]:
    """A line of the open game that reaches the target, one move per line."""
    ok, line = winnable(rules, deck, 300_000, target=target)
    if not ok:
        return []
    s = new_game(rules, deck)
    out = []
    for player, kind, card in line:
        if card == TRASH and kind != "pass":
            card = next(x for x in s.hands[player] if x % rules.ranks < s.board[x // rules.ranks])
        what = "hints (passes)" if kind == "pass" else f"{kind}s {name(card, rules, colors)}"
        out.append(f"{s.turn + 1:>3}. {names[player]} {what}")
        if kind == "pass":
            t = next(q for q in range(rules.players) if q != player and s.hands[q])
            a = hint(t, "rank", s.hands[t][0] % rules.ranks + 1)
        else:
            i = s.hands[player].index(card)
            a = play(i) if kind in ("play", "misplay") else discard(i)
        step(s, a)
    return out


def drops(states: list[State], acts: list[Action], start: int, budgets=BUDGETS):
    """The turns after which the ceiling fell, each with the score it fell below, latest
    target first; None for a turn the searches could not settle."""
    rules, deck = states[0].rules, states[0].deck
    out = []
    hi = len(states) - 1
    for target in range(states[-1].score + 1, start + 1):
        t = first_below(rules, deck, states, target, 0, hi, budgets)
        out.append((target, t))
        if t is None:
            break
        hi = t
    return out


def report(rules, deck, colors, names, states=None, acts=None, show_line=False, file=sys.stdout):
    c = ceiling(rules, deck)
    say = lambda *a: print(*a, file=file)  # noqa: E731
    how = "exact" if c.exact else f"between {c.low} and {c.high}; a search ran out of nodes"
    say(f"Best possible score: {c.high if c.exact else f'{c.low} to {c.high}'} ({how}).")
    for sentence in explain(rules, deck, c, colors):
        say(f"  {sentence}")
    if show_line and c.low:
        say(f"A line that scores {c.low}, with every card in view:")
        for x in describe_line(rules, deck, c.low, colors, names):
            say(f"  {x}")
    if states and acts:
        final = states[-1]
        say(f"Recorded score: {final.score} over {len(acts)} turns.")
        if final.score >= c.low:
            say("  The team reached the ceiling.")
        elif c.exact:
            for target, t in sorted(drops(states, acts, c.low), key=lambda x: x[1] or 0):
                if t is None:
                    say(f"  the turn the ceiling fell below {target} could not be settled")
                    continue
                before, a = states[t - 1], acts[t - 1]
                kind = action_kind(before, states[t], a)
                who = names[before.current]
                if a.kind == "hint":
                    what = "hinted"
                else:
                    card = name(before.hands[before.current][a.index], rules, colors)
                    what = {
                        "play": f"played {card}",
                        "strike": f"misplayed {card}",
                        "critical discard": f"discarded {card}, the last copy",
                        "discard": f"discarded {card}",
                    }[kind]
                left = before.deck_left
                say(
                    f"  turn {t}: {who} {what} ({left} cards in the deck, {before.hints} "
                    f"hints); the ceiling fell below {target}"
                )
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--deck", help="50 cards from the top, e.g. 'R1 Y3 ...'")
    src.add_argument("--json", help="a hanab.live JSON export (- for standard input)")
    src.add_argument("--seed", type=int, help="a seeded deal of this package")
    ap.add_argument("--players", type=int, default=2, choices=range(2, 6))
    ap.add_argument("--line", action="store_true", help="print a line that reaches the ceiling")
    args = ap.parse_args(argv)
    if args.json:
        text = sys.stdin.read() if args.json == "-" else Path(args.json).read_text()
        game = parse_live(json.loads(text))
        states, acts = live_states(game)
        print(f"{game.rules.players} players: {', '.join(game.names)}")
        report(game.rules, game.deck, LIVE_COLORS, game.names, states, acts, args.line)
        return
    rules = Rules(players=args.players)
    deck = seeded_deck(rules, args.seed) if args.seed is not None else parse_deck(args.deck, rules)
    names = [f"player {i + 1}" for i in range(rules.players)]
    report(rules, deck, COLORS, names, show_line=args.line)


if __name__ == "__main__":
    main()
