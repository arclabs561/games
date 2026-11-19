"""A text interface for language-model players, with no model attached.

`observation(state, player, events)` writes what one player may know as plain text: the
board, tokens, discards, the other hands, the hints received on each own card, the recent
actions, and the legal moves in a fixed syntax. With `deduction=True` it adds, for each own
card, the values still possible after hints and card counting, as the "Sherlock" setting of
Ramesh et al., "Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents"
(arXiv:2601.18077), supplies. `parse_action` reads the last move written in that syntax.

`play_text_game` runs a game with any function from prompt to reply. The tests use scripted
replies. `h2.play_game` is the same loop with a record of each move; it plays hosted models,
and the README reports the results.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from hanabi.rules import (
    COLOR_NAMES,
    Action,
    Event,
    Rules,
    State,
    card_name,
    color_of,
    copies,
    discard,
    hint,
    legal,
    legal_actions,
    new_game,
    play,
    rank_of,
    step,
)

COLOR_WORDS = ["red", "yellow", "green", "blue", "white"]

RULES_TEXT = """You are playing Hanabi, a cooperative card game. Five colors (red, yellow, green,
blue, white), ranks 1 to 5; each color has three 1s, two each of 2, 3 and 4, and one 5. You see
the other hands but not your own. The team builds one pile per color from 1 up to 5. On your
turn you either play a card (a wrong play costs a life), discard a card (gains a hint token; not
allowed when all 8 tokens are available), or spend a token to give a hint: name one color or
one rank to another player, which shows all their cards of that color or rank. The game ends
when the third life is lost (score 0), when all 25 cards are played, or one round after the deck
runs out. Score: cards played."""


def move_text(action: Action, rules: Rules) -> str:
    """The syntax `parse_action` reads: PLAY 2, DISCARD 1, HINT 3 RED, HINT 3 4."""
    if action.kind == "hint":
        what = COLOR_WORDS[action.value] if action.attr == "color" else str(action.value)
        return f"HINT {action.target} {what.upper()}"
    return f"{action.kind.upper()} {action.index + 1}"


def event_text(event: Event, rules: Rules) -> str:
    a = event.action
    if a.kind == "hint":
        what = COLOR_WORDS[a.value] if a.attr == "color" else f"rank {a.value}"
        where = ", ".join(str(i + 1) for i in event.touched)
        return f"Player {event.player} told player {a.target}: {what} at position(s) {where}."
    name = card_name(event.card, rules)
    if a.kind == "discard":
        return f"Player {event.player} discarded card {a.index + 1}, a {name}."
    outcome = "successfully" if event.success else "but it did not fit and cost a life"
    return f"Player {event.player} played card {a.index + 1}, a {name}, {outcome}."


def mask_text(mask: int, names) -> str:
    return "".join(str(n) for i, n in enumerate(names) if mask >> i & 1)


def possible_values(state: State, player: int) -> list[list[int]]:
    """For each own card, the card values left by hints and by counting the cards the player
    sees (other hands, board, discards). Counting ignores what other players' actions imply."""
    rules = state.rules
    seen: Counter[int] = Counter(state.discards)
    for c in range(rules.colors):
        for r in range(1, state.board[c] + 1):
            seen[c * rules.ranks + r - 1] += 1
    for p, hand in enumerate(state.hands):
        if p != player:
            seen.update(hand)
    out = []
    for cm, rm in state.knowledge[player]:
        vals = [
            t
            for t in range(rules.colors * rules.ranks)
            if cm >> color_of(t, rules) & 1
            and rm >> (rank_of(t, rules) - 1) & 1
            and seen[t] < copies(t, rules)
        ]
        out.append(vals)
    return out


def observation(
    state: State, player: int, events: list[Event], deduction: bool = False, last: int = 10
) -> str:
    """Everything player `player` may know, as text."""
    rules = state.rules
    lines = [RULES_TEXT, "", f"You are player {player} of {rules.players}."]
    board = ", ".join(f"{COLOR_WORDS[c]} {state.board[c]}" for c in range(rules.colors))
    lines.append(f"Piles: {board}. Score so far: {sum(state.board)}.")
    lines.append(
        f"Hint tokens: {state.hints} of {rules.max_hints}. Lives: {state.lives}. "
        f"Cards left in the deck: {state.deck_left}."
    )
    if state.turns_left >= 0:
        lines.append(f"The deck is empty; turns left in the game: {state.turns_left}.")
    gone = " ".join(card_name(c, rules) for c in sorted(state.discards)) or "none"
    lines.append(f"Discarded or misplayed: {gone}.")
    for p in range(rules.players):
        if p != player:
            cards = " ".join(card_name(c, rules) for c in state.hands[p])
            lines.append(f"Player {p} holds (oldest first): {cards}.")
    lines.append("Your cards, oldest first, with what the hints you received allow:")
    poss = possible_values(state, player) if deduction else None
    for i, (cm, rm) in enumerate(state.knowledge[player]):
        text = (
            f"  card {i + 1}: colors {mask_text(cm, COLOR_NAMES[: rules.colors])}, "
            f"ranks {mask_text(rm, range(1, rules.ranks + 1))}"
        )
        if poss is not None:
            text += "; could be " + " ".join(card_name(t, rules) for t in poss[i])
        lines.append(text)
    if events:
        lines.append("Recent actions:")
        lines += ["  " + event_text(e, rules) for e in events[-last:]]
    moves = ", ".join(move_text(a, rules) for a in legal_actions(state))
    lines.append(f"Legal moves: {moves}.")
    lines.append("Think it through, then end with one line holding only your move.")
    return "\n".join(lines)


MOVE = re.compile(r"\b(PLAY|DISCARD)\s+(\d+)\b|\bHINT\s+(\d+)\s+([A-Z]+|\d)\b", re.IGNORECASE)


def parse_action(text: str, state: State) -> Action | None:
    """The last move written in the reply, if it is legal now; None otherwise."""
    rules = state.rules
    for m in reversed(list(MOVE.finditer(text))):
        if m.group(1):
            i = int(m.group(2)) - 1
            a = play(i) if m.group(1).upper() == "PLAY" else discard(i)
        else:
            target, what = int(m.group(3)), m.group(4).lower()
            if what.isdigit():
                a = hint(target, "rank", int(what))
            elif what in COLOR_WORDS[: rules.colors]:
                a = hint(target, "color", COLOR_WORDS.index(what))
            else:
                continue
        return a if legal(state, a) else None
    return None


def play_text_game(
    rules: Rules,
    deck: list[int],
    complete: Callable[[str], str],
    deduction: bool = False,
    fallback: Callable[[State], Action] | None = None,
) -> tuple[State, int]:
    """Play one game where each move comes from `complete(prompt)`. An unreadable or illegal
    reply gets the fallback move (by default the first legal discard, else the first legal
    move). Returns the final state and the number of fallbacks."""
    state = new_game(rules, deck)
    events: list[Event] = []
    misses = 0
    while not state.over:
        reply = complete(observation(state, state.current, events, deduction))
        action = parse_action(reply, state)
        if action is None:
            misses += 1
            if fallback is not None:
                action = fallback(state)
            else:
                moves = legal_actions(state)
                action = next((a for a in moves if a.kind == "discard"), moves[0])
        events.append(step(state, action))
    return state, misses
