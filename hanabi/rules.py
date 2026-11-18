"""The rules of Hanabi as small functions, with seeded deals.

Vocabulary used throughout the package:

- A **card** is an int, `color * ranks + (rank - 1)`, so that a whole deck is a list of ints.
  `color_of` and `rank_of` take it apart; ranks are printed 1 to 5.
- A **deck** is the list of cards in the order they are dealt. The first `players * hand_size`
  cards go to the hands, in blocks (the first block to player 0), and the rest are drawn from
  the front.
- The **board** holds, for each color, the highest rank played so far (0 when none).
- The **score** is the sum of the board, 25 at most in the standard game.

The rules follow the official rule sheet and the Hanabi Learning Environment (Bard et al.,
2020): 8 hint tokens and 3 lives; a hint names one color or one rank and must touch at least
one card of the player it is given to; a discard is not allowed while all 8 tokens are on the
table; a played 5 returns a token; the game ends at 25, when the third life is lost (score 0),
or one full round after the last card is drawn, so that every player, including the one who
drew it, has one more turn. `Rules` has switches for the readings that other sources use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import NamedTuple

COLOR_NAMES = "RYGBW"  # red, yellow, green, blue, white
STANDARD_COUNTS = (3, 2, 2, 2, 1)  # copies of ranks 1 to 5 in each color


@dataclass(frozen=True)
class Rules:
    """One reading of the rules. The defaults are the standard game for `players` players."""

    players: int = 2
    colors: int = 5
    counts: tuple[int, ...] = STANDARD_COUNTS  # copies of each rank, lowest rank first
    hand_size: int = 0  # 0 means the standard size: 5 cards for 2 or 3 players, 4 for 4 or 5
    max_hints: int = 8
    lives: int = 3
    discard_when_full: bool = False  # allow a discard while all hint tokens are on the table
    zero_on_last_life: bool = True  # losing the last life scores 0 (HLE); False keeps the board

    @property
    def ranks(self) -> int:
        return len(self.counts)

    @property
    def hand(self) -> int:
        if self.hand_size:
            return self.hand_size
        return 5 if self.players <= 3 else 4

    @property
    def max_score(self) -> int:
        return self.colors * self.ranks

    @property
    def deck_size(self) -> int:
        return self.colors * sum(self.counts)


def color_of(card: int, rules: Rules) -> int:
    return card // rules.ranks


def rank_of(card: int, rules: Rules) -> int:
    """Rank from 1 up."""
    return card % rules.ranks + 1


def make_card(color: int, rank: int, rules: Rules) -> int:
    return color * rules.ranks + rank - 1


def card_name(card: int, rules: Rules) -> str:
    """'R1' for the red 1, and so on."""
    return f"{COLOR_NAMES[color_of(card, rules)]}{rank_of(card, rules)}"


def full_deck(rules: Rules) -> list[int]:
    """Every card once per copy, sorted by color, then rank."""
    return [
        make_card(c, r + 1, rules)
        for c in range(rules.colors)
        for r, n in enumerate(rules.counts)
        for _ in range(n)
    ]


def copies(card: int, rules: Rules) -> int:
    """How many copies of this card the deck holds."""
    return rules.counts[card % rules.ranks]


def seeded_deck(rules: Rules, seed: int) -> list[int]:
    """The deck for deal number `seed`: `random.Random(seed)` shuffles the sorted deck."""
    deck = full_deck(rules)
    random.Random(seed).shuffle(deck)
    return deck


class Drand48:
    """The C library's drand48, so that decks match those of the HanSim C++ simulator."""

    A, C, MASK = 0x5DEECE66D, 0xB, (1 << 48) - 1

    def __init__(self, seed: int):
        self.x = ((seed & 0xFFFFFFFF) << 16) | 0x330E

    def __call__(self) -> float:
        self.x = (self.A * self.x + self.C) & self.MASK
        return self.x / (1 << 48)


def hansim_decks(seed: int = 100):
    """The decks HanSim deals after srand48(seed), one game after another (5 players)."""
    rng = Drand48(seed)
    rules = Rules(players=5)
    while True:
        deck = full_deck(rules)
        for j in range(len(deck) - 1, -1, -1):
            k = int(rng() * (j + 1))
            deck[k], deck[j] = deck[j], deck[k]
        yield deck


class Action(NamedTuple):
    """kind is 'play', 'discard' or 'hint'. A hint has target, attr ('color' or 'rank'), value."""

    kind: str
    index: int = -1
    target: int = -1
    attr: str = ""
    value: int = -1


def play(i: int) -> Action:
    return Action("play", i)


def discard(i: int) -> Action:
    return Action("discard", i)


def hint(target: int, attr: str, value: int) -> Action:
    """value is a color index (0 to 4) or a rank (1 to 5)."""
    return Action("hint", target=target, attr=attr, value=value)


@dataclass
class State:
    """A game in progress. Hands list the oldest card first; a drawn card goes at the end.

    `knowledge[p][i]` is a pair of bit masks: the colors and the ranks (bit r - 1) that the
    hints so far leave possible for card i of player p, as the Hanabi Learning Environment
    tracks them. It ignores card counting; strategies keep their own records.
    """

    rules: Rules
    deck: list[int]
    hands: list[list[int]] = field(default_factory=list)
    knowledge: list[list[tuple[int, int]]] = field(default_factory=list)
    board: list[int] = field(default_factory=list)
    pos: int = 0  # index of the next card to draw
    hints: int = 0
    lives: int = 0
    current: int = 0
    turns_left: int = -1  # turns still to come after the deck ran out; -1 while it has cards
    discards: list[int] = field(default_factory=list)  # discarded and misplayed cards, in order
    turn: int = 0
    over: bool = False

    @property
    def score(self) -> int:
        if self.lives == 0 and self.rules.zero_on_last_life:
            return 0
        return sum(self.board)

    @property
    def deck_left(self) -> int:
        return len(self.deck) - self.pos


def new_game(rules: Rules, deck: list[int]) -> State:
    """Deal the hands in blocks from the front of the deck."""
    assert sorted(deck) == full_deck(rules), "not a full deck for these rules"
    h, p = rules.hand, rules.players
    unknown = ((1 << rules.colors) - 1, (1 << rules.ranks) - 1)
    return State(
        rules=rules,
        deck=list(deck),
        hands=[list(deck[i * h : (i + 1) * h]) for i in range(p)],
        knowledge=[[unknown] * h for _ in range(p)],
        board=[0] * rules.colors,
        pos=p * h,
        hints=rules.max_hints,
        lives=rules.lives,
    )


def playable(card: int, board: list[int], rules: Rules) -> bool:
    return rank_of(card, rules) == board[color_of(card, rules)] + 1


def touched(card: int, attr: str, value: int, rules: Rules) -> bool:
    """Does a hint of this color or rank touch the card?"""
    if attr == "color":
        return color_of(card, rules) == value
    return rank_of(card, rules) == value


def legal(state: State, action: Action) -> bool:
    """Is the action allowed for the current player?"""
    rules, hand = state.rules, state.hands[state.current]
    if state.over:
        return False
    if action.kind == "play":
        return 0 <= action.index < len(hand)
    if action.kind == "discard":
        room = rules.discard_when_full or state.hints < rules.max_hints
        return room and 0 <= action.index < len(hand)
    if action.kind == "hint":
        if state.hints < 1 or not 0 <= action.target < rules.players:
            return False
        if action.target == state.current:
            return False
        if action.attr == "color":
            if not 0 <= action.value < rules.colors:
                return False
        elif action.attr != "rank" or not 1 <= action.value <= rules.ranks:
            return False
        return any(touched(c, action.attr, action.value, rules) for c in state.hands[action.target])
    return False


def legal_actions(state: State) -> list[Action]:
    """Every legal action for the current player, plays first, then discards, then hints."""
    rules, n = state.rules, len(state.hands[state.current])
    actions = [play(i) for i in range(n)] + [discard(i) for i in range(n)]
    for t in range(rules.players):
        actions += [hint(t, "color", v) for v in range(rules.colors)]
        actions += [hint(t, "rank", v) for v in range(1, rules.ranks + 1)]
    return [a for a in actions if legal(state, a)]


class Event(NamedTuple):
    """What everyone sees after an action: the card shown (or -1) and, for a hint, the
    positions it touched in the target's hand."""

    player: int
    action: Action
    card: int = -1
    success: bool = False
    touched: tuple[int, ...] = ()


def _remove_and_draw(state: State, p: int, i: int) -> int:
    card = state.hands[p].pop(i)
    state.knowledge[p].pop(i)
    if state.pos < len(state.deck):
        state.hands[p].append(state.deck[state.pos])
        rules = state.rules
        state.knowledge[p].append(((1 << rules.colors) - 1, (1 << rules.ranks) - 1))
        state.pos += 1
        if state.pos == len(state.deck):
            state.turns_left = rules.players + 1  # this turn, then one more for each player
    return card


def step(state: State, action: Action) -> Event:
    """Carry out a legal action for the current player and pass the turn."""
    if not legal(state, action):
        raise ValueError(f"illegal action {action} for player {state.current}")
    rules, p = state.rules, state.current
    if action.kind == "hint":
        state.hints -= 1
        hits = []
        for i, c in enumerate(state.hands[action.target]):
            cm, rm = state.knowledge[action.target][i]
            bit = 1 << (action.value if action.attr == "color" else action.value - 1)
            hit = touched(c, action.attr, action.value, rules)
            if action.attr == "color":
                cm = cm & bit if hit else cm & ~bit
            else:
                rm = rm & bit if hit else rm & ~bit
            state.knowledge[action.target][i] = (cm, rm)
            if hit:
                hits.append(i)
        event = Event(p, action, touched=tuple(hits))
    else:
        card = _remove_and_draw(state, p, action.index)
        success = False
        if action.kind == "discard":
            state.discards.append(card)
            state.hints = min(state.hints + 1, rules.max_hints)
        elif playable(card, state.board, rules):
            success = True
            state.board[color_of(card, rules)] += 1
            if rank_of(card, rules) == rules.ranks:
                state.hints = min(state.hints + 1, rules.max_hints)
        else:
            state.discards.append(card)
            state.lives -= 1
        event = Event(p, action, card, success)
    state.turn += 1
    if state.turns_left > 0:
        state.turns_left -= 1
    if state.lives == 0 or sum(state.board) == rules.max_score or state.turns_left == 0:
        state.over = True
    state.current = (p + 1) % rules.players
    return event


def play_game(rules: Rules, deck: list[int], choose, max_turns: int = 1000) -> State:
    """Play a game where `choose(state)` returns the current player's action."""
    state = new_game(rules, deck)
    while not state.over and state.turn < max_turns:
        step(state, choose(state))
    return state
