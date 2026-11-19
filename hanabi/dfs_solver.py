"""A second open-game solver, written apart from `hanabi.solver` to check it.

It shares no search code with `hanabi.solver`: it has its own state, its own moves, its
own bounds and a different search. It uses only the deck and the rules from
`hanabi.rules`, and its winning lines are replayed there.

The search is plain depth-first search with a table of states already found lost. There is
no discrepancy limit and no restart. A state is cut off when one of three tests shows it
lost:

- *missing card*: some card still needed has no copy left in the hands or the deck.
- *schedule*: count the plays as unit jobs on one machine. Each play or discard draws one
  card while the deck lasts, so the k-th such action from now can play only cards held now
  or drawn by the first k - 1 actions, and after the last draw there are at most as many
  plays as turns left. Within a color the cards go in order. Greedy scheduling (play some
  available card whenever one exists) completes the most jobs: at the last slot where
  greedy finds nothing to play, every unfinished color waits for a card not yet drawn,
  so no schedule has finished more cards by then, and greedy plays in every slot after it.
- *free order*: the game in which hint tokens are unlimited, so that a pass is always
  allowed and the players may act in any order until the last card is drawn; lives are
  unlimited too, so a misplay is a discard. After the last draw, the remaining turns go
  in the real order, each player playing at most one card from their own hand. Every
  winning line of the real game, with its passes left out, wins the free-order game. Unlike
  the pooled game of `hanabi.solver`, it keeps each card in the hand of the player who drew
  it, and it keeps the order of the final round.

The free-order game is itself solved by depth-first search with a table, with the schedule
test as its only cut.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from hanabi.rules import (
    Action,
    Rules,
    State,
    color_of,
    discard,
    hint,
    legal,
    new_game,
    play,
    step,
)

GONE = 99  # a card that can never be played; all such cards are alike

type Pile = tuple[int, ...]  # the height of each color
type Holding = tuple[tuple[int, ...], ...]  # each player's hand, sorted


class Game:
    """One deal, seen by the second solver. Cards are pairs folded into ints c * R + r."""

    def __init__(self, rules: Rules, deck: list[int]):
        self.rules = rules
        self.deck = tuple(deck)
        self.R = rules.ranks
        self.C = rules.colors
        self.P = rules.players
        self.H = rules.hand
        self.goal = rules.max_score

    def needed(self, card: int, pile: Pile) -> bool:
        return card != GONE and card % self.R >= pile[card // self.R]

    def can_play(self, card: int, pile: Pile) -> bool:
        return card != GONE and card % self.R == pile[card // self.R]

    def tidy(self, cards, pile: Pile) -> tuple[int, ...]:
        """The hand with dead cards, and second copies within the hand, written as GONE:
        only one copy of a card is ever played, so a second copy in the same hand is as
        useless as a dead card."""
        out, seen = [], set()
        for c in cards:
            out.append(c if self.needed(c, pile) and c not in seen else GONE)
            seen.add(c)
        return tuple(sorted(out))

    def start(self) -> tuple[Pile, Holding, int]:
        pile = (0,) * self.C
        hands = tuple(
            self.tidy(self.deck[p * self.H : (p + 1) * self.H], pile) for p in range(self.P)
        )
        return pile, hands, self.P * self.H


# --- tests that a state is lost -----------------------------------------------------------


def missing_card(g: Game, pile: Pile, hands: Holding, pos: int) -> bool:
    """Is every copy of some needed card gone?"""
    left = set(g.deck[pos:])
    for h in hands:
        left.update(h)
    for c in range(g.C):
        for r in range(pile[c], g.R):
            if c * g.R + r not in left:
                return True
    return False


def schedule_plays(g: Game, pile: Pile, hands: Holding, pos: int, last_turns: int) -> int:
    """The most plays possible from here, by greedy scheduling of unit jobs (see above).

    last_turns is the number of turns that follow the last draw."""
    have = Counter(c for h in hands for c in h if c != GONE)
    top = list(pile)
    plays = 0

    def play_one() -> bool:
        for c in range(g.C):
            if top[c] < g.R and have[c * g.R + top[c]]:
                have[c * g.R + top[c]] -= 1
                top[c] += 1
                return True
        return False

    for k in range(pos, len(g.deck)):  # each action draws deck[k] after it acts
        if play_one():
            plays += 1
        have[g.deck[k]] += 1
    for _ in range(last_turns):
        if play_one():
            plays += 1
    return plays


def one_hand_winnable(g: Game, pile: Pile, pool: tuple[int, ...], pos: int, memo) -> bool:
    """The one-hand game: all hands held as one, any order, no tokens, and at most
    `g.P` plays after the last draw. It relaxes the free-order game further.

    Here a playable card may be played at once (the same card is drawn either way, and the
    played card frees the pile for others), a dead card or a second copy is the best
    discard, and otherwise any distinct card may go."""
    pile_l, cards, p = list(pile), list(pool), pos
    progress = True
    while progress and p < len(g.deck):
        progress = False
        for c in cards:
            if g.can_play(c, tuple(pile_l)):
                cards.remove(c)
                pile_l[c // g.R] += 1
                cards.append(g.deck[p])
                p += 1
                progress = True
                break
    pile = tuple(pile_l)
    if sum(pile) == g.goal:
        return True
    if p == len(g.deck):
        tops = list(pile)
        for _ in range(g.P):
            for c in range(g.C):
                if tops[c] < g.R and c * g.R + tops[c] in cards:
                    tops[c] += 1
                    break
        return sum(tops) == g.goal
    seen: set[int] = set()
    tidy = []
    for c in cards:
        tidy.append(GONE if not g.needed(c, pile) or c in seen else c)
        seen.add(c)
    key = ("one hand", pile, tuple(sorted(tidy)), p)
    if key in memo:
        return memo[key]
    if missing_card(g, pile, (tuple(tidy),), p):
        memo[key] = False
        return False
    choices = [GONE] if GONE in tidy else sorted(set(tidy))
    ok = False
    for c in choices:
        rest = list(tidy)
        rest.remove(c)
        rest.append(g.deck[p])
        if one_hand_winnable(g, pile, tuple(rest), p + 1, memo):
            ok = True
            break
    memo[key] = ok
    return ok


def turned(hands: Holding) -> Holding:
    """The hands rotated to the least rotation. The free-order game before the last draw
    does not change when the seats are rotated."""
    return min(hands[i:] + hands[:i] for i in range(len(hands)))


class Budget:
    """A count of states that a search may still expand."""

    def __init__(self, n: int):
        self.n = n

    def spend(self) -> bool:
        self.n -= 1
        return self.n >= 0


def free_order_winnable(
    g: Game, pile: Pile, hands: Holding, pos: int, cur: int, left: int, memo, budget=None
):
    """Can the free-order game (see above) be won from here? True, False, or None when the
    budget (a `Budget`, or None for no limit) runs out. `left` is the number of turns after
    the last draw still to come, or -1 while the deck has cards; `cur` matters only then.
    `memo` holds the states already settled."""
    if sum(pile) == g.goal:
        return True
    if left >= 0:
        return final_round(g, pile, hands, cur, left)
    key = (pile, turned(hands), pos)
    if key in memo:
        return memo[key]
    need = g.goal - sum(pile)
    if (
        missing_card(g, pile, hands, pos)
        or schedule_plays(g, pile, hands, pos, g.P) < need
        or not one_hand_winnable(g, pile, tuple(c for h in hands for c in h), pos, memo)
    ):
        memo[key] = False
        return False
    if budget is not None and not budget.spend():
        return None
    unsure = False
    # Plays first. A player who holds a dead card discards nothing else: the hand that
    # keeps the other card can do whatever the hand that keeps the dead card can.
    tries = [(p, c) for p in range(g.P) for c in sorted(set(hands[p])) if g.can_play(c, pile)]
    for p in range(g.P):
        rest_cards = [c for c in sorted(set(hands[p]), reverse=True) if not g.can_play(c, pile)]
        tries += [(p, c) for c in ([GONE] if GONE in rest_cards else rest_cards)]
    for p, card in tries:
        new_pile = pile
        if g.can_play(card, pile):
            new_pile = list(pile)
            new_pile[card // g.R] += 1
            new_pile = tuple(new_pile)
        rest = list(hands[p])
        rest.remove(card)
        rest.append(g.deck[pos])
        new_hands = tuple(g.tidy(rest if q == p else hands[q], new_pile) for q in range(g.P))
        nxt = (p + 1) % g.P
        last = pos + 1 == len(g.deck)
        ok = free_order_winnable(
            g, new_pile, new_hands, pos + 1, nxt, g.P if last else -1, memo, budget
        )
        if ok:
            memo[key] = True
            return True
        if ok is None:
            unsure = True
            if budget is not None and budget.n < 0:
                return None
    if unsure:
        return None
    memo[key] = False
    return False


def final_round(g: Game, pile: Pile, hands: Holding, cur: int, left: int) -> bool:
    """With no cards left to draw, `left` turns in order from player `cur`, each player
    playing at most one card from their own hand: can the goal be reached?"""
    if sum(pile) == g.goal:
        return True
    if left == 0:
        return False
    nxt = (cur + 1) % g.P
    for card in set(hands[cur]):
        if g.can_play(card, pile):
            new_pile = list(pile)
            new_pile[card // g.R] += 1
            rest = list(hands[cur])
            rest.remove(card)
            new_hands = tuple(tuple(rest) if q == cur else hands[q] for q in range(g.P))
            if final_round(g, tuple(new_pile), new_hands, nxt, left - 1):
                return True
    return final_round(g, pile, hands, nxt, left - 1)


def free_order_test(rules: Rules, deck: list[int], budget: int = 0):
    """Can the free-order game of this deal be won? True, False, or None when `budget`
    states (0 for no limit) were not enough."""
    g = Game(rules, deck)
    pile, hands, pos = g.start()
    return free_order_winnable(g, pile, hands, pos, 0, -1, {}, Budget(budget) if budget else None)


def certificates(rules: Rules, deck: list[int], budget: int = 0) -> list[str]:
    """The tests of this module that show at the start that the deal is lost. The
    free-order game is searched with at most `budget` states (0 for no limit)."""
    g = Game(rules, deck)
    pile, hands, pos = g.start()
    out = []
    if missing_card(g, pile, hands, pos):
        out.append("missing")
    if schedule_plays(g, pile, hands, pos, g.P) < g.goal:
        out.append("schedule")
    if (
        free_order_winnable(g, pile, hands, pos, 0, -1, {}, Budget(budget) if budget else None)
        is False
    ):
        out.append("free-order")
    return out


# --- the search ----------------------------------------------------------------------------


class OutOfNodesError(Exception):
    pass


def state_key(g: Game, s: State) -> tuple:
    """What decides the rest of the open game: hint knowledge, the discard pile and the
    order of cards in a hand do not, and dead cards are all alike."""
    pile = tuple(s.board)
    return (
        pile,
        tuple(g.tidy(h, pile) for h in s.hands),
        s.pos,
        s.hints,
        s.lives,
        s.current,
        s.turns_left,
    )


def candidate_moves(g: Game, s: State) -> list[Action]:
    """The legal moves of the current player, one per distinct card and kind, with every
    hint as one pass, in the order tried: plays (higher ranks first), discards of dead
    cards, discards of cards of which someone holds another copy, a pass, other discards,
    then misplays."""
    pile = tuple(s.board)
    hand = s.hands[s.current]
    first: dict[int, int] = {}
    for i, c in enumerate(hand):
        first.setdefault(c if g.needed(c, pile) else GONE, i)
    held = Counter(c for h in s.hands for c in h)
    order = sorted(first, key=lambda c: (-(c % g.R) if c != GONE else 0, c))
    plays = [play(first[c]) for c in order if g.can_play(c, pile)]
    dead = [first[c] for c in order if c == GONE]
    spare = [first[c] for c in order if c != GONE and held[c] > 1]
    other = [first[c] for c in order if c != GONE and held[c] == 1]
    out = plays + [discard(i) for i in dead + spare]
    q = next((q for q in range(g.P) if q != s.current and s.hands[q]), None)
    if q is not None:
        out.append(hint(q, "color", color_of(s.hands[q][0], g.rules)))
    out += [discard(i) for i in other]
    if s.lives > 1:
        out += [play(i) for i in dead + spare + other if not g.can_play(hand[i], pile)]
    return [a for a in out if legal(s, a)]


def solve(
    rules: Rules,
    deck: list[int],
    nodes: int = 0,
    cuts: str = "time",
    free_nodes: int = 0,
    stats: dict | None = None,
):
    """Depth-first search of the open game over `rules.step`, with a table of states found
    lost. Returns (True, actions), (False, None), or (None, None) when more than `nodes`
    states (0 for no limit) were expanded.

    `cuts` picks the tests that cut off a state: "none"; "time", the schedule test only, as
    plan H1 specifies; or "all", which adds the missing-card and free-order tests (each
    free-order test may expand at most `free_nodes` states, 0 for no limit, and cuts
    nothing when it runs out). If `stats` is a dict, its "nodes" entry counts the states
    expanded."""
    g = Game(rules, deck)
    lost: set = set()
    free_memo: dict = {}
    count = 0
    actions: list[Action] = []

    def search(s: State) -> bool:
        nonlocal count
        if s.score == g.goal:
            return True
        if s.over:
            return False
        key = state_key(g, s)
        if key in lost:
            return False
        count += 1
        if nodes and count > nodes:
            raise OutOfNodesError
        pile, hands = key[0], key[1]
        final = g.P if s.turns_left < 0 else s.turns_left
        if cuts != "none" and schedule_plays(g, pile, hands, s.pos, final) < g.goal - sum(pile):
            lost.add(key)
            return False
        if cuts == "all":
            budget = Budget(free_nodes) if free_nodes else None
            if (
                missing_card(g, pile, hands, s.pos)
                or free_order_winnable(
                    g, pile, hands, s.pos, s.current, s.turns_left, free_memo, budget
                )
                is False
            ):
                lost.add(key)
                return False
        for a in candidate_moves(g, s):
            t = copy_state(s)
            step(t, a)
            actions.append(a)
            if search(t):
                return True
            actions.pop()
        lost.add(key)
        return False

    try:
        if search(new_game(rules, deck)):
            return True, list(actions)
        return False, None
    except OutOfNodesError:
        return None, None
    finally:
        if stats is not None:
            stats["nodes"] = stats.get("nodes", 0) + min(count, nodes or count)


def copy_state(s: State) -> State:
    return replace(
        s,
        hands=[list(h) for h in s.hands],
        knowledge=[list(k) for k in s.knowledge],
        board=list(s.board),
        discards=list(s.discards),
    )


def replay_actions(rules: Rules, deck: list[int], actions) -> int:
    """Play the actions with the rules engine and return the final score."""
    s = new_game(rules, deck)
    for a in actions:
        step(s, a)
    return s.score
