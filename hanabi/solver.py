"""Which deals can reach a perfect score when everyone sees everything?

In the **open game** every player sees all hands and the order of the deck (van den Bergh,
2015, calls such games open; Baffier et al., 2017, call the one-player form solitaire
Hanabi). A hint then carries no information, so it is a **pass** that costs a token. A deal
is **winnable** if some sequence of legal actions reaches the maximum score. No strategy for
the real game can win a deal that is not winnable, so the winnable share is an upper bound on
any strategy's perfect-game rate.

Two engines answer the question for one deal:

- `brute_winnable` searches every legal action of `hanabi.rules` (each distinct hint
  separately) and remembers exact states it has finished. It is slow and simple.
- `winnable` searches the open game directly: it plays playable cards first, treats a hint as
  a pass, never discards the last copy of a card still needed, writes every card that can no
  longer be played as one "trash" symbol, and cuts off states that fail the bounds below.
  It returns a winning line, which `replay` checks with the rules engine.

**Bounds** that prove a state cannot be won (each is a certificate when it holds at the
start):

- *missing card*: every copy of a card still needed has been discarded.
- *time*: each play or discard draws a card while the deck lasts, and after the last card
  each player has one turn. So with nd cards left the team has at most nd + (turns after the
  deck runs out) plays, and the i-th draw from now can only play cards already in a hand or
  among the first i - 1 cards of the deck. `time_bound` takes the minimum over i.
- *storage*: once all copies of a needed card are out of the deck and the card cannot have
  been played yet, a copy must sit in some hand. If more such cards are needed at one moment
  than the hands can hold, the deal is lost.
"""

from __future__ import annotations

from collections import Counter

from hanabi.rules import Action, Rules, State, hint, legal_actions, new_game, step

TRASH = -1

type Board = tuple[int, ...]
type Hands = tuple[tuple[int, ...], ...]


# --- the brute-force engine ----------------------------------------------------------------


def state_key(s: State) -> tuple:
    """Everything that matters in the open game (hint knowledge does not)."""
    return (
        tuple(s.board),
        tuple(tuple(sorted(h)) for h in s.hands),
        s.pos,
        s.hints,
        s.lives,
        s.current,
        s.turns_left,
        tuple(sorted(s.discards)),
    )


def copy_state(s: State) -> State:
    return State(
        rules=s.rules,
        deck=s.deck,
        hands=[list(h) for h in s.hands],
        knowledge=[list(k) for k in s.knowledge],
        board=list(s.board),
        pos=s.pos,
        hints=s.hints,
        lives=s.lives,
        current=s.current,
        turns_left=s.turns_left,
        discards=list(s.discards),
        turn=s.turn,
        over=s.over,
    )


def brute_winnable(rules: Rules, deck: list[int], target: int = 0) -> bool:
    """Exhaustive search over every legal action of the rules engine: can the score reach
    the target (by default the maximum)?"""
    failed: set[tuple] = set()
    target = target or rules.max_score

    def search(s: State) -> bool:
        if s.score >= target:
            return True
        if s.over:
            return False
        key = state_key(s)
        if key in failed:
            return False
        for a in legal_actions(s):
            t = copy_state(s)
            step(t, a)
            if search(t):
                return True
        failed.add(key)
        return False

    return search(new_game(rules, deck))


# --- the fast engine -----------------------------------------------------------------------


class Deal:
    """Precomputed facts about one deal for the fast engine."""

    def __init__(self, rules: Rules, deck: list[int], target: int = 0):
        self.rules = rules
        self.deck = list(deck)
        self.nr = rules.ranks
        self.np = rules.players
        self.h = rules.hand
        self.max = rules.max_score
        self.target = target or rules.max_score  # the score to reach
        self.copies = [rules.counts[t % self.nr] for t in range(rules.colors * self.nr)]

    def is_trash(self, card: int, board: Board) -> bool:
        return card == TRASH or card % self.nr < board[card // self.nr]

    def canon(self, hand, board: Board) -> tuple[int, ...]:
        return tuple(sorted(TRASH if self.is_trash(c, board) else c for c in hand))


def chain_heights(board: Board, cards, nr: int) -> list[int]:
    """For each color, the height the board could reach using these cards, in any order."""
    have = set(cards)
    out = list(board)
    for c in range(len(board)):
        while out[c] < nr and c * nr + out[c] in have:
            out[c] += 1
    return out


def time_bound(d: Deal, board: Board, hands: Hands, pos: int, final_turns: int) -> int:
    """The most cards the team could still play: the time bound described above.

    final_turns is the number of turns after the deck runs out (the number of players while it has cards)."""
    nr, deck = d.nr, d.deck
    nd = len(deck) - pos
    have = {c for h in hands for c in h if c != TRASH}
    heights = chain_heights(board, have, nr)
    base = sum(board)
    best = nd  # i = 0: every draw plays a card
    # g = cards playable using hands and deck[pos : pos + i - 1]; bound = g + (nd - i)
    for i in range(1, nd + 1):
        g = sum(heights) - base
        best = min(best, g + nd - i)
        card = deck[pos + i - 1]
        have.add(card)
        c = card // nr
        while heights[c] < nr and c * nr + heights[c] in have:
            heights[c] += 1
    g_all = sum(heights) - base
    return min(best + final_turns, g_all)


def storage_violation(d: Deal, board: Board, hands: Hands, pos: int) -> int:
    """The first number of draws i at which more needed cards must be held than fit, or 0.

    Just after the i-th draw, the board can only have used cards from the hands and the
    first i - 1 drawn cards. A needed card above those heights, with no copy left in the
    deck but some copy already drawn, must be in a hand."""
    nr, deck, cap = d.nr, d.deck, d.np * d.h
    nd = len(deck) - pos
    seen = {c for h in hands for c in h if c != TRASH}
    left = Counter(deck[pos:])
    heights = chain_heights(board, seen, nr)
    for i in range(1, nd + 1):
        card = deck[pos + i - 1]
        left[card] -= 1
        drawn = seen | {card}
        must = 0
        for c in range(len(board)):
            for r in range(heights[c], nr):
                v = c * nr + r
                if left[v] == 0 and v in drawn:
                    must += 1
        if must > cap:
            return i
        seen.add(card)
        c = card // nr
        while heights[c] < nr and c * nr + heights[c] in seen:
            heights[c] += 1
    return 0


def score_cap(d: Deal, board: Board, hands: Hands, pos: int) -> int:
    """The highest score the cards not yet discarded allow: in each color, the board plus
    the run of ranks above it of which some copy is left. Below the maximum exactly when
    every copy of some needed card is gone."""
    nr = d.nr
    avail = {c for h in hands for c in h}
    avail.update(d.deck[pos:])
    return sum(chain_heights(board, avail, nr))


def pool_winnable(d: Deal, board: Board, hands: Hands, pos: int, final_turns: int, memo=None):
    """The **pooled game**, a relaxation: one player holds all hands together, with no turn
    order and no tokens, so each play or discard draws a card and after the last card any
    `final_turns` cards may be played. Every winning line of the real game is one here.

    In the pooled game playing a playable card at once is never worse (an exchange
    argument: play it now and make the discard later), and discarding trash or a second copy
    of a held card is never worse than discarding anything else, so the only choice is which
    card to throw away when nothing is playable and nothing is useless."""
    nr, deck, n = d.nr, d.deck, len(d.deck)
    if memo is None:
        memo = {}

    def norm(b, pool):
        out, seen = [], set()
        for c in pool:
            if c == TRASH or c % nr < b[c // nr] or c in seen:
                out.append(TRASH)
            else:
                seen.add(c)
                out.append(c)
        return tuple(sorted(out))

    def go(board_in: Board, pool_in: tuple[int, ...], p: int) -> bool:
        b = list(board_in)
        pool = list(pool_in)
        while p < n:  # play everything playable, drawing after each play
            for c in pool:
                if c != TRASH and c % nr == b[c // nr]:
                    pool.remove(c)
                    b[c // nr] += 1
                    pool.append(deck[p])
                    p += 1
                    break
            else:
                break
        b = tuple(b)
        if sum(b) >= d.target:
            return True
        if p >= n:
            return endgame(b, pool)
        key = (b, norm(b, pool), p)
        if key in memo:
            return memo[key]
        memo[key] = False
        avail = Counter(c for c in pool if c != TRASH)
        avail.update(deck[p:])
        pool_n = list(norm(b, pool))
        if TRASH in pool_n:
            choices = [TRASH]
        else:
            # never throw away the last copy of a needed card
            choices = [
                c
                for c in sorted(set(pool_n), key=lambda c: -(c % nr))
                if avail[c] > 1 or d.target < d.max
            ]
        for c in choices:
            rest = list(pool_n)
            rest.remove(c)
            rest.append(deck[p])
            if go(b, tuple(rest), p + 1):
                memo[key] = True
                return True
        return False

    def endgame(b: Board, pool) -> bool:
        need = d.target - sum(b)
        heights = chain_heights(b, [c for c in pool if c != TRASH], nr)
        return sum(heights) - sum(b) >= need and need <= final_turns

    return go(board, tuple(c for h in hands for c in h), pos)


def root_certificates(rules: Rules, deck: list[int]) -> list[str]:
    """The bounds that prove at the start that the deal cannot be won."""
    d = Deal(rules, deck)
    board = (0,) * rules.colors
    n = rules.players * rules.hand
    hands = tuple(tuple(deck[i * rules.hand : (i + 1) * rules.hand]) for i in range(rules.players))
    out = []
    if time_bound(d, board, hands, n, rules.players) < d.max:
        out.append("time")
    if storage_violation(d, board, hands, n):
        out.append("storage")
    if not pool_winnable(d, board, hands, n, rules.players):
        out.append("pooled")
    return out


class BudgetError(Exception):
    pass


INF = 10**9
SCHEDULE = (0, 1, 2, 4, 8, INF)


def winnable(
    rules: Rules,
    deck: list[int],
    budget: int = 0,
    bounds: bool = True,
    pass_first: bool = True,
    schedule=SCHEDULE,
    target: int = 0,
    stats: dict | None = None,
):
    """Fast search. Returns (True, line), (False, None), or (None, None) when the node budget
    (0 for none) runs out. A line is a list of open-game moves (player, kind, card).

    The search is limited discrepancy search (Harvey and Ginsberg, 1995): children are tried
    in the heuristic order below, and a pass with limit k may take, summed along a line, at
    most k steps away from the first child. The limits grow through `schedule`, which ends
    with no limit, so a deal is declared lost only after a complete search. With a target
    below the maximum it asks whether that score can be reached instead. If `stats` is a
    dict, its "nodes" entry is increased by the number of states expanded."""
    d = Deal(rules, deck, target)
    nr, np, h, maxh = d.nr, d.np, d.h, rules.max_hints
    failed: dict = {}  # state -> largest limit it failed under (INF: failed outright)
    pool_memo: dict = {}
    nodes = 0

    def children(board, hands, pos, hints, lives, cur, tl):
        """Moves in heuristic order, each with the state it leads to."""
        hand = hands[cur]
        nxt = (cur + 1) % np
        ntl = tl - 1 if tl > 0 else tl
        drawn = pos < len(d.deck)
        out = []

        def move(kind, card, new_board, new_hints, new_lives):
            rest = list(hand)
            rest.remove(card)
            if drawn:
                rest.append(d.deck[pos])
                npos, ntl2 = pos + 1, (np if pos + 1 == len(d.deck) else ntl)
            else:
                npos, ntl2 = pos, ntl
            nh = list(hands)
            nh[cur] = rest
            nh = tuple(d.canon(x, new_board) for x in nh)
            out.append((kind, card, (new_board, nh, npos, new_hints, new_lives, nxt, ntl2)))

        distinct = sorted(set(hand))
        # 1. play a playable card
        for card in distinct:
            if card != TRASH and card % nr == board[card // nr]:
                nb = list(board)
                nb[card // nr] += 1
                five = card % nr == nr - 1 and hints < maxh
                move("play", card, tuple(nb), hints + five, lives)
        can_discard = hints < maxh or rules.discard_when_full
        avail = Counter(c for x in hands for c in x)
        held = avail.copy()
        avail.update(d.deck[pos:])

        def next_copy(card):
            try:
                return d.deck.index(card, pos)
            except ValueError:
                return len(d.deck)

        # discard first what is needed latest, and whose other copy comes soonest; never the
        # last copy of a needed card
        order = sorted(
            (c for c in distinct if c != TRASH and (avail[c] > 1 or d.target < d.max)),
            key=lambda c: (-(c % nr - board[c // nr]), next_copy(c), c),
        )
        free = [c for c in order if held[c] > 1]
        useful = [c for c in order if held[c] == 1]
        up = min(hints + 1, maxh)
        # 2. discard trash, or a card of which someone holds another copy
        if can_discard:
            if TRASH in hand:
                move("discard", TRASH, board, up, lives)
            for c in free:
                move("discard", c, board, up, lives)

        def passes():
            if hints > 0 and any(hands[p] for p in range(np) if p != cur):
                out.append(("pass", TRASH, (board, hands, pos, hints - 1, lives, nxt, ntl)))

        # 3 and 4. pass, or discard a card with another copy left in the deck
        if pass_first:
            passes()
        if can_discard:
            for c in useful:
                move("discard", c, board, up, lives)
        if not pass_first:
            passes()
        # 5. misplay trash or a card with another copy left (costs a life, never the last)
        if lives > 1:
            for c in ([TRASH] if TRASH in hand else []) + order:
                if c == TRASH or c % nr != board[c // nr]:
                    move("misplay", c, board, hints, lives - 1)
        return out

    line: list[tuple[int, str, int]] = []

    def search(state, disc) -> tuple[bool, bool]:
        """(won, exact): exact means a failure holds without any limit."""
        nonlocal nodes
        board, hands, pos, hints, lives, cur, tl = state
        score = sum(board)
        if score >= d.target:
            return True, True
        if tl == 0:
            return False, True
        seen = failed.get(state, -1)
        if seen >= disc:
            return False, seen == INF
        nodes += 1
        if budget and nodes > budget:
            raise BudgetError
        if bounds:
            final = np if tl < 0 else tl
            if (
                score_cap(d, board, hands, pos) < d.target
                or time_bound(d, board, hands, pos, final) < d.target - score
                or not pool_winnable(d, board, hands, pos, final, pool_memo)
            ):
                failed[state] = INF
                return False, True
        cut = False
        for i, (kind, card, child) in enumerate(children(*state)):
            if i > disc:
                cut = True
                break
            line.append((cur, kind, card))
            ok, exact = search(child, disc - i)
            if ok:
                return True, True
            line.pop()
            cut = cut or not exact
        failed[state] = disc if cut else INF
        return False, not cut

    board0 = (0,) * rules.colors
    hands0 = tuple(d.canon(deck[i * h : (i + 1) * h], board0) for i in range(np))
    start = (board0, hands0, np * h, maxh, rules.lives, 0, -1)
    try:
        for limit in schedule:
            ok, exact = search(start, limit)
            if ok:
                return True, list(line)
            if exact:
                return False, None
    except BudgetError:
        return None, None
    finally:
        if stats is not None:
            stats["nodes"] = stats.get("nodes", 0) + min(nodes, budget or nodes)
    return None, None


def replay(rules: Rules, deck: list[int], line) -> int:
    """Play a winning line with the rules engine and return the score. Trash and passes are
    resolved to a concrete card and a legal hint."""
    s = new_game(rules, deck)
    for player, kind, card in line:
        assert s.current == player, "line out of turn"
        hand = s.hands[player]
        if kind == "pass":
            target = next(p for p in range(rules.players) if p != player and s.hands[p])
            a = hint(target, "rank", hand_rank(s.hands[target][0], rules))
        else:
            if card == TRASH:
                i = next(
                    i for i, c in enumerate(hand) if c % rules.ranks < s.board[c // rules.ranks]
                )
            else:
                i = hand.index(card)
            a = Action("play" if kind in ("play", "misplay") else "discard", i)
        step(s, a)
    return s.score


def hand_rank(card: int, rules: Rules) -> int:
    return card % rules.ranks + 1
