"""Does every game of Egyptian Ratscrew end? The game as a finite Markov chain.

With slaps decided at random, the only chance in a game is who wins each slap; given those
winners, play is fixed by the order of the cards. So the game is a Markov chain on a finite
set of positions (`State`), and the questions are which positions can reach the end of the
game, and how fast the chance of still playing falls. This module builds the chain for small decks and
answers both exactly; `play_from` replays `game.play_game` move by move as a check.

    uv run python -m egyptian_ratscrew.termination
"""

import random
import sys
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from itertools import combinations_with_replacement, permutations, product
from math import factorial, prod
from typing import NamedTuple

import numpy as np

from egyptian_ratscrew.game import CHANCES, Card, deal, slappable


class State(NamedTuple):
    """A position before a card is laid: the hands (top card first), the pile (bottom card
    first), whose turn it is, who laid the royal now being answered (None if no challenge is
    open), and how many cards the current player still owes that challenger."""

    hands: tuple[tuple[Card, ...], ...]
    pile: tuple[Card, ...]
    player: int
    challenger: int | None
    owed: int


def start(hands: Sequence[Sequence[Card]]) -> State:
    """The state at the start of a game: player 0 lays first."""
    return State(tuple(map(tuple, hands)), (), 0, None, 0)


def ended(state: State) -> bool:
    """The game is over: one player holds every card not on the pile, and no challenge is open."""
    return state.challenger is None and sum(map(bool, state.hands)) == 1


def moves(state: State, slaps: bool = True) -> list[tuple[int | None, State]]:
    """The states after the next card, by slap winner. Without a slap the list has one entry,
    labeled None; after a slappable card it has one entry per player, the slap's winner.
    The rules are those of `game.play_game` with no misslaps, step for step; with
    slaps=False they are those of Beggar-my-neighbour."""
    hands = [list(h) for h in state.hands]
    pile = list(state.pile)
    player, challenger, owed = state.player, state.challenger, state.owed
    players = len(hands)

    def next_with_cards(i: int) -> int:
        for step in range(1, players):
            if hands[(i + step) % players]:
                return (i + step) % players
        return i

    def collected(winner: int) -> State:
        new = [tuple(h) for h in hands]
        new[winner] = new[winner] + tuple(pile)
        return State(tuple(new), (), winner, None, 0)

    if not hands[player]:
        if challenger is not None:  # cannot answer: the challenger takes the pile
            hands[challenger].extend(pile)
            pile.clear()
            player, challenger, owed = challenger, None, 0
        else:
            player = next_with_cards(player)
    card = hands[player].pop(0)
    pile.append(card)
    if slaps and slappable(pile):
        return [(i, collected(i)) for i in range(players)]
    if card in CHANCES:
        challenger, owed = player, CHANCES[card]
        player = next_with_cards(player)
        if player == challenger:
            return [(None, collected(challenger))]
    elif challenger is not None:
        owed -= 1
        if owed == 0 or not hands[player]:
            return [(None, collected(challenger))]
    else:
        player = next_with_cards(player)
    return [(None, State(tuple(map(tuple, hands)), tuple(pile), player, challenger, owed))]


def play_from(
    state: State, slap_shares: Sequence[float], rng: random.Random, max_cards: int = 100_000
) -> tuple[int, int | None]:
    """Play on from `state`, drawing slap winners as `game.play_game` does. Returns (cards
    laid, winner), with winner None if the game has not ended after max_cards cards."""
    players = range(len(slap_shares))
    for laid in range(max_cards):
        if ended(state):
            return laid, next(i for i, h in enumerate(state.hands) if h)
        options = moves(state)
        if len(options) == 1:
            state = options[0][1]
        else:
            state = options[rng.choices(players, weights=slap_shares)[0]][1]
    return max_cards, None


def deals(deck: Sequence[Card], players: int = 2) -> Iterator[tuple[State, int]]:
    """Every distinct starting state from shuffling `deck` and dealing it round the table, with
    the number of orders of the deck's cards that give it (cards of the same rank are told
    apart, so every order is equally likely). The weights sum to len(deck)!."""
    copies = prod(factorial(deck.count(c)) for c in set(deck))
    for order in sorted(set(permutations(deck)), key=repr):
        yield start([order[i::players] for i in range(players)]), copies


class Chain(NamedTuple):
    """The Markov chain on every state reachable from some deal. States are numbered; `edges[k]`
    lists (slap winner or None, next state) for state k, `ended[k]` says whether state k is
    over, and `deals` maps each starting state's number to its weight."""

    states: list[State]
    edges: list[list[tuple[int | None, int]]]
    ended: list[bool]
    deals: dict[int, int]


def chain(deck: Sequence[Card], players: int = 2) -> Chain:
    """Build the chain by breadth-first search from every deal of `deck`."""
    index: dict[State, int] = {}
    states: list[State] = []
    starts: dict[int, int] = {}

    def number(s: State) -> int:
        if s not in index:
            index[s] = len(states)
            states.append(s)
        return index[s]

    for s, weight in deals(deck, players):
        k = number(s)
        starts[k] = starts.get(k, 0) + weight
    edges: list[list[tuple[int | None, int]]] = []
    done: list[bool] = []
    k = 0
    while k < len(states):
        s = states[k]
        over = ended(s)
        done.append(over)
        edges.append([] if over else [(w, number(t)) for w, t in moves(s)])
        k += 1
    return Chain(states, edges, done, starts)


def can_end(c: Chain) -> list[bool]:
    """For each state, whether some choice of slap winners leads to the end of the game.
    Search backward from the ended states."""
    back: list[list[int]] = [[] for _ in c.states]
    for k, out in enumerate(c.edges):
        for _, t in out:
            back[t].append(k)
    good = list(c.ended)
    queue = deque(k for k, over in enumerate(good) if over)
    while queue:
        t = queue.popleft()
        for k in back[t]:
            if not good[k]:
                good[k] = True
                queue.append(k)
    return good


def slap_free_cycle(state: State, limit: int = 1_000_000) -> list[State] | None:
    """If play from `state` never meets a slappable pile and never ends, the states of the loop
    it falls into; otherwise None. Without slaps play is fixed, so this follows one path."""
    seen: dict[State, int] = {}
    path: list[State] = []
    for _ in range(limit):
        if ended(state):
            return None
        if state in seen:
            return path[seen[state] :]
        seen[state] = len(path)
        path.append(state)
        options = moves(state)
        if len(options) > 1:
            return None
        state = options[0][1]
    raise RuntimeError("no loop or slap within the limit")


def one_player_wins_slaps(state: State, player: int) -> bool:
    """Whether the game from `state` ends when `player` wins every slap. Play is then fixed,
    so it either ends or returns to a state it has been in."""
    seen = set()
    while not ended(state):
        if state in seen:
            return False
        seen.add(state)
        options = moves(state)
        state = options[0][1] if len(options) == 1 else options[player][1]
    return True


def beggar_loop_tricks(state: State, limit: int = 1_000_000) -> list[tuple[Card, ...]]:
    """Play Beggar-my-neighbour (slaps ignored) from `state` until it returns to a state it has
    been in, and list the piles picked up between the two visits. An error if it ends."""
    seen: dict[State, int] = {}
    tricks: list[tuple[Card, ...]] = []
    for _ in range(limit):
        if ended(state):
            raise ValueError("this game ends")
        if state in seen:
            return tricks[seen[state] :]
        seen[state] = len(tricks)
        if not state.hands[state.player] and state.challenger is not None:
            tricks.append(state.pile)  # collected before the next card
        (_, after), card = moves(state, slaps=False)[0], _next_card(state)
        if not after.pile:
            pile = state.pile if state.hands[state.player] else ()
            tricks.append((*pile, card))
        state = after
    raise RuntimeError("no loop within the limit")


def _next_card(state: State) -> Card:
    """The card the next move lays."""
    if state.hands[state.player]:
        return state.hands[state.player][0]
    if state.challenger is not None:
        return (state.hands[state.challenger] + state.pile)[0]
    players = len(state.hands)
    for step in range(1, players):
        hand = state.hands[(state.player + step) % players]
        if hand:
            return hand[0]
    raise ValueError("nobody can lay a card")


def royal_slap(trick: Sequence[Card]) -> bool:
    """Whether a trick has a royal pair or a royal sandwich: a royal laid on the same royal,
    or on one card that covers the same royal. Such a pile can be slapped whatever ranks the
    other cards have."""
    return any(card in CHANCES and card in trick[max(0, i - 2) : i] for i, card in enumerate(trick))


assert royal_slap(("Q", 5, "Q")) and royal_slap((2, "J", "J")) and not royal_slap((5, 4, 5))


def survival(c: Chain, slap_shares: Sequence[float], cards: int) -> np.ndarray:
    """P(the game lasts more than t cards) for t = 0, 1, ..., cards, from a random deal,
    by pushing the exact distribution over states forward one card at a time."""
    total = sum(c.deals.values())
    p = np.zeros(len(c.states))
    for k, w in c.deals.items():
        p[k] = w / total
    src, dst, prob = _transitions(c, slap_shares)
    alive = ~np.array(c.ended)
    out = [p[alive].sum()]
    for _ in range(cards):
        p = np.bincount(dst, weights=p[src] * prob, minlength=len(p)) * alive
        out.append(p.sum())
    return np.array(out)


def _transitions(c: Chain, slap_shares: Sequence[float]) -> tuple[np.ndarray, ...]:
    """The chain's moves as arrays: from, to, and probability, one entry per edge."""
    src, dst, prob = [], [], []
    total = sum(slap_shares)
    for k, out in enumerate(c.edges):
        for w, t in out:
            src.append(k)
            dst.append(t)
            prob.append(1.0 if w is None else slap_shares[w] / total)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), np.array(prob)


def decay_rate(c: Chain, slap_shares: Sequence[float], steps: int = 4_000) -> float:
    """The spectral radius of the chain among unfinished states: the factor by which the
    chance of still playing eventually falls per card. Found by power iteration on the lazy
    chain (I + Q) / 2, whose largest eigenvalue (1 + rho) / 2 is the only one of its size,
    so the iteration settles even when play runs in cycles."""
    src, dst, prob = _transitions(c, slap_shares)
    alive = ~np.array(c.ended)
    n = len(c.states)
    u = alive.astype(float)  # P(a game from each state lasts more than t cards), rescaled
    scale = 1.0
    for _ in range(steps):
        nxt = (u + np.bincount(src, weights=prob * u[dst], minlength=n) * alive) / 2
        scale = nxt.max()
        u = nxt / scale
    return 2 * float(scale) - 1


def uniform_bound(c: Chain, slap_shares: Sequence[float]) -> tuple[int, float]:
    """The explicit bound behind the theorem: L, the most cards any state needs to reach the
    end along its best path, and delta, the least chance over all states of ending within L
    cards. Then P(game lasts more than m L cards) <= (1 - delta)^m from every state. Every
    state must be able to end."""
    assert all(can_end(c)), "some state cannot reach the end"
    src, dst, prob = _transitions(c, slap_shares)
    n = len(c.states)
    over = np.array(c.ended)
    # d[k]: fewest cards from state k to the end, by backward search
    back: list[list[int]] = [[] for _ in range(n)]
    for k, out in enumerate(c.edges):
        for _, t in out:
            back[t].append(k)
    d = [0 if e else -1 for e in c.ended]
    queue = deque(k for k in range(n) if c.ended[k])
    while queue:
        t = queue.popleft()
        for k in back[t]:
            if d[k] < 0:
                d[k] = d[t] + 1
                queue.append(k)
    steps = max(d)
    f = over.astype(float)  # f[k] = P(ended within t cards from state k)
    for _ in range(steps):
        f = np.where(over, 1.0, np.bincount(src, weights=prob * f[dst], minlength=n))
    return steps, float(f.min())


def decks_with_traps(cards: int, players: int) -> list[tuple[Card, ...]]:
    """Every deck of `cards` cards, with at least one royal and at most four of a rank, that
    deals into or reaches a trapped position. Number cards only matter through which are
    equal, so the ranks 2 to 5 stand for every choice of up to four number cards."""
    ranks: tuple[Card, ...] = (2, 3, 4, 5, "J", "Q", "K", "A")
    found = []
    for deck in combinations_with_replacement(ranks, cards):
        if any(r in CHANCES for r in deck) and all(deck.count(r) <= 4 for r in deck):
            if not all(can_end(chain(deck, players))):
                found.append(deck)
    return found


def tail_rate(lengths: Sequence[int], after: int) -> tuple[float, float, float]:
    """The chance per card that a game still going after `after` cards ends with the next
    card, if that chance is constant: games ending later, over cards laid later. Returns the
    estimate and a 95% interval, treating the number of endings as Poisson."""
    endings = sum(n > after for n in lengths)
    exposure = sum(n - after for n in lengths if n > after)
    half = 1.96 * endings**0.5 / exposure
    return endings / exposure, endings / exposure - half, endings / exposure + half


DECKS: tuple[tuple[Card, ...], ...] = (
    (2, 2, 3, 3, "J", "J"),
    (2, 2, 3, 3, "J", "Q"),
    (2, 2, 3, 3, 4, 4, "J", "J"),
    (2, 2, 3, 3, 4, 4, "J", "Q"),
    (2, 2, 2, 3, 3, 3, "J", "J"),
    (2, 2, 3, 3, "J", "J", "Q", "Q"),
    (2, 2, 3, 3, "J", "J", "A", "A"),
    (2, 2, 3, 3, "J", "Q", "K", "A"),
)

# Loops of Beggar-my-neighbour with a full deck, from Casella et al. (2024): a deal that
# reaches their 62-trick cycle, and two states that are not deals. Player 0 lays first; "-"
# is any card that is not a royal.
LOOPS: tuple[tuple[str, str], ...] = (
    ("---K---Q-KQAJ-----AAJ--J--", "----------Q----KQ-J-----KA"),
    ("--------------------J--Q-Q-Q-Q-K-KJ--K-K-A-A-A-AJ-", "J-"),
    ("--Q-K-J-----------------KJ--Q----A-Q-A-AJ--Q-K-A-K", "J-"),
)


# The "pieces" of Casella et al. (2024, Appendix C): stretches of a hand that, placed between
# jacks, keep Beggar-my-neighbour cycling. Their template deck is piece J piece J piece J -
# against J -, with player 0 laying first.
PIECES: tuple[str, ...] = (
    "--", "--Q-Q", "-----K", "--------", "--K---Q-KQQ", "--Q-Q-Q-Q-K-K", "--K---A----AA",
    "--------A-KAA", "--A-----A-A--A", "--A----Q----QQ", "--------A----AA", "--Q----Q----Q-Q",
    "--Q--Q-Q-QQ-Q-Q", "--------Q-Q-Q-Q", "--------Q----Q-Q", "--K-A--Q-K--Q--QQ",
    "--Q----A-Q-A-A---", "--------------K---", "--------Q----Q----", "-----------------Q-",
    "--------------------", "--------K----KQQ-KQQ", "--------K----KK--Q-K--Q--QQ",
    "--------K---------Q-Q-K---Q-KKQ", "--------------------A----------A---A",
    "--------------------Q----------Q----Q-Q",
)  # fmt: skip


def template_decks() -> Iterator[str]:
    """Every full deck of the template: three pieces (repeats allowed, any order), with each
    royal of a piece replaced by any of Q, K and A, so that the deck has four of each royal and
    36 other cards. Each deck once, as player 0's hand; player 1 holds J -."""
    seen = set()
    for trio in combinations_with_replacement(PIECES, 3):
        if "".join(trio).count("-") != 34 or sum(len(p) for p in trio) != 46:
            continue
        for order in sorted(set(permutations(trio))):
            joined = "J".join(order) + "J-"
            royals = [i for i, card in enumerate(joined) if card not in "J-"]
            for faces in product("QKA", repeat=len(royals)):
                if any(faces.count(r) != 4 for r in "QKA"):
                    continue
                cards = list(joined)
                for i, face in zip(royals, faces, strict=True):
                    cards[i] = face
                deck = "".join(cards)
                if deck not in seen:
                    seen.add(deck)
                    yield deck


def fast_beggar_loop(first: str, second: str, limit: int = 1_000_000) -> list[str] | None:
    """`beggar_loop_tricks` for two players written for speed, on strings: the tricks of the
    loop, or None if the game ends."""
    hands = [first, second]
    pile, player, challenger, owed = "", 0, -1, 0
    seen: dict[tuple, int] = {}
    tricks: list[str] = []
    for _ in range(limit):
        if challenger < 0 and not (hands[0] and hands[1]):
            return None
        key = (hands[0], hands[1], pile, player, challenger, owed)
        if key in seen:
            return tricks[seen[key] :]
        seen[key] = len(tricks)
        if not hands[player]:  # cannot answer: the challenger takes the pile
            hands[challenger] += pile
            tricks.append(pile)
            pile, player, challenger, owed = "", challenger, -1, 0
        card, hands[player] = hands[player][0], hands[player][1:]
        pile += card
        winner = -1
        if card in CHANCES:
            challenger, owed = player, CHANCES[card]
            player = 1 - player
            if not hands[player]:
                winner = challenger
        elif challenger >= 0:
            owed -= 1
            if owed == 0 or not hands[player]:
                winner = challenger
        else:
            player = 1 - player
        if winner >= 0:
            hands[winner] += pile
            tricks.append(pile)
            pile, player, challenger, owed = "", winner, -1, 0
    raise RuntimeError("no loop within the limit")


def template_search() -> Counter:
    """Play every template deck as Beggar-my-neighbour and sort them: decks whose game ends,
    decks that loop with a royal pair or sandwich somewhere in the loop, and decks that loop
    without one. About ten minutes."""
    counts: Counter = Counter()
    for deck in template_decks():
        tricks = fast_beggar_loop(deck, "J-")
        if tricks is None:
            counts["ends"] += 1
        else:
            counts["royal slap" if any(map(royal_slap, tricks)) else "no royal slap"] += 1
    return counts


def describe(deck: Sequence[Card], players: int, shares: Sequence[float]) -> dict:
    """Every exact figure the README gives for one small deck."""
    c = chain(deck, players)
    good = can_end(c)
    trapped = [k for k, g in enumerate(good) if not g]
    row = {
        "states": len(c.states),
        "trapped": len(trapped),
        "trapped orders": sum(w for k, w in c.deals.items() if not good[k]),
        "orders": sum(c.deals.values()),
        "trapped on slap-free loops": sum(
            slap_free_cycle(c.states[k]) is not None for k in trapped
        ),
        "ended by one player's slaps": sum(
            any(one_player_wins_slaps(c.states[k], p) for p in range(players))
            for k in range(len(c.states))
            if good[k]
        ),
        "can end": sum(good),
    }
    if not trapped:
        steps, delta = uniform_bound(c, shares)
        alive = survival(c, shares, 2_000)
        row |= {
            "L": steps,
            "delta": delta,
            "bound rate": (1 - delta) ** (1 / steps),
            "rate": decay_rate(c, shares),
            "mean cards": float(alive.sum()),
        }
    return row


def with_ranks(hands: Sequence[str], rng: random.Random) -> list[list[Card]]:
    """Hands in Beggar-my-neighbour notation with the 36 cards that are not royals ("-")
    given the ranks 2 to 10, four of each, in random order."""
    others = [rank for rank in range(2, 11) for _ in range(4)]
    rng.shuffle(others)
    ranks = iter(others)
    return [[next(ranks) if card == "-" else card for card in hand] for hand in hands]


def main() -> None:
    for players in (2, 3):
        smaller = [d for n in range(2, 6) for d in decks_with_traps(n, players)]
        print(f"{players} players: decks of 2 to 5 cards with a trapped position: {smaller}")
        print(f"  decks of 6 cards: {decks_with_traps(6, players)}")
    trap = start([("J", 2, 3), (2, "J", 3)])
    print("J 2 3 against 2 J 3: a loop of", len(slap_free_cycle(trap) or ()), "states, no slap")
    for players in (2, 3):
        for deck in DECKS if players == 2 else DECKS[:4]:
            row = describe(deck, players, [1] * players)
            print(f"{players} players, deck {' '.join(map(str, deck))}: {row}")

    deck = (2, 2, 3, 3, 4, 4, "J", "J")
    c = chain(deck)
    alive = survival(c, [1, 1], 2_000)
    rng = random.Random(1)
    lengths = []
    for _ in range(100_000):
        order = list(deck)
        rng.shuffle(order)
        lengths.append(play_from(start([order[0::2], order[1::2]]), [1, 1], rng)[0])
    print(f"deck {' '.join(map(str, deck))}, ending chance per card after 30 cards:")
    print(f"  exact {alive[30] / alive[30:].sum():.5f}, simulated", tail_rate(lengths, 30))
    for t in (100, 200, 400):
        print(
            f"  exact chance that a game still going after {t} cards ends with the next:",
            f"{1 - alive[t + 1] / alive[t]:.5f}",
        )
    print(f"  in the limit, exact: {1 - decay_rate(c, [1, 1]):.5f}")

    for hands in LOOPS:
        tricks = beggar_loop_tricks(start(hands))
        print(
            f"loop of {sum(map(len, tricks))} cards in {len(tricks)} tricks,",
            f"{sum(map(royal_slap, tricks))} with a royal pair or sandwich",
        )
    rng = random.Random(2)
    games = [play_from(start(with_ranks(LOOPS[0], rng)), [1, 1], rng) for _ in range(10_000)]
    print(
        "the looping deal with random ranks and slaps:",
        sum(w is not None for _, w in games),
        "of 10,000 games end; longest",
        max(n for n, _ in games),
    )

    rng = random.Random(2026)
    lengths = [play_from(start(deal(2, rng)), [1, 1], rng)[0] for _ in range(100_000)]
    for after in (300, 600):
        print(f"full deck, ending chance per card after {after} cards:", tail_rate(lengths, after))
    if "--template" in sys.argv:
        print("template decks:", template_search())


if __name__ == "__main__":
    main()
