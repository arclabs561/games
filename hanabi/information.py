"""The information strategy of Cox, De Silva, DeOrsey, Kenter, Retter and Tobin (2015).

C. Cox et al., "How to Make the Perfect Fireworks Display: Two Strategies for Hanabi",
*Mathematics Magazine* 88(5), 2015, 323-336. Their simulator, HanSim
(github.com/rjtobin/HanSim), defines the details the paper leaves out; this module follows
that code, quirks included, and says where.

How it works, in the paper's words where possible:

- Every card has a **possibility table**: the set of (color, rank) values that public
  information leaves for it. Here a table is a bit mask over the 25 card values.
- A hint **targets** one card in every other hand: the card whose table gives the highest
  chance of being playable, counting the copies not yet accounted for.
- The targeted card's table is cut into at most eight **hint sets**: all dead values
  together, as many single values as possible, and the rest in sets of at most eight.
- The **value of a hand** is the number of the hint set that holds its targeted card. The
  hinter sends the sum of the values she sees, mod 8, as one of eight hints (rank or color
  hint to the player 1 to 4 seats to her left). Each other player subtracts the values he
  sees and learns his own (hat guessing).
- On her turn a player plays a card she knows is playable; otherwise she discards a known
  dead card while fewer than five cards are in the discard pile; otherwise she hints if a
  token is left; otherwise she discards, in order of preference, a dead card, a duplicate,
  a card known not to be critical, a card not known to be critical, and her first card.

The paper reports, for five players, a mean score of 24.68 and a perfect game 76% of the
time (10^6 games), and 24.87 and 91% when players see their own cards.

For other numbers of players this module uses 2(P - 1) hint values and hint sets of up to
max(2(P - 1), ceil(n / 2(P - 1))) values, a generalization the paper does not make.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from statistics import mean

from hanabi.rules import (
    Action,
    Event,
    Rules,
    State,
    color_of,
    copies,
    discard,
    hansim_decks,
    hint,
    new_game,
    play,
    rank_of,
    seeded_deck,
    step,
)

NAN = float("nan")


def bits(mask: int):
    """The card values in a mask, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def known(mask: int) -> bool:
    return mask != 0 and mask & (mask - 1) == 0


class InfoTeam:
    """All players of the information strategy. Each decision uses only what the player
    could know: public records, the other hands, and her own decoding of hints."""

    def __init__(self, rules: Rules, cheat: bool = False, hansim: bool = True):
        self.rules = rules
        self.cheat = cheat
        # HanSim discards a known dead card even while all 8 tokens are on the table, which
        # the official rules forbid. hansim=False skips that discard when the tokens are full.
        self.hansim = hansim
        self.n_cards = rules.colors * rules.ranks
        self.all = (1 << self.n_cards) - 1
        self.slots = rules.hand
        self.hats = 2 * (rules.players - 1)
        # HanSim fills a partition rank by rank, and color by color within a rank.
        self.order = [c * rules.ranks + r for r in range(rules.ranks) for c in range(rules.colors)]
        self.illegal = False

    # --- public records -------------------------------------------------------------------

    def start(self, state: State):
        p = self.rules.players
        self.pub = [[self.all] * self.slots for _ in range(p)]
        self.used: Counter[int] = Counter()  # played, discarded and misplayed cards

    def board_masks(self, board: list[int]) -> tuple[int, int]:
        """(dead, playable) masks for this board."""
        dead = playable = 0
        for t in range(self.n_cards):
            r, c = rank_of(t, self.rules), color_of(t, self.rules)
            if r <= board[c]:
                dead |= 1 << t
            elif r == board[c] + 1:
                playable |= 1 << t
        return dead, playable

    def eliminate(self, tables: list[list[int]], sizes: list[int]):
        """Card counting: a value whose copies are all used or known is ruled out for every
        card not yet known, until nothing changes (HanSim's updateData)."""
        while True:
            seen = Counter(self.used)
            for p, row in enumerate(tables):
                for c in range(sizes[p]):
                    if known(row[c]):
                        seen[row[c].bit_length() - 1] += 1
            full = 0
            for t, n in seen.items():
                if n >= copies(t, self.rules):
                    full |= 1 << t
            changed = False
            for p, row in enumerate(tables):
                for c in range(sizes[p]):
                    if not known(row[c]) and row[c] & full:
                        row[c] &= ~full
                        changed = True
            if not changed:
                return

    def private(self, state: State, me: int) -> list[list[int]]:
        """Player me's tables: public ones, the other hands as seen, then card counting."""
        tables = [list(row) for row in self.pub]
        for p, hand in enumerate(state.hands):
            if p != me or self.cheat:
                for c, card in enumerate(hand):
                    tables[p][c] = 1 << card
        self.eliminate(tables, [len(h) for h in state.hands])
        return tables

    def prob_playable(self, p: int, c: int, playable_mask: int) -> float:
        """HanSim's probPlayable, in percent, from public tables. NaN if no copy is left."""
        mask = self.pub[p][c]
        seen = Counter(self.used)
        for row in self.pub:  # every slot, as HanSim loops over four
            for m in row:
                if known(m):
                    seen[m.bit_length() - 1] += 1
        total = good = 0
        for t in bits(mask):
            left = copies(t, self.rules) - seen[t]
            total += left
            if playable_mask >> t & 1:
                good += left
        if total == 0:
            return NAN
        return (good / total) / 0.01

    def target(self, p: int, hand_len: int, playable_mask: int) -> int:
        """The card a hint targets in player p's hand (HanSim's highestProbCard)."""
        best, best_card = 0.0, 0
        for c in range(hand_len):
            if known(self.pub[p][c]):
                continue
            prob = self.prob_playable(p, c, playable_mask)
            if prob > best:
                best, best_card = prob, c
        return best_card

    def partition(self, mask: int, dead_mask: int) -> dict[int, int]:
        """Hint set (0 up) of each value in the table: dead values together, as many single
        values as allowed, then sets of `block` values (HanSim's smart8Partition)."""
        h = self.hats
        dead = mask & dead_mask
        alive = [t for t in self.order if mask >> t & 1 and not dead >> t & 1]
        elements = len(alive) + (1 if dead else 0)
        block = max(h, math.ceil(elements / h))
        singles = 0
        for i in range(h + 1):
            if i + block * (h - i) >= elements:
                singles = i
        out = {t: 0 for t in bits(dead)}
        current, filled = (2 if dead else 1), 0
        for t in alive:
            out[t] = current - 1
            if current <= singles:
                current += 1
            else:
                filled += 1
                if filled == block:
                    filled, current = 0, current + 1
        return out

    # --- hints ----------------------------------------------------------------------------

    def encode(self, giver: int, number: int, state: State) -> Action:
        """Hint number 0 to 2(P-1) - 1: rank hints first, then color hints, to the player
        1, 2, ... seats to the left. The value is the last card whose public rank (or color)
        is unknown, else the first card (HanSim)."""
        others = self.rules.players - 1
        target = (number % others + giver + 1) % self.rules.players
        attr = "rank" if number < others else "color"
        hand = state.hands[target]

        def value(card):
            return rank_of(card, self.rules) if attr == "rank" else color_of(card, self.rules)

        v = value(hand[0])
        for c, card in enumerate(hand):
            mask = self.pub[target][c]
            vals = {value(t) for t in bits(mask)}
            if len(vals) != 1:
                v = value(card)
        return hint(target, attr, v)

    def decode(self, giver: int, action: Action) -> int:
        p = self.rules.players
        number = (action.target - giver) % p - 1
        return number + (p - 1 if action.attr == "color" else 0)

    def hand_values(self, state: State, giver: int) -> dict[int, tuple[int, dict[int, int]]]:
        """For every player but the giver: targeted card, and the partition of its table."""
        dead, playable_mask = self.board_masks(state.board)
        out = {}
        for p in range(self.rules.players):
            if p != giver:
                c = self.target(p, len(state.hands[p]), playable_mask)
                out[p] = (c, self.partition(self.pub[p][c], dead))
        return out

    # --- the action algorithm -------------------------------------------------------------

    def choose(self, state: State) -> Action:
        me, rules = state.current, self.rules
        self.eliminate(self.pub, [len(h) for h in state.hands])
        mine = self.private(state, me)[me]
        own = mine if self.cheat else self.pub[me]
        dead, playable_mask = self.board_masks(state.board)
        n = len(state.hands[me])
        for c in range(n):
            if mine[c] & ~playable_mask == 0:
                return play(c)
        room = self.hansim or state.hints < rules.max_hints
        if len(state.discards) < 5 and room:
            for c in range(n):
                if own[c] & ~dead == 0:
                    return discard(c)
        if state.hints > 0:
            values = self.hand_values(state, me)
            total = 0
            for p, (c, part) in values.items():
                total += part[state.hands[p][c]]
            return self.encode(me, total % self.hats, state)
        return discard(self.fallback_discard(state, me, own, dead))

    def fallback_discard(self, state: State, me: int, own: list[int], dead: int) -> int:
        """Steps 4 to 7; HanSim loops over all four slots even when the hand is shorter."""
        for c in range(self.slots):
            if own[c] & ~dead == 0:
                return c
        counts: Counter[int] = Counter()
        for row in self.pub if not self.cheat else self.private(state, me):
            for m in row:
                if known(m):
                    counts[m] += 1
        for c in range(self.slots):
            if known(own[c]) and counts[own[c]] > 1:
                return c
        crit = [self.critical(m) for m in own]
        for c in range(self.slots):
            if crit[c] is False:
                return c
        for c in range(self.slots):
            if crit[c] is not True:
                return c
        return 0

    def critical(self, mask: int) -> bool | None:
        """HanSim's isCritical: None unless the card is known."""
        if not known(mask):
            return None
        t = mask.bit_length() - 1
        r, n = rank_of(t, self.rules), self.used[t]
        if r == 1:
            return n == 2
        if r == self.rules.ranks:
            return True
        return n == 1

    # --- updates after an action ----------------------------------------------------------

    def observe(self, state_before: State, event: Event, values: dict | None):
        """Update the public tables. For a hint, `values` holds every hand's targeted card
        and partition, computed before the hint; each player decodes her own class, which
        equals the class of her true card."""
        a, rules = event.action, self.rules
        if a.kind == "hint":
            number = self.decode(event.player, a)
            hands = state_before.hands
            assert values is not None
            seen = sum(values[p][1][hands[p][values[p][0]]] for p in values)
            assert seen % self.hats == number, "hint does not decode"
            for p, (c, part) in values.items():
                cls = part[hands[p][c]]
                keep = 0
                for t, k in part.items():
                    if k == cls:
                        keep |= 1 << t
                self.pub[p][c] &= keep
            target = a.target
            for c in range(len(hands[target])):
                cards = 0
                for t in range(self.n_cards):
                    hit = (
                        color_of(t, rules) == a.value
                        if a.attr == "color"
                        else rank_of(t, rules) == a.value
                    )
                    if hit == (c in event.touched):
                        cards |= 1 << t
                self.pub[target][c] &= cards
        else:
            row = self.pub[event.player]
            del row[a.index]
            row.append(self.all)
            self.used[event.card] += 1


def play_info_game(rules: Rules, deck: list[int], cheat=False, hansim=True) -> State:
    """One game of the information strategy. Sets state.lives to 0 (score 0) if a player
    picks an empty slot, as HanSim scores an illegal action."""
    team = InfoTeam(rules, cheat, hansim)
    state = new_game(rules, deck)
    team.start(state)
    while not state.over:
        action = team.choose(state)
        if action.kind != "hint" and action.index >= len(state.hands[state.current]):
            team.illegal = True
            state.lives, state.over = 0, True
            break
        values = None
        if action.kind == "hint":
            team.eliminate(team.pub, [len(h) for h in state.hands])
            values = team.hand_values(state, state.current)
        before = State(rules, state.deck, [list(h) for h in state.hands])
        event = step(state, action)
        team.observe(before, event, values)
    return state


def summarize(scores: list[int]) -> str:
    n = len(scores)
    m = mean(scores)
    sd = math.sqrt(sum((s - m) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
    perfect = sum(s == 25 for s in scores) / n
    pe = math.sqrt(perfect * (1 - perfect) / n)
    return (
        f"mean {m:.3f} ± {1.96 * sd / math.sqrt(n):.3f}, "
        f"perfect {100 * perfect:.2f}% ± {196 * pe:.2f}% (n = {n}, 95% intervals)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=2000, help="games per setting")
    parser.add_argument("--players", type=int, nargs="*", default=[5])
    parser.add_argument("--cheat", action="store_true", help="players see their own cards")
    parser.add_argument("--official", action="store_true", help="no discards at 8 tokens")
    parser.add_argument("--hansim-decks", action="store_true", help="HanSim's decks, seed 100")
    args = parser.parse_args()
    for p in args.players:
        rules = Rules(players=p, discard_when_full=not args.official)
        decks = hansim_decks(100) if args.hansim_decks else None
        t0 = time.time()
        scores = []
        for i in range(args.n):
            deck = next(decks) if decks else seeded_deck(rules, i)
            scores.append(play_info_game(rules, deck, args.cheat, not args.official).score)
        hist = Counter(scores)
        print(f"{p} players: {summarize(scores)}; {time.time() - t0:.0f} s")
        print("  scores:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
