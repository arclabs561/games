"""One real search card, modeled exactly with its costs: Pot of Prosperity.

The card's text, as Yugipedia gives it from Konami's card database (entry 15756), read
2026-11-20 and paraphrased here:

- As a cost, banish 3 or 6 cards of your choice from your Extra Deck, face-down.
- Then reveal ("excavate") that many cards from the top of the deck, add 1 of them to the
  hand, and put the rest on the bottom.
- Only 1 can be activated per turn, and the player cannot draw cards by card effects for the
  whole turn in which it is activated. It is Limited: a deck holds at most 1 copy.

What `draws.py` leaves out, and this module keeps:

- **The Extra Deck cost.** Revealing 6 costs 6 of the 15 Extra Deck cards for the rest of the
  duel, so a deck that needs its Extra Deck may reveal only 3. Both are computed.
- **No draws that turn.** A hand that holds Prosperity and a draw-two must choose one. If it
  plays the draw-two, Prosperity cannot be activated afterwards; if it plays Prosperity, the
  draw-two cannot draw. The player chooses before seeing the deck, and takes the choice
  with the higher chance of meeting the need (`best`), or always one of them (`policy`).
- **One copy.** A second copy is not allowed, so `draws.py`'s three lookers are not a deck
  that can be built with this card.

The draw-two stays generic (draw 2, one per turn, no cost), as in `draws.py`, so that the
example isolates what this card's own costs do. In this example's model (`Engine` with
`draw2` and one looker) the generic looker is played after the draw-two, which the real
card forbids; the difference between the two is the price of that rule.

`p_prosperity` is exact, by summing over hands and over revealed cards. `p_brute` lists
every order of a small deck, and `simulate` deals seeded decks; both share no code with the
exact sum except the needs.
"""

import argparse
from collections.abc import Iterator
from fractions import Fraction
from itertools import permutations
from math import comb
from random import Random

from yugioh.draws import Engine, p_engine
from yugioh.odds import FIRST, SECOND, Deck, Hand, Need, starter, starter_and_extender

# Kinds of card, in this order: the four roles, then Prosperity and the generic draw-two.
STARTER, EXTENDER, BRICK, OTHER, PROSPERITY, DRAW2 = range(6)
POLICIES = ("best", "prosperity", "draw2")


def kinds(deck: Deck, draw2: int) -> tuple[int, ...]:
    """Cards of each kind; Prosperity (one copy) and the draw-twos replace other cards."""
    s, e, b, o = deck.counts()
    if o < 1 + draw2:
        raise ValueError("Prosperity and the draw-twos do not fit among the other cards")
    return (s, e, b, o - 1 - draw2, 1, draw2)


def multisets(counts: tuple[int, ...], n: int) -> Iterator[tuple[tuple[int, ...], int]]:
    """Every draw of n cards by kind, with its number of ways (multivariate hypergeometric)."""
    if len(counts) == 1:
        if n <= counts[0]:
            yield (n,), comb(counts[0], n)
        return
    for k in range(min(n, counts[0]) + 1):
        for rest, ways in multisets(counts[1:], n - k):
            yield (k, *rest), comb(counts[0], k) * ways


def roles(cards: tuple[int, ...]) -> Hand:
    """A draw by kind seen as a hand by role: Prosperity and draw-twos left in hand are other."""
    return (cards[0], cards[1], cards[2], cards[3] + cards[4] + cards[5])


def minus(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def p_reveal(rest: tuple[int, ...], hand: Hand, look: int, need: Need) -> Fraction:
    """Chance that one of the top `look` cards of `rest` completes the need when added."""
    total = sum(rest)
    look = min(look, total)
    good = 0
    for seen, ways in multisets(rest, look):
        r = roles(seen)
        if any(r[i] and need(tuple(h + (j == i) for j, h in enumerate(hand))) for i in range(4)):
            good += ways
    return Fraction(good, comb(total, look))


def p_draw_two(rest: tuple[int, ...], hand: Hand, need: Need) -> Fraction:
    """Chance that the hand meets the need after drawing the top two of `rest`."""
    good = 0
    for drawn, ways in multisets(rest, min(2, sum(rest))):
        r = roles(drawn)  # a drawn Prosperity cannot be activated; a second draw-two is dead
        if need(tuple(h + x for h, x in zip(hand, r, strict=True))):
            good += ways
    return Fraction(good, comb(sum(rest), min(2, sum(rest))))


def choose(rest, cards, look: int, need: Need, policy: str) -> Fraction:
    """The chance for one opening hand, playing Prosperity or a draw-two as the policy says."""
    base = roles(cards)
    if need(base):
        return Fraction(1)
    played = (base[0], base[1], base[2], base[3] - 1)  # the card played leaves the hand
    options = {}
    if cards[PROSPERITY]:
        options["prosperity"] = p_reveal(rest, played, look, need)
    if cards[DRAW2]:
        options["draw2"] = p_draw_two(rest, played, need)
    if not options:
        return Fraction(0)
    if policy == "best" or policy not in options:
        return max(options.values())
    return options[policy]


def p_prosperity(
    deck: Deck, n: int, need: Need, look: int = 6, draw2: int = 0, policy: str = "best"
) -> Fraction:
    """Chance that the hand meets the need with Prosperity and `draw2` draw-twos: exact."""
    counts = kinds(deck, draw2)
    total = comb(sum(counts), n)
    p = Fraction(0)
    for cards, ways in multisets(counts, n):
        p += ways * choose(minus(counts, cards), cards, look, need, policy)
    return p / total


def play(order: list[int], n: int, look: int, need: Need, choice: str) -> bool:
    """One shuffled deck played with a fixed choice: 'prosperity', 'draw2' or 'none'."""
    hand = order[:n]
    count = [hand.count(k) for k in range(6)]
    base = (count[0], count[1], count[2], count[3] + count[4] + count[5])
    if need(base):
        return True
    if choice == "prosperity" and count[PROSPERITY]:
        for card in order[n : n + look]:
            if card < 4 and need(tuple(h + (i == card) for i, h in enumerate(base))):
                return True
        return False
    if choice == "draw2" and count[DRAW2]:
        drawn = order[n : n + 2]
        h = list(base)
        for card in drawn:
            h[card if card < 4 else OTHER] += 1
        return need(tuple(h))
    return False


def hand_choice(order: list[int], n: int, preferred: str) -> str:
    """The fixed choice for a hand: the preferred card if it is in hand, else the other one."""
    hand = order[:n]
    other = "draw2" if preferred == "prosperity" else "prosperity"
    kind = {"prosperity": PROSPERITY, "draw2": DRAW2}
    if kind[preferred] in hand:
        return preferred
    return other if kind[other] in hand else "none"


def p_brute(counts: tuple[int, ...], n: int, look: int, need: Need) -> Fraction:
    """The 'best' chance by listing every order of a small deck, choosing per hand."""
    cards = [k for k, c in enumerate(counts) for _ in range(c)]
    orders = list(permutations(range(len(cards))))
    by_hand: dict[tuple[int, ...], list[list[int]]] = {}
    for perm in orders:
        order = [cards[i] for i in perm]
        key = tuple(sorted(perm[:n]))  # the physical cards in hand
        by_hand.setdefault(key, []).append(order)
    good = Fraction(0)
    for decks in by_hand.values():
        best = max(
            Fraction(sum(play(o, n, look, need, c) for o in decks), len(decks))
            for c in ("prosperity", "draw2", "none")
        )
        good += best * len(decks)
    return good / len(orders)


def simulate(
    deck: Deck, n: int, need: Need, look: int, draw2: int, preferred: str, trials: int, seed: int
) -> int:
    """Seeded shuffled decks, out of `trials`, that meet the need with a fixed preference."""
    rng = Random(seed)
    cards = [k for k, c in enumerate(kinds(deck, draw2)) for _ in range(c)]
    hits = 0
    for _ in range(trials):
        rng.shuffle(cards)
        hits += play(cards, n, look, need, hand_choice(cards, n, preferred))
    return hits


EXAMPLE = Deck(40, 12, 9, 3)
ROWS = (
    # (label, draw-twos, reveal, policy)
    ("Prosperity, reveal 6", 0, 6, "best"),
    ("Prosperity, reveal 3", 0, 3, "best"),
    ("Prosperity (reveal 6) and 3 draw-twos", 3, 6, "best"),
    ("Same, always playing Prosperity first", 3, 6, "prosperity"),
    ("Same, always playing a draw-two first", 3, 6, "draw2"),
)


def table() -> list[tuple[str, Fraction, Fraction, Fraction]]:
    """The README's rows: going first (starter; starter and extender) and going second."""
    out = []
    for label, d2, look, policy in ROWS:
        out.append(
            (
                label,
                p_prosperity(EXAMPLE, FIRST, starter, look, d2, policy),
                p_prosperity(EXAMPLE, FIRST, starter_and_extender, look, d2, policy),
                p_prosperity(EXAMPLE, SECOND, starter, look, d2, policy),
            )
        )
    return out


def upper_bounds() -> list[tuple[str, Fraction, Fraction, Fraction]]:
    """The same decks in `draws.py`'s model, which lets a looker follow a draw-two."""
    out = []
    for label, eng in (
        ("generic look-at-6, 1 copy", Engine(lookers=1, look=6)),
        ("generic look-at-6 and 3 draw-twos, 1 copy", Engine(draw2=3, lookers=1, look=6)),
    ):
        out.append(
            (
                label,
                p_engine(EXAMPLE, eng, FIRST, starter),
                p_engine(EXAMPLE, eng, FIRST, starter_and_extender),
                p_engine(EXAMPLE, eng, SECOND, starter),
            )
        )
    return out


def worth(p: Fraction, n: int = FIRST) -> Fraction:
    """Starters in place of other cards in the example deck that give chance p of a starter."""
    from yugioh.odds import p_hand

    values = [p_hand(Deck(40, EXAMPLE.starters + k, 9, 3), n, starter) for k in range(8)]
    for k in range(7):
        if values[k] <= p <= values[k + 1]:
            return k + (p - values[k]) / (values[k + 1] - values[k])
    raise ValueError("worth more than seven starters")


def report() -> str:
    """The text of results/prosperity.md."""
    from yugioh.odds import p_hand

    lines = [
        "# Pot of Prosperity in the example deck",
        "",
        "Written by `uv run python -m yugioh.prosperity --write`. Exact. The deck has 40 cards:",
        "12 starters, 9 extenders and 3 bricks; added cards replace other cards. Worth in",
        "starters: how many starters in place of other cards give the same chance of a starter",
        "going first, by linear interpolation.",
        "",
        "| Deck | Starter, first | Starter and extender, first | Starter, second | Worth in starters |",
        "|---|---:|---:|---:|---:|",
    ]
    none = (
        "no draw or search card",
        p_hand(EXAMPLE, FIRST, starter),
        p_hand(EXAMPLE, FIRST, starter_and_extender),
        p_hand(EXAMPLE, SECOND, starter),
    )
    for label, a, b, c in [none, *table(), *upper_bounds()]:
        lines.append(
            f"| {label} | {float(a):.2%} | {float(b):.2%} | {float(c):.2%} | {float(worth(a)):.2f} |"
        )
    lines += ["", "Exact fractions, going first, at least one starter:", ""]
    for label, a, _, _ in [none, *table(), *upper_bounds()]:
        lines.append(f"- {label}: {a.numerator:,}/{a.denominator:,}")
    return "\n".join(lines) + "\n"


def simulation(trials: int = 200_000, seed: int = 1) -> str:
    """The text of results/prosperity_simulation.md: seeded decks against the exact chances."""
    from yugioh.odds import wilson

    lines = [
        "# Pot of Prosperity: simulation",
        "",
        "Written by `uv run python -m yugioh.prosperity --write`. The example deck with Pot of",
        f"Prosperity (revealing 6) and 3 draw-2 cards, going first, a starter. Seed {seed},",
        f"{trials:,} shuffled decks per row, Wilson 95% intervals.",
        "",
        "| Fixed choice | Hits | 95% interval | Exact | Exact in interval |",
        "|---|---:|---|---:|---|",
    ]
    for preferred in ("prosperity", "draw2"):
        hits = simulate(EXAMPLE, FIRST, starter, 6, 3, preferred, trials, seed)
        lo, hi = wilson(hits, trials)
        exact = p_prosperity(EXAMPLE, FIRST, starter, 6, 3, preferred)
        inside = "yes" if lo <= exact <= hi else "no"
        lines.append(f"| {preferred} first | {hits:,} | {lo:.2%} to {hi:.2%} | "
                     f"{float(exact):.2%} | {inside} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write", action="store_true", help="write results/prosperity*.md")
    args = ap.parse_args()
    text, sim = report(), simulation(args.trials, args.seed)
    print(text)
    print(sim)
    if args.write:
        results = Path(__file__).parent / "results"
        (results / "prosperity.md").write_text(text)
        (results / "prosperity_simulation.md").write_text(sim)


if __name__ == "__main__":
    main()
