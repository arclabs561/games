"""A calculator for the opening hand: exact chances for one deck, going first and second.

uv run python -m yugioh.calc --starters 12 --extenders 9 --bricks 3
uv run python -m yugioh.calc --deck 43 --starters 12 --draw2 3 --exact
"""

import argparse
from fractions import Fraction

from yugioh.draws import Engine, p_engine
from yugioh.odds import FIRST, NEEDS, SECOND, Deck, fewest

TARGETS = (Fraction(8, 10), Fraction(9, 10), Fraction(95, 100))


def parser() -> argparse.ArgumentParser:
    """The command line."""
    p = argparse.ArgumentParser(prog="python -m yugioh.calc", description=__doc__.split("\n")[0])
    p.add_argument("--deck", type=int, default=40, help="cards in the main deck (40 to 60)")
    p.add_argument("--starters", type=int, default=0)
    p.add_argument("--extenders", type=int, default=0)
    p.add_argument("--bricks", type=int, default=0)
    p.add_argument("--cantrips", type=int, default=0, help="draw 1, any number per turn")
    p.add_argument("--draw2", type=int, default=0, help="draw 2, one per turn")
    p.add_argument("--lookers", type=int, default=0, help="look at the top cards, take one")
    p.add_argument("--look", type=int, default=6, help="cards a looker reveals")
    p.add_argument("--exact", action="store_true", help="also print the fractions")
    return p


def percent(p: Fraction) -> str:
    """A chance as a percentage with two decimals."""
    return f"{float(p):.2%}"


def report(args: argparse.Namespace) -> list[str]:
    """The lines the calculator prints."""
    if not 40 <= args.deck <= 60:
        raise SystemExit("a main deck has 40 to 60 cards")
    deck = Deck(args.deck, args.starters, args.extenders, args.bricks)
    engine = Engine(args.cantrips, args.draw2, args.lookers, args.look)
    other = deck.other - engine.cards
    parts = [f"{args.starters} starters", f"{args.extenders} extenders", f"{args.bricks} bricks"]
    if engine.cards:
        for count, kind in ((engine.cantrips, "draw-1"), (engine.draw2, "draw-2")):
            parts += [f"{count} {kind}"] * (count > 0)
        parts += [f"{engine.lookers} look-at-{engine.look}"] * (engine.lookers > 0)
    lines = [f"Deck of {deck.size}: {', '.join(parts)}, {other} other."]
    if engine.cards:
        lines.append("Draw cards are played at once and at no cost (see yugioh/draws.py).")
    lines += ["", f"{'Exact chance that the hand has':36} {'going first':>12} {'going second':>13}"]
    for label, need in NEEDS.items():
        a, b = (p_engine(deck, engine, n, need) for n in (FIRST, SECOND))
        lines.append(f"{label:36} {percent(a):>12} {percent(b):>13}")
        if args.exact:
            lines.append(f"  first {a}, second {b}")
    if not engine.cards:
        lines += ["", f"Fewest starters for at least one in hand, in {deck.size} cards:"]
        lines.append(f"{'':14}" + "".join(f"{float(t):>6.0%}" for t in TARGETS))
        for label, n in (("going first", FIRST), ("going second", SECOND)):
            lines.append(f"{label:14}" + "".join(f"{fewest(t, deck.size, n):>6}" for t in TARGETS))
    return lines


def main(argv: list[str] | None = None) -> None:
    """Print the report for the deck on the command line."""
    print("\n".join(report(parser().parse_args(argv))))


if __name__ == "__main__":
    main()
