"""Card facts used to test the embeddings: colors, card type, mana value, creature type.

None of these facts is shown to the embedding methods, which see only which cards share
decks. They are used afterwards, to ask what a card's neighbors have in common.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

WUBRG = "WUBRG"
PAIRS = ["WU", "UB", "BR", "RG", "WG", "WB", "UR", "BG", "WR", "UG"]  # the ten, WUBRG order

# Bloomburrow's ten archetypes, one creature type per color pair, as listed by the set's
# designer (M. Rosewater, "The Bloom of the Burrow, Part 1", Making Magic, 2024).
BLB_TRIBES = {
    "Bird": "WU",
    "Rat": "UB",
    "Lizard": "BR",
    "Raccoon": "RG",
    "Rabbit": "WG",
    "Bat": "WB",
    "Otter": "UR",
    "Squirrel": "BG",
    "Mouse": "WR",
    "Frog": "UG",
}


@dataclass(frozen=True)
class Card:
    """One card of the vocabulary, with the facts the tests use."""

    name: str
    colors: str  # color identity in WUBRG order; "" for colorless
    mana_value: int
    kind: str  # Creature, Instant, Sorcery, Enchantment, Artifact, Land, Planeswalker, Battle
    subtypes: tuple[str, ...]
    rarity: str
    decks: int  # fit decks holding the card
    pair: str  # the two-color pair it is played in most, among fit decks holding it
    pair_share: float  # the share of its two-color decks in that pair


def wubrg(s: str) -> str:
    """Colors in WUBRG order: 'GB' -> 'BG'."""
    return "".join(sorted(set(s or ""), key=WUBRG.index))


def kind_of(types: str) -> str:
    """The main card type from a type line such as 'Legendary Creature - Bat Cleric'."""
    main = types.split(" - ")[0].split()
    for k in (
        "Creature",
        "Land",
        "Planeswalker",
        "Battle",
        "Instant",
        "Sorcery",
        "Artifact",
        "Enchantment",
    ):
        if k in main:
            return k
    return main[-1] if main else ""


def subtypes_of(types: str) -> tuple[str, ...]:
    """The subtypes after the dash: 'Creature - Bat Cleric' -> ('Bat', 'Cleric')."""
    return tuple(types.split(" - ", 1)[1].split()) if " - " in types else ()


FIELDS = [
    "name",
    "colors",
    "mana_value",
    "kind",
    "subtypes",
    "rarity",
    "decks",
    "pair",
    "pair_share",
]


def write_cards(path: Path, cards: list[Card]) -> None:
    """Write the vocabulary as CSV, sorted by name."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for c in sorted(cards, key=lambda c: c.name):
            w.writerow(
                [
                    c.name,
                    c.colors,
                    c.mana_value,
                    c.kind,
                    " ".join(c.subtypes),
                    c.rarity,
                    c.decks,
                    c.pair,
                    f"{c.pair_share:.4f}",
                ]
            )


def read_cards(path: Path) -> list[Card]:
    """Read a vocabulary written by `write_cards`."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Card(
                r["name"],
                r["colors"],
                int(r["mana_value"]),
                r["kind"],
                tuple(r["subtypes"].split()),
                r["rarity"],
                int(r["decks"]),
                r["pair"],
                float(r["pair_share"]),
            )
            for r in csv.DictReader(f)
        ]


def tribe_partner(card: Card, tribes: dict[str, str]) -> str | None:
    """For a one-color creature of an archetype's creature type, the archetype's other color.

    A black Bat (white-black) gives 'W'. Cards of two archetype types, or whose color is
    not one of their archetype's two, give None.
    """
    if len(card.colors) != 1 or card.kind != "Creature":
        return None
    pairs = {tribes[t] for t in card.subtypes if t in tribes}
    if len(pairs) != 1:
        return None
    pair = pairs.pop()
    if card.colors not in pair:
        return None
    return pair.replace(card.colors, "")
