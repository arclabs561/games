"""The rules of solitaire Yahtzee, as small functions on one roll and one score card.

A **roll** is the dice showing, as a sorted tuple such as (1, 1, 6, 6, 6); the order of the dice
does not matter to the score, so the 7,776 ordered rolls of five dice fold into 252 sorted ones.
A **keep** is the sorted tuple of dice set aside before a reroll; it may be empty or the whole
roll. A **category** is a box on the score card. `Rules` collects everything that varies
between the versions studied here: the number of dice and faces, the categories, the two bonuses
and the Joker rule.

The official rules are those of the Hasbro instructions (2003, https://www.hasbro.com/common/
instruct/40958.pdf), read as Verhoeff does (https://www-set.win.tue.nl/~wstomv/misc/yahtzee/
rules.html):

- Three rolls per turn; any dice may be kept before each reroll. Thirteen turns, one box each.
- 35 points if the six upper boxes total 63 or more.
- 100 points for each further Yahtzee once the Yahtzee box holds 50 (the **Yahtzee bonus**).
- A Yahtzee scores in full (25, 30 or 40) in Full House, Small Straight or Large Straight once
  both the Yahtzee box and the upper box of its face are filled, with zeros allowed (the
  **Joker**).

With `joker="free"`, as in Verhoeff's and Glenn's solutions, the player may still put such a
Yahtzee in any open box. With `joker="forced"` the order of the Hasbro text applies: the upper
box of its face if it is open, else any open lower box, else any open upper box (for zero).
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from math import factorial, prod

type Roll = tuple[int, ...]

UPPER = ("aces", "twos", "threes", "fours", "fives", "sixes")
LOWER = (
    "three_kind",
    "four_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yahtzee",
    "chance",
)
CATEGORIES = UPPER + LOWER
JOKER_SCORES = {"full_house": 25, "small_straight": 30, "large_straight": 40}


@dataclass(frozen=True)
class Rules:
    """One version of the game. The defaults are the official rules."""

    dice: int = 5
    faces: int = 6
    rolls: int = 3
    categories: tuple[str, ...] = CATEGORIES
    threshold: int = 63  # upper total that earns the upper bonus
    upper_bonus: int = 35
    yahtzee_bonus: int = 100
    joker: str = "free"  # "free", "forced" or "none"

    def __post_init__(self):
        assert self.joker in ("free", "forced", "none"), self.joker
        assert len(set(self.categories)) == len(self.categories)
        assert all(c in CATEGORIES for c in self.categories)
        assert all(upper_face(c) <= self.faces for c in self.categories if c in UPPER)


def upper_face(category: str) -> int:
    """The face an upper box counts: 1 for aces, ..., 6 for sixes."""
    return UPPER.index(category) + 1


OFFICIAL = Rules()


def multisets(faces: int, size: int) -> list[Roll]:
    """Every sorted roll of `size` dice with `faces` faces."""
    return list(combinations_with_replacement(range(1, faces + 1), size))


assert len(multisets(6, 5)) == 252 and len(multisets(6, 2)) == 21


def probability(dice: Roll, faces: int) -> Fraction:
    """The chance that rolling len(dice) dice shows exactly these values, in any order."""
    counts = [dice.count(v) for v in set(dice)]
    return Fraction(factorial(len(dice)) // prod(factorial(c) for c in counts), faces ** len(dice))


assert probability((1, 1, 2), 6) == Fraction(3, 216)


def is_yahtzee(roll: Roll) -> bool:
    """All dice show the same face."""
    return len(set(roll)) == 1


def run(roll: Roll) -> int:
    """The length of the longest run of consecutive faces in the roll."""
    faces = set(roll)
    return max(next(k for k in range(1, len(roll) + 2) if v + k not in faces) for v in faces)


assert run((1, 2, 3, 3, 5)) == 3 and run((2, 3, 4, 5, 5)) == 4


def score(roll: Roll, category: str) -> int:
    """The points the roll earns in the box, without bonuses or Joker."""
    counts = sorted((roll.count(v) for v in set(roll)), reverse=True)
    if category in UPPER:
        v = upper_face(category)
        return v * roll.count(v)
    match category:
        case "three_kind":
            return sum(roll) if counts[0] >= 3 else 0
        case "four_kind":
            return sum(roll) if counts[0] >= 4 else 0
        case "full_house":
            return 25 if counts == [3, 2] else 0  # a Yahtzee is not a full house
        case "small_straight":
            return 30 if run(roll) >= 4 else 0
        case "large_straight":
            return 40 if run(roll) >= 5 else 0
        case "yahtzee":
            return 50 if is_yahtzee(roll) else 0
        case "chance":
            return sum(roll)
    raise ValueError(category)


assert score((1, 1, 6, 6, 6), "sixes") == 18 and score((1, 1, 6, 6, 6), "full_house") == 25
assert score((2, 3, 4, 4, 6), "small_straight") == 0 and score((1, 2, 3, 4, 6), "small_straight")
assert score((5, 5, 5, 5, 5), "full_house") == 0


def joker(rules: Rules, roll: Roll, filled: frozenset[str]) -> bool:
    """Whether the roll scores in full in the straights and Full House as a Joker."""
    if rules.joker == "none" or not is_yahtzee(roll) or "yahtzee" not in filled:
        return False
    upper = UPPER[roll[0] - 1]
    return upper in filled or upper not in rules.categories


def box_score(rules: Rules, roll: Roll, category: str, filled: frozenset[str]) -> int:
    """The points written in the box, Joker included, Yahtzee bonus not."""
    if category in JOKER_SCORES and joker(rules, roll, filled):
        return JOKER_SCORES[category]
    return score(roll, category)


def bonus(rules: Rules, roll: Roll, yahtzee_50: bool) -> int:
    """The Yahtzee bonus for this roll, given whether the Yahtzee box holds 50."""
    return rules.yahtzee_bonus if yahtzee_50 and is_yahtzee(roll) else 0


def legal(rules: Rules, roll: Roll, open_boxes: tuple[str, ...]) -> tuple[str, ...]:
    """The open boxes the roll may go in."""
    filled = frozenset(rules.categories) - set(open_boxes)
    if rules.joker != "forced" or not is_yahtzee(roll) or "yahtzee" not in filled:
        return open_boxes
    upper = UPPER[roll[0] - 1]
    if upper in open_boxes:
        return (upper,)
    lower = tuple(c for c in open_boxes if c in LOWER)
    return lower or open_boxes


assert legal(Rules(joker="forced"), (4,) * 5, ("fours", "chance", "yahtzee")) == (
    "fours",
    "chance",
    "yahtzee",
)  # the Yahtzee box is open, so no Joker yet
assert legal(Rules(joker="forced"), (4,) * 5, ("fours", "chance")) == ("fours",)
assert legal(Rules(joker="forced"), (4,) * 5, ("aces", "chance")) == ("chance",)
