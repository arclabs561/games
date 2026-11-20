"""Strategies a person can follow at the table, written as short ordered lists of rules.

A **strategy card** has two lists. The **keep list** says which dice to keep before a reroll;
the **box list** says where to score the final roll. Each list is read from the top, and the
first rule whose conditions all hold is followed. A keep rule names a **keep**, a way of choosing
dice that a person can apply at a glance ("the most common face", "the longest run"). A box rule
names a box. When no box rule applies, the roll goes where its points minus the box's **par**
are largest, with par a short table of round numbers (`PAR`). When no keep rule applies, the
most common face is kept.

Conditions are **tests**, each a few words about the dice or the score card: "three or more of
a kind", "its upper box is open", "one roll left", "on pace for the bonus". The **most common
face** is the face with the most dice, the higher face on a tie. **Pace** is the upper total
minus three times the faces of the filled upper boxes: three of each face makes 63 exactly, so a
pace of 0 or more means the bonus is on track.

A card fixes every choice of the game, so `solver.solve(rule=card)` gives its exact expected
score, and `policy.graph(..., rule=card)` its exact score distribution.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from yahtzee.rules import CATEGORIES, JOKER_SCORES, OFFICIAL, UPPER, bonus, box_score
from yahtzee.solver import Tables, terminal

N = len(CATEGORIES)
BOX = {c: i for i, c in enumerate(CATEGORIES)}
SHORT = {
    "aces": "1s",
    "twos": "2s",
    "threes": "3s",
    "fours": "4s",
    "fives": "5s",
    "sixes": "6s",
    "three_kind": "3K",
    "four_kind": "4K",
    "full_house": "FH",
    "small_straight": "SS",
    "large_straight": "LS",
    "yahtzee": "Y",
    "chance": "C",
}


class Dice:
    """Facts about each of the 252 rolls, as arrays in the order of `Tables.rolls`."""

    def __init__(self, t: Tables):
        rolls = t.rolls
        self.count = np.array([[r.count(f) for f in range(1, 7)] for r in rolls])
        # The most common face, the higher one on a tie.
        self.face = np.array([max(range(1, 7), key=lambda f: (r.count(f), f)) for r in rolls])
        self.top = self.count.max(axis=1)
        shape = [sorted((r.count(f) for f in set(r)), reverse=True) for r in rolls]
        self.two_pair = np.array([s[:2] == [2, 2] for s in shape])
        self.full_house = np.array([s == [3, 2] for s in shape])
        self.run = np.array([longest_run(r)[1] for r in rolls])
        self.total = np.array([sum(r) for r in rolls])
        self.high = np.array([max(r) for r in rolls])
        self.ones = self.count[:, 0]
        # Keeps, as indices into t.keeps.
        k = t.keep_index
        self.keep = {
            "all": np.array([k[r] for r in rolls]),
            "none": np.full(len(rolls), k[()]),
            "most": np.array([k[(f,) * r.count(f)] for f, r in zip(self.face, rolls, strict=True)]),
            "run": np.array([k[run_keep(r)] for r in rolls]),
            "pairs": np.array([k[pairs_keep(r)] for r in rolls]),
            "high": np.array([k[(max(r),)] for r in rolls]),
            "fives": np.array([k[tuple(d for d in r if d >= 5)] for r in rolls]),
            "most+high": np.array([k[most_high_keep(r)] for r in rolls]),
        }
        # All dice of each face, for keeps that depend on the score card.
        self.of_face = np.array([[k[(f,) * r.count(f)] for r in rolls] for f in range(1, 7)])


def longest_run(roll) -> tuple[int, int]:
    """The highest face and length of the longest run of consecutive faces (the higher run)."""
    faces = set(roll)
    best = (0, 0)
    for v in sorted(faces):
        n = next(k for k in range(1, 7) if v + k not in faces)
        if v - 1 not in faces and n >= best[1]:
            best = (v + n - 1, n)
    return best


assert longest_run((1, 2, 3, 5, 6)) == (3, 3) and longest_run((1, 2, 4, 5, 6)) == (6, 3)


def run_keep(roll) -> tuple[int, ...]:
    """One die of each face in the longest run."""
    end, n = longest_run(roll)
    return tuple(range(end - n + 1, end + 1))


assert run_keep((1, 1, 3, 4, 5)) == (3, 4, 5) and run_keep((2, 3, 4, 5, 5)) == (2, 3, 4, 5)


def pairs_keep(roll) -> tuple[int, ...]:
    """Every die of every face shown at least twice (two pair, a full house, or the group)."""
    return tuple(d for d in roll if roll.count(d) >= 2) or (max(roll),)


assert pairs_keep((2, 2, 5, 5, 6)) == (2, 2, 5, 5) and pairs_keep((1, 2, 3, 4, 6)) == (6,)


def most_high_keep(roll) -> tuple[int, ...]:
    """The most common face, and a 5 or 6 beside it if there is one."""
    f = max(range(1, 7), key=lambda v: (roll.count(v), v))
    rest = [d for d in roll if d != f and d >= 5]
    return tuple(sorted((f,) * roll.count(f) + tuple(rest[-1:])))


@dataclass
class Card:
    """The score-card facts for a batch of states, shaped (B, 1) to broadcast against rolls."""

    t: Tables
    states: np.ndarray

    @cached_property
    def mask(self):
        return (self.states // (2 * self.t.width))[:, None]

    @cached_property
    def total(self):
        return ((self.states // 2) % self.t.width)[:, None]

    @cached_property
    def flag(self):
        return (self.states % 2)[:, None] == 1

    @cached_property
    def open(self):
        """(B, 13): whether each box is open."""
        bits = 1 << np.arange(N)
        return (self.mask & bits[None, :]) == 0

    @cached_property
    def pace(self):
        """Upper total minus three times the faces of the filled upper boxes."""
        faces = np.arange(1, 7)
        return self.total - 3 * ((~self.open[:, :6]) * faces).sum(axis=1, keepdims=True)

    @cached_property
    def secured(self):
        return self.total >= 63

    @cached_property
    def turns_left(self):
        return self.open.sum(axis=1, keepdims=True)


# Tests: each maps (dice facts, card facts) to booleans that broadcast to (B, R).
def _face_open(d: Dice, c: Card):
    return np.take_along_axis(
        c.open[:, :6], np.broadcast_to(d.face - 1, (len(c.states), len(d.face))), axis=1
    )


TESTS = {
    "five of a kind": lambda d, c: d.top[None] == 5,
    "four or more of a kind": lambda d, c: d.top[None] >= 4,
    "three or more of a kind": lambda d, c: d.top[None] >= 3,
    "a pair or more": lambda d, c: d.top[None] >= 2,
    "two pair": lambda d, c: d.two_pair[None],
    "a full house": lambda d, c: d.full_house[None],
    "five in a row": lambda d, c: d.run[None] == 5,
    "four or more in a row": lambda d, c: d.run[None] >= 4,
    "three or more in a row": lambda d, c: d.run[None] >= 3,
    "the most common face is 4 or more": lambda d, c: d.face[None] >= 4,
    "the most common face is 3 or less": lambda d, c: d.face[None] <= 3,
    "the most common face is 1": lambda d, c: d.face[None] == 1,
    "the most common face is 5 or more": lambda d, c: d.face[None] >= 5,
    "the most common face is 2 or less": lambda d, c: d.face[None] <= 2,
    "its upper box is open": _face_open,
    "on pace for the bonus": lambda d, c: (c.pace >= 0) & ~c.secured,
    "the bonus is made": lambda d, c: c.secured,
    "the dice total 22 or more": lambda d, c: d.total[None] >= 22,
    "the dice total 25 or more": lambda d, c: d.total[None] >= 25,
    "one roll left": lambda d, c: np.ones((1, 1), bool),  # replaced per call; see `keeps`
    "two rolls left": lambda d, c: np.ones((1, 1), bool),
    "Yahtzee box holds 50": lambda d, c: c.flag,
    "nine or more boxes open": lambda d, c: c.turns_left >= 9,
    "six or fewer boxes open": lambda d, c: c.turns_left <= 6,
    "three or fewer boxes open": lambda d, c: c.turns_left <= 3,
}
for _c in CATEGORIES:
    TESTS[f"{SHORT[_c]} open"] = lambda d, c, i=BOX[_c]: c.open[:, i : i + 1]
    TESTS[f"{SHORT[_c]} filled"] = lambda d, c, i=BOX[_c]: ~c.open[:, i : i + 1]


@dataclass(frozen=True)
class Keep:
    """If every test holds, keep these dice."""

    tests: tuple[str, ...]
    keep: str  # a key of Dice.keep, or "most open" (the most common face with its box open)

    def __str__(self):
        return f"{' and '.join(self.tests) or 'otherwise'}: keep {KEEP_WORDS[self.keep]}"


@dataclass(frozen=True)
class Box:
    """If every test holds and the box is open, score the roll there."""

    tests: tuple[str, ...]
    box: str  # a category, or "face" for the upper box of the most common face

    def __str__(self):
        where = "its upper box" if self.box == "face" else SHORT[self.box]
        return f"{' and '.join(self.tests) or 'otherwise'}: score in {where}"


KEEP_WORDS = {
    "all": "all five",
    "none": "nothing",
    "most": "the most common face",
    "run": "the longest run",
    "pairs": "every pair",
    "high": "the highest die",
    "fives": "every 5 and 6",
    "most+high": "the most common face and a 5 or 6",
    "most open": "the most common face whose upper box is open",
}

# Par: what each box is worth when it is left open, in round numbers.
PAR = {
    "aces": 3,
    "twos": 6,
    "threes": 9,
    "fours": 12,
    "fives": 15,
    "sixes": 18,
    "three_kind": 22,
    "four_kind": 13,
    "full_house": 23,
    "small_straight": 29,
    "large_straight": 33,
    "yahtzee": 17,
    "chance": 22,
}


@dataclass(frozen=True)
class Strategy:
    """A strategy card: the keep list, the box list, and the par table with its bonus weight.

    For an upper box, the roll's points minus par is multiplied by `weight` while the bonus is
    still open, since each pip ahead of or behind pace changes the chance of the bonus.
    """

    keep_rules: tuple[Keep, ...] = ()
    box_rules: tuple[Box, ...] = ()
    par: tuple[tuple[str, float], ...] = tuple(PAR.items())
    weight: float = 1.0
    default: str = "most"
    _cache: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def dice(self, t: Tables) -> Dice:
        if "dice" not in self._cache:
            self._cache["dice"] = Dice(t)
        return self._cache["dice"]

    def size(self) -> int:
        return len(self.keep_rules) + len(self.box_rules)

    def test(self, name: str, d: Dice, c: Card, left: int | None = None) -> np.ndarray:
        if name == "one roll left":
            return np.full((1, 1), left == 1)
        if name == "two rolls left":
            return np.full((1, 1), left == 2)
        return TESTS[name](d, c)

    def keeps(self, t: Tables, states: np.ndarray, left: int) -> np.ndarray:
        return self.keep_choice(t, states, left)[0]

    def keep_choice(self, t: Tables, states: np.ndarray, left: int):
        """(B, R) the keep after each roll, and (B, R) the number of the rule that chose it."""
        d, c = self.dice(t), Card(t, states)
        shape = (len(states), len(t.rolls))
        out, fired = np.full(shape, -1), np.full(shape, -1)
        for i, rule in enumerate((*self.keep_rules, Keep((), self.default))):
            hit = (fired < 0) & self.holds(rule.tests, d, c, left)
            out = np.where(hit, self.keep_of(rule.keep, d, c), out)
            fired = np.where(hit, i, fired)
        return out, fired

    def holds(self, tests, d: Dice, c: Card, left: int | None = None) -> np.ndarray:
        hit = np.ones((len(c.states), len(d.face)), bool)
        for name in tests:
            if name != "it scores":
                hit = hit & self.test(name, d, c, left)
        return hit

    def keep_of(self, keep: str, d: Dice, c: Card) -> np.ndarray:
        if keep != "most open":
            return d.keep[keep][None, :]
        # The most common face among the open upper boxes, else the most common face.
        score = d.count * 10 + np.arange(1, 7)  # (R, 6): count first, then the higher face
        score = np.where(c.open[:, None, :6], score[None], -1)  # (B, R, 6)
        face = score.argmax(axis=2)
        pick = d.of_face[face, np.arange(len(d.face))[None, :]]
        return np.where(score.max(axis=2) >= 10, pick, d.keep["most"][None, :])

    def boxes(self, t: Tables, states: np.ndarray) -> np.ndarray:
        chosen = self.box_choice(t, states)[0]
        allow = np.zeros((*chosen.shape, N), bool)
        np.put_along_axis(allow, chosen[:, :, None], True, axis=2)
        return allow

    def box_choice(self, t: Tables, states: np.ndarray):
        """(B, R) the box for each final roll, and the number of the rule that chose it."""
        d, c = self.dice(t), Card(t, states)
        nb, nr = len(states), len(t.rolls)
        points = self.points(t, d, c)  # (B, R, 13)
        chosen, fired = np.full((nb, nr), -1), np.full((nb, nr), -1)
        for i, rule in enumerate(self.box_rules):
            hit = (fired < 0) & self.holds(rule.tests, d, c)
            box = self.box_of(rule.box, d, nb)
            hit = hit & np.take_along_axis(c.open, box, axis=1)
            if "it scores" in rule.tests:
                hit = hit & (np.take_along_axis(points, box[:, :, None], axis=2)[:, :, 0] > 0)
            chosen = np.where(hit, box, chosen)
            fired = np.where(hit, i, fired)
        rest = fired < 0
        chosen = np.where(rest, self.margin(points, c).argmax(axis=2), chosen)
        fired = np.where(rest, len(self.box_rules), fired)
        return chosen, fired

    @staticmethod
    def box_of(box: str, d: Dice, nb: int) -> np.ndarray:
        if box == "face":
            return np.broadcast_to(d.face[None, :] - 1, (nb, len(d.face)))
        return np.full((nb, len(d.face)), BOX[box])

    def margin(self, points: np.ndarray, c: Card, par=None, weight=None) -> np.ndarray:
        """(B, R, 13): points minus par, weighted in the upper boxes, -inf where filled."""
        par = np.array([dict(self.par if par is None else par)[k] for k in CATEGORIES], float)
        weight = self.weight if weight is None else weight
        margin = points - par
        w = np.where(c.secured, 1.0, weight)[:, :, None]  # (B, 1, 1)
        margin = np.concatenate([margin[:, :, :6] * w, margin[:, :, 6:]], axis=2)
        return np.where(c.open[:, None, :], margin, -np.inf)

    def points(self, t: Tables, d: Dice, c: Card) -> np.ndarray:
        """(B, R, 13): the points each roll writes in each box, Joker included.

        The Yahtzee bonus is left out: it is the same whichever box the roll goes in.
        """
        pts = np.broadcast_to(t.score[None].astype(float), (len(c.states), *t.score.shape)).copy()
        y = t.yahtzee[None, :]
        face_open = np.take_along_axis(
            c.open[:, :6], np.broadcast_to(d.face - 1, (len(c.states), len(d.face))), axis=1
        )
        joker = y & ~c.open[:, BOX["yahtzee"] : BOX["yahtzee"] + 1] & ~face_open
        for name, v in JOKER_SCORES.items():
            pts[:, :, BOX[name]] = np.where(joker, v, pts[:, :, BOX[name]])
        return pts

    def lines(self) -> list[str]:
        """The card as text, one rule per line."""
        out = ["Keep (first rule that applies):"]
        out += [f"  {i}. {r}" for i, r in enumerate(self.keep_rules, 1)]
        out += [f"  {len(self.keep_rules) + 1}. otherwise: keep {KEEP_WORDS[self.default]}"]
        out += ["Score (first rule that applies):"]
        out += [f"  {i}. {r}" for i, r in enumerate(self.box_rules, 1)]
        par = ", ".join(f"{SHORT[k]} {v:g}" for k, v in self.par)
        out += [f"  otherwise: the box with the most points minus par ({par})"]
        if self.weight != 1:
            out += [
                f"  (upper boxes: points minus par times {self.weight:g} until the bonus is made)"
            ]
        return out


UPPER_FACE = {c: i + 1 for i, c in enumerate(UPPER)}
Y = BOX["yahtzee"]


@dataclass
class Moves:
    """A card's moves in a batch of states: the keeps, the box and what the box earns."""

    keeps: list[np.ndarray]  # (B, R) the keep after the first roll, then after the second
    box: np.ndarray  # (B, R) the box for each final roll
    gain: np.ndarray  # (B, R) the points written plus any Yahtzee bonus
    next: np.ndarray  # (B, R) the state after the turn


def decision_key(t: Tables, states: np.ndarray) -> np.ndarray:
    """What a card's tests can see of each state: the open boxes, the 50, pace and bonus.

    States with the same key get the same moves, so each key is worked out once.
    """
    c = Card(t, states)
    return (c.mask * 2 + c.flag) * 4 + ((c.pace >= 0) & ~c.secured) * 2 + c.secured


def moves(card: Strategy, t: Tables, states: np.ndarray, dedupe: bool = True) -> Moves:
    """Every move the card makes in these states. No values are needed: a card fixes them all."""
    rules = t.rules
    if dedupe:
        uniq, first, inv = np.unique(
            decision_key(t, states).ravel(), return_index=True, return_inverse=True
        )
        rep = states[first]
    else:
        rep, inv = states, np.arange(len(states))
    d, c = card.dice(t), Card(t, rep)
    box = card.box_choice(t, rep)[0]
    points = np.take_along_axis(card.points(t, d, c), box[:, :, None], axis=2)[:, :, 0]
    keeps = [card.keeps(t, rep, left) for left in range(rules.rolls - 1, 0, -1)]
    keeps = [np.broadcast_to(k, box.shape)[inv] for k in keeps]
    box, points = box[inv], points[inv].astype(np.int64)
    c = Card(t, states)
    flag = c.flag
    bonus = np.where(t.yahtzee[None, :] & flag, rules.yahtzee_bonus, 0)
    up = np.where(box < 6, points, 0)
    total = np.minimum(c.total + up, t.cap)
    fifty = flag | ((box == Y) & (points == 50))
    nxt = t.index(c.mask | (1 << box), total, fifty.astype(np.int64))
    return Moves(keeps, box, points + bonus, nxt)


def final_chances(t: Tables, keeps: list[np.ndarray]) -> np.ndarray:
    """(B, R): the chance of each final roll when the turn's keeps are made."""
    nb, nk = keeps[0].shape[0], len(t.keeps)
    q = np.broadcast_to(t.first, keeps[0].shape)
    rows = np.arange(nb)[:, None] * nk
    for keep in keeps:
        z = np.bincount((rows + keep).ravel(), weights=q.ravel(), minlength=nb * nk)
        q = z.reshape(nb, nk) @ t.P
    return q


def expected_score(card: Strategy, t: Tables | None = None, batch: int = 2048) -> float:
    """The card's exact expected score, by following it forward from the empty score card.

    Each state carries its chance of being reached. The score is the sum, over states and final
    rolls, of the chance of reaching them times what the turn earns, plus the upper bonus.
    """
    t = t or Tables(OFFICIAL)
    states, occ = np.array([t.index(0, 0, 0)]), np.array([1.0])
    total = 0.0
    for _ in range(N):
        nxt, mass = [], []
        for lo in range(0, len(states), batch):
            s, o = states[lo : lo + batch], occ[lo : lo + batch]
            m = moves(card, t, s)
            w = o[:, None] * final_chances(t, m.keeps)
            total += float((w * m.gain).sum())
            nxt.append(m.next.ravel())
            mass.append(w.ravel())
        states, inv = np.unique(np.concatenate(nxt), return_inverse=True)
        occ = np.bincount(inv, np.concatenate(mass), len(states))
    upper = (states // 2) % t.width
    return total + float(occ[upper == t.cap].sum()) * t.rules.upper_bonus


def card_values(card: Strategy, t: Tables | None = None, batch: int = 2048):
    """The card's expected score still to come from every reachable state, backwards.

    The same numbers as `solver.solve(rule=card)`, computed without the solver's maxima.
    """
    t = t or Tables(OFFICIAL)
    values = terminal(t)
    for filled in range(N - 1, -1, -1):
        states = t.states(filled)
        for lo in range(0, len(states), batch):
            s = states[lo : lo + batch]
            m = moves(card, t, s)
            v = m.gain + values[m.next]
            rows = np.arange(len(s))[:, None]
            for keep in reversed(m.keeps):
                v = (v @ t.P.T)[rows, keep]
            values[s] = v @ t.first
    return t, values


def play(card: Strategy, t: Tables, games: int, seed: int) -> np.ndarray:
    """Seeded games played by the card, scored with the functions of `rules`, not the solver's.

    The dice are drawn as in `simulate.deal`: a row of five per game, turn and roll, of which a
    reroll of j dice uses the first j.
    """
    rules = t.rules
    rng = np.random.default_rng(seed)
    dice = rng.integers(1, 7, size=(games, N, rules.rolls, 5))
    roll_index = {r: i for i, r in enumerate(t.rolls)}
    filled = [frozenset[str]() for _ in range(games)]
    upper = np.zeros(games, dtype=np.int64)
    fifty = np.zeros(games, dtype=bool)
    score = np.zeros(games, dtype=np.int64)
    for turn in range(N):
        mask = np.array([sum(1 << BOX[c] for c in f) for f in filled])
        states = t.index(mask, np.minimum(upper, t.cap), fifty.astype(np.int64))
        uniq, inv = np.unique(states, return_inverse=True)
        rolls = [tuple(sorted(dice[g, turn, 0])) for g in range(games)]
        for level in range(1, rules.rolls):
            keep = card.keeps(t, uniq, rules.rolls - level)
            new = []
            for g in range(games):
                kept = t.keeps[keep[inv[g], roll_index[rolls[g]]]]
                assert all(kept.count(v) <= rolls[g].count(v) for v in kept)
                rest = tuple(dice[g, turn, level, : 5 - len(kept)])
                new.append(tuple(sorted(kept + rest)))
            rolls = new
        box = card.box_choice(t, uniq)[0]
        for g in range(games):
            c = CATEGORIES[box[inv[g], roll_index[rolls[g]]]]
            assert c not in filled[g]
            points = box_score(rules, rolls[g], c, filled[g])
            score[g] += points + bonus(rules, rolls[g], bool(fifty[g]))
            if c in UPPER:
                upper[g] += points
            if c == "yahtzee" and points == 50:
                fifty[g] = True
            filled[g] = filled[g] | {c}
    return score + np.where(upper >= rules.threshold, rules.upper_bonus, 0)


def _k(*a: str) -> Keep:
    return Keep(tuple(a[:-1]), a[-1])


# A starting card for the search: keep made hands, runs and groups, else the open face.
BASE = Strategy(
    keep_rules=(
        _k("five of a kind", "all"),
        _k("five in a row", "LS open", "all"),
        _k("four or more in a row", "LS open", "run"),
        _k("four or more in a row", "SS open", "all"),
        _k("three or more of a kind", "most"),
        _k("three or more in a row", "SS open", "run"),
    ),
    default="most open",
)
