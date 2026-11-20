"""One Pokémon against another, at four levels of detail.

The team study (``teams.py``) asks which team of six is best. Its unit is a **set**: a Pokémon
with one ability and four damaging moves, fixed in advance. A **matchup** M(p, q) is the share
of its HP that the winner of a one-on-one fight between sets p and q has left when the loser
faints, positive when p wins: M is in [-1, 1] and M(q, p) = -M(p, q). The levels differ only in
how the fight is computed. Each is exact in rationals.

1. **Types.** Each attacks with the better of its own types; the damage per hit is the type
   multiplier. Both have the same HP and stats and hit at the same time, so if p deals a and
   takes b with a > b, p wins with 1 - b/a of its HP left (Schmitz's duel, ``equilibrium.py``).
2. **Moves.** The same duel, but the damage per hit is the best of the set's four moves:
   power × expected hits × accuracy × 1.5 for a move of the user's own type × multiplier.
3. **Abilities.** Level 2, with the ability effects in ``LEVEL3_ABILITIES``: immunities
   (Levitate, Flash Fire, ...), resistances (Thick Fat, Filter, ...) and boosts (Adaptability,
   Technician, ...).
4. **Stats.** Level 100, every IV 31, 84 EVs in each stat, a neutral nature, no item. The
   damage of a move is the game's formula, averaged over its 16 random rolls and its accuracy
   (no critical hits). The two fight in turns: higher priority moves first, then higher
   speed, and a speed tie is a coin flip. Each uses, every turn, its move with the most
   expected damage. HP falls by the expected damage, recoil and drain apply, and a move's
   stat changes to its user apply (Close Combat, Draco Meteor). The abilities of level 3 apply,
   plus those in ``LEVEL4_ABILITIES`` (Huge Power, Intimidate, Multiscale, the Ruin abilities,
   ...). Nothing else is modeled: no status, weather, terrain, items, switching or
   Terastallization.

**Sets.** A Pokémon's ability is the first in slot order that the model represents (else its
first). Its four moves are those, among the moves ``teams_data.py`` keeps, that maximize the
sum over the pool of min(1, expected damage of its best move / opponent's HP) at level 4: the
share of each opponent's HP its best hit takes, capped at a KO. Ties go to the larger uncapped
sum, then to alphabetical order. The same sets are used at every level, so a level changes only
how a fight is scored.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from pathlib import Path

from pokemon import pokedex, usage
from pokemon.equilibrium import duel_score, multiplier

HERE = Path(__file__).resolve().parent
MOVES_CSV = HERE / "teams_moves.csv"
POKEMON_CSV = HERE / "teams_pokemon.csv"

LEVEL = 100
IV, EV = 31, 84
STAB = Fraction(3, 2)
LEVELS = (1, 2, 3, 4)
TURN_LIMIT = 100

# Level 3: abilities that change a matchup without stats.
IMMUNE = {
    "levitate": "ground",
    "earth-eater": "ground",
    "flash-fire": "fire",
    "well-baked-body": "fire",
    "water-absorb": "water",
    "storm-drain": "water",
    "dry-skin": "water",
    "volt-absorb": "electric",
    "lightning-rod": "electric",
    "motor-drive": "electric",
    "sap-sipper": "grass",
}
RESIST = {  # defender's ability: {attacking type: multiplier}
    "thick-fat": {"fire": Fraction(1, 2), "ice": Fraction(1, 2)},
    "heatproof": {"fire": Fraction(1, 2)},
    "water-bubble": {"fire": Fraction(1, 2)},
    "purifying-salt": {"ghost": Fraction(1, 2)},
    "dry-skin": {"fire": Fraction(5, 4)},
}
FILTER = ("filter", "solid-rock", "prism-armor")  # super-effective damage × 3/4
TYPE_BOOST = {  # attacker's ability: (type, multiplier)
    "transistor": ("electric", Fraction(13, 10)),
    "dragons-maw": ("dragon", Fraction(3, 2)),
    "steelworker": ("steel", Fraction(3, 2)),
    "steely-spirit": ("steel", Fraction(3, 2)),
    "rocky-payload": ("rock", Fraction(3, 2)),
    "water-bubble": ("water", Fraction(2)),
}
ATE = {"aerilate": "flying", "pixilate": "fairy", "refrigerate": "ice", "galvanize": "electric"}
SCRAPPY = ("scrappy", "minds-eye")
MOLD_BREAKER = ("mold-breaker", "teravolt", "turboblaze")
LEVEL3_ABILITIES = frozenset(
    {
        *IMMUNE,
        *RESIST,
        *FILTER,
        *TYPE_BOOST,
        *ATE,
        *SCRAPPY,
        *MOLD_BREAKER,
        "wonder-guard",
        "adaptability",
        "technician",
        "tinted-lens",
        "skill-link",
    }
)
# Level 4: abilities that need stats, categories or HP.
RUIN = {"sword-of-ruin": "def", "beads-of-ruin": "spd", "tablets-of-ruin": "atk",
        "vessel-of-ruin": "spa"}  # fmt: skip
LEVEL4_ABILITIES = frozenset(
    {
        "huge-power",
        "pure-power",
        "intimidate",
        "multiscale",
        "shadow-shield",
        "fur-coat",
        "ice-scales",
        *RUIN,
    }
)
DEFENSIVE = frozenset(
    {*IMMUNE, *RESIST, *FILTER, "wonder-guard", "multiscale", "fur-coat", "ice-scales"}
)  # what Mold Breaker ignores; Shadow Shield is not ignored
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


@dataclass(frozen=True)
class Move:
    name: str
    type: str
    category: str  # physical or special
    power: int
    accuracy: Fraction  # 1 for a move that never misses
    priority: int
    hits: Fraction
    drain: Fraction  # share of damage dealt healed; negative for recoil
    boosts: tuple[tuple[str, int], ...]  # stat changes to the user


@dataclass(frozen=True)
class Pokemon:
    identifier: str
    showdown: str
    typing: tuple[str, ...]
    base: tuple[int, ...]  # hp, atk, def, spa, spd, spe
    abilities: tuple[str, ...]
    moves: tuple[str, ...]  # damaging moves the rule keeps

    @property
    def stats(self) -> dict[str, int]:
        """Level 100, IVs 31, 84 EVs each, neutral nature."""
        out = {}
        for k, b in zip(STATS, self.base, strict=True):
            core = (2 * b + IV + EV // 4) * LEVEL // 100
            out[k] = 1 if (k == "hp" and b == 1) else core + (LEVEL + 10 if k == "hp" else 5)
        return out


@dataclass(frozen=True)
class Set:
    pokemon: Pokemon
    ability: str
    moves: tuple[Move, ...] = field(default=())

    @property
    def name(self) -> str:
        return self.pokemon.identifier


# Data.


@cache
def all_moves() -> dict[str, Move]:
    """Every kept move of ``teams_moves.csv``."""
    out = {}
    with MOVES_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kept"] != "1":
                continue
            out[r["move"]] = Move(
                name=r["move"],
                type=r["type"],
                category=r["category"],
                power=int(r["power"]),
                accuracy=Fraction(int(r["accuracy"]), 100) if r["accuracy"] else Fraction(1),
                priority=int(r["priority"]),
                hits=Fraction(r["hits"]),
                drain=Fraction(r["drain"]),
                boosts=tuple((k, int(v)) for k, v in (b.split(":") for b in r["boosts"].split())),
            )
    return out


@cache
def all_pokemon() -> tuple[Pokemon, ...]:
    """Every row of ``teams_pokemon.csv`` the study keeps, most used first."""
    moves = all_moves()
    out = []
    with POKEMON_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["excluded"]:
                continue
            form = usage.form(r["identifier"])
            typing = pokedex.typing(form, 9)
            assert typing is not None
            out.append(
                Pokemon(
                    identifier=r["identifier"],
                    showdown=r["showdown"],
                    typing=typing,
                    base=tuple(pokedex.base_stats(form, 9)[k] for k in pokedex.STATS),
                    abilities=tuple(r["abilities"].split()),
                    moves=tuple(m for m in r["moves"].split() if m in moves),
                )
            )
    return tuple(out)


def pool(n: int) -> tuple[Pokemon, ...]:
    """The n most used Pokémon."""
    return all_pokemon()[:n]


def choose_ability(p: Pokemon) -> str:
    """The first ability in slot order that the model represents, else the first."""
    modeled = LEVEL3_ABILITIES | LEVEL4_ABILITIES
    return next((a for a in p.abilities if a in modeled), p.abilities[0])


# Levels 1 to 3: damage rates without stats.


def effectiveness(move_type: str, attacker: Set, defender: Set, level: int) -> Fraction:
    """The type multiplier of a move against the defender, with level-3 abilities from 3 on."""
    if level < 3:
        return multiplier(move_type, defender.pokemon.typing)
    a, d = attacker.ability, defender.ability
    ignores = a in MOLD_BREAKER and d in DEFENSIVE
    m = Fraction(1)
    for t in defender.pokemon.typing:
        x = multiplier(move_type, (t,))
        if x == 0 and t == "ghost" and move_type in ("normal", "fighting") and a in SCRAPPY:
            x = Fraction(1)
        m *= x
    if not ignores:
        if IMMUNE.get(d) == move_type:
            return Fraction(0)
        if d == "wonder-guard" and m <= 1:
            return Fraction(0)
    return m


def move_type(m: Move, s: Set, level: int) -> str:
    """The move's type after an -ate ability (level 3 on)."""
    if level >= 3 and m.type == "normal" and s.ability in ATE:
        return ATE[s.ability]
    return m.type


def power_modifier(m: Move, s: Set, level: int) -> Fraction:
    """Technician, -ate and type-boosting abilities, as a factor on damage (level 3 on)."""
    if level < 3:
        return Fraction(1)
    a, f = s.ability, Fraction(1)
    if a == "technician" and m.power <= 60:
        f *= Fraction(3, 2)
    if a in ATE and m.type == "normal":
        f *= Fraction(6, 5)
    if a in TYPE_BOOST and TYPE_BOOST[a][0] == move_type(m, s, level):
        f *= TYPE_BOOST[a][1]
    return f


def hits(m: Move, s: Set, level: int) -> Fraction:
    """Expected hits; Skill Link makes a 2-to-5-hit move hit 5 times (level 3 on)."""
    if level >= 3 and s.ability == "skill-link" and m.hits == Fraction(31, 10):
        return Fraction(5)
    return m.hits


def stab(m: Move, s: Set, level: int) -> Fraction:
    """1.5 for a move of the user's own type, 2 with Adaptability (level 3 on), else 1."""
    if move_type(m, s, level) not in s.pokemon.typing:
        return Fraction(1)
    return Fraction(2) if level >= 3 and s.ability == "adaptability" else STAB


def defense_modifier(t: str, eff: Fraction, attacker: Set, defender: Set) -> Fraction:
    """Level-3 resistances and Filter, Tinted Lens on the attacker's side."""
    f = Fraction(1)
    if not (attacker.ability in MOLD_BREAKER and defender.ability in DEFENSIVE):
        f *= RESIST.get(defender.ability, {}).get(t, Fraction(1))
        if defender.ability in FILTER and eff > 1:
            f *= Fraction(3, 4)
    if attacker.ability == "tinted-lens" and 0 < eff < 1:
        f *= 2
    return f


def rate(m: Move, attacker: Set, defender: Set, level: int) -> Fraction:
    """Damage per turn of one move at levels 2 and 3, in units of power."""
    t = move_type(m, attacker, level)
    eff = effectiveness(t, attacker, defender, level)
    f = defense_modifier(t, eff, attacker, defender) if level >= 3 else Fraction(1)
    return (
        m.power
        * hits(m, attacker, level)
        * m.accuracy
        * stab(m, attacker, level)
        * power_modifier(m, attacker, level)
        * eff
        * f
    )


def best_rate(attacker: Set, defender: Set, level: int) -> Fraction:
    """The attacker's damage per turn at a level: its best own type (1) or best move (2, 3)."""
    if level == 1:
        return max(multiplier(t, defender.pokemon.typing) for t in attacker.pokemon.typing)
    return max((rate(m, attacker, defender, level) for m in attacker.moves), default=Fraction(0))


# Level 4: the damage formula.


def modify(value: int, num: int, den: int = 1) -> int:
    """Showdown's ``modify``: value × modifier, the modifier rounded to a multiple of 1/4096,
    the product rounded half down."""
    mod = (num * 4096 * 2 + den) // (2 * den)  # round(num/den × 4096)
    return (value * mod + 2047) // 4096


def stage(stat: int, s: int) -> int:
    """A stat at a stage from -6 to 6."""
    return stat * (2 + s) // 2 if s >= 0 else stat * 2 // (2 - s)


@dataclass(frozen=True)
class Fighter:
    """A set in a level-4 fight: its stats after abilities that act on entry."""

    set: Set
    stats: dict[str, int]
    start: dict[str, int]  # stages at the start (Intimidate)


def fighters(a: Set, b: Set) -> tuple[Fighter, Fighter]:
    """Both sides' stats, with the Ruin abilities and Intimidate applied."""

    def stats(me: Set, foe: Set) -> dict[str, int]:
        st = dict(me.pokemon.stats)
        if foe.ability in RUIN and me.ability != foe.ability:
            k = RUIN[foe.ability]
            st[k] = modify(st[k], 3, 4)
        if me.ability in ("huge-power", "pure-power"):
            st["atk"] *= 2
        return st

    def start(foe: Set) -> dict[str, int]:
        return {"atk": -1} if foe.ability == "intimidate" else {}

    return Fighter(a, stats(a, b), start(b)), Fighter(b, stats(b, a), start(a))


def damage_rolls(
    m: Move,
    attacker: Fighter,
    defender: Fighter,
    a_stages: dict[str, int],
    d_stages: dict[str, int],
    d_full_hp: bool,
) -> list[int]:
    """Damage of one hit for each of the 16 random rolls (85% to 100%), no critical hit."""
    a, d = attacker.set, defender.set
    t = move_type(m, a, 4)
    eff = effectiveness(t, a, d, 4)
    if eff == 0:
        return [0] * 16
    ignores = a.ability in MOLD_BREAKER and d.ability in DEFENSIVE
    atk_key, def_key = ("atk", "def") if m.category == "physical" else ("spa", "spd")
    power = m.power
    if a.ability == "technician" and power <= 60:
        power = modify(power, 3, 2)
    if a.ability in ATE and m.type == "normal":
        power = modify(power, 6, 5)
    if not ignores and d.ability == "dry-skin" and t == "fire":
        power = modify(power, 5, 4)
    atk = stage(attacker.stats[atk_key], a_stages.get(atk_key, 0))
    if a.ability in TYPE_BOOST and TYPE_BOOST[a.ability][0] == t:
        x = TYPE_BOOST[a.ability][1]
        atk = modify(atk, x.numerator, x.denominator)
    if not ignores:
        r = RESIST.get(d.ability, {}).get(t)
        if r is not None and d.ability != "dry-skin":
            atk = modify(atk, r.numerator, r.denominator)
    dfn = stage(defender.stats[def_key], d_stages.get(def_key, 0))
    if not ignores and d.ability == "fur-coat" and m.category == "physical":
        dfn *= 2
    base = (2 * LEVEL // 5 + 2) * power * atk // dfn // 50 + 2
    final = Fraction(1)
    if not ignores:
        if d.ability in FILTER and eff > 1:
            final *= Fraction(3, 4)
        if d.ability == "ice-scales" and m.category == "special":
            final *= Fraction(1, 2)
        if d.ability == "multiscale" and d_full_hp:
            final *= Fraction(1, 2)
    if d.ability == "shadow-shield" and d_full_hp:
        final *= Fraction(1, 2)
    if a.ability == "tinted-lens" and eff < 1:
        final *= 2
    out = []
    for r in range(85, 101):
        x = base * r // 100
        st = stab(m, a, 4)
        if st != 1:
            x = modify(x, st.numerator, st.denominator)
        e = eff
        while e >= 2:
            x, e = x * 2, e / 2
        while e < 1:
            x, e = x // 2, e * 2
        if final != 1:
            x = modify(x, final.numerator, final.denominator)
        out.append(max(x, 1))
    return out


def expected_damage(
    m: Move,
    attacker: Fighter,
    defender: Fighter,
    a_stages: dict[str, int] | None = None,
    d_stages: dict[str, int] | None = None,
    d_full_hp: bool = True,
) -> Fraction:
    """Mean damage of one use: over the 16 rolls, times accuracy and expected hits."""
    rolls = damage_rolls(m, attacker, defender, a_stages or {}, d_stages or {}, d_full_hp)
    return Fraction(sum(rolls), 16) * m.accuracy * hits(m, attacker.set, 4)


@dataclass
class Side:
    fighter: Fighter
    hp: Fraction
    stages: dict[str, int]

    @property
    def max_hp(self) -> int:
        return self.fighter.stats["hp"]


def choose(me: Side, foe: Side) -> tuple[Move, Fraction] | None:
    """The move with the most expected damage now (the first such in set order)."""
    best = None
    for m in me.fighter.set.moves:
        d = expected_damage(m, me.fighter, foe.fighter, me.stages, foe.stages, foe.hp == foe.max_hp)
        if best is None or d > best[1]:
            best = (m, d)
    return best


def act(me: Side, foe: Side, move: Move, dmg: Fraction) -> None:
    """One use of a move: damage, then recoil or drain, then stat changes to the user."""
    dealt = min(dmg, foe.hp)
    foe.hp -= dealt
    if move.drain < 0:
        me.hp -= dealt * -move.drain
    elif move.drain > 0:
        me.hp = min(Fraction(me.max_hp), me.hp + dealt * move.drain)
    for k, v in move.boosts:
        me.stages[k] = max(-6, min(6, me.stages.get(k, 0) + v))


def outcome(a: Side, b: Side) -> Fraction | None:
    """The signed share of HP the winner keeps, or None while both stand."""
    if a.hp <= 0 and b.hp <= 0:
        return Fraction(0)
    if b.hp <= 0:
        return a.hp / a.max_hp
    if a.hp <= 0:
        return -(b.hp / b.max_hp)
    return None


def copy(s: Side) -> Side:
    return Side(s.fighter, s.hp, dict(s.stages))


def race(a: Side, b: Side, turn: int = 0) -> tuple[Fraction, Fraction]:
    """(M, expected turns) of a level-4 fight from this state, speed ties split evenly."""
    if turn >= TURN_LIMIT:
        return Fraction(0), Fraction(turn)
    ca, cb = choose(a, b), choose(b, a)
    if ca is None or cb is None or (ca[1] == 0 and cb[1] == 0):
        return Fraction(0), Fraction(turn)

    def key(s: Side, c: tuple[Move, Fraction]) -> tuple[int, int]:
        return c[0].priority, stage(s.fighter.stats["spe"], s.stages.get("spe", 0))

    ka, kb = key(a, ca), key(b, cb)
    orders = [(True,)] if ka > kb else [(False,)] if kb > ka else [(True,), (False,)]
    total_m, total_t = Fraction(0), Fraction(0)
    for (a_first,) in orders:
        x, y = copy(a), copy(b)
        sequence = [(x, y, ca), (y, x, cb)] if a_first else [(y, x, cb), (x, y, ca)]
        result = None
        for me, foe, (mv, _) in sequence:
            # Both chose at the start of the turn; damage is computed when the move is used.
            dmg = expected_damage(mv, me.fighter, foe.fighter, me.stages, foe.stages,
                                  foe.hp == foe.max_hp)  # fmt: skip
            act(me, foe, mv, dmg)
            result = outcome(x, y)
            if result is not None:
                break
        if result is not None:
            m, t = result, Fraction(turn + 1)
        else:
            m, t = race(x, y, turn + 1)
        total_m += m
        total_t += t
    n = len(orders)
    return total_m / n, total_t / n


def level4(a: Set, b: Set) -> tuple[Fraction, Fraction]:
    """(M, expected turns) of the level-4 fight between two sets."""
    fa, fb = fighters(a, b)
    sa = Side(fa, Fraction(fa.stats["hp"]), dict(fa.start))
    sb = Side(fb, Fraction(fb.stats["hp"]), dict(fb.start))
    return race(sa, sb)


# Sets.


def hit_share(m: Move, a: Set, d: Set) -> Fraction:
    """Expected damage of one use at level 4, at full HP and neutral stages, over d's HP."""
    fa, fd = fighters(a, d)
    return expected_damage(m, fa, fd) / fd.stats["hp"]


def choose_moves(p: Pokemon, opponents: Sequence[Pokemon]) -> tuple[Move, ...]:
    """The four moves of the rule in the module's docstring, in alphabetical order."""
    moves = [all_moves()[m] for m in p.moves]
    me = Set(p, choose_ability(p))
    foes = [Set(q, choose_ability(q)) for q in opponents if q != p]
    share = {m: [hit_share(m, me, f) for f in foes] for m in moves}
    # Keep one of any moves with the same shares, then drop moves another beats everywhere.
    distinct = {}
    for m in sorted(moves, key=lambda m: m.name):
        distinct.setdefault(tuple(share[m]), m)
    cands = list(distinct.values())
    cands = [
        m
        for m in cands
        if not any(
            o is not m and all(x >= y for x, y in zip(share[o], share[m], strict=True))
            for o in cands
        )
    ]
    if len(cands) <= 4:
        return tuple(sorted(cands, key=lambda m: m.name))

    def score(combo: tuple[Move, ...]) -> tuple[Fraction, Fraction]:
        best = [max(share[m][i] for m in combo) for i in range(len(foes))]
        return sum((min(Fraction(1), x) for x in best), Fraction(0)), sum(best, Fraction(0))

    # Combinations come in alphabetical order, and max keeps the first of equal scores.
    best = max(itertools.combinations(sorted(cands, key=lambda m: m.name), 4), key=score)
    return tuple(best)


@cache
def sets(n: int) -> tuple[Set, ...]:
    """The fixed sets of the n most used Pokémon, chosen against each other."""
    ps = pool(n)
    return tuple(Set(p, choose_ability(p), choose_moves(p, ps)) for p in ps)


# Matchups.


def matchup(a: Set, b: Set, level: int) -> Fraction:
    """M(a, b) at a level."""
    if a.pokemon == b.pokemon:
        return Fraction(0)
    if level == 4:
        return level4(a, b)[0]
    return duel_score(best_rate(a, b, level), best_rate(b, a, level))


def matrix(ss: Sequence[Set], level: int) -> list[list[Fraction]]:
    """M over a list of sets, computed once per unordered pair."""
    n = len(ss)
    g = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g[i][j] = matchup(ss[i], ss[j], level)
            g[j][i] = -g[i][j]
    return g
