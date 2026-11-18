# Normal effectiveness are not written.
gen6 = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "dragon": 0.5,
        "steel": 2.0,
    },
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {
        "water": 2.0,
        "electric": 0.5,
        "grass": 0.5,
        "ground": 0.0,
        "flying": 2.0,
        "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5,
        "water": 2.0,
        "grass": 0.5,
        "poison": 0.5,
        "ground": 2.0,
        "flying": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "dragon": 0.5,
        "steel": 0.5,
    },
    "ice": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 0.5,
        "ground": 2.0,
        "flying": 2.0,
        "dragon": 2.0,
        "steel": 0.5,
    },
    "fighting": {
        "normal": 2.0,
        "ice": 2.0,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "ghost": 0.0,
        "dark": 2.0,
        "steel": 2.0,
        "fairy": 0.5,
    },
    "poison": {
        "grass": 2.0,
        "poison": 0.5,
        "ground": 0.5,
        "rock": 0.5,
        "ghost": 0.5,
        "steel": 0.0,
        "fairy": 2.0,
    },
    "ground": {
        "fire": 2.0,
        "electric": 2.0,
        "grass": 0.5,
        "poison": 2.0,
        "flying": 0.0,
        "bug": 0.5,
        "rock": 2.0,
        "steel": 2.0,
    },
    "flying": {
        "electric": 0.5,
        "grass": 2.0,
        "fighting": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "steel": 0.5,
    },
    "psychic": {
        "fighting": 2.0,
        "poison": 2.0,
        "psychic": 0.5,
        "dark": 0.0,
        "steel": 0.5,
    },
    "bug": {
        "fire": 0.5,
        "grass": 2.0,
        "fighting": 0.5,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 2.0,
        "ghost": 0.5,
        "dark": 2.0,
        "steel": 0.5,
        "fairy": 0.5,
    },
    "rock": {
        "fire": 2.0,
        "ice": 2.0,
        "fighting": 0.5,
        "ground": 0.5,
        "flying": 2.0,
        "bug": 2.0,
        "steel": 0.5,
    },
    "ghost": {
        "normal": 0.0,
        "psychic": 2.0,
        "ghost": 2.0,
        "dark": 0.5,
    },
    "dragon": {
        "dragon": 2.0,
        "steel": 0.5,
        "fairy": 0.0,
    },
    "dark": {
        "fighting": 0.5,
        "psychic": 2.0,
        "ghost": 2.0,
        "dark": 0.5,
        "fairy": 0.5,
    },
    "steel": {
        "fire": 0.5,
        "water": 0.5,
        "electric": 0.5,
        "ice": 2.0,
        "rock": 2.0,
        "steel": 0.5,
        "fairy": 2.0,
    },
    "fairy": {
        "fire": 0.5,
        "fighting": 2.0,
        "poison": 0.5,
        "dragon": 2.0,
        "dark": 2.0,
        "steel": 0.5,
    },
}

# Earlier charts. Steel and Dark arrived in Generation 2 and Fairy in Generation 6. Six
# multipliers differ from the Generation 6 chart: (attacker, defender, multiplier, the last
# generation in which it applies). Source: PokeAPI's type_efficacy_past.csv (see the README).
INTRODUCED = {"steel": 2, "dark": 2, "fairy": 6}
PAST = (
    ("bug", "poison", 2.0, 1),
    ("poison", "bug", 2.0, 1),
    ("ghost", "psychic", 0.0, 1),
    ("ice", "fire", 1.0, 1),
    ("ghost", "steel", 0.5, 5),
    ("dark", "steel", 0.5, 5),
)


def chart(generation: int) -> dict[str, dict[str, float]]:
    """The chart in force in a generation (1 to 9), over the types that exist then.

    Like ``gen6``, a pair at 1.0 is not written.
    """
    if not 1 <= generation <= 9:
        raise ValueError(f"no generation {generation}")
    types = [t for t in gen6 if INTRODUCED.get(t, 1) <= generation]
    out = {a: {d: m for d, m in gen6[a].items() if d in types} for a in types}
    for attacker, defender, multiplier, last in PAST:
        if generation <= last and attacker in out and defender in out:
            out[attacker][defender] = multiplier
            if multiplier == 1.0:
                del out[attacker][defender]
    return out
