# The chart as a game, generation by generation

Detail for the README's section [The chart as a game](../README.md#the-chart-as-a-game). Every number here is exact.

## The games on earlier charts

Generation 1 had 15 types, and four of its multipliers differ: Bug and Poison hit each other for double damage, Ghost did nothing to Psychic, and Ice did normal damage to Fire. Generations 2 to 5 added Steel and Dark, and in them Steel resisted Ghost and Dark. Generation 6 added Fairy and dropped those two resistances. All exact; `uv run --group pokemon python -m pokemon.equilibrium` prints them.

| | Generation 1 (15 types) | Generations 2–5 (17 types) | Generation 6 on (18 types) |
|---|---|---|---|
| Attack game, value | 1 | 52363/49845 ≈ 1.050 | 4158/3979 ≈ 1.045 |
| Defender's mix leads with | Ghost 1/2, Normal 1/4, Psychic 1/4 | Dark 26.8%, Ghost 24.0% | Normal 28.4%, Fairy 14.9% |
| Duel, types played | 9, led by Dragon 26.6%, Psychic 19.7%, Rock 18.4% | 9, led by Dragon 22.9%, Steel 17.7% | 11, led by Steel 23.6%, Dragon 17.5% |
| Duel over all typings | 16 typings, not unique | 17, unique, led by Electric/Flying 16.0% | 23, unique, led by Electric/Flying 12.9% |
| Log-ratio, k = 1, 2, 3, 4, 6, 10 | unique for each k | unique for each k | unique at k = 1, a segment for k ≥ 2 |

- **Generation 1.** Nothing resisted Dragon, so Dragon alone guarantees the attacker an average multiplier of 1, and the value is exactly 1. The attacker's optimal strategy is not unique: Fire's weight can be anything from 0 to 32/97. In the duel, Psychic is second (94/477), since only Bug hit it for double damage and Ghost could not hit it at all. Normal, Fire, Electric, Grass, Poison and Ghost are never played. Over all 120 typings, Water/Dragon never loses a duel. So playing it alone is optimal, though no Pokémon had that typing until Generation 2.
- **Steel and Dark arrive (Generation 2).** Steel goes straight to second place in the duel (48/271), and Psychic, Rock and Bug drop out of it. Dragon stays first. The defender in the attack game now leans on Dark and Ghost, which together take just over half the weight. Every equilibrium of the duel and the log-ratio game is unique.
- **Fairy arrives, and Steel loses two resistances (Generation 6).** Steel moves to first place in the duel and Dragon to second, and Fairy enters at 3.8%. Normal becomes the defender's main type in the attack game. The log-ratio game's optimal set becomes a segment for k ≥ 2. The two changes pull in opposite directions. Applied alone to the Generations 2–5 chart, the loss of Steel's resistances puts Dark first in the duel (16.9%), while Fairy alone leaves Steel first (22.3%). `generation_6_split()` builds both charts.

## The typings of each generation

The Pokémon data covers every generation through the newest games (Generation 9, including Legends: Z-A and its Mega Evolutions). A form counts as of generation g if it was introduced in g or earlier. Its typing is the one it had in g's games: Clefairy is Normal through Generation 5 and Fairy from 6. Each generation uses its own chart. Exact.

| Generation | Forms | Typings | Duel over existing typings |
|---|---|---|---|
| 1 | 151 | 33 | unique, 13 typings, led by Ice/Psychic 19.7% |
| 2 | 251 | 55 | Water/Dragon alone is optimal; not unique |
| 3 | 392 | 75 | not unique; one equilibrium leads with Water/Ground |
| 4 | 508 | 88 | unique, 15, led by Water/Ground 19.2% |
| 5 | 673 | 110 | unique, 19, led by Water/Ground 16.0% |
| 6 | 811 | 133 | unique, 21, led by Water/Ground 10.1% |
| 7 | 955 | 146 | unique, 27, led by Water/Ground 10.8% |
| 8 | 1,106 | 155 | unique, 29, led by Water/Ground 11.6% |
| 9 | 1,303 | 162 | unique, 23, led by Electric/Flying 12.7% |

- In Generations 2 and 3, Water/Dragon (Kingdra's typing) loses no duel against any typing that exists: its worst result is a tie. Dragon/Steel arrives in Generation 4 and beats it.
- From Generation 7 to 9, 16 typings were added, among them Fire/Grass, Electric/Fighting and Fighting/Fairy, and all three enter the duel's equilibrium. Seven typings leave it, among them Flying/Dragon and Normal/Fairy. The Generation 9 equilibrium has 23 typings, as the 171-typing game's does, and 21 of them are the same. The 171-typing equilibrium also uses Fire/Fairy, which no Pokémon has, and Fighting/Ghost; the Generation 9 one uses Water/Dragon and Fighting/Fairy in their place.
- Nine of the 171 typings are still unused: Bug/Dragon, Fire/Fairy, Ground/Fairy, Ice/Poison, Normal/Bug, Normal/Ice, Normal/Rock, Normal/Steel and Rock/Ghost.

