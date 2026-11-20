# The chart as a game

The attack game, the duel and its comparison with Schmitz, the log-ratio game, rock-paper-scissors triangles, and earlier charts. Moved unchanged from the [Pokémon README](../README.md), which keeps the short answers.


`equilibrium.py` treats the chart as a two-player zero-sum game and solves it with linear programming (scipy's HiGHS). The floating-point solution only finds the support. The weights are then solved exactly with fractions on that support, and every equilibrium inequality is checked exactly. Where the equilibrium is unique, an exact argument shows it: no unused strategy ties, and the equations on the support have one solution. Where that argument does not apply, the least and greatest weight of a strategy over all optimal strategies are found by LP and then proved exactly: the primal and dual solutions are rebuilt with fractions, checked, and their values must meet. Tests are in `tests/test_pokemon_equilibrium.py`.

**Attack game.** One player picks an attacking type, the other a defending type, and the payoff is the multiplier. The attacker can guarantee an average multiplier of 4158/3979 (about 1.045), and the equilibrium is unique. The attacker mixes 11 types, led by Ground (19.5%). The defender mixes 11, led by Normal (28.4%), Fairy (14.9%) and Ghost (14.6%). Normal suits the defender because only Fighting is super-effective against it and Ghost cannot touch it. If the defender may pick any of the 171 typings, the value drops to 13207/13501 (about 0.978). Flying/Steel then leads the defender's mix at 19.9%.

**Duel game.** This is the model of Schmitz, ["The Objectively Best Pokemon Type, According to Game Theory"](https://quantimschmitzcom.wordpress.com/2021/07/30/the-objectively-best-pokemon-type-according-to-game-theory/) (2021). Two Pokémon that differ only in type each attack with a move of their own type. If one deals multiplier a and takes b, with a > b, it wins and keeps 1 − b/a of its HP. The winner scores that share and the loser scores its negative; equal multipliers score 0. The game is symmetric, so its value is 0. The equilibrium is unique and matches every weight Schmitz published, to his three decimal places: Steel 627/2653 (0.236), Dragon 0.175, Water 0.161, Ground 0.127, Electric 0.083, Grass 0.070, Flying 0.068, Ghost 0.040, Fairy 0.038, Dark 0.002, Fighting 0.001. The same seven types get no weight: Normal, Fire, Poison, Psychic, Bug, Rock and Ice. Each of the seven loses on average against the equilibrium mix, so none ties.

Schmitz's follow-up, ["The best Pokemon type combinations according to game theory"](https://quantimschmitzcom.wordpress.com/2023/08/12/the-best-pokemon-type-combinations-according-to-game-theory/) (2023), plays the same duel between typings, each using the better of its two move types. He reports 27 typings with positive weight, averaged over 30 equilibria, led by Water/Flying and Poison/Dark. With the model as his text describes it, the 171-typing game has a unique equilibrium on 23 typings. It is led by Electric/Flying (12.9%), Grass/Rock (11.8%) and Water/Ground (11.5%), and every typing in it has two types. Water/Flying and Poison/Dark both lose against it, by 0.062 and 0.020. Five of his top ten are in it. Because this equilibrium is unique, averaging over several would not change it, so his game or his solver differs from this one somewhere. Where is not known: his table of matchup scores is an image that was not read here, and his published repository holds no Pokémon matrix. Restricted to the 146 typings that some Pokémon or form had as of Generation 7, the equilibrium is again unique, on 27 typings led by Water/Ground. Water/Flying is not among them.

**Log-ratio game.** The payoff is log2(a/b) for single types, which makes the payoffs integers. An immunity needs a finite value for log2(0); call it −k. With k = 1, which counts an immunity as a resistance, the equilibrium is unique. The seven types left out are Normal, Ice, Poison, Flying, Psychic, Bug and Rock, so Flying replaces Fire in Schmitz's list. With k = 2, 3, 4, 6 or 10 the optimal strategies form a line segment. At one end is a mix without Ghost; at the other end, Ghost takes weight t and the rest of the mix is scaled by 1 − t. Ghost ties that mix, so adding it costs nothing, up to t = 5/79 (about 6.3%) at k = 2, 154/1915 at k = 3, 142/2431 at k = 4, 118/3583 at k = 6 and 9/557 at k = 10. The types never played are eight at k = 2 and k = 10 and six at k = 3, 4 and 6. Which types get zero weight depends on the scoring, not only on the chart. That the optimal set is a segment is proved exactly: the two ends are optimal, and the equations every optimal strategy must satisfy leave one free direction.

**Rock-paper-scissors triangles.** A triangle is three types where each is super-effective against the next: a → b → c → a. There are 16, counted once each, by checking every ordered triple and again with networkx's cycle search. In five of them each type also resists the type it hits:

- Fire → Grass → Water
- Fire → Steel → Rock
- Grass → Ground → Poison
- Fighting → Rock → Flying
- Fighting → Dark → Psychic

No two different types are super-effective against each other; only Dragon and Ghost are super-effective against themselves.

### Earlier charts and each generation's typings

Generation 1 had 15 types, and four of its multipliers differ: Bug and Poison hit each other for double damage, Ghost did nothing to Psychic, and Ice did normal damage to Fire. Generations 2 to 5 added Steel and Dark, and in them Steel resisted Ghost and Dark. Generation 6 added Fairy and dropped those two resistances. `uv run --group pokemon python -m pokemon.equilibrium` prints the same games on each chart; all are exact.

| | Generation 1 (15 types) | Generations 2–5 (17 types) | Generation 6 on (18 types) |
|---|---|---|---|
| Attack game, value | 1 | 52363/49845 ≈ 1.050 | 4158/3979 ≈ 1.045 |
| Duel, types played | 9, led by Dragon 26.6% | 9, led by Dragon 22.9% | 11, led by Steel 23.6% |
| Duel over all typings | 16 typings, not unique | 17, unique, led by Electric/Flying 16.0% | 23, unique, led by Electric/Flying 12.9% |

- In Generation 1 nothing resisted Dragon, so the attack game's value is exactly 1, and Water/Dragon never loses a duel against any of the 120 typings.
- Steel arrives second in the duel in Generation 2 and moves to first in Generation 6, when Fairy arrives and Steel loses two resistances. The two changes pull in opposite directions.
- With each generation's own Pokémon and chart, the duel over existing typings is unique from Generation 4 on, led by Water/Ground through Generation 8 and by Electric/Flying in Generation 9. In Generations 2 and 3, Water/Dragon (Kingdra's typing) loses no duel.
- Nine of the 171 typings are still unused, among them Fire/Fairy, which the 171-typing equilibrium plays.

[`results/generations.md`](generations.md) has the full tables, the defender's mixes and the per-generation detail.
