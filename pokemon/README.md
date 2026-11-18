# Pokémon

Two separate analyses share this folder: the type chart, and catch odds for the Emerald Kaizo ROM hack.

## The type chart

`pokemon_types.py` holds the Generation 6 chart: 18 types, and the multiplier when each attacks each.

- No set of six attacking types is super-effective against every type, and ten sets of seven are. Fighting and Ground are in all ten. Exact, by checking every combination.
- Among the type combinations that Pokémon in `pokemon_gen7.csv` have, Fairy/Steel takes the least damage summed over the 18 attacking types: 13.25, against 18 for a type with no weaknesses or resistances.

## The chart as a game

`equilibrium.py` treats the chart as a two-player zero-sum game and solves it with linear programming (scipy's HiGHS). The floating-point solution only finds the support. The weights are then solved exactly with fractions on that support, and every equilibrium inequality is checked exactly. Where the equilibrium is unique, an exact argument shows it: no unused strategy ties, and the equations on the support have one solution. A second LP check, the least and greatest weight of each type over all optimal strategies, agrees. Tests are in `tests/test_pokemon_equilibrium.py`.

**Attack game.** One player picks an attacking type, the other a defending type, and the payoff is the multiplier. The attacker can guarantee an average multiplier of 4158/3979 (about 1.045), and the equilibrium is unique. The attacker mixes 11 types, led by Ground (19.5%). The defender mixes 11, led by Normal (28.4%), Fairy (14.9%) and Ghost (14.6%). Normal suits the defender because only Fighting is super-effective against it and Ghost cannot touch it. If the defender may pick any of the 171 typings (18 single types and 153 pairs), the value drops to 13207/13501 (about 0.978). Flying/Steel then leads the defender's mix at 19.9%.

**Duel game.** This is the model of Schmitz, ["The Objectively Best Pokemon Type, According to Game Theory"](https://quantimschmitzcom.wordpress.com/2021/07/30/the-objectively-best-pokemon-type-according-to-game-theory/) (2021). Two Pokémon that differ only in type each attack with a move of their own type. If one deals multiplier a and takes b, with a > b, it wins and keeps 1 − b/a of its HP. The winner scores that share and the loser scores its negative; equal multipliers score 0. The game is symmetric, so its value is 0. The equilibrium is unique and matches every weight Schmitz published, to his three decimal places: Steel 627/2653 (0.236), Dragon 0.175, Water 0.161, Ground 0.127, Electric 0.083, Grass 0.070, Flying 0.068, Ghost 0.040, Fairy 0.038, Dark 0.002, Fighting 0.001. The same seven types get no weight: Normal, Fire, Poison, Psychic, Bug, Rock and Ice. Each of the seven loses on average against the equilibrium mix, so none ties.

Schmitz's follow-up, ["The best Pokemon type combinations according to game theory"](https://quantimschmitzcom.wordpress.com/2023/08/12/the-best-pokemon-type-combinations-according-to-game-theory/) (2023), plays the same duel between typings, each using the better of its two move types. He reports 27 typings with positive weight, averaged over 30 equilibria, led by Water/Flying and Poison/Dark. With the model as his text describes it, the 171-typing game has a unique equilibrium on 23 typings. It is led by Electric/Flying (12.9%), Grass/Rock (11.8%) and Water/Ground (11.5%), and every typing in it has two types. Water/Flying and Poison/Dark both lose against it, by 0.062 and 0.020. Five of his top ten are in it. Because this equilibrium is unique, averaging over several would not change it, so his game or his solver differs from this one somewhere. Where is not known: his table of matchup scores is an image that was not read here, and his published repository holds no Pokémon matrix. Restricted to the 146 typings that some Pokémon in `pokemon_gen7.csv` has, the equilibrium is again unique, on 27 typings led by Water/Ground. Water/Flying is not among them.

**Log-ratio game.** The payoff is log2(a/b) for single types, which makes the payoffs integers. An immunity needs a finite value for log2(0); call it −k. With k = 1, which counts an immunity as a resistance, the equilibrium is unique. The seven types left out are Normal, Ice, Poison, Flying, Psychic, Bug and Rock, so Flying replaces Fire in Schmitz's list. With k = 2, 3, 4, 6 or 10 the optimal strategies form a polytope rather than a point, and Ghost's weight ranges from 0 up to about 6% at k = 2. Six to eight types are never played, depending on k. Which types get zero weight depends on the scoring, not only on the chart.

**Rock-paper-scissors triangles.** A triangle is three types where each is super-effective against the next: a → b → c → a. There are 16, counted once each, by checking every ordered triple and again with networkx's cycle search. In five of them each type also resists the type it hits:

- Fire → Grass → Water
- Fire → Steel → Rock
- Grass → Ground → Poison
- Fighting → Rock → Flying
- Fighting → Dark → Psychic

No two different types are super-effective against each other; only Dragon and Ghost are super-effective against themselves.

## Other analyses

`graph.py` writes the chart as a directed graph, with 18 types and the 120 links that are not neutral, to `display/graph.json`, and serves `display/index.html`, which draws it. `graph_analysis.py` ranks the types by PageRank and betweenness with networkx. `pokemon.ipynb` scores types, type pairs, Pokémon and random teams.

## Emerald Kaizo

`emerald_kaizo/` models IVs, stats and catch odds, to decide how many wild Pokémon of a species are worth catching for a given role. For example, a nature chosen in advance turns up 4% of the time, so 74 catches give a 95% chance of at least one. `iv_analysis.md` is the generated report, `lab_notes.md` the working notes, and `rom_fingerprint.md` identifies the ROM that was analyzed, which is not included.

## Data

- `Pokemon.csv`: 800 Pokémon and forms through Generation 6 (number 721), with types and base stats. It matches `Pokemon.csv` in the Kaggle dataset [Pokemon with stats](https://www.kaggle.com/datasets/abcsds/pokemon) (version 2), which is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Verified: the license shown on that page, and the file's header, first rows and size (44,028 bytes against the 44.03 kB listed). Not verified: a byte-for-byte comparison, since Kaggle needs a login to download. That page says the data was gathered from pokemon.com, pokemondb.net and Bulbapedia.
- `pokemon_gen7.csv`: 1,061 Pokémon and forms up to National Pokédex number 802, with types, abilities and base stats. It is `pokemon.csv` from the Kaggle dataset [Pokemon Sun and Moon (Gen 7) Stats](https://www.kaggle.com/datasets/mylesoneill/pokemon-sun-and-moon-gen-7-stats) (version 2). Verified: the column count (26), the first ten column names, the first rows, the id range (1 to 1062) and the size (249,988 bytes against the 249.99 kB listed). The file is byte-identical to copies committed in March 2017 to [bdemeshev/pr201](https://github.com/bdemeshev/pr201/blob/master/2016/datasets/pokemon.csv) and in September 2017 to [prasertcbs/basic-dataset](https://github.com/prasertcbs/basic-dataset/blob/master/PokemonSunMoonGen7/pokemon.csv), whose commit names the Kaggle dataset. The current copy in [ScottPolhemus/dex](https://github.com/ScottPolhemus/dex/blob/master/data/sm-pokemon.csv), a file first committed in December 2016, differs only in its line endings. The Kaggle page lists its license as "Other (specified in description)", but the description names no license, so its terms are unknown. Inferred, not verified: the Kaggle dataset is the origin of the copies, as its description and the one commit message suggest.

[PokeAPI](https://github.com/PokeAPI/pokeapi) publishes types, stats and species flags as CSV files under a BSD-style license, and the notebook already takes its list of legendary species from there. It has the four columns of `pokemon_gen7.csv` that the notebook reads (species, both types, and the stat total, as a sum), so it could replace that file. Both data files are left as they are, so the notebook's outputs do not change.

Type icon designs: NovaWarp's [Poketypes + Faketypes](https://www.deviantart.com/novawarp/art/Poketypes-%2B-Faketypes-2.1-286933205) and DrewLinne's [Pokemon Types](https://www.deviantart.com/drewlinne/art/Pokemon-Types-374262033).

```bash
uv run python -m pokemon.graph                       # writes display/graph.json, serves the page on port 8000
uv run --group pokemon python -m pokemon.graph_analysis
uv run --group pokemon python -m pokemon.equilibrium   # the games above, in a few seconds
uv run python -m pokemon.emerald_kaizo.iv_analysis   # regenerates emerald_kaizo/iv_analysis.md
uv run --group pokemon --with jupyter jupyter lab pokemon/pokemon.ipynb
```
