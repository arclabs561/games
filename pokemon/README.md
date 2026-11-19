# Pokémon

Two separate analyses share this folder: the type chart, and catch odds for the Emerald Kaizo ROM hack.

## The type chart

`pokemon_types.py` holds the Generation 6 chart: 18 types, and the multiplier when each attacks each. `chart(g)` gives the chart of any generation. Generation 1 had 15 types, and four of its multipliers differ: Bug and Poison hit each other for double damage, Ghost did nothing to Psychic, and Ice did normal damage to Fire. Generations 2 to 5 added Steel and Dark, and in them Steel resisted Ghost and Dark. Generation 6 added Fairy and dropped those two resistances. The charts agree pair by pair with PokeAPI's `type_efficacy.csv` and `type_efficacy_past.csv` and with Pokémon Showdown's per-generation chart files, for every generation from 1 to 9 (`sources.py`, below).

- No set of six attacking types is super-effective against every single type, and ten sets of seven are. Fighting and Ground are in all ten. Exact, by checking every combination.
- Among the typings that some Pokémon or form has, Steel/Fairy takes the least damage summed over the 18 attacking types: 53/4 = 13.25, against 18 for a type with no weaknesses or resistances. This holds from Generation 6, when Fairy arrived, through Generation 9. In Generations 2 to 5 the least is Flying/Steel's 12, out of 17. In Generation 1 it is 14 out of 15, shared by Psychic and Electric/Flying. Exact, `tests/test_pokemon_equilibrium.py`.

## The chart as a game

`equilibrium.py` treats the chart as a two-player zero-sum game and solves it with linear programming (scipy's HiGHS). The floating-point solution only finds the support. The weights are then solved exactly with fractions on that support, and every equilibrium inequality is checked exactly. Where the equilibrium is unique, an exact argument shows it: no unused strategy ties, and the equations on the support have one solution. Where that argument does not apply, the least and greatest weight of a strategy over all optimal strategies are found by LP and then proved exactly: the primal and dual solutions are rebuilt with fractions, checked, and their values must meet. So every claim below about uniqueness, about types never played, and about weight ranges is exact. Tests are in `tests/test_pokemon_equilibrium.py`.

**Attack game.** One player picks an attacking type, the other a defending type, and the payoff is the multiplier. The attacker can guarantee an average multiplier of 4158/3979 (about 1.045), and the equilibrium is unique. The attacker mixes 11 types, led by Ground (19.5%). The defender mixes 11, led by Normal (28.4%), Fairy (14.9%) and Ghost (14.6%). Normal suits the defender because only Fighting is super-effective against it and Ghost cannot touch it. If the defender may pick any of the 171 typings (18 single types and 153 pairs), the value drops to 13207/13501 (about 0.978). Flying/Steel then leads the defender's mix at 19.9%.

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

## The games on earlier charts

The same games, played on the chart of Generation 1 and on that of Generations 2 to 5. All exact; `uv run --group pokemon python -m pokemon.equilibrium` prints them.

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
- As of Generation 7 the data gives exactly the 146 typings of the file it replaced (below), so the Generation 7 results above and in the notebook did not change.

## Does the equilibrium predict what people play?

No. This is plan P1 (`experiments/plans/P1.md`, with its amendments), run on Smogon's monthly usage statistics for the Pokémon Showdown ladder. The primary test uses `gen9ou` at rating cutoff 1500, with the twelve months 2025-11 to 2026-10 pooled with equal weight. A typing's usage share is the summed team share of its Pokémon, divided by 6. Its availability is the number of Pokémon and forms with that typing listed in any of the twelve files; 444 forms with 147 typings are listed. The equilibrium is the duel game's over those 147 typings, on the Generation 6 chart. It is unique, on 21 typings, led by Grass/Rock (13.6%) and Water/Ground (11.9%).

The plan's test compares two Spearman correlations with usage over the 147 typings, one for equilibrium weight and one for availability. Δρ is the first minus the second. The interval is a 95% percentile bootstrap over typings, with 10,000 resamples and seed 0.

| Correlation with usage share, over 147 typings | ρ | 95% interval |
|---|---|---|
| Duel equilibrium weight | 0.08 | −0.10 to 0.25 |
| Number of Pokémon available | 0.40 | 0.25 to 0.53 |
| Highest base-stat total among them (secondary) | 0.59 | 0.46 to 0.69 |
| **Δρ, equilibrium minus availability (primary)** | **−0.32** | **−0.56 to −0.08** |

- **The primary result.** The interval for Δρ lies below 0. The plan counts the equilibrium as predicting use beyond availability only if the interval lies above 0, so it does not. The number of Pokémon with a typing predicts its use better than its equilibrium weight does, and base stats predict better still.
- **Why.** 126 of the 147 typings get weight 0, so the equilibrium ranks most typings as tied. The most used typing, Fighting/Ground at 5.4% of team slots, is one of them: it loses 0.22 on average against the equilibrium mix, mostly to Ground/Flying, Electric/Flying and Water/Ground. Of the ten most used typings, three are in the support.
- **Support mass (secondary).** The 21 typings in the support take 20.3% of usage and account for 11.0% of the available Pokémon.
- **Stability (secondary).** Δρ is between −0.34 and −0.30 in each of the twelve months, with the pooled year's equilibrium and availability. At the other cutoffs it is −0.34 (cutoff 0, interval −0.58 to −0.10), −0.27 (1695, −0.50 to −0.04) and −0.34 (1825, −0.58 to −0.09).
- **The attack game (secondary).** The defender's mix over the 147 typings gives Δρ = −0.30 (−0.52 to −0.09).
- **Generation 1 (secondary, no test).** `gen1ou` uses all 33 Generation 1 typings, and its duel equilibrium is the unique one on 13 typings in the table above. At cutoff 1500, ρ is 0.30 for the equilibrium, 0.34 for availability and 0.51 for base stats; Δρ = −0.04 (−0.52 to 0.46). Generation 1 has no abilities, items or Terastallization, but its equilibrium does no better there.

These comparisons are outside the plan and are exploratory:

- The log-ratio game over the 147 typings gives Δρ = −0.34 (k = 1) and −0.29 (k = 2), both with intervals below 0.
- Over the 18 single types, each Pokémon's share is split equally between its types. Then the duel equilibrium's ρ with `gen9ou` usage is 0.61 (0.19 to 0.83), against 0.51 (0.01 to 0.84) for availability, and Δρ = 0.10 (−0.33 to 0.61). With 18 types the intervals are too wide to separate the two.

**How the numbers are known.** The shares are sums of the published values, and the equilibria and their uniqueness are exact. Each correlation is exact given those inputs. The intervals come from the bootstrap. `scipy.stats.spearmanr` gives the same Δρ, to 15 decimal places. The bootstrap code agrees with exact enumeration of all 256 resamples on four typings (`tests/test_pokemon_usage.py`). Every number is in `results/p1.json`, and the per-typing table is in `results/usage-typings.json`.

**What the chart leaves out.** Base stats, abilities, moves, items and Terastallization all shape what people use, and the chart models none of them. A null result was the expected outcome. Spearman's ρ also treats all 126 typings with weight 0 as tied, so the equilibrium can only separate its support from the rest.

## How much of the chart is a ranking

`hodge.py` splits the chart into a ranking part and a cyclic part, following Jiang, Lim, Yao and Ye, ["Statistical ranking and combinatorial Hodge theory"](https://doi.org/10.1007/s10107-010-0419-x), *Mathematical Programming* 127 (2011) 203–244. The flow between types i and j is A(i, j) = log2(E(i, j)/E(j, i)), where E(i, j) is the multiplier when i attacks j; it is the payoff of the log-ratio game. Any such flow is the sum of a gradient s(i) − s(j), the least-squares fit of one score per type, and a remainder with no net flow at any type. On the complete graph the score is the row mean, s(i) = (1/18) Σⱼ A(i, j), and the remainder is a sum of cycles around triangles, with no "harmonic" part left over. The code checks both facts exactly rather than assuming them, and a test compares the scores with numpy's least squares.

- **The ranking part is small.** It carries 14/297 (4.7%) of the squared norm with k = 1 and 89/1701 (5.2%) with k = 2. The share grows with k, to 601/7317 (8.2%) at k = 10, and tends to 5/54 (9.3%) as k grows without bound. It is a ratio of two quadratics in k, computed exactly. The other 91–95% is cyclic. On the Generations 2–5 chart the share is 128/2465 (5.2%) at k = 1; on Generation 1's, 22/515 (4.3%).
- **What it ranks first.** Steel, for each k tried and on every chart since Generation 2. A type's score is what its attacks gain, in doublings, minus what attacks against it gain, averaged over the types. Steel hits 3 types hard and 4 weakly (net −1), and takes 3 super-effective hits, 10 resisted ones and one immunity, so its score is (6 + k)/18. Its rank comes from defense. Grass ranks last at k = 1 to 3, with −5/18. On Generation 1's chart, Psychic ranks first (tied with Dragon at k = 1).
- **What the two parts mean as games.** In the ranking part alone, Steel beats or ties every type, so pure Steel is optimal. In the cyclic part alone, every type scores 0 against the uniform mix, so the uniform mix is optimal. The real chart sits between the two. That is consistent with what the equilibria show: Steel has the most weight in the duel, but the weight is spread over 10 or 11 types. This is a reading of the numbers, not a theorem. Candogan, Menache, Ozdaglar and Parrilo, ["Flows and decompositions of games: harmonic and potential games"](https://doi.org/10.1287/moor.1110.0500), *Mathematics of Operations Research* 36 (2011) 474–503, split a whole game this way into a potential and a harmonic part. Their split works on the game's strategy profiles, not on the types, so it is a related construction, not the same one.

## Other analyses

`graph.py` writes the chart as a directed graph, with 18 types and the 120 links that are not neutral, to `display/graph.json`, and serves `display/index.html`, which draws it. `graph_analysis.py` ranks the types by PageRank and betweenness with networkx. `pokemon.ipynb` scores types, type pairs, Pokémon and random teams.

## Emerald Kaizo

`emerald_kaizo/` models IVs, stats and catch odds, to decide how many wild Pokémon of a species are worth catching for a given role. For example, a nature chosen in advance turns up 4% of the time, so 74 catches give a 95% chance of at least one. `iv_analysis.md` is the generated report, `lab_notes.md` the working notes, and `rom_fingerprint.md` identifies the ROM that was analyzed, which is not included.

## Data

- `pokeapi_forms.csv`: 1,303 Pokémon and forms, from Bulbasaur through the newest games. Each row has the form's identifier, its species and National Pokédex number, the generation it was introduced in, legendary, mythical and Mega flags, its typing in each generation from 1 to 9 (blank before it existed), and its current base stats. `pokeapi_stats_past.csv` has the 235 earlier base stats, such as Generation 1's single Special stat and the stat changes of Generation 6. `pokedex.py` reads both.
- `Pokemon.csv`: 800 Pokémon and forms through Generation 6 (number 721), with types and base stats. It matches `Pokemon.csv` in the Kaggle dataset [Pokemon with stats](https://www.kaggle.com/datasets/abcsds/pokemon) (version 2), which is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Verified: the license shown on that page, and the file's header, first rows and size (44,028 bytes against the 44.03 kB listed). Not verified: a byte-for-byte comparison, since Kaggle needs a login to download. That page says the data was gathered from pokemon.com, pokemondb.net and Bulbapedia.

**Where `pokeapi_forms.csv` comes from.** `sources.py` builds both PokeAPI files from the CSV files of [PokeAPI](https://github.com/PokeAPI/pokeapi) at commit `e98f2ad8`, and checks the SHA-256 of each download. It leaves out Gigantamax, totem and Let's Go partner (`-starter`) rows, which are battle states of a species rather than forms with their own entry. It copies identifiers and numbers only: no names, Pokédex text or other prose. `uv run --group pokemon python -m pokemon.sources --check` downloads the inputs again, checks the type charts, and fails if either file would change.

**License.** PokeAPI's repository has one license file, [`LICENSE.md`](https://github.com/PokeAPI/pokeapi/blob/master/LICENSE.md), copied here as `PokeAPI-LICENSE.md`. It is a BSD-style license, and its text speaks of source code and binary forms; it does not say whether it covers the data. PokeAPI's data began as a copy of the [veekun/pokedex](https://github.com/veekun/pokedex) data ([pokeapi.co/about](https://pokeapi.co/about)). The veekun README licenses its code under MIT and says the data "is the intellectual property of Nintendo, Creatures, inc., and GAME FREAK, inc." and is used "at your own legal risk". PokeAPI's contributors added the data for Generation 8 on, partly scraped from other sites. So no license grants the data itself. What is committed here is a table of facts (types, base stats, dates of introduction), with credit to PokeAPI and veekun and PokeAPI's notice.

**Smogon's usage statistics.** `results/usage-typings.json` is derived from Smogon's monthly usage statistics for Pokémon Showdown ([smogon.com/stats](https://www.smogon.com/stats/)), the "chaos" JSON files for `gen9ou` and `gen1ou` from 2025-11 to 2026-10. It covers all four rating cutoffs of each format, 96 files in all. The license of these statistics is unknown: none was found on the stats pages or in the FAQ thread about them on Smogon's forums. So the raw files are not committed. `usage.py` downloads them, records each file's URL, SHA-256, size and number of battles, and keeps only usage shares by typing and by type, with the number of Pokémon behind each and the highest total of base stats among them. The machine that wrote this README cannot reach smogon.com, so the download ran on GitHub Actions, on the branch `compute/pokemon-p1`. Each file lists only Pokémon with at least 0.01% weighted usage; the listed shares sum to between 5.974 and 5.9996 per file, against 6 for a full count. Showdown names are mapped to the identifiers of `pokeapi_forms.csv`. Every name in the files was checked against Showdown's `pokedex.ts` (MIT license, the pinned commit of `sources.py`), with its typing in the format's generation. The same holds for every Showdown species except the Arceus and Silvally formes (`python -m pokemon.usage check`).

**The file it replaces.** Until this change the notebook used `pokemon_gen7.csv`, 1,061 rows from the Kaggle dataset [Pokemon Sun and Moon (Gen 7) Stats](https://www.kaggle.com/datasets/mylesoneill/pokemon-sun-and-moon-gen-7-stats), whose license is unknown: the page lists "Other (specified in description)", and the description names none. It also held Pokédex text. Taken as of Generation 7, the PokeAPI data has 955 forms and exactly the same 146 typings as that file, so every result about typings is unchanged, in the README and in the notebook. The notebook's results for single Pokémon did change, because the two files count forms differently: Blissey's stat-total percentile is now 83.2 (was 81.4), and Toxapex's combined percentile 90.2 (was 92.3). The notebook now names forms by PokeAPI identifier, such as `pumpkaboo-small`. It uses base stats as of Generation 7, and it takes legendary, mythical and Mega flags from the data rather than from a list typed into it.

```bash
uv run python -m pokemon.graph                       # writes display/graph.json, serves the page on port 8000
uv run --group pokemon python -m pokemon.graph_analysis
uv run --group pokemon python -m pokemon.equilibrium   # the games above, in about a minute
uv run --group pokemon python -m pokemon.hodge         # the ranking and cyclic parts, in seconds
uv run --group pokemon python -m pokemon.sources       # rebuilds the data files from PokeAPI
uv run python -m pokemon.usage build                  # downloads Smogon's files, writes results/usage-typings.json
uv run --group pokemon python -m pokemon.p1            # plan P1 from that table, in about 30 seconds
uv run python -m pokemon.emerald_kaizo.iv_analysis   # regenerates emerald_kaizo/iv_analysis.md
uv run --group pokemon --with jupyter jupyter lab pokemon/pokemon.ipynb
```
