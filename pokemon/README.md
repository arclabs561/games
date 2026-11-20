# Pokémon

Which attacking types cover the type chart, which typings take the least damage, what the chart's equilibrium is when it is played as a game, and whether that equilibrium predicts what people play. Separately, `emerald_kaizo/` works out how many wild Pokémon to catch in the Emerald Kaizo ROM hack.

- No six attacking types are super-effective against all 18 types; ten sets of seven are, and all ten include Fighting and Ground. Steel/Fairy takes the least damage. Exact.
- Played as a duel, the chart has a unique equilibrium on 11 types that matches every weight Schmitz (2021) published. Exact.
- A ranking of the types, led by Steel, accounts for only 4.7% to 5.2% of the chart; the rest is cyclic. Exact.
- The equilibrium does not predict which typings people use on the Showdown ladder: the number of Pokémon with a typing predicts its use better. Bootstrap interval, below.

![The Generation 6 type chart: attacking types down the side, defending types across the top. Colored cells are 2 (red), 1/2 (light blue) and 0 (black); blank cells are 1.](figures/type-chart.svg)

*The Generation 6 chart, written by `figures.py`. Each row is an attacking type. Only 120 of the 324 cells are not neutral, and the table is not symmetric.*

## Rules as modeled

- **The chart.** `pokemon_types.py` holds the Generation 6 chart: 18 types, and the multiplier when each attacks each. `chart(g)` gives the chart of any generation (below). A move against a Pokémon with two types gets the product of the two multipliers, so it can be 4, 2, 1, 1/2, 1/4 or 0.
- **Typings.** A typing is one type or a pair, 171 in all. The analyses use either all 171, or those that some Pokémon or form has in a given generation (146 as of Generation 7, 162 as of Generation 9).
- **What is left out.** Base stats, abilities, moves, items, the bonus for a move of the user's own type and Terastallization are not modeled, except that the notebook's Pokémon scores use the total of the base stats. A typing is assumed to attack with moves of its own types.
- **Arbitrary scores.** The notebook's coverage score and its attack and defense scores are sums chosen for simplicity; its rankings of Pokémon depend on them, and are not facts about the games.

## Results

| Result | How it is known |
|---|---|
| No set of six attacking types is super-effective against every type; ten sets of seven are, and Fighting and Ground are in all ten. | Exact: every set, checked a second time against the chart directly |
| Among existing typings, Steel/Fairy takes the least damage summed over the 18 attacking types: 53/4, against 18 for a type with no weaknesses or resistances. It holds from Generation 6 to 9. In Generations 2 to 5 the least is Flying/Steel's 12 of 17; in Generation 1, 14 of 15, shared by Psychic and Electric/Flying. | Exact |
| Attack game: the attacker can guarantee an average multiplier of 4158/3979 ≈ 1.045, with a unique equilibrium led by Ground (attacker) and Normal (defender). Over all 171 defending typings the value is 13207/13501 ≈ 0.978. | Exact: LP support, then fractions and a uniqueness certificate |
| Duel game: unique equilibrium on 11 types, led by Steel 627/2653 ≈ 0.236; the 7 types left out and every weight match Schmitz (2021) to three decimals. | Exact; agrees with a cited source |
| Duel over all 171 typings: unique, on 23 typings led by Electric/Flying (12.9%). Schmitz (2023) reports 27 typings led by Water/Flying; both of his leaders lose against this equilibrium. | Exact; disagrees with a cited source, cause unknown |
| Log-ratio game: unique at k = 1; for k = 2, 3, 4, 6 and 10 the optimal strategies form a segment, along which Ghost's weight runs from 0 to a bound such as 5/79. | Exact, with the segment proved |
| 16 rock-paper-scissors triangles of super-effective types, 5 of them with each type also resisting the one it hits. | Exact, by two methods |
| The ranking part of the chart carries 14/297 (4.7%) of its squared norm with k = 1 and 89/1701 (5.2%) with k = 2; Steel ranks first. | Exact, checked against least squares |
| Equilibrium weight against usage on `gen9ou`: Spearman's ρ is 0.08, against 0.40 for the number of Pokémon with each typing. The difference, Δρ, is −0.32, with interval −0.56 to −0.08, so the equilibrium does not predict use beyond availability. | Bootstrap over typings, 10,000 resamples, seed 0, 95% interval (plan P1) |
| Fighting/Ice has the most conflicting types; Ground/Ice and Fighting/Ice attack best; Ice/Rock defends worst. | Exact, given the notebook's sums |
| With the notebook's mix of base stats and typing, Empoleon is the best final starter of Generations 1 to 7, and Celesteela the best Pokémon new in Generation 7 that is not legendary. | Exact, given an arbitrary score |
| Emerald Kaizo: a nature chosen in advance turns up 4% of the time, so 74 catches give a 95% chance of at least one. | Exact under the independent uniform IV model |

## How the results are checked

- **Covers.** `covers(k)` is compared in the tests with a search that reads the chart directly, and the notebook checks that no cover stays one after any of its types is dropped.
- **Equilibria.** The floating-point LP only finds the support. The weights are then solved exactly with fractions on that support, and every equilibrium inequality is checked exactly. Uniqueness, unused types and weight ranges are proved exactly, as described under [The chart as a game](#the-chart-as-a-game).
- **Charts by generation.** The charts agree pair by pair with PokeAPI's `type_efficacy.csv` and `type_efficacy_past.csv` and with Pokémon Showdown's per-generation chart files, for every generation from 1 to 9 (`sources.py`).
- **Usage statistics.** `scipy.stats.spearmanr` gives the same Δρ to 15 decimal places. The bootstrap code agrees with an exact enumeration of all 256 resamples on four typings.
- **Ranking part.** The scores are compared with numpy's least squares, and the absence of a harmonic part is checked exactly, not assumed.
- **Notebook.** `pokemon.ipynb` imports `scoring.py`, which `tests/test_pokemon_scoring.py` tests, including the percentile rank against scipy's. CI reruns the notebook and compares its text and figures, and checks that the SVGs here are what `figures.py` writes.

## The type chart and its typings

`pokemon.ipynb` asks the questions a player would ask of the chart, in order: how many types there are, the smallest cover, a score for coverage, typings that take the least, typings whose two types disagree, attack and defense scores of typings, Pokémon and random teams, and the best starter.

- **Covering.** No six types cover the chart. Of the ten covers of seven, Fighting and Ground are in all of them: Fighting is the only type super-effective against Normal, and only Ground hits Electric hard. The notebook also scores a set of attacking types: two points per type hit hard, minus one per type that resists, minus two per immune type. The score reaches at most 18, first with four sets of six types, and all 16 best sets include Ground and Fairy.
- **Defense.** The ten typings that take the least are led by Steel/Fairy (53/4), Flying/Steel (27/2) and Dragon/Steel and Ghost/Steel (57/4 each); nine of the ten include Steel.
- **Scores.** Over the 146 typings that exist as of Generation 7, attack and defense rescaled to run from 0 to 1 have correlation −0.20. Normal is at percentile 2.1 of their sum. Averaging over a random team of six typings (10,000 teams, seed 42) narrows the spread by a factor of about 2.5, close to √6.
- **Pokémon.** The total of Blissey's base stats is at percentile 83.2 among the 955 forms. Toxapex's typing is at percentile 81.2 among typings, and its combined score at 90.2. Celesteela and Blacephalon, Ultra Beasts that the data does not mark as legendary, lead the forms that are neither legendary nor Mega.

![Fairy/Steel: the best multiplier it deals to each type, sum 23, and the multiplier each type deals it, sum 53/4.](figures/steel-fairy.svg)

*Steel/Fairy, the typing that takes the least. The top row is what it deals with moves of its own types; the bottom row is what moves of each type deal it, with two immunities (Poison and Dragon) and one 1/4 (Bug).*

## The chart as a game

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

[`results/generations.md`](results/generations.md) has the full tables, the defender's mixes and the per-generation detail.

## Does the equilibrium predict what people play?

No. This is plan P1 (`experiments/plans/P1.md`, with its amendments), run on Smogon's monthly usage statistics for the Pokémon Showdown ladder. The primary test uses `gen9ou` at rating cutoff 1500, with the twelve months 2025-11 to 2026-10 pooled with equal weight. A typing's usage share is the summed team share of its Pokémon, divided by 6. Its availability is the number of Pokémon and forms with that typing listed in any of the twelve files: 444 forms with 147 typings. The equilibrium is the duel game's over those 147 typings, on the Generation 6 chart; it is unique, on 21 typings.

The test compares two Spearman correlations with usage over the 147 typings, one for equilibrium weight and one for availability. Δρ is the first minus the second. The interval is a 95% percentile bootstrap over typings, with 10,000 resamples and seed 0.

| Correlation with usage share, over 147 typings | ρ | 95% interval |
|---|---|---|
| Duel equilibrium weight | 0.08 | −0.10 to 0.25 |
| Number of Pokémon available | 0.40 | 0.25 to 0.53 |
| Highest base-stat total among them (secondary) | 0.59 | 0.46 to 0.69 |
| **Δρ, equilibrium minus availability (primary)** | **−0.32** | **−0.56 to −0.08** |

- **The primary result.** The interval for Δρ lies below 0. The plan counts the equilibrium as predicting use beyond availability only if the interval lies above 0, so it does not.
- **Why.** 126 of the 147 typings get weight 0, so the equilibrium ranks most typings as tied. The most used typing, Fighting/Ground at 5.4% of team slots, is one of them.
- **Secondary results.** Δρ is between −0.34 and −0.30 in each of the twelve months and at every other rating cutoff but one (−0.27 at 1695). The attack game gives −0.30. In `gen1ou`, Δρ is −0.04 with an interval from −0.52 to 0.46.

Base stats, abilities, moves, items and Terastallization all shape what people use, and the chart models none of them, so a null result was the expected outcome. [`results/p1.md`](results/p1.md) has the secondary and exploratory results in full.

## How much of the chart is a ranking

`hodge.py` splits the chart into a ranking part and a cyclic part, following Jiang, Lim, Yao and Ye, ["Statistical ranking and combinatorial Hodge theory"](https://doi.org/10.1007/s10107-010-0419-x), *Mathematical Programming* 127 (2011) 203–244. The flow between types i and j is A(i, j) = log2(E(i, j)/E(j, i)), where E(i, j) is the multiplier when i attacks j; it is the payoff of the log-ratio game. Any such flow is the sum of a gradient s(i) − s(j), the least-squares fit of one score per type, and a remainder with no net flow at any type. On the complete graph the score is the row mean, s(i) = (1/18) Σⱼ A(i, j), and the remainder is a sum of cycles around triangles, with no "harmonic" part left over. The code checks both facts exactly rather than assuming them.

- **The ranking part is small.** It carries 14/297 (4.7%) of the squared norm with k = 1 and 89/1701 (5.2%) with k = 2. The share grows with k, to 601/7317 (8.2%) at k = 10, and tends to 5/54 (9.3%) as k grows without bound. It is a ratio of two quadratics in k, computed exactly. The other 91–95% is cyclic. On the Generations 2–5 chart the share is 128/2465 (5.2%) at k = 1; on Generation 1's, 22/515 (4.3%).
- **What it ranks first.** Steel, for each k tried and on every chart since Generation 2. A type's score is what its attacks gain, in doublings, minus what attacks against it gain, averaged over the types. Steel hits 3 types hard and 4 weakly (net −1), and takes 3 super-effective hits, 10 resisted ones and one immunity, so its score is (6 + k)/18. Its rank comes from defense. Grass ranks last at k = 1 to 3, with −5/18. On Generation 1's chart, Psychic ranks first (tied with Dragon at k = 1).
- **What the two parts mean as games.** In the ranking part alone, Steel beats or ties every type, so pure Steel is optimal. In the cyclic part alone, every type scores 0 against the uniform mix, so the uniform mix is optimal. The real chart sits between the two. That is consistent with what the equilibria show: Steel has the most weight in the duel, but the weight is spread over 10 or 11 types. This is a reading of the numbers, not a theorem. Candogan, Menache, Ozdaglar and Parrilo, ["Flows and decompositions of games: harmonic and potential games"](https://doi.org/10.1287/moor.1110.0500), *Mathematics of Operations Research* 36 (2011) 474–503, split a whole game this way into a potential and a harmonic part. Their split works on the game's strategy profiles, not on the types, so it is a related construction, not the same one.

## Emerald Kaizo

Emerald Kaizo is a ROM hack of Pokémon Emerald. Its question is a player's too, but it is not about the type chart: how many wild Pokémon of a species are worth catching to get one that fits a role.

- **Model.** Six IVs, each independently uniform on 0 to 31, and one of 25 natures, also uniform. The tables of which species appear in grass, and failed throws, are not modeled, so the counts are per catch of the target species.
- **Results.** One chosen nature turns up with chance 1/25 = 4%, so a player expects 25 catches, and 74 give a 95% chance of at least one. Two IVs that matter at 16 or more, with one of two acceptable natures, have chance 2%: 50 catches expected, 149 for 95%. Exact under the model; `iv_analysis.md` has the table and a simulation as a check.
- **Caveat.** The game's random number generator is deterministic, so encounters at the same timing are not independent; the uniform model is a marginal approximation.

`iv_analysis.md` is the generated report and `lab_notes.md` the working notes. `rom_fingerprint.md` identifies the ROM that was analyzed, which is not included.

## Data

| Source | What it is | License | Committed |
|---|---|---|---|
| [PokeAPI](https://github.com/PokeAPI/pokeapi) CSV files, commit `e98f2ad8` | `pokeapi_forms.csv` (1,303 forms, typings in each generation, current base stats) and `pokeapi_stats_past.csv` (235 earlier base stats) | BSD-style for the code; no license grants the data (below) | yes, facts only |
| Kaggle, [Pokemon with stats](https://www.kaggle.com/datasets/abcsds/pokemon) (version 2) | `Pokemon.csv`, 800 forms through Generation 6 | CC0 1.0 | yes |
| Smogon [usage statistics](https://www.smogon.com/stats/) | monthly "chaos" files for `gen9ou` and `gen1ou`, 2025-11 to 2026-10 | unknown | no; only derived shares |
| Pokémon Showdown `pokedex.ts` | names and typings, to check the mapping of usage names | MIT | no |

**PokeAPI.** `sources.py` builds both PokeAPI files from the CSV files of PokeAPI and checks the SHA-256 of each download. It leaves out Gigantamax, totem and Let's Go partner (`-starter`) rows, which are battle states of a species rather than forms with their own entry. It copies identifiers and numbers only: no names, Pokédex text or other prose. `uv run --group pokemon python -m pokemon.sources --check` downloads the inputs again, checks the type charts, and fails if either file would change. PokeAPI's repository has one license file, [`LICENSE.md`](https://github.com/PokeAPI/pokeapi/blob/master/LICENSE.md), copied here as `PokeAPI-LICENSE.md`. It is a BSD-style license, and its text speaks of source code and binary forms; it does not say whether it covers the data. PokeAPI's data began as a copy of the [veekun/pokedex](https://github.com/veekun/pokedex) data ([pokeapi.co/about](https://pokeapi.co/about)). The veekun README licenses its code under MIT and says the data "is the intellectual property of Nintendo, Creatures, inc., and GAME FREAK, inc." and is used "at your own legal risk". PokeAPI's contributors added the data for Generation 8 on, partly scraped from other sites. So no license grants the data itself. What is committed here is a table of facts (types, base stats, dates of introduction), with credit to PokeAPI and veekun and PokeAPI's notice.

**Kaggle.** `Pokemon.csv` matches the file in the Kaggle dataset, which is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Verified: the license shown on that page, and the file's header, first rows and size (44,028 bytes against the 44.03 kB listed). Not verified: a byte-for-byte comparison, since Kaggle needs a login to download. That page says the data was gathered from pokemon.com, pokemondb.net and Bulbapedia. The notebook only shows it for comparison.

**Smogon.** `results/usage-typings.json` is derived from the "chaos" JSON files for `gen9ou` and `gen1ou`, all four rating cutoffs of each format, 96 files in all. No license was found on the stats pages or in the FAQ thread about them on Smogon's forums. So the raw files are not committed. `usage.py` downloads them, records each file's URL, SHA-256, size and number of battles, and keeps only usage shares by typing and by type, with the number of Pokémon behind each and the highest total of base stats among them. The machine that wrote this README cannot reach smogon.com, so the download ran on GitHub Actions, on the branch `compute/pokemon-p1`. Each file lists only Pokémon with at least 0.01% weighted usage; the listed shares sum to between 5.974 and 5.9996 per file, against 6 for a full count. Showdown names are mapped to the identifiers of `pokeapi_forms.csv`. Every name in the files was checked against Showdown's `pokedex.ts`, at the pinned commit of `sources.py`, with its typing in the format's generation. The same holds for every Showdown species except the Arceus and Silvally formes (`python -m pokemon.usage check`).

**The file the PokeAPI data replaced.** The notebook once used `pokemon_gen7.csv`, 1,061 rows from the Kaggle dataset [Pokemon Sun and Moon (Gen 7) Stats](https://www.kaggle.com/datasets/mylesoneill/pokemon-sun-and-moon-gen-7-stats). Its license is unknown: the page lists "Other (specified in description)", and the description names none. It also held Pokédex text. Taken as of Generation 7, the PokeAPI data has 955 forms and exactly the same 146 typings as that file, so every result about typings is unchanged. The results for single Pokémon did change, because the two files count forms differently: Blissey's stat-total percentile is now 83.2 (was 81.4), and Toxapex's combined percentile 90.2 (was 92.3).

## Related work

- Schmitz (2021 and 2023), cited above: the duel over single types is reproduced exactly; the duel over typings is not, and the cause is open.
- Jiang, Lim, Yao and Ye (2011) give the ranking split used by `hodge.py`; Candogan, Menache, Ozdaglar and Parrilo (2011) give a related split of whole games.

## Questions left open

- **Schmitz's typing duel.** Why his 27-typing equilibrium differs from the unique one here. His matchup table is an image; reading it would settle whether the game or the solver differs.
- **Weighted scores.** The notebook's sums treat every attacking type as equally likely. Weighting by Showdown usage would change which typings and Pokémon rank first.
- **The best team.** A team of six chosen for coverage and defense together, with real movesets rather than one move per type.
- **Emerald Kaizo encounters.** How often each species appears, and the generator's timing, which the independent model leaves out.

## Usage

```bash
uv run --group pokemon --with jupyter jupyter lab pokemon/pokemon.ipynb
uv run --group pokemon python -m pokemon.figures       # the SVGs in figures/
uv run python -m pokemon.graph                         # writes display/graph.json, serves the page on port 8000
uv run --group pokemon python -m pokemon.graph_analysis
uv run --group pokemon python -m pokemon.equilibrium   # the games above, in about a minute
uv run --group pokemon python -m pokemon.hodge         # the ranking and cyclic parts, in seconds
uv run --group pokemon python -m pokemon.sources       # rebuilds the data files from PokeAPI
uv run python -m pokemon.usage build                   # downloads Smogon's files, writes results/usage-typings.json
uv run --group pokemon python -m pokemon.p1            # plan P1 from that table, in about 30 seconds
uv run python -m pokemon.emerald_kaizo.iv_analysis     # regenerates emerald_kaizo/iv_analysis.md
```

## Files

| File | Contents |
|---|---|
| `pokemon_types.py` | the Generation 6 chart, and `chart(g)` for every generation |
| `scoring.py`, `figures.py` | covers, attack and defense scores, and the pictures; used by `pokemon.ipynb` |
| `pokedex.py`, `sources.py` | reading and building the PokeAPI files |
| `equilibrium.py`, `hodge.py` | the chart as a game, and its ranking split |
| `usage.py`, `p1.py`, `results/` | plan P1 and its results |
| `graph.py`, `graph_analysis.py`, `display/` | the chart as a directed graph, with PageRank and betweenness |
| `emerald_kaizo/` | IV and catch odds for Emerald Kaizo |
