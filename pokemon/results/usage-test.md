# Does the equilibrium predict what people play?

Plan P1 in short; [p1.md](p1.md), written by `p1.py`, has every secondary and exploratory result. Moved unchanged from the [Pokémon README](../README.md), which keeps the short answers.


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

Base stats, abilities, moves, items and Terastallization all shape what people use, and the chart models none of them, so a null result was the expected outcome. [`results/p1.md`](p1.md) has the secondary and exploratory results in full.
