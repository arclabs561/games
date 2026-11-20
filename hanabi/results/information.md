# The information strategy

The hat-guessing strategy of Cox et al. (2015), reproduced against HanSim, and its generalization to three and four players. The [Hanabi README](../README.md) gives the short answers.


In the information strategy, a hint is read as a number from 0 to 7 rather than for its literal content: a rank or color hint to the player one to four seats to the left. Every player computes, from public information, which card in each other hand to "target" and splits that card's remaining possibilities into eight sets. The hinter sends the sum, mod 8, of the set numbers they see. Each other player subtracts the numbers they see and learns which set holds their own targeted card. This is the hat-guessing trick. `information.py` gives the details and the order in which a player plays, discards or hints.

Cox et al. describe the strategy in prose and point to their simulator, [HanSim](https://github.com/rjtobin/HanSim), for the rest. `information.py` is a new implementation that follows HanSim's code, including its quirks: the targeted card is chosen by HanSim's estimate of the chance it is playable, the possibility sets are filled rank by rank, and the fallback discards look at all four slots even when a hand is shorter. HanSim lets a player discard while all 8 tokens are on the table. `hansim_decks` reproduces the C library's `drand48` shuffle, so both programs can play the same decks.

| Five players | Mean score | Perfect games | How it is known |
|---|---:|---:|---|
| Cox et al. (2015), Figure 8 | 24.68 | 76% | their simulation, 10⁶ games |
| HanSim, compiled from their repository | 24.668 | 75.44% | its first 10,000 games |
| This implementation, the same decks | 24.680 ± 0.004 | 76.31% ± 0.26% | 100,000 games |
| This implementation, official discard rule | 24.686 ± 0.013 | 76.76% ± 0.83% | 10,000 seeded deals |
| Players who see their own cards: paper | 24.87 | 91% | their simulation, 10⁶ games |
| Players who see their own cards: here | 24.866 ± 0.009 | 90.70% ± 0.57% | 10,000 games |

Intervals here and below are 95% intervals. On the first 10,000 HanSim decks this implementation scores exactly what HanSim scores, game by game (`hansim_check.py` rebuilds HanSim and repeats the comparison). The published 24.68 and 76% are reproduced. HanSim's rule on discards changes nothing measurable: with the official rule the strategy scores the same within the interval. The paper remarks that the strategy never makes an error. The tests play 30 seeded deals each for three, four and five players, and no game loses a life.

The paper treats five players only. With 2(P − 1) hint values for P players and larger possibility sets where needed, the same code scores 24.564 ± 0.016 with 71.42% ± 0.89% perfect for four players and 22.254 ± 0.032 with 6.57% ± 0.49% for three (10,000 seeded deals each, HanSim's discard rule). This version keeps the official rule that a hint must touch a card. It is not the first to go beyond five players. Bouzy (2017) extends the hat strategy to other numbers of players by allowing hints that touch no card. Wu's [hanabi.rs](https://github.com/WuTheFWasThat/hanabi.rs) (WTFWThat, 2016) is a stronger information strategy for two to five players: it reports 24.7946 (84.48%) and 24.9356 (95.05%) on 20,000 seeds for three and four players, and 24.9223 (94.04%) for five.
