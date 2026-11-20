# Rule variants and greedy play

What each bonus rule is worth, and Glenn's greedy strategy against optimal play. The [Yahtzee README](../README.md) gives the short answers. "Exact" means computed by the dynamic program without sampling, in double precision (good to about 10⁻¹³).

## What each rule is worth

The value of a rule is the optimal score with it minus the optimal score without it. Each row is exact.

| Rules | Optimal score | Change |
|---|---:|---:|
| official (upper bonus, Yahtzee bonus, Joker) | 254.5896 | |
| no Joker | 253.9702 | −0.6194 |
| no Yahtzee bonus | 246.0873 | −8.5023 |
| no Yahtzee bonus, no Joker | 245.8708 | −8.7188 |
| no upper bonus | 237.8425 | −16.7471 |
| no upper bonus, no Joker | 237.3669 | −17.2227 |
| no upper bonus, no Yahtzee bonus | 229.7389 | −24.8507 |
| none of the three | 229.6385 | −24.9511 |
| forced Joker instead of free choice | 254.5877 | −0.0019 |

- Optimal play earns the upper bonus's 35 points in 68.12% of games, 23.84 points on average. The rule is worth only 16.75. Without the bonus, the best play scores 237.84 in the other boxes. Chasing the bonus lowers that to 230.75, so it costs 7.09 points elsewhere.
- In the same way, the Yahtzee bonus pays 9.58 points on average but is worth 8.50.
- The Joker is worth 0.62 points with the Yahtzee bonus and 0.22 without it.
- The rules interact only a little. The upper bonus is worth 16.75 with the Yahtzee rules and 16.23 without them.
- Forced placement of a Yahtzee costs 0.0019 points on average against free choice.

## Optimal against greedy

Glenn's ([2007](https://doi.org/10.1109/CIG.2007.368089)) **greedy** strategy plays each turn for the most points that turn, Yahtzee bonus included, and ignores the upper bonus. It also chooses its keeps to maximize that turn's expected points. When several boxes tie, for example when every open box scores zero, it takes the first one on the score card. Its expected score is 217.3074 (exact), 37.28 points below optimal. Its standard deviation is 48.91 and its median 208. It earns the upper bonus in only 2.62% of games.

Glenn reports 218.54 for greedy play. He does not say how ties are broken, and the rule for ties matters. With ties going to the last box on the card the value is 217.44. Glenn's rule-based strategy has a fallback order of boxes that begins with Aces and Chance. With ties broken in that order, greedy play scores 219.05.

In 20,000 seeded games (seed 1), both strategies play the same dice wherever they reroll the same number of dice. Optimal play averages 254.80 ± 0.83 and greedy 217.42 ± 0.68. The difference is 37.38 ± 0.76 points. All three intervals contain the exact values. Optimal play scores more than greedy in 79% of these games and less in 20%.
