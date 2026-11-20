# Calibration of the rating from one game: diagnosis on pilot games

The plan, with its amendments, is `calibration-plan.md`. Every number here is in `calibration-pilot.json`, which the diagnosis job wrote from 998 stratified player-games (499 games, September 2025, files 32 to 39) and 598 population player-games (300 games, the same files). All of it is exploratory: it is the pilot. The test result is in the README and in `calibration-test.json`.

## Which hypotheses held

| Hypothesis | Check on pilot games | Verdict |
|---|---|---|
| 1. The flat prior and the grid's edge at 600 | The likelihood's maximum, which has no prior, is off by +178 (600 to 1000) and +150 (1000 to 1400). The posterior mean is off by +209 and +163. | Not the cause: the maximum keeps 85% to 92% of the bias. |
| 2. Maia-3's rating bins and clipping at 600 | The code has no bins: a rating is a blend of two learned vectors, clamped to 0 to 5,000. Per 100 points, a move's log-probability changes 2.4 times as much at 600 to 700 as at 1500 to 1600, and 4.1 times at 400 to 500. | Not the cause: Maia-3 tells low ratings apart more sharply, not less. |
| 3. Maia-3's rating scale differs from the players' | Pooled over players in 100-point bins, the moves are likeliest at a rating 120 to 170 above the true one below 1000, and 55 to 160 above it from 1000 to 1400. From 1400 to 2300 the gap is under 60. The mean score at the true rating is +0.019 per move (0.015 to 0.023) in the lowest band and +0.006 (0.003 to 0.008) in the next. | The cause: the likelihood itself is shifted. |
| 4. Time control and clock | Mean error below 1400: +120 at 180+0 (110 player-games), +190 at 180+2 (50) and at 300+0 (148), +260 at 300+3 (86). Moves made with more than half the starting time left score +0.014 (0.011 to 0.017), the rest +0.009 (0.006 to 0.013). | A real part: the bias at 300+3 is twice that at 180+0. The 140-point gap is about four times the standard error that the errors' spread gives (about 36, not bootstrapped). |
| 5. Opening moves in short games | The lowest band's mean score is positive in every phase: +0.020 (plies 0 to 9), +0.025 (10 to 19), +0.022 (20 to 39), +0.016 (40 on). Without each player's first 10 moves the bias grows, to +267 and +168. | Not the cause. |
| 6. New or improving accounts | 37 of 998 player-games were by players whose rating rose 100 or more by the end of the month. Below 1400 their error is +239 (20 player-games), and +183 for the other 380. The lowest band's median error is +160, against a mean of +209. | A small part at most: the rest are biased almost as much. |
| 7. Dependent moves make the likelihood too sharp | Mean observed information at the true rating over the variance of the total score (Godambe's ratio): 0.58, or 0.54 without centering. By band, 0.57 to 0.75. | The cause of the coverage fault. The power fitted for 95% coverage, 0.56, agrees. |
| 8. The bias costs coverage | Coverage after each band's mean error is subtracted: 89.4%, against 87.2% before. | A small part: about 2 points of the 8 missing. |

The score of a move is the slope of its log-probability at the true rating, per 100 points. It is zero on average if the model at that rating describes the players rated there. Intervals on scores are 95% bootstrap intervals over games (2,000 resamples, seeds 30 to 32).

## Which moves carry the information

Each move's log-likelihood ratio between the true rating plus 200 and minus 200:

- **Top moves.** Moves that were Maia-3's top choice at the true rating (19,230 of 34,702) have a mean absolute ratio of 0.147, the other moves 0.363. The top moves still push upward on average (+0.076): playing the likeliest move looks stronger.
- **Phase.** Moves in plies 0 to 19 carry 0.28 each, plies 20 to 39 0.26, and plies 40 on 0.21.

## Case files

The ten largest errors below 1400 are players rated 828 to 1,310 whom the old method placed at 1,900 to 2,800. Seven of the ten played five-minute games, and nine of the ten had 22 to 33 moves. In the largest, a player rated 828 wins with a mating attack in 24 moves. Forced recaptures (Kxf7 at 0.995) say nothing. A few developing moves (Nc6, g6) are more than twice as likely for a 2,600 as for an 800, and the posterior has two peaks, near 1,050 and at the top of the grid. The case files contain the games, so they stay in the private runs dataset (`runs/chess/calib-pilot-diagnose/cases.txt`).

## Repairs, cross-validated on the pilot

Twofold, by file parity, on the 998 stratified player-games. Bias is the mean error below 1400.

| Arm | Log score | Coverage | Bias below 1400 | Mean absolute error |
|---|---:|---:|---:|---:|
| F0 old method | −2.397 | 88.2% | +182 | 201 |
| F1 grid 400 to 3000 | −2.476 | 87.2% | +186 | 220 |
| F2 tempered (τ 0.56) | −2.376 | 94.5% | +205 | 219 |
| F3 game mixture (σ 150) | −2.393 | 94.1% | +185 | 219 |
| F4 move mixture (σ 100) | −2.476 | 87.2% | +190 | 221 |
| F5 rating map | −2.432 | 90.2% | +100 | 214 |
| F6 correction after the fact | −2.677 | 90.0% | +14 | 230 |
| F7 rating map, tempered (τ 0.68) | −2.370 | 94.8% | +114 | 211 |
| F8 correction, tempered | −2.630 | 96.8% | +16 | 230 |

The rule in the plan chose F2: F7's log score is better by 0.006, within one standard error (0.025), and F2 has one constant fewer. The amendment scores the bias question with F7. The correction after the fact (F6, F8) removes the mean bias but not the spread: its errors grow and its log score is the worst, because it stretches the scale where the estimate is flat.

The population prior helps on average and hurts at the ends. On the population pilot sample, F2 with the prior has a mean absolute error of 172 against 195 for F0, with 95% coverage. But players above 1800 are pulled down by 117 to 237 points, and those below 1000 up by 309.
