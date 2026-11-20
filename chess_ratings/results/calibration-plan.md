# Plan: why the rating from one game is biased and overconfident, and a fix scored once

Written 2026-11-20, before any test game below is read. The estimate in the README (exploratory, 500 player-games from file 40 of September 2025) has two faults. Its 95% intervals hold the true rating 86% of the time, and players below 1400 get estimates about 200 points too high. This plan names the possible causes and the check for each one. It fixes the candidate repairs and the rule that picks one, and it sets the test that scores the pick once. Diagnosis and fitting use pilot games only. The test month is new and is read once, after the method is frozen by a dated amendment below.

This plan sits in the game's folder rather than in `experiments/plans/`, and it is committed on a branch, not merged, before the test run. Under `experiments/METHOD.md` that makes the test result exploratory, and the README will say so.

## Data

- **Pilot games:** September 2025, files 32 to 39 of 62 (Lichess/standard-chess-games). The old pilot used files 30 and 31, the old test file 40, and the blunder study file 50; none of these is used here.
- **Test games:** August 2025, files 40 and 41 of 66. Maia-3 trained on Lichess blitz games from January 2023 to July 2025, so August is the first month after its training data. The paper adds very high and very low rated games from August and September 2025 to one of its test sets, so these months were not used for training.
- **Who qualifies** (as before): rated blitz; ended normally or on time; neither rating moved by more than 15 points; at least 40 half-moves; clock comments present.
- **Two samples in each split.** The *stratified* sample has 100 games in each of the five bands of the mean rating (600 to 1000, 1000 to 1400, 1400 to 1800, 1800 to 2200, 2200 to 2700), as in the README. The *population* sample has 300 qualifying games in hash order, whatever their ratings, and so follows Lichess's rating distribution. The pilot draws both from files 32 to 39. The test draws the stratified sample from August file 40 and the population sample from file 41. Order within a sample is by md5(salt || Site), with salts `chess-calibration-pilot-2026-11-20` and `chess-calibration-test-2026-11-20`.
- **Moves used:** as before, each player's moves before they first have under 30 seconds.
- **Likelihoods** are computed once per player-game on a wide grid, ratings 400 to 3000 in steps of 100 (27 points), with both players given the same rating and seven earlier positions. Lichess ratings do not go below 400. The old grid (600 to 2600) is a subset, so the old method is recomputed exactly on the same queries.

## Hypotheses for the bias, and the check for each on pilot games

1. **The prior and the grid.** The old estimate is a posterior mean under a flat prior on 600 to 2600. A wide, flat likelihood is pulled toward the middle of the grid and cut off at 600. *Check:* bias by band of the likelihood's maximum and of the posterior mean, on the wide grid and on the old one. If the maximum on the wide grid has less than half the old bias below 1400, the prior and grid explain most of it.
2. **Maia-3's rating input: bins and clipping.** Read in the code at commit 1e13597, Maia-3 has no bins. A rating is clamped to 0 to 5,000 and enters as a linear blend of two learned vectors. Training resampled games into 100-point bins of the mean rating from 600 to 2600, plus one bin below 600 and one above 2600, so ratings below 600 were seen but pooled. *Check:* the mean absolute slope of each move's log-probability, per 100 points, in each grid cell. If it falls below a third of its value at 1500 in the cells under 1000, the model barely tells low ratings apart there.
3. **Maia-3's rating scale differs from the players'.** For true ratings in 100-point bins, find the rating at which Maia-3 gives the pooled moves their highest likelihood. Also compute the mean score (slope of the log-likelihood) at the true rating, which is zero in expectation if the model is right. *Check:* if that best rating is more than 100 above the true one in the bins below 1400, the bias is in the likelihood.
4. **Time control and the clock, which Maia-3 does not see** (its released models are built without clock inputs). *Check:* bias in the two lower bands by time control (180+0, 180+2, 300+0, 300+3, other) and by whether the mover had more or less than half the starting time left. A difference above 100 points between groups, larger than twice its bootstrap standard error, counts as an effect.
5. **Opening moves in short games.** Book moves are played alike at every rating and may carry little information, or the wrong information. *Check:* the mean score at the true rating, by ply (0 to 9, 10 to 19, 20 to 39, 40 and later), and the bias when each player's first 10 moves are dropped. A change of more than 50 points in the lower bands counts.
6. **New or improving accounts of strong players.** A player whose rating is about to rise plays above it. *Check:* each pilot player's other rated blitz games in September 2025 (all 62 files): the number of games and the change from this game's rating to the last of the month. Compare the bias of players whose rating then rose by 100 or more with that of the rest, and compare mean with median error. Usernames are used inside the job only and never stored.

## Hypotheses for the coverage

7. **Moves are not independent given the positions.** Then the curvature of the log-likelihood overstates the information. *Check:* the variance across player-games of the total score at the true rating, against the mean observed information there. Their ratio estimates the power τ to which the likelihood should be raised: Godambe's sandwich for a composite likelihood.
8. **The bias itself.** A 200-point shift with intervals 720 wide loses coverage on its own. *Check:* coverage after only the bias is removed.

## Case files and moves

The pilot job writes a case file for the ten player-games with the largest errors in the two lower bands and for one random player-game per band. A case file shows the posterior curve, every move with Maia-3's probability at the true rating and at the estimate, the move's score, the ply and the clock. Case files go to the private dataset, since they contain games; only summaries are committed.

## Candidate repairs

Each uses the wide grid and the same per-move log-probabilities.

- **F0** the old method: flat prior on 600 to 2600, posterior mean and central 95% interval.
- **F1** flat prior on 400 to 3000.
- **F2** F1 with the likelihood raised to a power τ, fitted so that pilot coverage is 95%.
- **F3** a rating mixture for the whole game: the player's strength in this game is spread normally around the rating, with standard deviation σ fitted on the pilot, instead of a point.
- **F4** a rating mixture for each move: each move's probability is Maia-3's averaged over ratings spread normally around the rating, with σ fitted on the pilot.
- **F5** a map of the rating scale: Maia-3 is queried at h(R) instead of R, where h is fitted on the pilot from hypothesis 3 (monotone, by isotonic regression, constant beyond the data).
- **F6** a correction fitted on the pilot after the fact: the estimate and the interval ends pass through the inverse of the pilot's mean estimate as a function of the true rating (isotonic).
- **F7** F5 with τ fitted after the map, and **F8** F6 with τ.
- **P** the population prior, which applies to any of the above: Lichess's rating distribution among qualifying players in the pilot files, for the game's time control (five groups as in hypothesis 4), smoothed by a normal kernel of 50 points.

## Rule that picks the method

- Every fitted constant is chosen by twofold cross-validation on the stratified pilot sample, with the folds made of even and odd files. The score is the mean log score of the true rating: the log of the posterior density at it, per 100 points.
- F0 to F8 are compared with a flat prior. The arm with the best cross-validated log score is chosen, provided its cross-validated coverage lies between 93% and 97%. If it does not, the next best arm that meets the coverage condition is chosen. An arm within one standard error of the best, with fewer fitted constants, is chosen over it.
- The chosen arm is refit on the whole stratified pilot sample, and its constants are written to `calibration-fit.json`. A dated amendment below names the arm and the file's SHA-256 before the test job starts.
- The population prior is then added to the chosen arm, fitted on the pilot population sample in the same way, for the population goal. The prior is not chosen by the test.

## Two goals, kept apart

A prior that matches the population makes the estimate more accurate on average over players, but it pulls players far from the middle toward it. That raises the bias at a given rating, which is the fault measured above. So the stratified sample is scored with a flat prior, and the population sample with the population prior as well as without it.

## Test, scored once

- **Primary 1, bias below 1400.** The mean error of the chosen arm, against F0, over the player-games in the two lower bands of the stratified test sample. The paired difference (|old bias| minus |new bias|) gets a 95% bootstrap interval, resampling games (10,000 resamples, seed 11), and a one-sided bootstrap p-value.
- **Primary 2, coverage.** The share of 95% intervals that hold the true rating, over the whole stratified test sample, with a Wilson interval. It is compared with F0 by McNemar's exact test on the same player-games. The two p-values are corrected by Holm's method.
- **Secondary**, without significance claims: mean absolute error, mean error, coverage and median interval width by band, with bootstrap intervals (seeds 12 to 20); the same for every arm F0 to F8; the population sample with and without the population prior; move matching at the true ratings.
- **Detectable effects**, from the README's numbers (standard deviation of errors about 250 points in the lower bands, 400 player-games there, in 200 games): about 2.80 × 250 / √200 ≈ 50 points for the bias, if pairs from a game are as correlated as they can be, and 35 if they are independent. For coverage, with 1,000 player-games, about 2 points. The pilot's standard deviations will update these in the amendment.
- **A null result** is reported in the README with its interval: if the bias below 1400 or the coverage does not improve, the README says so and keeps the old numbers as the main ones.

## Compute and cost

Maia3-5M runs on Hugging Face Jobs, cpu-upgrade (8 vCPU), as before: `python -m chess_ratings.job calib-records --split pilot|test --sample stratified|population`, one job per split and sample, each with a timeout of 2 hours. The diagnosis and fitting run as `calib-diagnose` on cpu-basic, with a 30-minute timeout, from the stored records. The old run did about 300 queries a second. Each split's two jobs need about 1.5 million queries, about 85 minutes of cpu-upgrade. The expected cost is under $0.15 and the cap is $0.50. Outputs go to the private dataset `arclabs561/games-runs` under `runs/chess/calib-*`; aggregates go to this folder.

## Amendments

None yet.
