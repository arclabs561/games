# Optimal play

The expected score and its check against published values, the distribution of the score, and how often optimal play makes a Yahtzee. Moved unchanged from the [Yahtzee README](../README.md), which keeps the short answers. "Exact" means computed by the dynamic program in double precision.

## The expected score

Optimal play scores 254.5896094820 on average. Verhoeff's [trivia page](https://www-set.win.tue.nl/~wstomv/misc/yahtzee/trivia.html) gives 254.5896, and Glenn's [technical report](https://www.cs.loyola.edu/~jglenn/research/optimal_yahtzee.pdf) gives 254.59. Verhoeff's [slides](http://www.yahtzee.org.uk/optimal_yahtzee_TV.pdf) also give the values of thirteen positions, and all thirteen agree to the two decimals given. Two of them, as expected final scores:

- After a first roll of 1 1 6 6 6, keeping 6 6 6 is worth 265.12, and taking the full house at once 253.91.
- After a second roll of 1 1 3 4 6, keeping 3 4 is worth 245.17 and keeping 1 1 is worth 245.14.

The solver visits 536,448 states between turns, the number that Verhoeff gives for the states that need computing. Without the Yahtzee bonus and the Joker the value is 245.8707745141, which Verhoeff and Glenn give as 245.87.

Verhoeff's trivia page also quotes an exact fraction for the optimal score, found by K. Kelly and J. Liese (["Let's Get Rolling! Exact Optimal Solitaire Yahtzee"](https://doi.org/10.1080/0025570X.2022.2055334), *Mathematics Magazine* 95, 2022). Its denominator is 2¹⁶⁹ · 3¹⁷⁹. The fraction equals 254.5893738837. That is 0.000236 below the value here and below Verhoeff's own 254.5896. None of the readings of the Joker tried here gives their number. The forced reading gives 254.5877 and no Joker gives 253.9702. Four other readings were tried and then dropped from the code; none came within 0.0002 of the fraction. Their paper was not read, so the rules they used are unknown here.

## The distribution of the score

Under optimal play the final score has standard deviation 59.6117, as Verhoeff gives. Its median is 248: P(score ≥ 248) = 0.5018, and P(score ≤ 248) = 0.5074. Verhoeff found the same median in a million simulated games. All the numbers in this section are exact.

| Quantile | 1% | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Score | 151 | 180 | 194 | 217 | 248 | 272 | 318 | 388 | 474 |

- The most likely single score is 266, with chance 0.0126.
- The score is 300 or more in 14.33% of games and 400 or more in 3.84%. It is 500 or more in 0.72%. The long right tail comes from Yahtzee bonuses.
- The least score optimal play can end with is 12, which Verhoeff also gives. Its chance is 2.9 · 10⁻⁴⁴. The most is 1,575, the largest score that the rules allow.
- Without the Yahtzee bonus and the Joker, the standard deviation drops to 39.8208 (Verhoeff: 39.82). The median is 247, and the score never exceeds 375.

`distribution.csv` lists the chance of every score.

The mean, standard deviation and share of zeros of each box also match Verhoeff's table to two decimals, for all thirteen boxes. The upper bonus is earned in 68.12% of games.

## How often optimal play makes a Yahtzee

- The Yahtzee box holds 50 at the end in 33.74% of games, and 0 in the rest.
- At least one turn ends with five of a kind in 36.73% of games, and 0.4640 turns do on average. Exactly one does in 28.40%, two in 7.13% and three in 1.09%.
- A Yahtzee bonus is earned in 8.245% of games, worth 9.58 points on average. A Joker is used 0.0373 times per game.

Verhoeff gives 63.24% for games with no Yahtzee rolled, where the figure here is 63.27%. He may count five of a kind after any roll, not only the last one, but this was not checked.
