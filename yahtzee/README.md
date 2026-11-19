# Yahtzee

In solitaire Yahtzee one player plays the thirteen turns of a game alone and tries to score as much as possible on average. The game is small enough to solve exactly by dynamic programming, as Verhoeff (1999), Woodward (2003) and Glenn (2006) did. This folder repeats that computation and checks it against brute force and the published values. It then asks what the mean leaves out: the distribution of the score, what each bonus rule is worth, what to keep after the first roll, how often optimal play makes a Yahtzee, and how far a greedy player falls behind.

"Exact" below means computed by the dynamic program in double precision. Changing the order of the sums moves the mean by about 10⁻¹³. "Simulated" numbers give the number of games and a 95% interval.

## Rules as modeled

The [Hasbro rules](https://www.hasbro.com/common/instruct/40958.pdf), as Verhoeff reads them on his [rules page](https://www-set.win.tue.nl/~wstomv/misc/yahtzee/rules.html):

- Each turn has up to three rolls of five dice. Before each reroll the player keeps any dice, not necessarily the ones kept before. The roll then goes in one open box, possibly for zero.
- The upper boxes score the dice of their face. Three and Four of a Kind and Chance score the sum of the dice, Full House 25, Small Straight 30, Large Straight 40 and Yahtzee 50. A Yahtzee is not a full house.
- 35 points if the six upper boxes total 63 or more.
- 100 points for each further Yahtzee once the Yahtzee box holds 50 (the **Yahtzee bonus**). It does not count toward the 63.
- The **Joker**: once the Yahtzee box is filled (with 50 or 0) and the upper box of the Yahtzee's face is filled, a Yahtzee scores 25, 30 and 40 in Full House and the straights.

Verhoeff and Glenn let the player put a Yahtzee in any open box (**free choice**). The Hasbro text instead forces the upper box of its face if that box is open, and otherwise a lower box. That **forced** reading is solved too.

## Results

### The expected score

Optimal play scores 254.5896094820 on average. Verhoeff's [trivia page](https://www-set.win.tue.nl/~wstomv/misc/yahtzee/trivia.html) gives 254.5896, and Glenn's [technical report](https://www.cs.loyola.edu/~jglenn/research/optimal_yahtzee.pdf) gives 254.59. Verhoeff's [slides](http://www.yahtzee.org.uk/optimal_yahtzee_TV.pdf) also give the values of thirteen positions, and all thirteen agree to the two decimals given. Two of them, as expected final scores:

- After a first roll of 1 1 6 6 6, keeping 6 6 6 is worth 265.12, and taking the full house at once 253.91.
- After a second roll of 1 1 3 4 6, keeping 3 4 is worth 245.17 and keeping 1 1 is worth 245.14.

The solver visits 536,448 states between turns, the number that Verhoeff gives for the states that need computing. Without the Yahtzee bonus and the Joker the value is 245.8707745141, which Verhoeff and Glenn give as 245.87.

Verhoeff's trivia page also quotes an exact fraction for the optimal score, found by K. Kelly and J. Liese (["Let's Get Rolling! Exact Optimal Solitaire Yahtzee"](https://doi.org/10.1080/0025570X.2022.2055334), *Mathematics Magazine* 95, 2022). Its denominator is 2¹⁶⁹ · 3¹⁷⁹. The fraction equals 254.5893738837. That is 0.000236 below the value here and below Verhoeff's own 254.5896. None of the readings of the Joker tried here gives their number. The forced reading gives 254.5877 and no Joker gives 253.9702. Four other readings were tried and then dropped from the code; none came within 0.0002 of the fraction. Their paper was not read, so the rules they used are unknown here.

### The distribution of the score

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

### How often optimal play makes a Yahtzee

- The Yahtzee box holds 50 at the end in 33.74% of games, and 0 in the rest.
- At least one turn ends with five of a kind in 36.73% of games, and 0.4640 turns do on average. Exactly one does in 28.40%, two in 7.13% and three in 1.09%.
- A Yahtzee bonus is earned in 8.245% of games, worth 9.58 points on average. A Joker is used 0.0373 times per game.

Verhoeff gives 63.24% for games with no Yahtzee rolled, where the figure here is 63.27%. He may count five of a kind after any roll, not only the last one, but this was not checked.

### What each rule is worth

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

### The best keep after the first roll

`openings.csv` gives, for each of the 252 first rolls of the game, the best keep and its expected final score, with the second-best keep. Counting the 7,776 equally likely ordered rolls:

- Only Yahtzees and large straights are kept whole, 246 of the 7,776 rolls. Some dice are always kept.
- 6,156 rolls hold a pair or more, and the best keep is every die of the most common face. That includes full houses: with 5 5 6 6 6, keep 6 6 6 (265.12) rather than the full house.
- In the other 900 rolls with a pair, the pair is given up:
  - for a straight draw. With 1 1 3 4 5, keep 3 4 5; with 2 3 4 5 5, keep 2 3 4 5.
  - for a single die of 4 or more, when the pair is of 1s. With 1 1 2 3 6, keep the 6; with 1 1 2 4 5, keep the 5.
- The closest decision is 1 1 3 4 6. Keeping the 4 is worth 249.8811 and keeping 3 4 is worth 249.8763. After the second roll the order flips, as in Verhoeff's example above.
- The worst first roll is 1 1 2 3 6, worth 249.83 when the 6 is kept (Verhoeff: 249.83).

### Optimal against greedy

Glenn's ([2007](https://doi.org/10.1109/CIG.2007.368089)) **greedy** strategy plays each turn for the most points that turn, Yahtzee bonus included, and ignores the upper bonus. It also chooses its keeps to maximize that turn's expected points. When several boxes tie, for example when every open box scores zero, it takes the first one on the score card. Its expected score is 217.3074 (exact), 37.28 points below optimal. Its standard deviation is 48.91 and its median 208. It earns the upper bonus in only 2.62% of games.

Glenn reports 218.54 for greedy play. He does not say how ties are broken, and the rule for ties matters. With ties going to the last box on the card the value is 217.44. Glenn's rule-based strategy has a fallback order of boxes that begins with Aces and Chance. With ties broken in that order, greedy play scores 219.05.

In 20,000 seeded games (seed 1), both strategies play the same dice wherever they reroll the same number of dice. Optimal play averages 254.80 ± 0.83 and greedy 217.42 ± 0.68. The difference is 37.38 ± 0.76 points. All three intervals contain the exact values. Optimal play scores more than greedy in 79% of these games and less in 20%.

## How it works

The state between turns is Glenn's triple: the set of filled boxes, the upper total capped at 63, and whether the Yahtzee box holds 50. A higher total earns no more, so 63 stands for 63 or more. There are 2¹³ · 64 · 2 = 1,048,576 triples, and 536,448 of them can occur. A total can be impossible for the filled upper boxes; for example, a total of 1 cannot come from the Twos box alone. And the Yahtzee box cannot hold 50 while it is open. Glenn says 1,260 of the 4,096 pairs of upper boxes and totals are impossible; the count here is 1,302. That gap was not resolved, but the state count matches Verhoeff's.

Within a turn, a roll is a sorted multiset of dice (252 of them) and a keep is a multiset of zero to five dice (462). `solver.py` solves each turn backwards:

1. For each final roll, it takes the best open box: its points plus the value of the state it leads to.
2. For each keep, it averages over the rerolls, as a product with the 462 × 252 matrix of reroll chances.
3. For each roll, it takes the best keep inside it. That maximum is built up one die at a time, from the keeps of four dice down to the empty keep.

It does this for 2,048 states at once with numpy, layer by layer from full score cards down to the empty one. It takes about 50 seconds on one core.

`policy.py` then follows the optimal choices forward from the empty card. It records, for each of the 284,312 states that optimal play reaches, every way the turn can end: the next state, the points and the chance. Summing over those edges backwards gives the exact distribution of the final score. Each state's distribution is a mixture of the next states' distributions, shifted by the points earned. The per-box statistics are sums of each state's chance of being reached times what its turn earns.

## Checks

- `brute.py` solves small games by plain recursion, with exact fractions. Its dice are ordered, a keep is any subset of their positions, and the state is the whole score card. It shares nothing with the solver except the scoring rules. The tests compare the two on seven games with two to five dice, two to four faces and three to five boxes, each with an upper bonus at a lower threshold. Three of those games differ only in their Joker reading and have three different values. The values from the two methods differ by less than 10⁻¹³.
- One-box games have closed forms. Chance alone is worth 70/3, and its distribution is that of five independent dice. Yahtzee alone matches an independent chain over the largest group kept. With only Aces open and an upper total of 62, the value is 5 · 91/216 + 35 · (1 − (5/6)¹⁵).
- Three numbers differ by less than 10⁻¹³: the solver's value, the mean of the computed distribution and the sum of the per-box means. Seeded simulations of small games match their exact distributions.
- The tests check the published values above against the full solve, and check `openings.csv` and `distribution.csv` against it.

Not done: exact rational arithmetic for the full game. That would test whether any decision is a near-tie that rounding gets wrong. It would also settle the comparison with Kelly and Liese's fraction.

## Other sources

Pawlewicz, ["Nearly Optimal Computer Play in Multi-player Yahtzee"](https://doi.org/10.1007/978-3-642-17928-0_23), *Computers and Games* 2010, treats the game for several players, where the aim is to win rather than to score. His [slides](https://mimuw.edu.pl/~pan/papers/yahtzee-cg2010.pdf) estimate that an optimal two-player strategy would need 12.7 trillion stored values. They also report that a nearly optimal player wins about 4 more games in 100 than the best human players of an online site. P. Woodward, "Yahtzee: The Solution", *Chance* 16(1), 2003, also solved the solitaire game; that article was not read.

## Files

- `rules.py` holds the scoring rules and `Rules`, which describes one variant of the game.
- `solver.py` holds the dynamic program, plus Glenn's greedy strategy as a variant with every state valued at zero.
- `policy.py` holds the states a strategy reaches, its exact score distribution and its per-box statistics.
- `brute.py` is the brute-force check on small games.
- `simulate.py` plays seeded games, many at once.
- `study.py` prints every number above and writes `openings.csv` and `distribution.csv`.

```bash
uv run python -m yahtzee.solver            # the optimal value; about a minute
uv run python -m yahtzee.study optimal     # checks, distribution, per box; about 6 minutes
uv run python -m yahtzee.study rules       # the table of rule values; about 6 minutes
uv run python -m yahtzee.study openings    # writes openings.csv
uv run python -m yahtzee.study greedy      # greedy play under three tie rules; about 5 minutes
uv run python -m yahtzee.study simulate    # 20,000 seeded games of each strategy; about 3 minutes
```
