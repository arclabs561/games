# How many turns does Tenzi take?

Each player rolls ten dice, keeps the dice that show one face, and rerolls the rest until all ten match. Counting the matched dice gives a Markov chain, which gives the answer exactly. Every turn count below includes the first roll.

- Keeping the face that is most common after the first roll, a game takes 15.35 turns on average. The median is 14, and the most likely length is 12.
- Switching faces whenever the rerolled dice show more of another face than are kept lowers the mean to 15.29 (15.2944).
- No strategy does better than switching. Solving the game as a Markov decision process, with every choice of dice to keep after every roll, gives the same mean exactly.
- Naming the face before the first roll, as some players do, raises the mean to 16.56 (16.5648), with standard deviation 6.83. The game then lasts as long as the slowest of ten dice, each of which needs a geometric number of rolls.
- A game takes more than 30 turns 3.3% of the time with the fixed face.

## Choosing the face before the first roll

The turns are the largest of ten independent Geometric(1/6) waits, so P(T > k) = 1 − (1 − (5/6)^k)^10. Expanding the power and summing over k gives the mean and variance in closed form. They agree exactly with the Markov chain started from zero matched dice.

Bacinski and Pennings, in the MathFest slides [How to win at Tenzi](http://sigmaa.maa.org/mcst/documents/TenziforMathfest.pdf), use this rule and report a mean of 15.565 turns with standard deviation 6.834. Their mean is the exact 16.565 less one. Their table of finishing times leaves out the first roll: beside each roll k it puts the exact chance of finishing on roll k + 1, to within 0.0006. Their state vectors do count it: S₄ gives 0.0014 for ten matched dice, which is P(T ≤ 4), and S₂₀ − S₁₉ gives 0.0401, which is P(T = 20). Shifting by one changes the mean but not the standard deviation, which is why their 6.834 matches.

## The race

With n players who roll together and play independently, the game ends on the first turn on which someone finishes. If one player is still playing after turn t with chance S(t), the first finish comes on turn t with chance S(t − 1)ⁿ − S(t)ⁿ. Two or more players tie on that turn unless exactly one finishes. The players with one die left at the end are those with nine dice matched on that turn; their expected number follows from the chain's distribution of matched dice after each turn.

| players | mean winning turn, face named first | slides | P(tie) | players with one die left | slides | mean winning turn, switching |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 12.90 | 13.0 | 0.048 | 0.52 | 0.53 | 11.68 |
| 3 | 11.41 | 11.4 | 0.080 | 0.91 | 0.91 | 10.23 |
| 4 | 10.54 | 10.6 | 0.103 | 1.22 | 1.21 | 9.39 |
| 6 | 9.52 | 9.6 | 0.137 | 1.70 | 1.71 | 8.41 |
| 10 | 8.48 | 8.5 | 0.181 | 2.41 | 2.37 | 7.42 |
| 20 | 7.37 | 7.4 | 0.241 | 3.57 | 3.58 | 6.38 |
| 30 | 6.84 | 6.8 | 0.276 | 4.38 | 4.36 | 5.89 |

The columns without a source are exact; the sums stop at 200 turns, where the chance of a longer game is below 10⁻¹⁴. The slides' columns are labeled with one player fewer than the rows here: they count the other players, not everyone. Their race means count the first roll, unlike their single-player mean. They differ from the exact values by up to 0.1 turns, and their one-die-left column by up to 0.05 players, as a simulation would. Their share of players with one die left divides by everyone, winner included. Their chart of a player who rolls faster gives a jump of 0.0235 at equal speeds, about half the exact chance of a tie between two players, 0.048. We could not reconcile that figure.

## The best strategy

After a roll, all that matters is how many dice show each face. The player then keeps some dice and rerolls the rest. Policy iteration with exact fractions starts from switching and finds no keep that improves on it, both when the kept dice must share one face and when any dice may be kept. Every other choice of dice is worse by at least 0.05 turns, and keeping one die is exactly as good as rerolling it. The expected turns still to come after keeping k dice of one face are:

| k | 0 or 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turns | 15.29 | 15.11 | 14.66 | 13.93 | 13.02 | 11.93 | 10.56 | 8.73 | 6.00 |

These fall as k grows, so keeping more dice of one face is always better. That is the switching strategy. On small games, up to four dice on three faces, the same method agrees with value iteration over every ordered roll and every subset of dice to keep.

## Checks

Each exact mean lies inside the 95% interval of a seeded simulation of 100,000 games (16.535 ± 0.042 with the face named first). A simulation of 20,000 races of four players agrees too: 10.515 ± 0.041 for the winning turn, 0.107 ± 0.004 for a tie and 1.208 ± 0.013 for players with one die left.

## Files

- `tenzi.py` computes the distributions and the race, runs the simulations, and writes `turns.csv` for the figure in `tenzi.pdf`.
- `optimal.py` solves the decision process and checks it against value iteration on small games.
- `monte_carlo.py` simulates a million games while keeping the shortest and longest ones, and checks the Gumbel-max trick for sampling a die whose faces are not equally likely.

```bash
uv run python -m tenzi.tenzi
uv run python -m tenzi.optimal
uv run --group tenzi python -m tenzi.tenzi --plot   # also a histogram with a fitted gamma density
uv run erase/paper/typst_vendor.py tenzi/tenzi.typ    # rebuilds tenzi.pdf
```
