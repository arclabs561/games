# Who is the better player?

Two friends play a game with no draws. One wins k of n games. What is the chance that this player is the better one?

With a uniform prior on the chance p of winning a game, the answer equals the chance of at most k heads in n + 1 fair coin flips. Six wins in ten games give 743/1024, about 0.726. Being 95% sure takes 8 wins in 10 games, 59 in 100, or 527 in 1,000. The curve is close to a logistic one, and its slope follows from the spread of the posterior.

Playing until one player is 95% sure, with a cap of N games, changes the picture for equal players. Someone is then declared better with chance 0.528 when N = 100 and 0.728 when N = 1,000, against 0.089 and 0.094 for one look after N games. When the first player wins each game with chance 0.55, the worse player is declared better with chance 0.120 (N = 100) or 0.123 (N = 1,000). Yet if p is drawn from the uniform prior, only 3.6% (N = 100) or 3.7% (N = 1,000) of verdicts are wrong. The stopping rule does not change the posterior, so each verdict keeps its 95%. All of these numbers are exact, from a dynamic program over (wins, games).

`peggle.pdf` explains and proves this. `peggle.py` checks the coin-flip identity against the Beta integral with exact fractions, computes the tables, and writes `better.csv` for the figure. `stopping.py` computes the optional-stopping chances exactly. The tests check them against brute force over all sequences of up to 12 games and against seeded simulation. They also check them against Table 1 of Armitage, McPherson and Rowe (1969), which gives the same chances for repeated binomial tests. Of its 34 entries for 2α = 0.01 and 0.05, 32 agree to the five printed decimals and the other two agree within 2·10⁻⁵. `python -m peggle.stopping` prints a 200,000-trial simulation for each of four cases; all eight exact values fall inside the 95% intervals.

```bash
uv run python -m peggle.peggle
uv run python -m peggle.stopping                       # about 10 seconds
uv run erase/paper/typst_vendor.py peggle/peggle.typ   # rebuilds peggle.pdf
```
