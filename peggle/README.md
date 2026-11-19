# Who is the better player?

Two friends play a game with no draws. One wins k of n games. What is the chance that this player is the better one?

With a uniform prior on the chance p of winning a game, the answer equals the chance of at most k heads in n + 1 fair coin flips. Six wins in ten games give 743/1024, about 0.726. Being 95% sure takes 8 wins in 10 games, 59 in 100, or 527 in 1,000. The curve is close to a logistic one, and its slope follows from the spread of the posterior.

Playing until one player is 95% sure, with a cap of N games, changes the picture for equal players. Someone is then declared better with chance 0.528 when N = 100 and 0.728 when N = 1,000, against 0.089 and 0.094 for one look after N games. When the first player wins each game with chance 0.55, the worse player is declared better with chance 0.120 (N = 100) or 0.123 (N = 1,000). Yet if p is drawn from the uniform prior, only 3.6% (N = 100) or 3.7% (N = 1,000) of verdicts are wrong. The stopping rule does not change the posterior, so each verdict keeps its 95%. All of these numbers are exact, from a dynamic program over (wins, games).

A rule that is safe for equal players needs a guarantee at p = 1/2 itself. The Bayes factor 2ⁿ / ((n + 1)·C(n, k)) says how many times likelier the record is if p is drawn from the uniform prior than if p = 1/2. When p = 1/2 it is a fair game that starts at 1 (a test martingale), so by Ville's inequality it ever reaches 20 with chance at most 1/20, whatever the stopping rule. Playing until it reaches 20 declares equal players unequal with chance 0.0276 (N = 100) or 0.0357 (N = 1,000), exactly, and 0.0387 at N = 10,000 in floating point. The price is speed: the rule needs 66 wins in 100 games where the 95% rule needs 59, and at p = 0.55 it reaches a verdict within 1,000 games only 51% of the time, after 745 games on average, though it then favors the worse player only 0.4% of the time.

With the cap fixed in advance, a stricter posterior threshold is just as safe. The chance of a verdict between equal players changes only at thresholds equal to a posterior that occurs, so bisection with the same dynamic program finds the least strict threshold that keeps it at or below 5%: 0.99704 for N = 100 and 0.9987003 for N = 1,000. The chance is then 0.0497 for both, exactly. These rules need 64 wins in 100 games or 548 in 1,000. At p = 0.55 they declare the better player better with chance 0.113 (N = 100) or 0.698 (N = 1,000), after 93.8 or 637.9 games on average, against 0.061 and 0.512 for the Bayes factor rule. They gain because they spend the whole 5% within the cap.

Wald's sequential test of p = 0.45 against p = 0.55 is "play until one player leads by 15 games", since the likelihood ratio depends only on the lead. By the gambler's ruin, at p = 0.55 it declares the worse player better with chance 9¹⁵/(9¹⁵ + 11¹⁵) = 0.0470, after 135.9 games on average, exactly. A lead of 14 would err 5.7% of the time. The 95% rule with N = 1,000 needs about as many games, 138.0, and errs 12.3% of the time. Wald's test does not test equality: at p = 1/2 it declares someone better after 225 games on average.

`peggle.pdf` explains and proves this. `peggle.py` checks the coin-flip identity against the Beta integral with exact fractions, computes the tables, and writes `better.csv` for the figure. `stopping.py` computes the optional-stopping chances exactly. The tests check them against brute force over all sequences of up to 12 games and against seeded simulation. They also check them against Table 1 of Armitage, McPherson and Rowe (1969), which gives the same chances for repeated binomial tests. Of its 34 entries for 2α = 0.01 and 0.05, 32 agree to the five printed decimals and the other two agree within 2·10⁻⁵. `python -m peggle.stopping` prints a 200,000-trial simulation for each of four cases; all eight exact values fall inside the 95% intervals.

```bash
uv run python -m peggle.peggle
uv run python -m peggle.stopping                       # about 10 seconds
uv run erase/paper/typst_vendor.py peggle/peggle.typ   # rebuilds peggle.pdf
```
