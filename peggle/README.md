# Who is the better player?

Two friends play a game with no draws. One wins k of n games. What is the chance that this player is the better one?

With a uniform prior on the chance p of winning a game, the answer equals the chance of at most k heads in n + 1 fair coin flips. Six wins in ten games give 743/1024, about 0.726. Being 95% sure takes 8 wins in 10 games, 59 in 100, or 527 in 1,000. The curve is close to a logistic one, and its slope follows from the spread of the posterior.

`peggle.pdf` explains and proves this. `peggle.py` checks the coin-flip identity against the Beta integral with exact fractions, computes the tables, and writes `better.csv` for the figure.

```bash
uv run python -m peggle.peggle
uv run erase/paper/typst_vendor.py peggle/peggle.typ   # rebuilds peggle.pdf
```
