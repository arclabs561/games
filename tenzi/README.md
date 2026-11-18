# How many turns does Tenzi take?

Each player rolls ten dice, keeps the dice that show one face, and rerolls the rest until all ten match. Counting the matched dice gives a Markov chain, which gives the answer exactly:

- Keeping the face that is most common after the first roll, a game takes 15.35 turns on average. The median is 14, and the most likely length is 12.
- Switching faces whenever the rerolled dice show more of another face than are kept lowers the mean to 15.29.
- A game takes more than 30 turns 3.3% of the time.

The exact means lie inside the 95% intervals of the simulated ones, from 100,000 games for each strategy.

- `tenzi.py` computes the distributions, runs the simulations, and writes `turns.csv` for the figure in `tenzi.pdf`.
- `monte_carlo.py` simulates a million games while keeping the shortest and longest ones, and checks the Gumbel-max trick for sampling a die whose faces are not equally likely.

```bash
uv run python -m tenzi.tenzi
uv run --group tenzi python -m tenzi.tenzi --plot   # also a histogram with a fitted gamma density
uv run erase/paper/typst_vendor.py tenzi/tenzi.typ    # rebuilds tenzi.pdf
```
