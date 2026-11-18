# Mastermind

A code is four pegs in six colors, 1,296 codes in all. Each strategy below is played against every code, so the numbers are exact:

| Strategy | Total guesses | Average | Worst case |
|---|---:|---:|---:|
| most parts | 5,668 | 4.373 | 6 |
| expected size | 5,696 | 4.395 | 6 |
| entropy | 5,722 | 4.415 | 6 |
| minimax | 5,801 | 4.476 | 5 |
| first consistent | 7,471 | 5.765 | 9 |

The totals match Table 7 of B. Kooi, "Yet another Mastermind strategy", *ICGA Journal* 28(1), 2005. Two strategies are on the Pareto frontier of average and worst case: most parts has the lowest average, and minimax is the only one that never needs a sixth guess.

- `mastermind.py` has the codemaker's reply, `score`, which everything else uses.
- `strategies.py` compiles each strategy into a guess tree and plays it against every code.
- `fallout3.py` narrows down the password in the Fallout 3 terminal hacking game, where the reply is how many letters are right and in place.

```bash
uv run python -m mastermind.strategies          # about half a minute
uv run python -m mastermind.fallout3 doctrines batteries scorpions generated   # asks how many letters were right
```
