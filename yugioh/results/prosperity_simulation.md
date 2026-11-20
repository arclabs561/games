# Pot of Prosperity: simulation

Written by `uv run python -m yugioh.prosperity --write`. The example deck with Pot of
Prosperity (revealing 6) and 3 draw-2 cards, going first, a starter. Seed 1,
200,000 shuffled decks per row, Wilson 95% intervals.

| Fixed choice | Hits | 95% interval | Exact | Exact in interval |
|---|---:|---|---:|---|
| prosperity first | 181,991 | 90.87% to 91.12% | 90.90% | yes |
| draw2 first | 181,198 | 90.47% to 90.73% | 90.52% | yes |
