# Going second, two roles and the Normal Summon: a seeded simulation

Written by `uv run python -m yugioh.second --simulate`. Seed 1, 200,000 hands per row, each decided by trying every assignment of the need's slots to distinct cards (`roles.fits_cards`), not by the search over groups that the exact values use.

| Need | Exact | Simulated | 95% interval | Exact inside |
|---|---:|---:|---:|---:|
| 12 starters, 12 hand traps: a starter and a hand trap, second | 75.6852% | 75.5260% | 75.3371% to 75.7139% | yes |
| example deck, 12 hand traps: a starter, an extender and a hand trap, second | 58.5214% | 58.4675% | 58.2514% to 58.6833% | yes |
| example deck, 3 starters also extend: a starter and an extender, first | 70.1280% | 70.2180% | 70.0172% to 70.4180% | yes |
| example deck, 8 starters and 3 extenders need the Normal Summon, first | 53.8557% | 53.9980% | 53.7795% to 54.2164% | yes |
