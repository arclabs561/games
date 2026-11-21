# Going second, two roles and the Normal Summon: a seeded simulation

Written by `uv run python -m yugioh.second --simulate`. Seed 1, 200,000 hands per row, each decided by trying every assignment of the need's slots to distinct cards (`roles.fits_cards`), not by the search over groups that the exact values use.

| Need | Exact | Simulated | 95% interval | Exact inside |
|---|---:|---:|---:|---:|
| 12 starters, 12 hand traps: a starter and a hand trap, second | 80.5785% | 80.4615% | 80.2871% to 80.6347% | yes |
| example deck, 12 hand traps: a starter, an extender and a hand trap, second | 62.8099% | 62.7645% | 62.5524% to 62.9761% | yes |
| example deck, 3 starters also extend: a starter and an extender, first | 70.1280% | 70.2180% | 70.0172% to 70.4180% | yes |
| example deck, 8 starters and 3 extenders need the Normal Summon, first | 53.8557% | 53.9980% | 53.7795% to 54.2164% | yes |
