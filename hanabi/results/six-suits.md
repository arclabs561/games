# Six colors: the open-game ceiling, and published strategies under it

With every card in view, what share of six-color deals can reach 30? The method is plan H1's, applied to a 60-card deck: the same two solvers, budgets, certificates and later attempts, on seeded deals 0 to 19,999 of `Rules(colors=6)` for each number of players. The sixth color has the same ten cards as the others. It is the purple suit of van Doorn's hat bot, whose documentation calls the variant "6 suits". This run was not planned in advance as H1 was. It is an exact computation per deal, with no cost and no learned model, and its decks are new: a seed shuffles a different deck with six colors than with five.

## The ceiling

| Players | Lost | Undecided | Winnable share | 95% interval | Mean score at most |
|---:|---:|---:|---|---|---|
| 2 | 562 | 2 | 0.9718 to 0.9719 | 0.9694 to 0.9741 | 29.9632 ± 0.0033 |
| 3 | 12 | 7 | 0.99905 to 0.9994 | 0.9985 to 0.9997 | 29.9992 ± 0.0005 |
| 4 | 4 | 3 | 0.99965 to 0.9998 | 0.9993 to 0.9999 | 29.9998 ± 0.0002 |
| 5 | 0 | 18 | 0.9991 to 1 | 0.9986 to 1 | 30.0000 ± 0.0000 |

Shares are of seeded deals 0 to 19,999; the interval is Wilson's at 95%, widened over the undecided deals as in H1. "Mean score at most" counts each undecided deal as 30 and each lost deal at its best score (an upper bound where a search ran out of budget).

The two solvers agree on every one of the 78,676 deals that both decide. Every lost deal has a certificate: of the 562 lost two-player deals, the pooled game proves 541, time 126, storage 90, the last-card test 535 and the free-order game 558, many of them more than once. With three and four players every lost deal fails the time bound. No five-player deal is proved lost; 18 stay undecided, and the free-order game is won for 10 of them.

Two-player deals are lost about twice as often as with five colors: 562 of 20,000 against 276. The best score of a lost two-player deal is 29 for 412 deals, 28 for 92, 27 for 34, 26 for 3 and 25 for 1; the 20 deals lost only by a later certificate are counted at 29.

The reruns at 5,000,000 nodes of the 56 deals that both solvers left undecided were still running when these numbers were taken: three of their 14 jobs had finished, settling 12 deals (six two-player deals lost by Solver B with all cuts, one lost and one won by Solver A, four five-player deals won). The rest count as undecided above, so a later summary can only narrow the ranges.

## Published strategies under the ceilings

Every published strategy must score below the ceiling of its own game. With six colors, van Doorn's hat bot (`fpvandoorn/hanabi`, `doc_hat_player.md`) wins 93.9% of four-player games and 95.1% of five-player games, losing 0.09 and 0.07 points on average against 30. Both lie below the ceiling. With four players the ceiling puts at most 0.0002 points out of reach on average, and with five no sampled deal is proved lost. So nearly all of the hat bot's loss could in principle be closed.

With five colors, hanabi.rs's information strategy (J. Wu; its README's table, reproduced here by building it) scores 24.7946, 24.9356 and 24.9223 with three to five players, and wins 84.48%, 95.05% and 94.04%. Plan H1's ceilings are 24.9990, 24.9998 and 24.9999, with winnable shares of at least 0.9988, 0.99965 and 0.99935. So the deal rules out at most 0.5%, 0.3% and 0.1% of hanabi.rs's gap to 25. Its cheating player, which sees its own cards but not the deck, wins 88.40% of two-player games against a winnable share of 0.9862.

| Strategy | Colors | Players | Mean | Perfect | Ceiling: mean at most | Ceiling: share winnable |
|---|---:|---:|---:|---:|---:|---|
| hanabi.rs, information | 5 | 3 | 24.7946 | 84.48% | 24.9990 | 0.9988 to 0.9992 |
| hanabi.rs, information | 5 | 4 | 24.9356 | 95.05% | 24.9998 | 0.99965 to 0.99985 |
| hanabi.rs, information | 5 | 5 | 24.9223 | 94.04% | 24.9999 | 0.99935 to 0.99995 |
| hanabi.rs, cheating | 5 | 2 | 24.8209 | 88.40% | 24.9838 | 0.9862 |
| van Doorn's hat bot | 6 | 4 | 29.91 | 93.9% | 29.9998 | 0.99965 to 0.9998 |
| van Doorn's hat bot | 6 | 5 | 29.93 | 95.1% | 30.0000 | 0.9991 to 1 |

The hanabi.rs means are over its seeds 0 to 19,999; the hat bot's are from its documentation, rounded there to 0.1 point of percent and 0.01 point. All of them come from rules that differ from `rules.py` in small ways (hanabi.rs keeps the board when the last life is lost; see [hat-gap.md](hat-gap.md)), which cannot raise a score above the open-game ceiling of the same deck.

## Reproducing

The records were made on GitHub Actions, branch `compute/hanabi-six-suits`, at commit `28a75f5` (the classification) and `f3d3d50` (three of the 14 rerun jobs, as of 2026-11-20 20:30 UTC) (the deals left undecided, at 5,000,000 nodes); `six-suits.json` keeps the summary and a row for every deal not won.

```bash
uv run python -m hanabi.upper_bound --sample 2 500 --start 0 --colors 6 --solver both --json OUT/p2-00000.json
uv run python -m hanabi.upper_bound --sample 2 0 --seeds 3137 --colors 6 --rerun --budgets 5000000 --json OUT/p2-rerun-a.json
uv run python -m hanabi.six_suits --runs OUT --out hanabi/results/six-suits.json
```
