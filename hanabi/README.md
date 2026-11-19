# Hanabi

Hanabi is a cooperative card game for two to five players. Each player sees the other hands but not their own, and the team tries to play the 50-card deck into five piles, one per color, in order from 1 to 5. A perfect game scores 25. This folder asks two questions:

- How well does the hat-guessing "information strategy" of Cox et al. (2015) play, and can its published numbers be reproduced?
- What share of deals can reach 25 at all, if every player could see every card, including the deck? No strategy for the real game wins a deal that fails this test, so the share bounds every agent's rate of perfect games.

## Rules as modeled

`rules.py` follows the official rule sheet and the Hanabi Learning Environment (Bard et al., 2020):

- Five colors; each has three 1s, two 2s, two 3s, two 4s and one 5. Hands hold five cards with two or three players and four with four or five.
- The team starts with 8 hint tokens and 3 lives. A hint names a color or a rank to another player and must touch at least one of their cards. It costs a token.
- A discard returns a token and is not allowed while all 8 tokens are on the table. A played 5 also returns a token.
- A card that does not fit its pile costs a life. Losing the third life ends the game with score 0.
- The game also ends at 25, or one round after the last card is drawn, so that every player, including the one who drew it, has one more turn.

`Rules` has switches for other readings: discards with all tokens on the table (as HanSim allows), keeping the board when the last life is lost, and smaller decks with fewer colors or ranks. A deal is reproducible: `seeded_deck(rules, seed)` shuffles with `random.Random(seed)`.

## The information strategy

In the information strategy, a hint is read as a number from 0 to 7 rather than for its literal content: a rank or color hint to the player one to four seats to the left. Every player computes, from public information, which card in each other hand to "target" and splits that card's remaining possibilities into eight sets. The hinter sends the sum, mod 8, of the set numbers she sees. Each other player subtracts the numbers he sees and learns which set holds his own targeted card. This is the hat-guessing trick. `information.py` gives the details and the order in which a player plays, discards or hints.

Cox et al. describe the strategy in prose and point to their simulator, [HanSim](https://github.com/rjtobin/HanSim), for the rest. `information.py` is a new implementation that follows HanSim's code, including its quirks: the targeted card is chosen by HanSim's estimate of the chance it is playable, the possibility sets are filled rank by rank, and the fallback discards look at all four slots even when a hand is shorter. HanSim lets a player discard while all 8 tokens are on the table. `hansim_decks` reproduces the C library's `drand48` shuffle, so both programs can play the same decks.

| Five players | Mean score | Perfect games | How it is known |
|---|---:|---:|---|
| Cox et al. (2015), Figure 8 | 24.68 | 76% | their simulation, 10⁶ games |
| HanSim, compiled from their repository | 24.668 | 75.44% | its first 10,000 games |
| This implementation, the same decks | 24.680 ± 0.004 | 76.31% ± 0.26% | 100,000 games |
| This implementation, official discard rule | 24.686 ± 0.013 | 76.76% ± 0.83% | 10,000 seeded deals |
| Players who see their own cards: paper | 24.87 | 91% | their simulation, 10⁶ games |
| Players who see their own cards: here | 24.866 ± 0.009 | 90.70% ± 0.57% | 10,000 games |

Intervals here and below are 95% intervals. On the first 10,000 HanSim decks this implementation scores exactly what HanSim scores, game by game (`hansim_check.py` rebuilds HanSim and repeats the comparison). The published 24.68 and 76% are reproduced. HanSim's rule on discards changes nothing measurable: with the official rule the strategy scores the same within the interval. The paper remarks that the strategy never makes an error. The tests play 30 seeded deals each for three, four and five players, and no game loses a life.

The paper treats five players only. With 2(P − 1) hint values for P players and larger possibility sets where needed, the same code scores 24.564 ± 0.016 with 71.42% ± 0.89% perfect for four players and 22.254 ± 0.032 with 6.57% ± 0.49% for three (10,000 seeded deals each, HanSim's discard rule). This generalization is ours. Stronger hat strategies exist for three and four players: [hanabi.rs](https://github.com/WuTheFWasThat/hanabi.rs) reports 24.7946 (84.48%) and 24.9356 (95.05%) on 20,000 seeds, and 24.9223 (94.04%) for five.

## Which deals can reach 25?

With every card in view, a hint carries no information; it only passes the turn and costs a token. Van den Bergh (2015) calls this the open game and shows that a deal can be won in the real game only if it can be won in the open game. Baffier et al. (2017) prove that deciding this is NP-complete, even with hands of two cards, so the answer here comes from search.

`solver.py` has two engines. `brute_winnable` tries every legal action of the rules engine. `winnable` searches the open game: it tries plays first, writes every card that can no longer be played as one "trash" symbol, never discards the last copy of a needed card, and cuts off states that fail three bounds:

- **Time.** Each play or discard draws a card while the deck lasts, and after that each player has one turn. The i-th draw from now can only play cards already in a hand or among the first i − 1 cards of the deck.
- **Storage.** Once every copy of a needed card is out of the deck, and the card cannot have been played yet, some hand must hold it. A deal fails if more such cards are needed at one moment than the hands can hold.
- **Pooled game.** Let one player hold all the hands together, with no turn order and no tokens. There, playing a playable card at once is never worse (play it now and make the discard later), so the only choice is which card to discard when nothing is playable. Every winning line of the real game wins the pooled game.

Each bound that holds at the start of a deal is a certificate that the deal is lost. A won deal comes with its winning line, which the rules engine replays to 25. On full deals `winnable` uses limited discrepancy search (Harvey and Ginsberg, 1995). It first stays close to the move ordering above, then allows more and more departures from it, until the search is complete.

`dfs_solver.py` is a second solver, written apart from `solver.py` and sharing no code with it except the rules engine. It is a plain depth-first search over `rules.step` with a table of states found lost, and by default its only bound is the time bound, written again as a schedule: count the plays as unit jobs on one machine, each released when its card is drawn, and schedule greedily. Greedy is optimal here. At the last slot where greedy has nothing to play, every unfinished color waits for a card not yet drawn, so no schedule has played more by then, and greedy plays in every later slot. The same file has two certificates that keep track of who holds what:

- **Last card.** The player who draws the last card has one turn left, and it is the last turn of the game. A won game ends with a 5, so a last card of lower rank is never played in a won game, and the deal can be won only if it can be won with that card thrown away. The test makes the last card dead and applies the time bound and the pooled game (written again) to the changed deal.
- **Free-order game.** Give the team unlimited tokens and lives. Then a pass is always allowed, so the players may act in any order until the last card is drawn, and a misplay is a discard. Each drawn card stays in the hand of the player who drew it, and after the last draw the remaining turns go in the real order, each player playing at most one card from their own hand. Every winning line of the real game, with its passes left out, wins this game. Unlike the pooled game, it keeps the hands apart and keeps the order of the final round. It is decided by search, with the time bound and the pooled game as cuts, so on large deals it can run out of budget.

### Small decks, every deal

`upper_bound.py --exact` enumerates every deal of a reduced deck. A deal is counted once per ordering of the shuffled deck, so the counts below are out of the number of distinct orderings.

| Deck | Players and hand | Winnable orderings | Share |
|---|---|---:|---:|
| one color, 10 cards | one player, hand 1, no hints | 5,934 of 75,600 | 0.0785 |
| one color, 10 cards | one player, hand 2 | 36,275 of 75,600 | 0.4798 |
| one color, 10 cards | one player, hand 3 | 59,484 of 75,600 | 0.7868 |
| one color, 10 cards | one player, hand 4 | 65,916 of 75,600 | 0.8719 |
| one color, 10 cards | one player, hand 5 | 66,360 of 75,600 | 0.8778 |
| one color, 10 cards | two players, hand 2 | 72,138 of 75,600 | 0.9542 |
| one color, 10 cards | two players, hand 3 | 72,006 of 75,600 | 0.9525 |
| two colors of ranks 1–3, 12 cards | two players, hand 3 | 3,326,400 of 3,326,400 | 1 |
| two colors of ranks 1–3, 12 cards | two players, hand 4 | 3,325,920 of 3,326,400 | 0.99986 |

The one-player rows use van den Bergh's model: one player, no hints, a discard always allowed. They match his Appendix A, row (3,2,2,2,1), to all four digits, and his count of 5,934 for a hand of one card. The other rows use the standard tokens and lives. These counts are exact. On every deal of four smaller decks, brute force gives the same answer as the fast engine. Brute force also rechecks every 200th deal of the rows above.

The certificates of `solver.py` do not explain every lost deal. With one color, two players and hands of two, 2,910 lost orderings fail the pooled or time bound, and 552 are lost only by turn order and tokens: search finds them lost, but none of those certificates does. One of them starts R1 R1 for the first player and R4 R5 for the second, with R1 R3 R3 R2 R2 R4 in the deck. With hands of three cards, 1,284 orderings are of this kind. A larger hand does not always help: hands of three leave a deck of four cards, and the team runs out of turns.

With two colors and hands of four, all 480 lost orderings are of this kind. In one, the first player holds R1 R1 R1 R2, the second R3 Y2 Y2 Y3, and the deck is R2 Y1 Y1 Y1. All six cards must be played with only four draws and two final turns, so every turn that draws must be a play. The second player must play R3, Y2 and Y3, and Y2 has to wait for a Y1 drawn late; the search confirms that the three plays do not fit into the turns that remain. The pooled game lets anyone play any held card in the final round, so it misses this.

The free-order game catches all of these. In every row of the table it proves every lost ordering lost, including the 552, 1,284 and 480 that the older certificates miss. Of those, the last-card test proves 240 of the 552 and none of the 1,284 or the 480; on the two-color deck the last card is often a 3, the top rank, and then the test says nothing. On every deal of these decks Solver B gives the same answer as `winnable` and reproduces every count in the table (`upper_bound.py --exact --second`); neither new certificate ever holds for a won deal.

### The full game, sampled

Plan H1 ([`experiments/plans/H1.md`](../experiments/plans/H1.md)) fixed this measurement before it was run: seeded deals 0 to 19,999 for each number of players, the official rules, and two solvers.

- **Solver A** is `solver.winnable`, as above, with 20,000 search nodes per attempt, then 300,000 for deals left undecided, then 5,000,000.
- **Solver B** is `dfs_solver.solve` with the time bound only, with 100,000 nodes, then 1,000,000.

A deal neither solver settles is "undecided". The winnable share then lies between the won share and the won-plus-undecided share, and the interval widens that range by sampling error (Wilson, 95%). The run used GitHub Actions at commit 2c2cc50 (reruns at 927ef71 and 43eb021); `results/h1.json` has a row for every deal.

| Players | Won | Lost | Undecided | Winnable share | 95% interval | Decided by both solvers |
|---:|---:|---:|---:|---|---|---:|
| 2 | 19,724 | 266 | 10 | 0.9862 to 0.9867 | 0.9845 to 0.9882 | 19,603 |
| 3 | 19,976 | 17 | 7 | 0.9988 to 0.9992 | 0.9982 to 0.9995 | 19,928 |
| 4 | 19,993 | 3 | 4 | 0.99965 to 0.99985 | 0.9993 to 0.9999 | 19,942 |
| 5 | 19,986 | 1 | 13 | 0.99930 to 0.99995 | 0.9988 to 1 | 19,923 |

The two solvers agree on every one of the 79,396 deals that both decide. Solver B decides fewer deals: its one bound cannot prove the deals that fail only storage or the pooled game, so it leaves most of those undecided, and some hard wins too. Seeds 0 to 1,999 give exactly what this README reported before from those seeds with Solver A alone: for two players 1,960 won, 38 lost and 2 undecided, and the same bound on the mean score, 24.9765 ± 0.008. Each seed gives the same order of cards for every number of players, so the four rows are not independent of one another; seed 18105, for instance, is lost with four players and with five.

Lost deals now exist with four and with five players, though rarely: 3 of 20,000 and 1 of 20,000. All fail the time bound. In the five-player deal (seed 18105) the first red 1 and the first yellow 1 are the 16th and 17th cards drawn, which leaves 13 draws and 5 final turns for the 10 red and yellow plays and the rest.

**Outside the plan.** The last-card and free-order certificates were added after the plan, and so was a third search, Solver B with all its tests as cuts, run on the deals the plan's steps left undecided. With them, every two-player deal is settled.

| Players | Lost | Undecided | Winnable share | 95% interval | Mean score at most |
|---:|---:|---:|---|---|---|
| 2 | 276 | 0 | 0.9862 | 0.9845 to 0.9877 | 24.9838 ± 0.0021 |
| 3 | 17 | 7 | 0.9988 to 0.9992 | 0.9982 to 0.9995 | 24.9990 ± 0.0005 |
| 4 | 3 | 4 | 0.99965 to 0.99985 | 0.9993 to 0.9999 | 24.9998 ± 0.0002 |
| 5 | 1 | 12 | 0.99935 to 0.99995 | 0.9989 to 1 | 24.9999 ± 0.0001 |

The eleven deals that the 2,000-deal run left undecided are now settled. Two-player seeds 202 and 1725 are lost: Solver A proves it with 5,000,000 nodes, and both fail the last-card and free-order tests. Of the other nine, six are won: five-player seeds 474, 545, 744 and 1219 by Solver A with 5,000,000 nodes, 612 by Solver B, and 987 by Solver B with all its cuts. Three stay undecided: seed 1144 with three players and seed 1493 with four and with five.

The deals still undecided are seeds 1144, 4071, 6398, 11087, 17647, 18356 and 19522 (three players), 1493, 11087, 12041 and 12385 (four), and 1493, 7748, 11087, 11220, 12385, 13315, 14194, 14915, 16269, 17008, 18356 and 19522 (five). No certificate proves any of them lost. The free-order game is won for 13 of them, and its search ran out of budget on the other 10.

**Certificates.** Each lost deal, counted under every certificate that proves it (so a row can add up to more than its total):

| Players | Lost | Time | Storage | Pooled game | Last card | Free-order game | None |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 276 | 167 | 15 | 258 | 259 | 276 | 0 |
| 3 | 17 | 17 | 0 | 17 | 17 | 17 | 0 |
| 4 | 3 | 3 | 0 | 3 | 3 | 3 | 0 |
| 5 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |

The free-order game proves every lost deal. The older certificates miss 18 two-player deals, the ones lost to turn order. These 18 have a common shape. Each needs a play on both turns of the final round: the free-order game with the final round pooled, so that any player may make its two plays from any hand, wins all 18, and with one pooled final play it loses all 18. And in 17 of them the last card of the deck is a 4. By the last-card test that 4 is never played, so the other 4 of its color must wait in a hand. The last-card test alone proves 16 of the 18.

The best score on a lost two-player deal is 24 for 225, 23 for 35, 22 for 5 and 20 for 1; deals lost only by a certificate outside the plan are counted at 24, an upper bound. So no strategy averages more than 24.9842 ± 0.0020 points with two players, counting the plan's undecided deals as 25, or 24.9838 ± 0.0021 with every deal settled. For three, four and five players the same bound is 24.9990 ± 0.0005, 24.9998 ± 0.0002 and 24.9999 ± 0.0001.

Cheating bots that see their own cards but not the deck give lower bounds on the same shares. Quuxplusone's [CheatBot](https://github.com/Quuxplusone/Hanabi) wins 93.92%, 98.37%, 97.94% and 96.47% of games for two to five players. The shares above sit at or above those, as they must.

**Departures from the plan.**

- The plan's Solver B was a CP-SAT model, with the depth-first search as the fallback if CP-SAT could not decide full deals in about a minute each. `solver_cp.py` is that model. On all 540 deals of three small decks its verdicts match brute force, but on five full two-player deals it reached no verdict within 60 seconds each, so Solver B is the fallback. OR-Tools is declared at the top of the file, as inline metadata, rather than in a `hanabi` dependency group, because adding it to the shared lock file would downgrade protobuf for another folder.
- The plan asked for the fallback to be written without reading `solver.py`. It was written after `solver.py` had been read, so its move order follows the same idea (plays first, then discards of useless cards, then passes). It shares no code with `solver.py`.
- The plan gave Solver B a time limit without a value. The node budgets above were chosen before the run.
- The plan's last step, Solver A with 5,000,000 nodes, ran in two batches. The first also ran the extra attempts and was cancelled when those proved too slow on some deals; its finished records were kept, and the 38 deals it had not finished ran again with Solver A alone. With the 40 jobs of the main run, this used 61 runner jobs, one more than the plan's cap; one of the 61 failed at once on a script error.
- Before the plan was read, an exploratory run classified seeds 0 to 21,999 with Solver A alone, with budgets up to 3,000,000 nodes. It is kept, labeled exploratory, in `results/h1-exploratory.json`. It uses the same seeds as the plan, which the shared method discourages; because both solvers are deterministic, the confirmatory run can only repeat its verdicts, and on every deal both runs settle they agree. The two new certificates were found by looking at the exploratory run's lost deals, so their counts above are exploratory in that sense too.

### Where published agents stand

The bound is optimistic. Real players do not see the deck order, so the best real strategy scores lower. Still, it says how much of each agent's gap to 25 is out of reach for anyone.

| Agent, two players | Mean | Perfect | Gap to 25 | Share of the gap that no agent can close |
|---|---:|---:|---:|---:|
| SAD with auxiliary task (Hu and Foerster, 2020) | 24.08 | 56.1% | 0.92 | 1.7% |
| OBL level 4 (Hu et al., 2021) | 24.10 | | 0.90 | 1.8% |
| IPPO, entropy 0.05 (Forkel et al., 2025) | 24.48 | | 0.52 | 3.0% |
| SAD with SPARTA search (Lerer et al., 2020) | 24.61 | 75.5% | 0.39 | 4.1% |
| o3, minimal prompt (Ramesh et al., 2026) | 15.9 | | 9.1 | 0.2% |

The last column is (25 − 24.9842) divided by the gap, with 24.9842 the two-player bound of plan H1 above; it carries that bound's uncertainty of about an eighth of its size. The 2,000-deal bound used before, 24.9765, gave shares about half again as large. Language models trail by a wide margin. In "Sparks of Cooperative Reasoning" (arXiv:2601.18077), the best of 17 models averages about 15 to 18 points over 10 seeded games per setting, and in LLM-Hanabi (arXiv:2510.04980) the best five-player score is DeepSeek-R1's 30.00 on a 100-point scale. The leaderboard of the AH2AC2 challenge (arXiv:2506.21490) reports 21.04 for OBL level 4 and 9.91 for DeepSeek-R1 with H-group conventions when paired with human proxies in two-player games. With five players the information strategy's gap is 0.32 points and hanabi.rs's is 0.08; only one five-player deal in 20,000 is lost, so in principle nearly all of it can be closed.

## Language-model players

Experiment H2 ([plan](../experiments/plans/H2.md), with two dated amendments) seats the same hosted model in both chairs of a two-player game, on seeded deals 0 to 29, and compares each score with the best score that the deal allows in the open game. The solver wins all 30 deals, so every deal's ceiling is 25 and the gap is 25 minus the score.

`llm.py` writes, as text, everything that one player may know: the board, the other hand, the hints on each own card, the recent actions and the legal moves. With `deduction=True` it also lists the values each own card can still be, after hints and card counting, like the "Sherlock" setting of Ramesh et al. The model's reply must end with a move such as `PLAY 2` or `HINT 1 RED`. A reply with no legal move gets a fallback: the first legal discard, or, when all eight tokens are on the table, the first legal move, a blind play of card 1.

The models are gpt-oss-120b (reasoning effort medium, through Novita) and DeepSeek-V3.2 (its default mode without reasoning, through DeepInfra), on Hugging Face Inference Providers at temperature 0 and seed 0, with replies capped at 4,096 tokens. Each game was played once.

| Two players | Deals | Gap to the ceiling | Mean score | Perfect games | Lost all lives | Fallbacks per game | Cost |
|---|---:|---|---:|---:|---:|---:|---:|
| gpt-oss-120b, deduction on | 30 | 18.63 (16.70 to 20.57) | 6.37 | 0 | 12 | 15.0 | $1.38 |
| gpt-oss-120b, deduction off | 24 | 18.46 (16.58 to 20.46) | 6.54 | 0 | 8 | 12.9 | $1.04 |
| DeepSeek-V3.2, deduction on | 30 | 25 | 0 | 0 | 30 | 0 | $0.53 |
| DeepSeek-V3.2, deduction off | 30 | 25 | 0 | 0 | 30 | 0.03 | $0.48 |

Intervals are 95% percentile bootstrap intervals over deals (10,000 resamples, seed 0). With no perfect game in 30, the Wilson interval for the perfect share runs from 0 to 11%. DeepSeek-V3.2 lost its three lives in every game, so its gap is 25 on every deal; the share of such games is 30 of 30, with a Wilson interval of 89% to 100%.

- **The primary result.** With deduction on, gpt-oss-120b plays 18.6 points below the ceiling (16.7 to 20.6). DeepSeek-V3.2 plays 25 points below it. The plan does not rank the two models.
- **Deduction made no difference that these games can see.** Paired on deals 0 to 23, gpt-oss-120b's gap with deduction off minus its gap with deduction on is −0.79 points (−3.71 to 2.25). A gain from deduction larger than 2.25 points lies outside the interval. For DeepSeek-V3.2 the difference is 0 on every deal.
- **Where gpt-oss-120b lost points.** All 450 fallbacks with deduction on came from replies that reached the 4,096-token cap before writing a move: 24% of turns. A reply averaged 2,773 tokens. The 18 games that kept a life averaged 10.6 points, and no game scored more than 13.
- **Where DeepSeek-V3.2 lost points.** It wrote a legal move in 1,545 of its 1,546 replies, so its losses are misplays. In the one game read through (seed 0, deduction on), a life was lost on a card about which the hints said only that it was not yellow; the reply called it a 1 that was safe to play.
- **Comparison, without new runs.** The ceiling is 25 on these deals, so the gaps compare with the "Gap to 25" column above: 0.39 to 0.92 for the reinforcement-learning agents and 9.1 for o3 with a minimal prompt. Those come from other deals, and for o3 another prompt. The information strategy at three players trails its ceiling by about 2.7 points (22.254 against 24.998).
- **Not run.** The three-player games with a model, a secondary part of the plan, were not played: the budget ran out. The run of gpt-oss-120b with deduction off covers 24 deals instead of 30, for the same reason. Both cuts were written into the plan before the last run's results were read.
- **The pilot** (seeds 1000 to 1003, exploratory) used the plan's original cap of 512 tokens. Most replies ran out of tokens before writing a move, the fallback's blind plays used up the lives, and all eight games scored 0 within 15 turns. The plan's amendment gives the counts.

The spending is metered, not billed: the providers' usage objects report tokens but no cost, so each call is priced at the provider's listed price times its reported tokens. The ten runs, pilot included, total $3.60 by that meter. Two probe jobs made about $0.002 of calls, and the CPU jobs that ran the games cost about $0.05 at $0.01 an hour. The cap was $4.

Limits: the prompt was fixed in advance and not tuned; every seat holds the same model, so this is self-play, not play with strangers; temperature 0 does not make these providers deterministic (one probe prompt sent twice to DeepSeek-V3.2 on DeepInfra at temperature 0 and seed 0, once directly and once through the router's automatic choice, got two different replies); and the 4,096-token cap still cuts off a quarter of gpt-oss-120b's turns. The cap and the prompt are part of what is measured.

Per-game scores, token counts and costs are in `results/h2-runs/`, one folder per run, and `results/h2.json` holds every number above. Every prompt and reply is kept in a private dataset, with the commit that each run used.

## Checks

- The rules engine has tests for each rule above, including the extra round after the last card and the score of 0 on the third life.
- The information strategy matches HanSim game by game on 10,000 decks; a test checks the first 100. A test also checks the partition of the paper's Figure 6.
- On every deal of four small decks, the two solver engines give the same answer, for the maximum score and for every lower target. Every won deal's line is replayed by the rules engine.
- Each certificate is checked against brute force on those decks: no deal with a certificate is won.
- Solver B (`dfs_solver.py`) agrees with brute force on every deal of the four smaller decks, with and without its bounds, and with `winnable` on every deal of the decks in the table. The two new certificates never hold for a won deal there. A test checks greedy scheduling against an exhaustive search of the schedule on 200 random deals.
- On the full game, Solver A and Solver B give the same verdict on every one of the 79,396 deals both decide (plan H1, below). Every win of either solver is replayed by the rules engine.
- The constraint model of `solver_cp.py` agrees with brute force on every deal of three small decks (540 deals); a test checks some of them when OR-Tools is installed.

The two solvers share the rules engine, so a bug there would fool both. The rules engine has its own tests, and the information strategy matches HanSim game by game, which checks the rules from outside.

## Open questions

- Settle the 23 deals still undecided (listed above). No certificate here proves any of them lost, and 13 are won in the free-order game, so many may be won; a search guided by the free-order game's winning line might find a real one.
- The free-order game drops tokens. No full deal here is lost to tokens alone, but some small decks are: with three players, hands of one card and one token, 24 of the 180 orderings of a four-card deck are lost though the free-order game wins them.
- The best real (not open-information) score for each deal is unknown; Endgame-Analyzer on [git.abstractnonsen.se](https://git.abstractnonsen.se/Hanabi/Endgame-Analyzer) solves the case where players know their own cards but not the deck, for endgames.

## Usage

Everything except `solver_cp.py` and `hosted.py` uses only the standard library; `solver_cp.py` needs OR-Tools (`uv run --with ortools ...`), and `hosted.py` needs the `hanabi` dependency group (`huggingface_hub`). Times are from one run on a shared two-core machine.

```bash
uv run python -m hanabi.information -n 2000               # five players, about 1 min
uv run python -m hanabi.information -n 100000 --hansim-decks   # the 100,000-game row (35 min)
uv run python -m hanabi.information -n 10000 --cheat --hansim-decks
uv run python -m hanabi.information -n 10000 --players 3 4
uv run python -m hanabi.hansim_check -n 10000            # needs git and g++
uv run python -m hanabi.upper_bound --exact --check 200   # small decks (about 5 min)
uv run python -m hanabi.upper_bound --sample 2 2000       # about 1 min; 4 or 5 players take 2
uv run python -m hanabi.upper_bound --sample 5 0 --budget 300000 --seeds 474 545   # about 1 min each
uv run python -m hanabi.upper_bound --exact --second    # Solver B on the small decks too
# plan H1, one block of 2,000 seeds (the run used 40 blocks on GitHub Actions)
uv run python -m hanabi.upper_bound --sample 2 2000 --start 4000 --solver both --json p2-4000.json
uv run python -m hanabi.upper_bound --sample 3 0 --seeds 1144 --rerun --budgets 5000000 --json p3-rerun.json
uv run python -m hanabi.h1 --runs <folder of blocks> --out hanabi/results/h1.json --all-rows
uv run python -m hanabi.upper_bound --sample 5 20000 --budget 300000   # a bigger run, hours
uv run --group hanabi python -m hanabi.h2 play --model openai/gpt-oss-120b --provider novita \
    --reasoning-effort medium --seeds 0-29 --deduction on --max-tokens 4096 --cap 2 --out OUT
uv run python -m hanabi.h2 report --runs hanabi/results/h2-runs --out hanabi/results/h2.json
```

`h2 play` needs `HF_TOKEN` and network access to Hugging Face; the runs above were launched with `experiments/hfrun.py` on Hugging Face Jobs. `h2 report` runs here in a few seconds.

## Files

| File | Contents |
|---|---|
| `rules.py` | Cards, decks, the rules as small functions, seeded deals, HanSim's shuffle |
| `information.py` | The information strategy of Cox et al., following HanSim |
| `hansim_check.py` | Builds HanSim from GitHub and compares scores game by game |
| `solver.py` | Open-game solvers (brute force and fast), the bounds, replay of winning lines |
| `upper_bound.py` | Exact counts on small decks, sampled verdicts on the full game, the H1 records |
| `dfs_solver.py` | Solver B: a second open-game solver, and the last-card and free-order certificates |
| `solver_cp.py` | The open game as a time-indexed constraint model for CP-SAT (small decks only) |
| `h1.py` | Merges the H1 records, checks that the solvers agree, and writes the summary |
| `results/h1.json` | Plan H1: the summary and one row per deal (players, seed, verdicts, certificates, best score) |
| `results/h1-exploratory.json` | The exploratory run before the plan: summary, and the deals not found won |
| `llm.py` | Text observations and move parsing for language-model players |
| `hosted.py` | Hosted models through Inference Providers, with retries and a cost meter |
| `h2.py` | Experiment H2: plays the games and writes the report |
| `results/` | `h2.json` and the per-game summaries of each H2 run |

## Sources

- C. Cox, J. De Silva, P. DeOrsey, F. H. J. Kenter, T. Retter, J. Tobin, "How to Make the Perfect Fireworks Display: Two Strategies for Hanabi", *Mathematics Magazine* 88(5), 2015. Code: github.com/rjtobin/HanSim (GPL; not copied here).
- M. van den Bergh, "Hanabi, a co-operative game of fireworks", bachelor thesis, Leiden University, 2015.
- J.-F. Baffier, M.-K. Chiu, Y. Diez, M. Korman, V. Mitsou, A. van Renssen, M. Roeloffzen, Y. Uno, "Hanabi is NP-hard, even for cheaters who look at their cards", *Theoretical Computer Science* 675, 2017.
- B. Bouzy, "Playing Hanabi Near-Optimally", *Advances in Computer Games*, 2017. He relaxes the hint rule (a hint may touch no card) and reports 24.74 for his "hat information" player with five players and 81.30% perfect games, and 92% perfect games with a depth-one search.
- N. Bard et al., "The Hanabi Challenge: A New Frontier for AI Research", *Artificial Intelligence* 280, 2020 (arXiv:1902.00506).
- H. Hu, J. Foerster, "Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning", ICLR 2020; A. Lerer, H. Hu, J. Foerster, N. Brown, "Improving Policies via Search in Cooperative Partially Observable Games", AAAI 2020; H. Hu et al., "Off-Belief Learning", ICML 2021.
- J. Forkel, C. Ruhdorfer, M. Beukman, A. Bulling, J. Foerster, "High entropy leads to symmetry-equivariant policies in Dec-POMDPs", arXiv:2511.22581, Table 1.
- M. Ramesh et al., "Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents", arXiv:2601.18077; F. Liang et al., "LLM-Hanabi", arXiv:2510.04980; T. Dizdarević et al., "Ad-Hoc Human-AI Coordination Challenge", arXiv:2506.21490.
- W. D. Harvey, M. L. Ginsberg, "Limited Discrepancy Search", IJCAI 1995.
