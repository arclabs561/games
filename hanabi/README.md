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

`upper_bound.py --sample` classifies seeded deals 0 to 1,999 for each number of players, with a budget of 20,000 search nodes per attempt. The deals left undecided were run again with 300,000 nodes (`--seeds`). A deal that still no attempt settles is "undecided". The winnable share then lies between the won share and the won-plus-undecided share; the interval widens that range by sampling error (Wilson, 95%).

| Players | Won | Lost | Undecided | Winnable share | 95% interval |
|---:|---:|---:|---:|---|---|
| 2 | 1,960 | 38 | 2 | 0.9800 to 0.9810 | 0.973 to 0.986 |
| 3 | 1,995 | 4 | 1 | 0.9975 to 0.9980 | 0.994 to 0.999 |
| 4 | 1,999 | 0 | 1 | 0.9995 to 1 | 0.997 to 1 |
| 5 | 1,993 | 0 | 7 | 0.9965 to 1 | 0.993 to 1 |

The larger budget settled 3 of 4 undecided three-player deals, 9 of 10 four-player deals and 11 of 18 five-player deals, all as wins, and neither two-player deal. The undecided seeds are 202 and 1725 (two players), 1144 (three), 1493 (four), and 474, 545, 612, 744, 987, 1219 and 1493 (five).

Every lost deal in this sample has a certificate. For two players, 25 of the 38 fail the time bound (for example, the 1s of some color come too late), 4 fail storage, and 9 fail only the pooled game. The 4 lost three-player deals fail the time bound. Every lost deal also fails the pooled game, so in this sample turn order and tokens never decide a deal on their own. They do on small decks, as shown above.

The best score on a lost two-player deal is 24 for 30 deals, 23 for 7 and 22 for 1. So no strategy averages more than 24.977 ± 0.008 points with two players, counting undecided deals as 25. For three players the same bound is 24.998 ± 0.002. No four- or five-player deal in the sample was shown lost.

Cheating bots that see their own cards but not the deck give lower bounds on the same shares. Quuxplusone's [CheatBot](https://github.com/Quuxplusone/Hanabi) wins 93.92%, 98.37%, 97.94% and 96.47% of games for two to five players. The shares above sit at or above those, as they must.

### Where published agents stand

The bound is optimistic. Real players do not see the deck order, so the best real strategy scores lower. Still, it says how much of each agent's gap to 25 is out of reach for anyone.

| Agent, two players | Mean | Perfect | Gap to 25 | Share of the gap that no agent can close |
|---|---:|---:|---:|---:|
| SAD with auxiliary task (Hu and Foerster, 2020) | 24.08 | 56.1% | 0.92 | 2.6% |
| OBL level 4 (Hu et al., 2021) | 24.10 | | 0.90 | 2.6% |
| IPPO, entropy 0.05 (Forkel et al., 2025) | 24.48 | | 0.52 | 4.5% |
| SAD with SPARTA search (Lerer et al., 2020) | 24.61 | 75.5% | 0.39 | 6.0% |
| o3, minimal prompt (Ramesh et al., 2026) | 15.9 | | 9.1 | 0.3% |

The last column is (25 − 24.9765) divided by the gap, with 24.9765 the two-player bound above; it carries that bound's uncertainty of about a third of its size. Language models trail by a wide margin. In "Sparks of Cooperative Reasoning" (arXiv:2601.18077), the best of 17 models averages about 15 to 18 points over 10 seeded games per setting, and in LLM-Hanabi (arXiv:2510.04980) the best five-player score is DeepSeek-R1's 30.00 on a 100-point scale. The leaderboard of the AH2AC2 challenge (arXiv:2506.21490) reports 21.04 for OBL level 4 and 9.91 for DeepSeek-R1 with H-group conventions when paired with human proxies in two-player games. With five players the information strategy's gap is 0.32 points and hanabi.rs's is 0.08; the sample shows no lost deal, so in principle nearly all of it can be closed.

## Language-model players

No language-model API is reachable from the machine this was written on, so this folder reports no language-model results. `llm.py` provides what an experiment needs:

- `observation(state, player, events)` writes what one player may know: piles, tokens, lives, the discards, the other hands, the hints on each of the player's own cards, the last ten actions, and the legal moves. With `deduction=True` it also lists, for each own card, the values that remain possible after hints and card counting, like the "Sherlock" setting of Ramesh et al.
- `parse_action(reply, state)` reads the last move written as `PLAY 2`, `DISCARD 1`, `HINT 3 RED` or `HINT 3 4`, and returns `None` if it is missing or illegal.
- `play_text_game(rules, deck, complete)` plays a game with any function from prompt to reply, and counts the replies that could not be used.

Plan for an experiment, to be run where a model is available:

1. Use seeded deals 0 to 99 for each number of players, the same for every model, and report each model's mean, its perfect-game share and the number of unusable replies.
2. Run each model with and without `deduction`, to compare with the Watson and Sherlock settings of Ramesh et al.
3. Pair one model with partners that play the information strategy, to separate a model's own play from its ability to read its partners.
4. For each game, report the solver's verdict on the deal, so that lost deals are not counted against the model.

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

- Settle the deals still undecided (listed above). They are three-, four- and five-player deals that no certificate here proves lost, and most of them are won in the free-order game, so they may well be won; a search that uses the free-order game's winning line as a guide might find a real one.
- The free-order game drops tokens. No full deal here is lost to tokens alone, but some small decks are: with three players, hands of one card and one token, 24 of the 180 orderings of a four-card deck are lost though the free-order game wins them.
- The best real (not open-information) score for each deal is unknown; Endgame-Analyzer on [git.abstractnonsen.se](https://git.abstractnonsen.se/Hanabi/Endgame-Analyzer) solves the case where players know their own cards but not the deck, for endgames.

## Usage

Everything except `solver_cp.py` uses only the standard library; `solver_cp.py` needs OR-Tools (`uv run --with ortools ...`). Times are from one run on a shared two-core machine.

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
```

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
