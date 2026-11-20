# Which deals can reach 25?

The open game, the two solvers and their certificates, and exact counts on every deal of small decks. Plan H1 on the full game is in [h1.md](h1.md). Moved unchanged from the [Hanabi README](../README.md), which keeps the short answers.


With every card in view, a hint carries no information; it only passes the turn and costs a token. Van den Bergh (2015) calls this the open game and shows that a deal can be won in the real game only if it can be won in the open game. Baffier et al. (2017) prove that deciding this is NP-complete, even with hands of two cards, so the answer here comes from search.

`solver.py` has two engines. `brute_winnable` tries every legal action of the rules engine. `winnable` searches the open game: it tries plays first, writes every card that can no longer be played as one "trash" symbol, never discards the last copy of a needed card, and cuts off states that fail three bounds:

- **Time.** Each play or discard draws a card while the deck lasts, and after that each player has one turn. The i-th draw from now can only play cards already in a hand or among the first i − 1 cards of the deck.
- **Storage.** Once every copy of a needed card is out of the deck, and the card cannot have been played yet, some hand must hold it. A deal fails if more such cards are needed at one moment than the hands can hold.
- **Pooled game.** Let one player hold all the hands together, with no turn order and no tokens. There, playing a playable card at once is never worse (play it now and make the discard later), so the only choice is which card to discard when nothing is playable. Every winning line of the real game wins the pooled game.

Each bound that holds at the start of a deal is a certificate that the deal is lost. A won deal comes with its winning line, which the rules engine replays to 25. On full deals `winnable` uses limited discrepancy search (Harvey and Ginsberg, 1995). It first stays close to the move ordering above, then allows more and more departures from it, until the search is complete.

`dfs_solver.py` is a second solver, written apart from `solver.py` and sharing no code with it except the rules engine. It is a plain depth-first search over `rules.step` with a table of states found lost, and by default its only bound is the time bound, written again as a schedule: count the plays as unit jobs on one machine, each released when its card is drawn, and schedule greedily. Greedy is optimal here. At the last slot where greedy has nothing to play, every unfinished color waits for a card not yet drawn, so no schedule has played more by then, and greedy plays in every later slot. The same file has two certificates that keep track of who holds what:

- **Last card.** The player who draws the last card has one turn left, and it is the last turn of the game. A won game ends with a 5, so a last card of lower rank is never played in a won game, and the deal can be won only if it can be won with that card thrown away. The test makes the last card dead and applies the time bound and the pooled game (written again) to the changed deal.
- **Free-order game.** Give the team unlimited tokens and lives. Then a pass is always allowed, so the players may act in any order until the last card is drawn, and a misplay is a discard. Each drawn card stays in the hand of the player who drew it, and after the last draw the remaining turns go in the real order, each player playing at most one card from their own hand. Every winning line of the real game, with its passes left out, wins this game. Unlike the pooled game, it keeps the hands apart and keeps the order of the final round. It is decided by search, with the time bound and the pooled game as cuts, so on large deals it can run out of budget.

## Small decks, every deal

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
