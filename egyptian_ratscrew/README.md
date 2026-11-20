# Egyptian Ratscrew

How likely is a challenge to be answered, how long does a game last, does it always end, and how much does slapping speed matter?

- With no slaps, a challenge is answered 29% of the time after a jack and 76% after an ace. Following the whole exchange, the player who laid a jack keeps the pile 81% of the time, and the player who laid an ace half the time. Exact, by recursion on the unseen cards.
- Two players who slap equally fast take a median of 289 cards per game, and four take 1,051. After 300 cards, 27% to 28% of the remaining two-player games end within each further 100 cards, so the tail is geometric rather than a power law. Simulated, 10,000 games each.
- Random slaps do not make every game end. With the six cards 2 2 3 3 J J, 16 of the 720 orders of the deck loop forever without a single slappable pile. A game ends with probability 1 exactly when it cannot reach such a trapped position, and then the chance that it lasts more than t cards is at most a constant times ρ^t for an explicit ρ < 1, so the tail must be geometric. Proved by hand, and exact for eight small decks by searching every position. For the full deck it is not known whether a trapped position exists. Every known loop of Beggar-my-neighbour lays a royal pair or sandwich, which can always be slapped; see [Does every game end?](#does-every-game-end).
- Winning 55% of the slaps wins 64% of the games, and winning 70% wins 90%. On the logit scale the win chance is close to a straight line, 5.50 times the difference in shares. Simulated, 20,000 games at each of nine shares.
- Laying the first card is a small disadvantage. With equal shares the first of two players wins 49.77% of games (95% interval 49.67% to 49.86%, 1,000,000 games). With four players no seat effect is detected (200,000 games). With slaps off, the game is Beggar-my-neighbour, and on random deals the first player wins 49.87% (49.77% to 49.97%, 1,000,000 deals). That is the direction [Andorno et al. (2026)](https://arxiv.org/abs/2602.23406) report for smaller decks. The same deals match [Truman Collins's](http://tkcs-collins.com/truman/byn/byn.shtml) figures from 369.4 billion deals: 2.695% end after 11 tricks against his 2.70150%, and 51.95% after 6 to 28 tricks against his "just over 50%".
- Misslaps, with a one-card penalty as in the [Bicycle rules](https://bicyclecards.com/how-to-play/egyptian-rat-screw), shift the win curve to the right; its slope on the logit scale falls only from 5.5 to 5.0. A fast player who misslaps after 1% of the cards that cannot be slapped needs 51.2% of the slaps to win half the games against a careful one; at 5% they need 56.1%, and at 10%, 61.7%. Simulated, 20,000 games per point; the break-even shares are read off fitted lines.
- For three and four players, give each player the strength exp(2.75 n s), where s is their slap share and n the number of players, and a player's chance of winning is their share of the total strength. The constant comes from the two-player games alone. It predicts win chances on 120 random share vectors with a root mean square error of 0.017 (three players) and 0.022 (four), against 0.005 from sampling alone. A power law refitted for each number of players does better (0.010 and 0.009). The entropy and the sum of squares of the shares rank the favorite's chances poorly. Simulated, 4,000 games per vector.

![Three piles of cards, bottom card on the left: 5 K 7 7 (a pair, slappable), 9 7 K 7 (a sandwich, slappable) and 9 7 K 8 (not slappable).](figures/slaps.svg)

*Piles that can and cannot be slapped, drawn by `figures.py`. Royals are shaded; the matching cards of a pair or sandwich are outlined.*

## Results

| Result | How it is known |
|---|---|
| The challenger keeps the pile with chance 0.811 after a jack, 0.674 after a queen, 0.574 after a king and 0.503 after an ace. | Exact, by recursion on counts of unseen cards; checked by brute force on small decks and by simulation (20,000 exchanges per royal, seed 1) |
| Counting only the first answer, as simple arithmetic does, gives 0.706 to 0.236: 13% below the exact answer for a jack and under half of it for an ace. | Exact |
| Two players with equal shares take a median of 289 cards, four take 1,051. Long two-player games end at a steady 0.0031 per card (0.0030 to 0.0032 after 300 cards). | Simulated, 10,000 games each, seeds 2 and 4 |
| Six cards, 2 2 3 3 J J, can loop forever with no slappable pile: 16 of the 720 orders deal into such a trap. | Exact, every position |
| Without a reachable trap the tail is geometric. On 2 2 3 3 J Q the exact rate is 0.891 per card and the mean 7.78 cards; 20,000 simulated games give 7.69 to 7.80. | Proved by hand; exact for small decks; simulated check, seed 5 |
| Whether some deal of the full deck reaches a trap. | Not known |
| Winning 55% of the slaps wins 64% of the games, 70% wins 90%; the scaled logistic strength predicts three- and four-player win chances with root mean square errors of 0.017 and 0.022. | Simulated, with seeds and intervals in `runs.json` |
| The first of two players wins 49.77% (49.67% to 49.86%); with slaps off, 49.87%, in line with Andorno et al. (2026) and Collins. | Simulated, 1,000,000 games each |

The notebook works through these in the order of Peter Norvig's [War notebook](https://github.com/norvig/pytudes/blob/main/ipynb/war.ipynb): simple arithmetic, brute force, simulation, and an exact method on an abstract deck, and it ends with a comparison of the two games. In War the arithmetic estimate of a sweep is within 1.7% of the exact answer; in Ratscrew the arithmetic misses the counter-challenges, which matter most after an ace. A game of War, in Norvig's simulation, has a median of 224 plays of at least two cards each, longer than a two-player game of Ratscrew.

`game.py` holds the rules and the exact recursion, `termination.py` the game as a Markov chain on positions, and `egyptian_ratscrew.ipynb` works through the questions. `study.py` plays the large runs in seeded chunks and records them in `runs.json`; the notebook reads that record, and the tests replay parts of it. Reaction times are not modeled: each slap goes to a player with a fixed chance. With slapping turned off the game is Beggar-my-neighbour, so the tests check the rules against record deals from [Richard Mann's list](https://richardpmann.com/beggar-my-neighbour-records.html), which end after exactly the published 8,344 and 7,225 cards, and against the first deal known never to end ([Casella et al., 2024](https://arxiv.org/abs/2403.13855)). For two older records the tricks and cards counted here, 712 and 6,913, agree with the BeggarMyPython checker that Mann's list links to, and differ by one from the list itself.

```bash
uv run python -m egyptian_ratscrew.game
uv run --group egyptian_ratscrew python -m egyptian_ratscrew.figures  # the SVGs in figures/
uv run --group egyptian_ratscrew --with jupyter jupyter lab egyptian_ratscrew/egyptian_ratscrew.ipynb
uv run python -m egyptian_ratscrew.study  # plays any run missing from runs.json; under an hour on two cores
uv run python -m egyptian_ratscrew.termination  # small decks, loops and tail rates; about five minutes
uv run python -m egyptian_ratscrew.termination --template  # adds the template search; ten more minutes
```

## Does every game end?

The geometric tail above is a simulated fact. This section asks when it is a theorem. The short answer: slaps bring chance into the game only when a pile can be slapped, so a deal whose play never offers a slap behaves as Beggar-my-neighbour and can loop forever. When no such position can be reached, the game ends with probability 1, and the tail is geometric with a rate that can be written down.

### The game as a Markov chain

A **position** is what `play_game` keeps track of before each card: the hands in order, the pile, whose turn it is, which player (if any) laid the royal now being answered, and how many cards are still owed. The positions considered are those reachable from some deal. Time is counted in cards laid. When a card makes the pile slappable, player i takes it with chance s_i, where every share s_i is positive and the shares add up to 1. Every other step is fixed by the rules. So the game is a Markov chain on a finite set of positions, and the only chance in it is who wins each slap. The game **ends** at a position where no challenge is open and one player holds every card not on the pile.

A position **can finish** if some run of slap winners leads from it to the end. A position that cannot finish is **trapped**. Being trapped depends only on which slap outcomes are possible, never on the shares, since every share is positive.

**Theorem 1.** Consider any deck, any number of players and slap shares s_1, ..., s_n that are all positive, and let s be the smallest. Let x be a position, and let R be the set of positions reachable from x.

1. If R contains a trapped position, the game from x goes on forever with positive probability.
2. If R contains no trapped position, write L for the largest, over positions y in R, of the fewest cards from y to the end. Write δ for the smallest, over y in R, of the chance that the game from y ends within L cards. Then δ ≥ s^L. For every t ≥ 0, the chance that the game from x lasts more than t cards is at most (1 − δ)^⌊t/L⌋, where ⌊t/L⌋ is t/L rounded down. If δ < 1, that bound is at most ρ\*^t / (1 − δ), where ρ\* = (1 − δ)^(1/L) < 1.

*Proof.* For the first part, the trapped position is reached with positive probability, and from it no run of slap winners reaches the end. For the second, take y in R. A shortest way from y to the end lays at most L cards. Each card on it is either forced or a slap won by a named player, which has chance at least s, so the game follows that way with chance at least s^L. This gives δ ≥ s^L. Wherever the game stands after mL cards, it is at a position of R. From there it ends within L more cards with chance at least δ, whatever happened before, because the chain forgets its past (the Markov property). So the chance of lasting more than (m + 1)L cards is at most 1 − δ times the chance of lasting more than mL cards, and induction on m gives (1 − δ)^m. For any t, take m = ⌊t/L⌋, which is at least t/L − 1. ∎

The second part is the standard fact that a finite absorbing Markov chain is absorbed with probability 1 (Kemeny and Snell, *Finite Markov Chains*, 1960, chapter 3), with the constants written out. The bound ρ\* is crude. The exact rate is ρ, the largest eigenvalue of the matrix Q of one-card chances among unfinished positions. The largest row sum of Q^t is the largest chance, over starting positions, of still playing after t cards, and its t-th root tends to ρ (Gelfand's formula for the spectral radius). By the Perron–Frobenius theory of nonnegative matrices (Seneta, *Non-negative Matrices and Markov Chains*, 1981, chapter 1), ρ is itself an eigenvalue, real and nonnegative, which is why power iteration finds it.

**Theorem 2.** Suppose that play from a position never makes the pile slappable and returns to a position it has been in. Then the position is trapped, whatever the shares.

*Proof.* Without a slappable pile there is no chance, so play is fixed, and a fixed play that returns to a position repeats forever. ∎

A trap needs at least six cards: no deck of five or fewer cards has a trapped position, for two or three players. Among decks of six cards, exactly those made of two jacks and four number cards, with no rank three or four times, have one. Both are computed by `decks_with_traps`. One example is the deck 2 2 3 3 J J, dealt as J 2 3 to player 0 and 2 J 3 to player 1, with player 0 laying first. Player 0 lays the J, player 1 pays the 2, and player 0 takes the trick J 2. The hands are now 2 3 J 2 and J 3. Player 0 leads the 2, player 1 answers with the J, player 0 pays the 3, and player 1 takes 2 J 3. Player 1 leads the 3, player 0 answers with the J, player 1 pays the 2, and player 0 takes 3 J 2. The hands are 2 3 J 2 and J 3 again, with player 0 to lead. No trick ever holds a pair or a sandwich. In the usual notation this is the deal J - - against - J -, the looping game of Beggar-my-neighbour with which [Casella et al. (2024)](https://arxiv.org/abs/2403.13855) begin their construction. With a 2 in place of the second 3, the trick 2 J 2 would be a sandwich and offer a slap.

![The deal J 2 3 against 2 J 3: the hands after each trick, and the trick picked up. After the third trick the hands are those after the first, so play loops with no pair or sandwich.](figures/trap.svg)

*The six-card trap. Each row shows both hands, top card on the left, and the trick just picked up; the second and fourth rows are the same position.*

### Small decks, every position

For small decks the whole chain can be built: start from every order of the deck, deal it round the table, and follow every slap outcome. Then a backward search from the ended positions finds every position that can finish. The table gives, for two players with equal shares, the number of positions, the orders of the deck (cards of equal rank told apart) that deal into a trap, the constants L and δ of Theorem 1, the bound ρ\*, the exact rate ρ per card, and the exact mean length of a game.

| deck | positions | trapped orders | L | δ | ρ\* | ρ | mean cards |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 2 3 3 J J | 804 | 16 of 720 | | | | | |
| 2 2 3 3 J Q | 1,528 | 0 of 720 | 12 | 1/4 | 0.9763 | 0.8909 | 7.78 |
| 2 2 3 3 4 4 J J | 27,222 | 0 of 40,320 | 25 | 1/8 | 0.9947 | 0.9168 | 16.24 |
| 2 2 3 3 4 4 J Q | 53,154 | 0 of 40,320 | 30 | 1/4 | 0.9905 | 0.8963 | 12.57 |
| 2 2 2 3 3 3 J J | 6,958 | 0 of 40,320 | 23 | 1/16 | 0.9972 | 0.9174 | 17.98 |
| 2 2 3 3 J J Q Q | 28,284 | 0 of 40,320 | 22 | 1/8 | 0.9939 | 0.8884 | 14.75 |
| 2 2 3 3 J J A A | 27,988 | 0 of 40,320 | 18 | 1/8 | 0.9926 | 0.8847 | 14.43 |
| 2 2 3 3 J Q K A | 99,348 | 0 of 40,320 | 18 | 1/2 | 0.9622 | 0.8706 | 9.33 |

With three players, 2 2 3 3 J J again has 16 trapped orders of 720 (84 trapped positions of 2,246). With three players there is no trapped position in the other three decks checked: 2 2 3 3 J Q, 2 2 3 3 4 4 J J and 2 2 3 3 4 4 J Q. The rates ρ for those three are 0.8327, 0.9327 and 0.8882.

Three things hold in every deck and number of players checked. They are computed, not proved for decks in general.

- The trapped positions are exactly those of Theorem 2: positions whose play never offers a slap and loops.
- From every position that can finish, some single player winning every slap finishes the game. A run of slaps won by one player is enough, and no mixing of winners is needed.
- The crude bound ρ\* lies well above the exact ρ, because δ is the chance of the single worst position.

The exact rates were checked in several ways. On the deck 2 2 3 3 J Q, power iteration agrees with a dense eigenvalue computation to 10^−9 for two share vectors, and the exact mean length agrees with the solution of the linear system for expected times. For t up to 6, a listing of every deal and every run of slap winners gives the same chances of lasting more than t cards. On the deck 2 2 3 3 4 4 J J, 100,000 simulated games (seed 1) give an ending chance per card, among games still going after 30 cards, of 0.0879 (95% interval 0.0861 to 0.0896). The exact value of the same quantity is 0.0871. The ending chance per card settles to its limit 1 − ρ = 0.0832 only after about 200 cards: the exact one-card ending chance after 100 cards is 0.0880, after 200 cards 0.0830 and after 400 cards 0.0833.

### The full deck

Theorem 1 applies to the full deck, but its condition cannot be checked by building the chain: the deals alone number 52!/(4!)^13, about 9.2 × 10^49 orders of the ranks. A trap would need one of two things: a loop of Beggar-my-neighbour in which no trick ever holds a pair or a sandwich, or a set of positions that every slap outcome stays inside without reaching the end. The small decks above have only the first kind. What is known about the first kind is this.

- [Casella et al. (2024)](https://arxiv.org/abs/2403.13855) give a deal that reaches a loop of 62 tricks. They also give two other full-deck positions that loop; for the second, three cards are removed from a 55-card version, as they describe. The loops lay 440, 624 and 628 cards, and 7 of the 62, 24 of the 72 and 15 of the 82 tricks hold a royal pair or a royal sandwich (Q - Q, for instance). Such a pile can be slapped whatever ranks the other cards have, so every pass round these loops offers slaps. Computed by `beggar_loop_tricks`.
- Their construction places "pieces" between jacks: player 0 holds piece J piece J piece J -, and player 1 holds J -. Taking every choice of three pieces from their list of 26 (Appendix C), in every order, and every way of making each royal of a piece a Q, K or A, gives 2,598,750 full decks. Of these, 103,992 loop, and every one of those loops lays a royal pair or sandwich. Computed by `template_search`, about ten minutes.
- With slaps, the looping deal leaves its loop. Giving its 36 other cards the ranks 2 to 10 at random and playing with equal shares, 10,000 of 10,000 games end, the longest after 3,551 cards. Simulated, seed 2.

So for the full deck, ending with probability 1 from every deal is supported by simulation and by the absence of slap-free loops among the loops checked. It is not proved. If it holds, Theorem 1 makes the tail geometric. The simulated rate is steady: among two-player games still going after 300 cards, the ending chance per card is 0.00318 (95% interval 0.00315 to 0.00321; 100,000 games, seed 2026), and after 600 cards it is 0.00316 (0.00311 to 0.00320). That is ρ ≈ 0.9968 per card, and 27.2% of the remaining games end within each further 100 cards (27.0% to 27.5%), in line with the 27% to 28% above. The small decks end far faster, with ρ near 0.9 per card, so their rates say nothing about the size of the full deck's rate. What they test is the method: on a deck where both can be computed, the simulated tail rate matches the exact one.

### What the slap randomness buys

Beggar-my-neighbour has no chance, so each deal either ends or loops. Whether some deal loops was asked by Conway and was open for decades. [Paulhus (1999)](https://doi.org/10.2307/2589054) built looping games for reduced decks. [Casella et al. (2024)](https://arxiv.org/abs/2403.13855) settled the question for the full deck with a deal that loops, and they list as still open how many loops can be reached from a deal and how many deals with equal hands loop. [Andorno et al. (2026)](https://arxiv.org/abs/2602.23406) find further loops with an automated search in standard and generalized settings. So termination for every deal is no longer an open problem; it is false.

Two papers add chance to games of this kind and prove that the expected length is finite. [Lakshtanov and Roshchina (2012)](https://arxiv.org/abs/1007.1371) show that War can loop under a fixed rule for the order in which a winner returns the two cards. If each player sometimes uses the other order, every position can reach the end, so the expected length is finite. [Aleksenko and Lakshtanov (2013)](https://arxiv.org/abs/1109.1460) prove the same for games of the Beggar-my-neighbour kind in which the player who leads each trick is chosen at random and each trick is shuffled before it goes under a hand. In both models chance acts at every trick. In Egyptian Ratscrew as modeled here, it acts only when a pile can be slapped. That is why the six-card trap exists here, while their theorems rule out traps in their models.

Precisely, random slaps buy three things and cannot buy a fourth.

1. The question becomes one of reachability: the game ends with probability 1 from every deal exactly when no trapped position can be reached from a deal (Theorem 1).
2. When that holds, the tail is geometric, with an explicit if crude bound, and the exact rate is the largest eigenvalue of the chain (Theorem 1 and the remark after it).
3. On small decks, a run of slaps won by one player finishes every game that can finish (computed).
4. Random slaps cannot rescue a position whose play never offers a slap (Theorem 2). For the full deck, a trap of that kind would be a loop of Beggar-my-neighbour, with the 36 other cards given ranks, in which no trick ever holds a pair or a sandwich of any rank. Every loop checked lays a royal pair or sandwich. Whether a loop without one exists is open as far as these sources show.

### How each claim is known

| claim | how it is known |
|---|---|
| Theorems 1 and 2 | proved by hand above |
| the six-card trap and its loop of six cards | hand trace above, and by `slap_free_cycle` |
| no trap with five or fewer cards; which six-card decks have one | exact: every deck of that size, by `decks_with_traps` |
| the table of small decks, the three-player figures, the three findings under the table | exact: every position built by `chain`, trapped ones found by backward search (`can_end`), constants by `uniform_bound`, rates by power iteration (`decay_rate`) |
| exact rates and lengths | checked against a dense eigenvalue solver, a linear solve and full enumeration on the deck 2 2 3 3 J Q (tests) |
| `moves` follows the rules of `play_game` | the same seeded games played both ways, for two to four players (tests) |
| royal slaps in the known loops, the template search | computed from the decks in Casella et al.; a fast loop finder checked against `beggar_loop_tricks` on a sample |
| tail rates for 2 2 3 3 4 4 J J and the full deck, games from the looping deal | simulated, with seeds and intervals as given |
| termination for the full deck | not known |
