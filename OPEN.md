# Open questions

70 open questions from 16 games' READMEs, collected by `tools/open_questions.py`. Edit the READMEs, not this file, then run `python3 tools/open_questions.py`; a test checks that this file is current.

Each status comes from the question's wording, by the rules in the script's docstring. It guides the choice of what to work on next and is not itself a finding.

| Status | Questions |
|---|---:|
| blocked on data | 9 |
| blocked on compute | 4 |
| partly answered | 12 |
| open | 45 |

No open-questions section: [Egyptian Ratscrew](egyptian_ratscrew/README.md), [Yahtzee](yahtzee/README.md).

| Game | Question | Status |
|---|---|---|
| [Common Words in Boggle](boggle/README.md) | **Findability and cancellation.** Whether a player finds a word, and whether an opponent finds it too, needs an experiment with people; the post describes one. | blocked on data |
| [Common Words in Boggle](boggle/README.md) | **Unseen words.** The exact count of possible words lets estimators of the number of unseen words (Good and Turing, Chao) be scored against the truth. That has not been done. | open |
| [Common Words in Boggle](boggle/README.md) | **Exact board probability of a word.** The chance of a whole board is a permanent. The chance that a board shows a given short word, such as "toe", is a finite sum over where the cubes carrying its letters land, but the paths overlap and the sum needs inclusion and exclusion. Only the simulated value is known here. | partly answered |
| [Common Words in Boggle](boggle/README.md) | **Other editions.** Big Boggle (5 by 5) and Super Big Boggle (6 by 6) have other cubes and minimum lengths. The published listings of the 5 by 5 set disagree on one cube. | open |
| [Common Words in Boggle](boggle/README.md) | **The best 5 by 5 board.** Boyan's board resists every one-square change. Changes of two squares, swaps and restarts, which Norvig lists, have not been tried. | partly answered |
| [Common Words in Boggle](boggle/README.md) | **The best board for `enable1.txt`.** Vanderkam's proof is for ENABLE2K. Whether `perslatgsineters` is also best for `enable1.txt` has not been checked. | partly answered |
| [Cannonballs](cannonballs/README.md) | Is the single ball the only square pile with a cube number of balls? Remainders rule out only two of the six cases for cubes. | partly answered |
| [Cannonballs](cannonballs/README.md) | Is there a short proof, using Pell equations and remainders, that 7,140 is the last number that is both a triangle and a tetrahedral pile? | open |
| [Cannonballs](cannonballs/README.md) | Which oblong piles hold a triangular number of balls? For a fixed short side, each extra ball of length adds a triangle, so the remainder method of the square case applies. | open |
| [Cannonballs](cannonballs/README.md) | How do the counts and touching pairs change for piles built in the hexagonal close packing? | open |
| [Chess: what a rating says, and what a player at a rating plays](chess_ratings/README.md) | How much of the gap between real and model settling comes from players who improve, and how much from strong players on new accounts? Splitting real players by their rating at game 61 against their first results would separate them. | open |
| [Chess: what a rating says, and what a player at a rating plays](chess_ratings/README.md) | Does Maia-3 expect fewer opening blunders when it sees the moves that led to a position? Running the common positions with each occurrence's history would tell. | open |
| [Chess: what a rating says, and what a player at a rating plays](chess_ratings/README.md) | Would Maia3-79M or Allie 2.1 estimate a rating from one game better than Maia3-5M, and is the error floor set by the model or by the noise in the rating itself (about 31 points)? | open |
| [Chess: what a rating says, and what a player at a rating plays](chess_ratings/README.md) | With one game a day instead of back to back, the deviation grows between games by 0.21436 σ² per day. How does that change the games needed to settle? | open |
| [Codenames](codenames/README.md) | Does the single-peak property of the value of a turn hold for every weighting? | open |
| [Codenames](codenames/README.md) | Why does a guesser trained on people make a worse giver? | open |
| [Codenames](codenames/README.md) | Would five epochs on the human turns alone give part of the teacher's gain, and does the gain grow with more teacher turns? | open |
| [Codenames](codenames/README.md) | Would language-model givers do better with a prompt that states the whole clue rule and the cost of the assassin, or with a retry after an illegal clue? | open |
| [Codenames](codenames/README.md) | Do language-model clues fare better with people than with embedding guessers? | blocked on data |
| [Codenames](codenames/README.md) | Would GRPO with an entropy bonus, more replies per board, or several gradient steps per batch keep exploring past the first listed clue? | open |
| [Codenames](codenames/README.md) | Why does gpt-oss-120b return empty replies on many test calls and almost no validation calls? | open |
| [Drive Ya Nuts](drive_ya_nuts/README.md) | How many classes of solvable sets are there when a nut may appear twice? A comment on the Stack Exchange question reports 6,901,100; it was not checked here. | partly answered |
| [Drive Ya Nuts](drive_ya_nuts/README.md) | How were the original nuts chosen? A designer who kept drawing until a set had one solution would find one after about 18.7 draws on average. | open |
| [Erase](erase/README.md) | A proof of Theorem 15 that does not check cases is still open. | open |
| [Erase](erase/README.md) | Do the thresholds for targets other than 10 hold for hands of more than 11 cards? The paper's induction to larger hands is worked out only for 10. | partly answered |
| [Erase](erase/README.md) | Does any target above 30 have a threshold above 9? Only 14 reaches 9 among the targets checked, because of one multiset of eight cards. | partly answered |
| [EV Nova and Naev](ev_nova/README.md) | Whether Naev's mean prices match the game. A running copy of Naev could check a few spobs. | blocked on data |
| [EV Nova and Naev](ev_nova/README.md) | Time. A jump in Naev takes flying time that depends on the system, and the source comments put a jump at about three periods and a landing at about one. Rates per period would need a model of flight time. | partly answered |
| [EV Nova and Naev](ev_nova/README.md) | Missions against trading. Cargo missions in both games pay for delivery and need no capital. They are not modeled. | open |
| [EV Nova and Naev](ev_nova/README.md) | Price drift in Naev, used on purpose. Prices in a system rise and fall together, with periods of a few hundred periods, so waiting for a good phase may beat flying at once. | open |
| [EV Nova and Naev](ev_nova/README.md) | Story changes. EV Nova's galaxy changes with control bits, and Naev's with its campaign. Only the start of an EV Nova game and Naev's base data are studied. EV Nova's disasters (`öops`), which shift one price for a while, are left out, as are hypergates and wormholes. | open |
| [EV Nova and Naev](ev_nova/README.md) | The prices of EV Nova's special commodities, and whether a player's rank changes commodity prices. | open |
| [EV Nova and Naev](ev_nova/README.md) | Legal limits. Some special commodities are contraband, and a player without enough standing with a Naev spob's faction cannot land there. | open |
| [FreeCell](freecell/README.md) | **The eight undecided levels.** Solver B can store 260 million positions on a runner with 16 GB, and that was not enough to decide them. A disk-backed search could decide them, and so could a win found by a better search. Deal 982 with two free cells is the hardest; Fish (2012) reports that it was intractable for a long time before it was shown impossible. | blocked on compute |
| [FreeCell](freecell/README.md) | **A short certificate with free cells.** No proof that 11982 is lost avoids listing positions. Sells (2026) tried counting free places at the moment a stuck card must move; he found that it fails once a column can be emptied. With a free cell, the only certificate here is the fence. A short argument like the stuck-card set would need to bound how long cards can stay in free cells. | partly answered |
| [FreeCell](freecell/README.md) | **Deals with no frozen card.** In 27 lost deals every card moves in some line. What makes deal 8927 lost with three free cells? | open |
| [FreeCell](freecell/README.md) | **Further deals.** Fish (2012) reports 82,126 deals lost with two free cells among the first 400,000. Pringle and Fish (2018) count 102,075 deals lost with four among all FreeCell Pro deals. Neither is rerun here. A survey of the first million deals with four free cells was started and cancelled before it finished; no result of it is reported. | blocked on compute |
| [Hanabi](hanabi/README.md) | Split the points between a team's score and its deal's ceiling into what hidden information costs and what play costs. That needs an agent that plays the real game well, scored on the same 3,079 decks: an H-group convention bot, or the challenge's human proxies, which are only reachable through its evaluation service. | blocked on data |
| [Hanabi](hanabi/README.md) | Trace the 67 human games whose drops a search could not settle, with larger budgets. | blocked on compute |
| [Hanabi](hanabi/README.md) | Settle the 23 deals still undecided (listed above). No certificate here proves any of them lost, and 13 are won in the free-order game, so many may be won; a search guided by the free-order game's winning line might find a real one. | partly answered |
| [Hanabi](hanabi/README.md) | The free-order game drops tokens. No full deal here is lost to tokens alone, but some small decks are: with three players, hands of one card and one token, 24 of the 180 orderings of a four-card deck are lost though the free-order game wins them. | partly answered |
| [Hanabi](hanabi/README.md) | The best real (not open-information) score for each deal is unknown; Endgame-Analyzer on [git.abstractnonsen.se](https://git.abstractnonsen.se/Hanabi/Endgame-Analyzer) solves the case where players know their own cards but not the deck, for endgames. | open |
| [Hanabi](hanabi/README.md) | Plan H4's policies were still improving when training stopped. Whether the gap between self-play and cross-play at α = 0.05 closes with more steps, as in Forkel et al., is open; 10⁹ steps a seed would take about 4.6 hours an arm on an A10G at the speed measured. | blocked on compute |
| [Hanabi](hanabi/README.md) | Whether playing the newest card is the habit that separates the clusters of low-entropy seeds could be tested by forcing or forbidding it in one policy and replaying the cross-play games. | open |
| [Maps](maps/README.md) | Does CIEDE2000 instead of Oklab distance change which shelf method wins? | open |
| [Maps](maps/README.md) | A NeRV sweep (λ KL(P‖Q) + (1 − λ) KL(Q‖P)) would trace the whole frontier between precision and recall on these data; λ = 1 is SNE, already implemented. | open |
| [Maps](maps/README.md) | How far apart are t-SNE's local minima on the Pokémon, over 20 seeds, in cost and in T and C? | open |
| [Maps](maps/README.md) | How do T and C of a SOM move with grid size, and does Kohonen's batch algorithm reach the same scores? | open |
| [Maps](maps/README.md) | How often does a random projection change which word a Codenames guesser picks? That asks about cosine rankings, a sharper test than distortion. | open |
| [Mastermind](mastermind/README.md) | How does the game choose a screen's words, and are duds chosen to share letters with the password? A list of real screens would settle whether the ENABLE screens here are easier or harder than the game's. | blocked on data |
| [Mastermind](mastermind/README.md) | ENABLE has many words that a game would not use, such as JAUK and WHIT. Would screens of common words change the table? | open |
| [Mastermind](mastermind/README.md) | Is there a rule for the first guess that keeps more of the best possible chance than minimax's 99.41%? | open |
| [Peggle: who is the better player?](peggle/README.md) | Who shoots first in a Peggle duel, and how large is that edge? The first-shooter edges here are assumed values, not measurements. | blocked on data |
| [Peggle: who is the better player?](peggle/README.md) | A player whose chance rises from 0.45 to 0.55 halfway through 1,000 games is usually declared the worse player by the 95% rule. Does some rule that looks only at wins and games avoid this, or does it need the order of the games? | open |
| [Pokémon](pokemon/README.md) | **Schmitz's typing duel.** Why his 27-typing equilibrium differs from the unique one here. His matchup table is an image; reading it would settle whether the game or the solver differs. | blocked on data |
| [Pokémon](pokemon/README.md) | **Weighted scores.** The notebook's sums treat every attacking type as equally likely. Weighting by Showdown usage would change which typings and Pokémon rank first. | open |
| [Pokémon](pokemon/README.md) | **Sets beyond damage.** Setup moves, recovery, hazards, items, Terastallization, and a policy that switches by choice. The six most used Pokémon do far better in full battles than under perfect switching, which suggests that what the sets leave out matters. | open |
| [Pokémon](pokemon/README.md) | **Full battles beyond 63 teams.** The full-battle equilibrium is over a fixed pool. A double oracle whose best response is searched in the simulator would say whether some other team beats it. | open |
| [Pokémon](pokemon/README.md) | **Level 4 played out.** The same expected-damage fights carried through six-against-six battles with the level-5 policy would separate the effect of perfect switching from that of Showdown's mechanics. | open |
| [Pokémon](pokemon/README.md) | **A larger pool.** 40 Pokémon by use; how the equilibria move with 60 or 80. | open |
| [Pokémon](pokemon/README.md) | **Emerald Kaizo encounters.** How often each species appears, and the generator's timing, which the independent model leaves out. | open |
| [Spot It](spot_it/README.md) | What is the largest maximal family of order 7 that is not a plane? Random searches found 37 cards, and Theorem 4 allows at most 47. | partly answered |
| [Spot It](spot_it/README.md) | Does a plane of order 12, a deck of 157 cards, exist? It is the smallest order still open. | open |
| [Spot It](spot_it/README.md) | Which two cards does a real box leave out? A card list of an actual box, with a license that allows checking it here, would settle it. | blocked on data |
| [Tenzi](tenzi/README.md) | The slides' chart of a player who rolls faster gives a jump of 0.0235 at equal speeds, about half the exact chance of a tie between two players, 0.048. Where does 0.0235 come from? | open |
| [Tenzi](tenzi/README.md) | What is the best strategy in a race against players who watch our dice and react to them? | open |
| [Yu-Gi-Oh! opening hands](yugioh/README.md) | Real draw cards have costs: banishing cards from the deck or the Extra Deck, or no further draws that turn. Modeling one specific card, cost included, would show how far below these upper bounds it falls. | open |
| [Yu-Gi-Oh! opening hands](yugioh/README.md) | A starter that needs the turn's one Normal Summon competes with another such starter. A need that counts Normal Summons would separate those hands. | open |
| [Yu-Gi-Oh! opening hands](yugioh/README.md) | "Worth in starters" is measured on one need. For "a starter and an extender" a looker could be worth more, since it can take whichever card is missing. | open |
| [Yu-Gi-Oh! opening hands](yugioh/README.md) | Ratios in real tournament decks, if a source with clear terms is found. | blocked on data |
