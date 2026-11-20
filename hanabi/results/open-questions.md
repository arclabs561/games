# Open questions

The full list; the README keeps the main ones. Moved unchanged from the [Hanabi README](../README.md), which keeps the short answers.


- Split the points between a team's score and its deal's ceiling into what hidden information costs and what play costs. That needs an agent that plays the real game well, scored on the same 3,079 decks: an H-group convention bot, or the challenge's human proxies, which are only reachable through its evaluation service.
- Trace the 67 human games whose drops a search could not settle, with larger budgets.
- Settle the 23 deals still undecided (listed above). No certificate here proves any of them lost, and 13 are won in the free-order game, so many may be won; a search guided by the free-order game's winning line might find a real one.
- The free-order game drops tokens. No full deal here is lost to tokens alone, but some small decks are: with three players, hands of one card and one token, 24 of the 180 orderings of a four-card deck are lost though the free-order game wins them.
- The best real (not open-information) score for each deal is unknown; Endgame-Analyzer on [git.abstractnonsen.se](https://git.abstractnonsen.se/Hanabi/Endgame-Analyzer) solves the case where players know their own cards but not the deck, for endgames.
- Plan H4's policies were still improving when training stopped. Whether the gap between self-play and cross-play at α = 0.05 closes with more steps, as in Forkel et al., is open; 10⁹ steps a seed would take about 4.6 hours an arm on an A10G at the speed measured.
- Whether playing the newest card is the habit that separates the clusters of low-entropy seeds could be tested by forcing or forbidding it in one policy and replaying the cross-play games.
