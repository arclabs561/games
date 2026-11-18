# Egyptian Ratscrew

How likely is a challenge to be answered, how long does a game last, and how much does slapping speed matter?

- With no slaps, a challenge is answered 29% of the time after a jack and 76% after an ace. Following the whole exchange, the player who laid a jack keeps the pile 81% of the time, and the player who laid an ace half the time. Exact, by recursion on the unseen cards.
- Two players who slap equally fast take a median of 289 cards per game, and four take 1,051. After 300 cards, 27% to 28% of the remaining two-player games end within each further 100 cards, so the tail is geometric rather than a power law. Simulated, 10,000 games each.
- Winning 55% of the slaps wins 64% of the games, and winning 70% wins 90%. Simulated, 4,000 games each.

`game.py` holds the rules and the exact recursion, and `egyptian_ratscrew.ipynb` works through the questions. Reaction times are not modeled: each slap goes to a player with a fixed chance. With slapping turned off the game is Beggar-my-neighbour, so the tests check the rules against two record deals from [Richard Mann's list](https://richardpmann.com/beggar-my-neighbour-records.html), which end after exactly the published 8,344 and 7,225 cards, and against the first deal known never to end ([Casella et al., 2024](https://arxiv.org/abs/2403.13855)).

```bash
uv run python -m egyptian_ratscrew.game
uv run --group egyptian_ratscrew --with jupyter jupyter lab egyptian_ratscrew/egyptian_ratscrew.ipynb
```
