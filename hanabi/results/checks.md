# Checks

Every check of the rules engine, the strategy, the solvers, the certificates and the experiments. Moved unchanged from the [Hanabi README](../README.md), which keeps the short answers.


- The rules engine has tests for each rule above, including the extra round after the last card and the score of 0 on the third life.
- The information strategy matches HanSim game by game on 10,000 decks; a test checks the first 100. A test also checks the partition of the paper's Figure 6.
- On every deal of four small decks, the two solver engines give the same answer, for the maximum score and for every lower target. Every won deal's line is replayed by the rules engine.
- Each certificate is checked against brute force on those decks: no deal with a certificate is won.
- Solver B (`dfs_solver.py`) agrees with brute force on every deal of the four smaller decks, with and without its bounds, and with `winnable` on every deal of the decks in the table. The two new certificates never hold for a won deal there. A test checks greedy scheduling against an exhaustive search of the schedule on 200 random deals.
- On the full game, Solver A and Solver B give the same verdict on every one of the 79,396 deals both decide (plan H1, below). Every win of either solver is replayed by the rules engine.
- The constraint model of `solver_cp.py` agrees with brute force on every deal of three small decks (540 deals); a test checks some of them when OR-Tools is installed.
- JaxMARL's two-color game and `rules.py` agree on legal moves, game length and score in 4,000 random games (`tests/test_hanabi_jax.py`), and on the score of each of the 1,024,000 games of plan H4's evaluation. Action numbers convert both ways for two to four players. A tiny training run, its evaluation and its summary run end to end on a CPU in the tests.
- The bootstrap of plan H4 is checked on matrices with known answers. The committed ceilings of the H4 test deals match a fresh solve on a sample of them.

- From games in progress, `winnable` agrees with brute force on every target from every position of random games on every deal of three small decks, and the free-order game with a target below 25 never rules out a score that brute force reaches there.
- Every one of the 3,079 human games replays in `rules.py` with every action legal and ends on its recorded score. No game scores above its deal's ceiling, and every game scored 25 has ceiling 25. A test recomputes the summary and the picture from the committed per-game records. Action numbers agree with `jax_bridge.py`, which plan H4 checked against JaxMARL. Exports in both clue styles of hanab.live replay in `ceiling.py`.

The two solvers share the rules engine, so a bug there would fool both. The rules engine has its own tests, and the information strategy matches HanSim game by game, which checks the rules from outside.
