# Why the three-player information strategy trails hanabi.rs

With three players, `information.py` averages 22.254 points and hanabi.rs's information strategy 24.7946, a gap of 2.54. The gap comes from the strategy, not the rules. Most of it, 2.25 points, remains when hanabi.rs is limited to the four hint values used here; it lies in what each hint asks. Detail below. Every number is simulated on seeded deals, with the sample size given; intervals are 95%.

## The two programs

- **hanabi.rs** (J. Wu, github.com/WuTheFWasThat/hanabi.rs, commit `ef77311`) was built here with cargo and run with `-g info` on its seeds 0 to 19,999. It reproduces its README's table to every digit: 22.5217, 24.7946, 24.9356 and 24.9223 for two to five players. Its repository has no license file, so none of its code is copied here; the one change made for this note is described in words.
- **`information.py`** generalizes Cox et al.'s five-player strategy to P players with 2(P − 1) hint values. With three players it scores 22.254 ± 0.032 on seeded deals 0 to 9,999; `hat_gap.py` reproduces that run game by game.

## The rules: no difference that matters

Reading hanabi.rs's `src/game.rs` and `src/main.rs` gives the rules it plays: 8 hint tokens and 3 lives; hands of 5 cards for two and three players and 4 for four and five; a hint must touch a card; no discard while all 8 tokens are on the table; a played 5 returns a token; the game ends one round after the last draw. These are the rules of `rules.py`, with one exception: when the last life is lost, hanabi.rs keeps the board as the score, where `rules.py` scores 0.

That exception does not touch this comparison. In hanabi.rs's 20,000 three-player games none loses all three lives (552 end with one life left, read from a lives histogram added to its output). `information.py`'s three-player run used HanSim's discard rule (a discard is allowed with 8 tokens on the table). Replayed with the official discard rule and the board kept on the last life, every one of the 10,000 deals scores the same, and no game loses a life.

## The strategies: two differences

1. **How many values a hint carries.** Cox et al. read a hint as one of 2(P − 1) numbers: a rank or a color hint to one of the other players, 4 numbers with three players. hanabi.rs also reads which cards a hint touches, which gives 3 or 4 numbers per other player (`get_info_per_player` in `src/strategies/information.rs`), so 6 to 8 with three players.
2. **What a hint asks.** In Cox et al.'s strategy, each player learns which of the sets holds one targeted card, the one most likely to be playable. The sets are drawn from the card's remaining identities. hanabi.rs first asks, in one sum, whether each of several cards is playable or dead (`AdditiveComboQuestion`), and only when a player already knows a playable card does it split one card's identities.

**Measuring the first difference in hanabi.rs.** `get_info_per_player` was changed to return at most 2, so that a hint carries 4 values with three players, as in Cox et al. Everything else was left alone, including the questions. Seeds 0 to 19,999:

| hanabi.rs, information strategy | 2 players | 3 players | 4 players | 5 players |
|---|---:|---:|---:|---:|
| As published (3 or 4 values per other player) | 22.5217 | 24.7946 | 24.9356 | 24.9223 |
| At most 2 values per other player | 20.3146 | 24.5037 | 24.8874 | 24.8969 |
| Perfect games, as published | 12.55% | 84.48% | 95.05% | 94.04% |
| Perfect games, at most 2 values | 0.39% | 66.85% | 91.44% | 91.84% |

With three players, the fewer values cost hanabi.rs 0.29 points. At 24.50 it still leads `information.py` by 2.25.

**Measuring the first difference in `information.py`.** `hat_gap.py` lets each hint carry a number modulo m, for m above the legal 4. This relaxation is not a legal strategy: it measures how much the number of values limits Cox et al.'s design. Seeded deals 0 to 9,999, HanSim's discard rule, run on GitHub Actions at commit `b3d6a47`:

| Hint values | 4 (legal) | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|
| Mean score | 22.254 ± 0.032 | 23.164 ± 0.029 | 23.774 ± 0.025 | 24.353 ± 0.019 | 24.623 ± 0.015 |
| Perfect games | 6.57% | 19.52% | 35.05% | 58.56% | 73.89% |

At 8 values, the most hanabi.rs's reading gives with three players, the relaxed strategy scores 23.77, still 1.02 below hanabi.rs. Cox et al.'s design needs about 16 values to come within 0.2 points; with five players it has 8 legal values, which is why the paper's case works.

## The finding

The 2.54-point gap is a strategy difference. Restricting the number of hint values costs hanabi.rs only 0.29 points, but costs Cox et al.'s design 1.52 points against 8 values. So the gap lies mostly in what a hint asks. Asking whether cards are playable or dead uses a few values well; splitting one card's identities into 4 sets does not. `information.py`'s three- and four-player numbers measure a weak generalization, not the limit of hat-guessing strategies.

## Reproducing

```bash
git clone https://github.com/WuTheFWasThat/hanabi.rs && cd hanabi.rs && git checkout ef77311
cargo build --release
./target/release/rust_hanabi -n 20000 -s 0 -t 4 -p 3 -g info
# for the limited run, make get_info_per_player in src/strategies/information.rs return at
# most 2 (min(2, its value)), rebuild, and run the same command
uv run python -m hanabi.hat_gap -n 10000 --out hanabi/results   # about 25 minutes
```
