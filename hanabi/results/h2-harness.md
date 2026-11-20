# H2's gpt-oss-120b numbers are partly limited by the harness (2026-11-20)

A dated note on plan H2. It changes no published number, and H2 was not run again.

**Finding.** In H2's primary arm, gpt-oss-120b (reasoning effort medium, through Novita) reached the 4,096-token reply cap on 450 of its 1,875 turns, 24%, before it wrote a move. Each such turn got the harness's fallback move instead of the model's. So the published gap to the ceiling, 18.63 points (16.70 to 20.57), measures the model and the cap together, not the model alone. DeepSeek-V3.2 never reached the cap in its confirmatory arms, and its numbers are not affected.

What the fallbacks did, from replaying every recorded game in `rules.py` (`python -m hanabi.h2 harness`, written to `h2-harness.json`; computed exactly, and each replay ends with the recorded score):

| Arm | Games | Turns | Fallbacks | Fallback discards | Fallback plays | Lives lost by fallbacks | Lives lost by the model's moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b, deduction on | 30 | 1,875 | 450 | 446 | 4 | 4 | 56 |
| gpt-oss-120b, deduction off | 24 | 1,489 | 309 | 305 | 4 | 4 | 42 |
| DeepSeek-V3.2, deduction on | 30 | 796 | 0 | 0 | 0 | 0 | 90 |
| DeepSeek-V3.2, deduction off | 30 | 750 | 1 | 1 | 0 | 0 | 90 |
| pilot at 512 tokens, both models | 8 | 65 | 50 | 10 | 40 | 24 | 0 |

In the arm with deduction off, one fallback came from a reply that ended without a move before the cap; the other 308 came from the cap. Every gpt-oss-120b game in both confirmatory arms had at least one fallback.

How to read it:

- The fallback is the first legal discard, or a blind play of card 1 when all eight hint tokens are on the table. At 4,096 tokens nearly all of gpt-oss-120b's fallbacks were discards. They cost few lives (4 of 60 with deduction on, none of them the third), and lives are not where the cap did its harm. Each fallback threw away the oldest card in the hand in place of the move the model was reasoning toward. That card may have been playable or the last of its kind. The replay counts these discards but cannot say which move the model would have made.
- The deduction comparison is affected too. Both of its arms lost about a fifth of their turns to the cap (24% with deduction on, 21% with it off). A prompt that makes the model reason longer loses more turns, so the comparison mixes the value of deduction help with its cost in reply length.
- A higher cap would not leave the scores as they are. Replies that reached the cap are censored at 4,096 tokens, so the logs do not show how long they would have run. Whether 8,192 or 16,384 tokens, or a lower reasoning effort, would raise the score is unknown until it is run.

**What changed.** `python -m hanabi.h2 play` now takes `--max-tokens` with a default of 16,384 (H2 passed 4,096 on the command line), and `--reasoning-effort` takes low, medium or high. Both are recorded in each run's `summary.json`. The cost meter reserves room for a full reply at the cap, so a larger cap lowers the number of games a fixed budget allows. Any rerun is a new plan or a dated amendment to H2, with its own cap.
