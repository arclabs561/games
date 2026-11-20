# Simulation

Written by `uv run python -m yugioh.tables --simulate`. The example deck of 40 cards
(12 starters, 9 extenders, 3 bricks), going first. Seed 1; Wilson intervals.

## Hands without draw cards: 1,000,000 hands, one sample for every need

| Need | Hits | 95% interval | Exact | Exact in interval |
|---|---:|---:|---:|---:|
| at least one starter | 851,044 | 85.0345% to 85.1740% | 85.0640% | yes |
| a starter and an extender | 609,843 | 60.8887% to 61.0799% | 61.0091% | yes |
| a starter and another engine card | 780,201 | 77.9388% to 78.1012% | 77.9954% | yes |
| no brick | 662,112 | 66.1184% to 66.3038% | 66.2449% | yes |
| a starter and no brick | 581,851 | 58.0884% to 58.2817% | 58.1706% | yes |

## With draw cards: 200,000 shuffled decks per row, dealt by code separate from the exact recursion

| Need | Hits | 95% interval | Exact | Exact in interval |
|---|---:|---:|---:|---:|
| 1 draw-2: starter | 173,225 | 86.463% to 86.761% | 86.597% | yes |
| 1 draw-2: starter and extender | 127,918 | 63.748% to 64.169% | 64.038% | yes |
| 2 draw-2: starter | 176,013 | 87.863% to 88.148% | 87.903% | yes |
| 2 draw-2: starter and extender | 133,347 | 66.467% to 66.880% | 66.676% | yes |
| 3 draw-2: starter | 178,181 | 88.953% to 89.226% | 89.008% | yes |
| 3 draw-2: starter and extender | 137,927 | 68.760% to 69.166% | 68.958% | yes |
| 3 look-at-6: starter | 183,144 | 91.449% to 91.693% | 91.499% | yes |
| 3 look-at-6: starter and extender | 149,575 | 74.597% to 74.977% | 74.804% | yes |
| 3 draw-2, 3 look-at-6: starter | 188,794 | 94.295% to 94.497% | 94.365% | yes |
| 3 draw-2, 3 look-at-6: starter and extender | 163,503 | 81.582% to 81.920% | 81.794% | yes |
| 3 draw-1, added (43 cards): starter | 170,092 | 84.889% to 85.202% | 85.064% | yes |
| 3 draw-1, added (43 cards): starter and extender | 121,996 | 60.784% to 61.212% | 61.009% | yes |
| 3 draw-2, added (43 cards): starter | 172,597 | 86.147% to 86.449% | 86.350% | yes |
| 3 draw-2, added (43 cards): starter and extender | 127,269 | 63.423% to 63.845% | 63.702% | yes |
| 3 look-at-6, added (43 cards): starter | 178,195 | 88.960% to 89.233% | 89.170% | yes |
| 3 look-at-6, added (43 cards): starter and extender | 139,095 | 69.345% to 69.749% | 69.652% | yes |
