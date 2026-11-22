# C10's P − F by slice

Exploratory. Written by `uv run python -m codenames.diagnose slices`. The slices were fixed in `diagnose.SLICES` before they were computed, but after C10's results were known. 300 test boards of seed 9004; words per turn. 'In' is the mean P − F on the boards in the slice, 'out' on the others; the difference has a 95% interval from resampling each group, 10,000 resamples, seed 0. Holm's correction is over all 18 tests at a family level of 0.05. The language-model values are one reply per clue.

| Slice | Boards in it |
|---|---:|
| embedding_families_agree: the strangers and the human-fitted guesser give P - F the same sign | 193 |
| flagged: F's clue or one of P's seven clues is illegal by the strict check | 112 |
| p_lower_count: P's count is below F's | 148 |
| p_nearer_assassin: P's clue is closer to the assassin than F's, by the proposer's cosine | 130 |
| p_rarer: P's clue is rarer than F's (lower wordfreq Zipf frequency) | 107 |

| Slice | Scored by | In | Out | In − out | p | Holm rejects |
|---|---|---:|---:|---|---:|---|
| embedding_families_agree | eat | -0.008 | 0.022 | -0.030 (-0.194 to 0.141) | 0.7272 | no |
| embedding_families_agree | llm | 0.101 | -0.196 | 0.297 (0.070 to 0.527) | 0.0116 | no |
| flagged | eat | -0.034 | 0.025 | -0.059 (-0.225 to 0.108) | 0.4714 | no |
| flagged | human | 0.158 | 0.130 | 0.029 (-0.147 to 0.213) | 0.7510 | no |
| flagged | llm | -0.045 | 0.019 | -0.063 (-0.328 to 0.221) | 0.6212 | no |
| flagged | strangers | 0.251 | 0.232 | 0.019 (-0.066 to 0.107) | 0.6702 | no |
| p_lower_count | eat | 0.108 | -0.099 | 0.207 (0.042 to 0.369) | 0.0124 | no |
| p_lower_count | human | 0.146 | 0.135 | 0.011 (-0.164 to 0.194) | 0.8892 | no |
| p_lower_count | llm | -0.162 | 0.148 | -0.310 (-0.547 to -0.070) | 0.0150 | no |
| p_lower_count | strangers | 0.307 | 0.173 | 0.134 (0.052 to 0.218) | 0.0016 | yes |
| p_nearer_assassin | eat | 0.190 | -0.140 | 0.330 (0.162 to 0.488) | 0.0000 | yes |
| p_nearer_assassin | human | 0.200 | 0.095 | 0.106 (-0.078 to 0.290) | 0.2554 | no |
| p_nearer_assassin | llm | -0.081 | 0.053 | -0.134 (-0.382 to 0.101) | 0.2658 | no |
| p_nearer_assassin | strangers | 0.339 | 0.164 | 0.175 (0.092 to 0.256) | 0.0000 | yes |
| p_rarer | eat | -0.236 | 0.135 | -0.371 (-0.550 to -0.191) | 0.0000 | yes |
| p_rarer | human | 0.126 | 0.148 | -0.023 (-0.214 to 0.172) | 0.8074 | no |
| p_rarer | llm | -0.033 | 0.010 | -0.043 (-0.291 to 0.193) | 0.7226 | no |
| p_rarer | strangers | 0.268 | 0.223 | 0.045 (-0.041 to 0.131) | 0.3032 | no |
