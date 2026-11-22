# C10 boards where the strangers and the language models disagree most

Exploratory. Written by `uv run python -m codenames.diagnose disagree`. Each row is a test board of seed 9004 and P − F under each kind of guesser, in words per turn; 'gap' is the strangers' P − F minus the language models'. The language-model value is the mean of two single replies, so a large gap is partly the draw of a reply. Case files for the first 10 are in `cases/`. The level is the worst over F's clue and P's seven.

| Board | Strangers | Human-fitted | EAT | Language models | Gap | F | P | Case |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 209 | 1.265 | 2.388 | 0.279 | 8.500 | -7.235 | legal | illegal | [c10-209](cases/c10-209.md) |
| 296 | 0.888 | -0.853 | -0.522 | 5.000 | -4.112 | legal | borderline | [c10-296](cases/c10-296.md) |
| 85 | -0.022 | -0.906 | 1.018 | 4.000 | -4.022 | legal | legal | [c10-085](cases/c10-085.md) |
| 224 | 0.496 | 0.615 | -1.000 | 4.500 | -4.004 | legal | illegal | [c10-224](cases/c10-224.md) |
| 230 | 1.054 | 2.057 | 0.000 | -2.500 | 3.554 | legal | legal | [c10-230](cases/c10-230.md) |
| 244 | 1.461 | 0.340 | 1.904 | -2.000 | 3.461 | legal | illegal | [c10-244](cases/c10-244.md) |
| 66 | 0.816 | 1.368 | -0.800 | 4.000 | -3.184 | legal | illegal | [c10-066](cases/c10-066.md) |
| 15 | 0.264 | 0.348 | -1.240 | -2.500 | 2.764 | legal | legal | [c10-015](cases/c10-015.md) |
| 243 | -0.280 | -1.474 | -0.606 | -3.000 | 2.720 | legal | illegal | [c10-243](cases/c10-243.md) |
| 89 | 1.007 | 1.704 | 0.254 | -1.500 | 2.507 | legal | legal | [c10-089](cases/c10-089.md) |
| 294 | -0.004 | -0.077 | 0.643 | 2.500 | -2.504 | legal | illegal |  |
| 44 | 0.432 | -1.079 | 1.460 | -2.000 | 2.432 | legal | legal |  |
| 65 | 0.393 | 0.679 | 0.187 | -2.000 | 2.393 | legal | legal |  |
| 269 | 0.842 | -0.122 | 0.052 | -1.500 | 2.342 | legal | illegal |  |
| 236 | 0.298 | 0.005 | -0.615 | -2.000 | 2.298 | legal | legal |  |
| 232 | 0.733 | 0.887 | -1.514 | -1.500 | 2.233 | legal | illegal |  |
| 123 | 1.133 | 1.134 | -0.454 | -1.000 | 2.132 | legal | legal |  |
| 31 | -0.066 | -0.755 | 0.000 | -2.000 | 1.934 | legal | borderline |  |
| 101 | 0.926 | -0.181 | 0.135 | -1.000 | 1.926 | legal | illegal |  |
| 225 | 0.825 | 1.497 | 0.401 | -1.000 | 1.825 | borderline | legal |  |
