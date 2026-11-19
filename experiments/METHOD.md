# Method shared by the plans

Each plan in `plans/` follows these rules. Where a plan departs from one, it says so and why.

## Before a run

- **The plan is merged first.** A plan is merged into `main` before any run that its results will come from. After that it is not edited. A change made later goes in a dated "Amendments" section at the end of the plan, with the reason, before the affected runs. The preregistration workshops at NeurIPS (preregister.science) use the same split: the protocol is fixed before the confirmatory experiments, and everything else is reported as exploratory.
- **Runs name a commit.** Every run records the full SHA it ran at (`hfrun.py` does this). Code changes after the plan is merged are allowed, but the plan names the module, the arguments and the outputs, and a run that changes them is an amendment.
- **Exploratory work is labeled.** Numbers already in a README, and pilots, are exploratory. They can motivate a plan but never count as its result. Seeds and boards already used for them are not reused for the confirmatory run.

## Design

- **One factor at a time.** Each comparison changes one thing: the training partners, the model, the split. Everything else stays fixed and is listed in the plan.
- **Paired designs and common random numbers.** Arms are compared on the same boards, deals, test turns and seeds. The difference is computed per unit (board, deal, turn or game) and then averaged, so the variance between units cancels. Bouthillier et al. (2021) recommend this pairing as the simplest way to gain power.
- **Seeds.** Seeds that pick the evaluation data (boards, deals) are separate from seeds that drive training. Evaluation seeds are the same for every arm. Training runs use at least five seeds where a run is cheap and at least three where it is not. One seed is not enough: Henderson et al. (2018) split ten runs of one configuration into two groups of five and found the two averages significantly different.
- **Held-out partners are the headline for coordination.** A giver or player is scored with partners it never trained with or tuned against. Its score with its own partner is reported too, but never as the main result. Hu et al. (2020) evaluate their agents by cross-play between runs that never trained together, and AH2AC2 (Dizdarević et al., 2025) scores agents against human proxies held behind an API.

## Analysis

- **One primary metric per question, fixed in advance**, with its test and its interval method. Where a plan has two primary hypotheses, both are tested with the Holm correction. Everything else is secondary and is reported without a significance claim.
- **Intervals.** 95% percentile bootstrap intervals with 10,000 resamples and a fixed resampling seed. The resampling unit is the unit that was sampled: games for human turns, which come in games, and boards or deals for simulated play. Where there are several tasks with a few seeds each, the bootstrap is stratified: seeds are resampled within each task, as Agarwal et al. (2021) do. They also recommend the interquartile mean over runs when runs are few, and a plan with five or more seeds per task reports it as a secondary metric. Proportions from independent trials use Wilson intervals, as the READMEs already do.
- **Test sets are used once.** A test set is evaluated once, with the configuration fixed in the plan. Choices such as temperatures, templates and learning rates are made on training or validation data. AH2AC2 enforces this with an evaluation key that can be used only once. Generated test data (boards, deals) is drawn fresh for the confirmatory run.

## Budgets and reporting

- **Costs are capped.** Each plan states its expected cost and a hard cap. Jobs set a timeout that enforces the cap. The total over all plans is more than $3, so each paid plan is approved separately before its runs.
- **Negative and null results are reported**, in the game's README, with the same detail as positive ones. A plan says in advance what will be reported if the result is null. A run that fails or is stopped is recorded with its reason, and no number is quoted from it.
- **Where results go.** Summaries with every quoted number go to the game's `results/`. Large or unpublishable artifacts go to the private dataset `arclabs561/games-runs`, as `README.md` describes.

## Sources

- R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021.
- X. Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021.
- P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger, "Deep Reinforcement Learning that Matters", AAAI 2018.
- H. Hu, A. Lerer, A. Peysakhovich, J. Foerster, "'Other-Play' for Zero-Shot Coordination", ICML 2020.
- T. Dizdarević et al., "Ad-Hoc Human-AI Coordination Challenge", arXiv:2506.21490.
- The NeurIPS 2020 and 2021 pre-registration workshops, preregister.science.
