# The best team of six

The pool, the sets, the five levels, the team game, full battles and the comparison with ladder use. [teams.md](teams.md), written by `teams.py`, has every table of the study. Moved unchanged from the [Pokémon README](../README.md), which keeps the short answers.


No single team is best at any level of detail. Each model's team game has a mixed equilibrium, and what it mixes changes as the model gets more detailed. Primarina survives every level; the Pokémon players use most do not appear until the last.

![Six cards, one per Pokémon of a team: Raging Bolt, Iron Treads, Ting-Lu, Zapdos, Moltres and Primarina, each colored by its types, with its ability and four moves colored by move type.](../figures/teams-cards.svg)

*The heaviest team (26.8%) of the full-battle equilibrium, drawn by `teams_figures.py`. Each card shows the set that every model uses: one ability and four damaging moves.*

**The pool and the sets.** The pool is the 40 most used Pokémon of `gen9ou` at rating 1500, from November 2025 to October 2026 (`teams_usage.py`). Two Ogerpon masks are left out, because Showdown keeps their forme only while they hold their mask. Each Pokémon gets one fixed set (`teams_model.py`):
- Level 100, every IV 31, 84 EVs in each stat, a neutral nature, no item, no Terastallization.
- Its ability is the first in slot order that the models represent.
- Its four moves are those that take the largest share of each opponent's HP per hit, summed over the pool and capped at a KO.
- Only moves with a fixed power count. Knock Off, Sucker Punch, U-turn, Freeze-Dry and Body Press are among those left out; `teams_moves.csv` gives each reason.

The ability is the one players use most for 24 of the 40. Of the four moves, 1.25 on average are among the four players use most. These sets only deal damage; nothing else a Pokémon can do is modeled.

**Five levels.** A **matchup** M(p, q) is the share of HP that the winner of a one-on-one fight keeps, positive when p wins. The levels differ only in how the fight is computed:

| Level | What the fight sees |
|---|---|
| 1. Types | Each attacks with its better own type; damage is the type multiplier, as in Schmitz's duel above |
| 2. Moves | Damage is power × hits × accuracy × 1.5 for the user's own type × multiplier, for the best of the four moves |
| 3. Abilities | Level 2 with immunities (Levitate, Flash Fire, ...), resistances (Thick Fat, Filter, ...) and boosts (Adaptability, Technician, ...) |
| 4. Stats | The game's damage formula averaged over its 16 rolls, turns in speed and priority order, recoil, drain and self stat drops; plus Intimidate, Multiscale and the Ruin abilities |
| 5. Simulator | Pokémon Showdown itself: 50 seeded battles per pair, each side using its most damaging move every turn |

**The team game.** A team scores against another team by perfect switching: each opposing Pokémon is met by the team's best answer to it, and each of the team's own Pokémon by the opponent's best answer. The score is the mean of the first minus the mean of the second. The game is symmetric, with value 0. It is solved by double oracle (McMahan, Gordon and Blum 2003). Finding the best team against a mix is a facility location problem, solved as an integer program. All 3,838,380 teams are then checked against the final mix, exactly.

| Level | Ranking share of M | Teams in the equilibrium | Heaviest team | The six most used, against it |
|---|---:|---:|---|---:|
| 1. Types | 12.7% | 3 | Kingambit, Dragonite, Iron Treads, Zapdos, Primarina, Tyranitar (50%) | −0.240 |
| 2. Moves | 14.3% | 11 | Great Tusk, Gholdengo, Raging Bolt, Slowking-Galar, Deoxys-Speed, Primarina (47%) | −0.228 |
| 3. Abilities | 14.2% | 11 | the same (36%) | −0.235 |
| 4. Stats | 22.4% | 15 | Zamazenta, Kyurem, Ting-Lu, Rillaboom, Deoxys-Speed, Primarina (25%) | −0.251 |
| 5. Simulator | 22.0% | 11 | Zamazenta, Raging Bolt, Ting-Lu, Iron Moth, Deoxys-Speed, Primarina (23%) | −0.340 |

*The ranking share is the Hodge split of the previous section, applied to M: the part that one score per Pokémon explains. The last column is the score of the team of the six most used Pokémon against the equilibrium; 0 would tie.*

**What survives and what flips.**
- *Survives:* Primarina is on the equilibrium team with chance 100%, 47%, 36%, 99% and 100% at levels 1 to 5. Mixing survives too, at every level and in full battles.
- *Mostly cyclic at every level:* one ranking explains 13% to 22% of the matchups, more than the 5% it explains of the type chart.
- *Flips between types and stats:* Kendall's τ between the rankings of levels 1 and 4 is 0.01. Zamazenta goes from 39th by types to 3rd by stats.
- *Small steps:* abilities change 39 of the 780 matchups (τ = 0.89 between levels 2 and 3). Level 5's ranking stays close to level 4's (τ = 0.86, interval 0.84 to 0.86).
- *Flips with the battle format:* under perfect switching, the six most used Pokémon are among the worst teams of the 63 in the cross-level tournament. In full battles they hold the second-largest weight.

![Rank of each of the 40 Pokémon by its ranking score at each level, and its rank by ladder use; the eight that rank highest in the simulator are colored.](../figures/teams-ranks.svg)

*Rankings across levels. Lines cross most between levels 3 and 4, where stats enter; the last column, ladder use, follows none of them.*

**Full battles.** A team game built on one-on-one matchups is still a model. So 63 teams play a round robin of whole six-against-six battles in Showdown, 20 battles per pair with sides alternating. The 63 are each level's equilibrium and best teams, the six most used Pokémon, and eight random teams. Both sides use the same policy:
- Lead with the Pokémon that matches up best, at level 4, against the opposing six.
- Each turn, use the most damaging move; never switch by choice.
- After a faint, send in the best level-4 answer to the opposing Pokémon.

The equilibrium of the 63-team matrix is unique, on 15 teams. The heaviest team (26.8%, interval 20.4% to 32.2%) is the one on the cards above. The six most used Pokémon weigh 13.7%, though the perfect-switching models rank that team 55th to 62nd of 63. One ranking explains 28.2% of the full-battle matrix. Levels 4 and 5 rank the 63 teams with τ = 0.51 against the full battles; levels 1 to 3 with τ between −0.08 and 0.07.

![The seven heaviest teams of the full-battle equilibrium on a circle, with arrows from each team to the teams it beats.](../figures/teams-graph.svg)

*The three heaviest teams form a cycle: T1 loses to the six most used (T2), which lose to T3, which loses to T1.*

**Against real use.** No level ranks the 40 Pokémon the way the ladder uses them. Spearman's ρ between use and ranking score is between 0.02 and 0.08 at every level. Between use and the chance of being on the equilibrium team, it is −0.20 to 0.30. Every interval spans 0. The models and players agree on a few names. Zamazenta, 5th by use, is on 77% and 79% of equilibrium teams at levels 4 and 5 and on 51% in full battles. Raging Bolt, 7th by use, is on 46% to 74% at levels 1, 2, 3 and 5 and in full battles. They disagree on the two most used, Great Tusk and Gholdengo. Both are on most equilibrium teams at levels 2 and 3 and on none at the other levels. In full battles they appear only as part of the six-most-used team. Walls and setup Pokémon are what damage-only sets cannot represent, so this gap is expected.

[`results/teams.md`](teams.md) has every table of the study.

**Simulator and license.** Pokémon Showdown is MIT-licensed (its `LICENSE` at the pinned commit). poke-env is also MIT (its `LICENSE` on GitHub and on PyPI, version 0.17.0), but it plays through a server, which picks each battle's random seed. The battles here use Showdown's own battle stream instead (`teams_showdown.js`), which takes a seed per battle, so every battle can be rerun.

## Does the equilibrium survive a policy that switches? (plan P2)

The equilibrium above is one of a game played by one policy: use the most damaging move and never switch by choice. Plan P2 ported the voluntary switch of poke-env's `SimpleHeuristicsPlayer` (poke-env 0.17.0, MIT) into the battle driver as policy S: leave a matchup worse than −2, or a defensive or main attacking stat lowered three stages, when a bench Pokémon has a positive matchup, and bring in the best one. The move choice stays the old policy's (D), so the two differ in switching only. Two new round robins of the same 63 teams ran on GitHub Actions, 40 battles per pair with sides alternating: M_S with both sides on S (label `p2-switch`, 78,120 battles) and M_D′ with both on D (label `p2-replicate`, 78,120 battles), a fresh replicate. The published equilibrium σ_D is held fixed. Its exploitability ε(σ_D) is the best payoff, in wins minus losses per battle, that any of the 63 teams gets against it. Summaries: `teams-battles-p2-switch.json`, `teams-battles-p2-replicate.json`, `teams-battles-p2-strength.json`, and every number below in [p2.json](p2.json), written by `python -m pokemon.p2`.

- **The equilibrium does not survive.** ε_S(σ_D) is 0.431 and ε_D′(σ_D) is 0.199, a difference of 0.232 (0.101 to 0.349; bootstrap over battles within each pair, both matrices resampled, 10,000 resamples, seed 0). The interval lies above 0, so by the plan's rule the team result depends on the policy. The team that exploits σ_D most under S is Ting-Lu, Iron Moth, Rillaboom, Deoxys-Speed, Primarina and Enamorus. The detectable effect, from the bootstrap's spread, is 0.18, larger than the 0.10 the plan expected.
- **S is the stronger policy here.** Each team played itself 40 times with S on one side and D on the other (2,520 battles, label `p2-strength`). S won 1,349 and D 1,171, with no ties: 53.5% (51.6% to 55.5%, Wilson).
- **The switching equilibrium.** M_S has a unique equilibrium on 11 teams, all with Primarina. The heaviest, Kyurem, Ting-Lu, Zapdos, Primarina, Enamorus and Iron Crown, weighs 23.5% (12.8% to 29.2%, 1,000 bootstrap resamples). It shares 7 teams with σ_D (Jaccard 0.37), the two mixtures are 0.62 apart in total variation, and the team of the six most used Pokémon gets no weight (0 to 0) against 13.7% in σ_D. One ranking explains 30.0% of M_S, against 28.2% for the published matrix, and the rankings of the 63 teams agree with τ = 0.76.
- **The replicate is noisier than the noise model.** Redrawing each pair's 20 published battles from its own shares gives ε(σ_D) of 0.111 on average (sd 0.036, 2,000 draws, seed 0), as the plan computed. The fresh replicate, with twice the battles per pair, gives 0.199. So σ_D is more exploitable in new battles than redrawing the old ones predicts: the published equilibrium is partly fitted to the noise of its own 20 battles per pair. The replicate's own equilibrium has 13 teams, shares 9 with σ_D (Jaccard 0.47, total variation 0.44), keeps the published heaviest team at 25.6% (18.5% to 35.1%), and ranks the teams with τ = 0.95 against the published matrix.
- **Checks.** No choice errors and no stat mismatches in any of the 158,760 battles, and no battle reached the 300-turn limit. The damage checks agree on all 217,589 eligible hits of M_D′ and 8,387 of the strength battles. In M_S, 229,818 agree and 5 differ: Garchomp's Earthquake on Hatterene (4) and Kingambit's Kowtow Cleave on Slowking-Galar (1). The pilot (200 battles of each policy, exploratory) had none; these cases are not explained yet. S makes 0.38 to 0.45 voluntary switches a battle, and battles last 15.9 turns under S against 15.5 under D.

This is still one heuristic. A switching rule tuned to these sets, or a learned policy, could move the equilibrium further, and the sets still have no setup, recovery, hazards or items.
