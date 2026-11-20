# Teams of six at five levels of detail

Written by `uv run --group pokemon python -m pokemon.teams` from the models in `teams_model.py` and the simulator runs summarized in `teams-duels.json` and `teams-battles.json`. Exact unless marked. The README's section on teams explains the models.

## The pool and its sets

The 40 most used Pokémon of `gen9ou` (rating 1500, 2025-11 to 2026-10) that can be played without an item. Use is the share of teams with the Pokémon (Smogon).

| # | Pokémon | use | ability here | most used ability | moves here |
|---:|---|---:|---|---|---|
| 1 | great-tusk | 32.4% | protosynthesis | protosynthesis | close-combat, head-smash, headlong-rush, ice-spinner |
| 2 | gholdengo | 21.9% | good-as-gold | goodasgold | focus-blast, make-it-rain, power-gem, shadow-ball |
| 3 | kingambit | 21.5% | defiant | supremeoverlord | brick-break, iron-head, kowtow-cleave, stone-edge |
| 4 | dragonite | 16.5% | multiscale | multiscale | earthquake, fire-blast, ice-beam, outrage |
| 5 | zamazenta | 15.6% | dauntless-shield | dauntlessshield | close-combat, crunch, ice-fang, stone-edge |
| 6 | iron-valiant | 15.5% | quark-drive | quarkdrive | close-combat, ice-punch, moonblast, throat-chop |
| 7 | raging-bolt | 14.5% | protosynthesis | protosynthesis | ancient-power, draco-meteor, earthquake, thunderbolt |
| 8 | slowking-galar | 14.1% | curious-medicine | regenerator | fire-blast, ice-beam, psychic, sludge-wave |
| 9 | dragapult | 13.8% | clear-body | infiltrator | fire-blast, outrage, shadow-ball, surf |
| 10 | hatterene | 12.5% | healer | magicbounce | dark-pulse, dazzling-gleam, giga-drain, psychic |
| 11 | corviknight | 12.3% | pressure | pressure | brave-bird, double-edge, iron-head, thief |
| 12 | kyurem | 11.9% | pressure | pressure | earth-power, focus-blast, ice-beam, outrage |
| 13 | cinderace | 11.6% | blaze | libero | flare-blitz, gunk-shot, low-sweep, scorching-sands |
| 14 | samurott-hisui | 11.2% | torrent | sharpness | drill-run, ice-beam, surf, throat-chop |
| 15 | gliscor | 11.1% | hyper-cutter | poisonheal | earthquake, gunk-shot, ice-fang, stone-edge |
| 16 | iron-treads | 10.8% | quark-drive | quarkdrive | earthquake, ice-spinner, iron-head, wild-charge |
| 17 | landorus-therian | 10.6% | intimidate | intimidate | earthquake, outrage, psychic, stone-edge |
| 18 | alomomola | 10.4% | healer | regenerator | ice-beam, liquidation, play-rough, psychic |
| 19 | ting-lu | 9.9% | vessel-of-ruin | vesselofruin | earthquake, stone-edge, throat-chop, zen-headbutt |
| 20 | darkrai | 9.2% | bad-dreams | baddreams | dark-pulse, focus-blast, ice-beam, psychic |
| 21 | glimmora | 9.0% | toxic-debris | toxicdebris | dazzling-gleam, earth-power, power-gem, sludge-wave |
| 22 | pecharunt | 9.0% | poison-puppeteer | poisonpuppeteer | gunk-shot, malignant-chain, shadow-ball |
| 23 | ceruledge | 8.4% | flash-fire | weakarmor | close-combat, dragon-claw, flare-blitz, shadow-claw |
| 24 | iron-moth | 8.1% | quark-drive | quarkdrive | dazzling-gleam, energy-ball, overheat, sludge-wave |
| 25 | rillaboom | 7.7% | overgrow | grassysurge | brick-break, double-edge, earthquake, wood-hammer |
| 26 | zapdos | 7.4% | pressure | static | brave-bird, heat-wave, hurricane, thunderbolt |
| 27 | walking-wake | 7.2% | protosynthesis | protosynthesis | double-edge, draco-meteor, flamethrower, surf |
| 28 | garganacl | 6.0% | purifying-salt | purifyingsalt | earthquake, hammer-arm, ice-punch, stone-edge |
| 29 | deoxys-speed | 5.9% | pressure | pressure | fire-punch, focus-blast, ice-beam, psycho-boost |
| 30 | moltres | 5.5% | pressure | flamebody | brave-bird, hurricane, overheat, scorching-sands |
| 31 | primarina | 5.5% | torrent | torrent | ice-beam, moonblast, shadow-ball, sparkling-aria |
| 32 | tyranitar | 5.5% | sand-stream | sandstream | crunch, earthquake, ice-beam, stone-edge |
| 33 | clefable | 5.3% | cute-charm | magicguard | fire-blast, ice-beam, moonblast, psychic |
| 34 | heatran | 5.2% | flash-fire | flashfire | earth-power, flash-cannon, overheat, power-gem |
| 35 | scizor | 5.0% | technician | technician | bug-bite, close-combat, iron-head, thief |
| 36 | tornadus-therian | 4.9% | regenerator | regenerator | bleakwind-storm, focus-blast, heat-wave, icy-wind |
| 37 | garchomp | 4.8% | sand-veil | roughskin | earthquake, fire-blast, outrage, stone-edge |
| 38 | enamorus-incarnate | 4.8% | cute-charm | contrary | earth-power, moonblast, mystical-fire, superpower |
| 39 | iron-crown | 4.6% | quark-drive | quarkdrive | bulldoze, focus-blast, psychic, tachyon-cutter |
| 40 | weezing-galar | 4.6% | levitate | neutralizinggas | dark-pulse, gunk-shot, overheat, strange-steam |

The ability here is the most used one for 24 of 40 Pokémon. Of the four moves here, on average 1.25 are among the Pokémon's four most used moves.

## The Pokémon game at each level

M(p, q) is the share of HP the winner of a one-on-one fight keeps (level 5: the mean over the simulated battles). The ranking share is the part of M's squared norm that one score per Pokémon explains (Hodge decomposition).

| level | ranking share | top five by ranking score | Pokémon in the equilibrium |
|---|---:|---|---:|
| 1: types | 12.7% | primarina, gliscor, landorus-therian, iron-treads, samurott-hisui | 9 |
| 2: moves | 14.3% | primarina, great-tusk, deoxys-speed, alomomola, walking-wake | 11 |
| 3: abilities | 14.2% | primarina, great-tusk, deoxys-speed, weezing-galar, alomomola | 11 |
| 4: stats | 22.4% | kyurem, walking-wake, zamazenta, darkrai, landorus-therian | 11 |
| 5: simulator | 22.0% | zamazenta, walking-wake, kyurem, darkrai, primarina | 11 |

Kendall's τ between the rankings of two levels:

| | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|
| 1 | 0.17 | 0.14 | 0.01 | 0.00 |
| 2 |  | 0.89 | 0.25 | 0.22 |
| 3 |  |  | 0.18 | 0.16 |
| 4 |  |  |  | 0.86 |

## The team game at each level

P(T, U) is the mean over U's six of T's best answer, minus the mean over T's six of U's best answer. The equilibrium is of the whole game over all 3,838,380 teams, found by double oracle and checked exactly against every team. The ladder is 20,000 teams drawn by use (seed 0).

| level | teams in the equilibrium | heaviest team (weight) | best against the ladder (score) | best against random teams (score) |
|---|---:|---|---|---|
| 1 | 3 | kingambit, dragonite, iron-treads, zapdos, primarina, tyranitar (50.0%) | samurott-hisui, iron-treads, zapdos, walking-wake, primarina, garchomp (0.312) | kingambit, raging-bolt, iron-treads, zapdos, walking-wake, primarina (0.297) |
| 2 | 11 | great-tusk, gholdengo, raging-bolt, slowking-galar, deoxys-speed, primarina (46.8%) | great-tusk, alomomola, zapdos, walking-wake, deoxys-speed, primarina (0.286) | great-tusk, gholdengo, alomomola, zapdos, deoxys-speed, primarina (0.269) |
| 3 | 11 | great-tusk, gholdengo, raging-bolt, slowking-galar, deoxys-speed, primarina (35.9%) | great-tusk, alomomola, zapdos, deoxys-speed, primarina, weezing-galar (0.283) | great-tusk, gholdengo, alomomola, zapdos, deoxys-speed, primarina (0.261) |
| 4 | 15 | zamazenta, kyurem, ting-lu, rillaboom, deoxys-speed, primarina (24.9%) | zamazenta, kyurem, landorus-therian, iron-moth, walking-wake, primarina (0.419) | zamazenta, kyurem, landorus-therian, darkrai, iron-moth, primarina (0.381) |
| 5 | 11 | zamazenta, raging-bolt, ting-lu, iron-moth, deoxys-speed, primarina (23.2%) | zamazenta, kyurem, iron-moth, walking-wake, deoxys-speed, primarina (0.406) | zamazenta, raging-bolt, kyurem, iron-moth, walking-wake, primarina (0.385) |

Chance that each Pokémon is on the equilibrium team (the eight largest):

| level | Pokémon |
|---|---|
| 1 | iron-treads 100.0%, zapdos 100.0%, primarina 100.0%, kingambit 50.0%, dragonite 50.0%, raging-bolt 50.0%, ting-lu 50.0%, tyranitar 50.0% |
| 2 | deoxys-speed 100.0%, great-tusk 95.5%, gholdengo 84.4%, slowking-galar 67.1%, raging-bolt 52.0%, tyranitar 48.0%, primarina 46.8%, zapdos 41.9% |
| 3 | deoxys-speed 100.0%, great-tusk 92.7%, gholdengo 88.9%, slowking-galar 59.3%, raging-bolt 55.8%, tyranitar 36.9%, primarina 35.9%, zapdos 30.3% |
| 4 | primarina 99.1%, zamazenta 76.7%, dragonite 61.1%, garchomp 58.6%, zapdos 58.2%, rillaboom 41.8%, ting-lu 34.8%, deoxys-speed 31.4% |
| 5 | deoxys-speed 100.0%, primarina 100.0%, zamazenta 78.6%, raging-bolt 74.4%, zapdos 62.2%, ting-lu 48.8%, walking-wake 42.5%, iron-treads 30.0% |

Solver details: rounds of double oracle, teams in the restricted game, whether its equilibrium is unique (exact certificate), and teams tied with the best score (0) against the mix.

| level | rounds | restricted teams | restricted equilibrium unique | teams scoring 0 |
|---|---:|---:|---|---:|
| 1 | 14 | 16 | yes | 76 |
| 2 | 30 | 32 | yes | 18 |
| 3 | 33 | 35 | no | 23 |
| 4 | 42 | 44 | no | 100 |
| 5 | 33 | 35 | no | 20 |

## One tournament across the levels

63 teams: the six most used Pokémon (team 0), each level's equilibrium teams and best teams, and random teams (seed 1). Each level plays the same round robin.

| level | ranking share | teams in the equilibrium | τ with full battles |
|---|---:|---:|---:|
| 1 | 60.8% | 3 |  |
| 2 | 78.4% | 11 |  |
| 3 | 78.8% | 11 |  |
| 4 | 82.1% | 15 |  |
| 5 | 80.1% | 11 |  |

## Agreement with use (exploratory)

Spearman's ρ over the 40 Pokémon between use and each level's ranking score, and between use and the chance of being on the equilibrium team, with 95% bootstrap intervals over Pokémon. The pool is the 40 most used, so this asks about order within the top of the ladder.

| level | ρ, ranking score | ρ, equilibrium team |
|---|---|---|
| 1 | 0.03 (-0.25 to 0.32) | 0.13 (-0.16 to 0.45) |
| 2 | 0.07 (-0.28 to 0.39) | 0.30 (-0.02 to 0.59) |
| 3 | 0.02 (-0.34 to 0.36) | 0.10 (-0.27 to 0.46) |
| 4 | 0.08 (-0.25 to 0.36) | -0.20 (-0.52 to 0.13) |
| 5 | 0.05 (-0.29 to 0.37) | -0.00 (-0.32 to 0.29) |

## Simulator checks

- One-on-one battles: 780 pairs, 50 battles each, seed 0. Damage checks: 35,758 hits agree with level 4's rolls, 0 differ. Stats that differ from Showdown's: 0. Choices Showdown refused: 0.
