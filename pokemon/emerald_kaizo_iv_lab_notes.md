# Emerald Kaizo IV Lab Notes

## Purpose

This lab studies whether catching several wild Pokemon is worth the time in
Emerald Kaizo. The useful output is not a universal "good IV" percentile. It
is a decision model for a particular species, role, encounter method, and
play style.

The current analysis is intentionally conservative about what is known from
the downloaded ROM. The file in Downloads has a verified Emerald base header,
but its Kaizo build identity is unresolved.

## Better Questions

These questions are more decision-relevant than "how many catches for IVs above
the 90th percentile?":

- What is the unit being counted: target-species catches, all grass encounters,
  successful ball throws, or minutes of play?
- Is this an ordinary playthrough, an IV-optimization run, a Nuzlocke, or an
  RNG-manipulation workflow?
- What species and encounter location are being analyzed?
- Which natures are acceptable for the intended role?
- Is one ability required, or are both abilities usable?
- Which stats actually matter for the role: one attacking stat, Speed, bulk,
  or several of them?
- Is the objective a battle outcome such as surviving a hit or securing an
  OHKO, rather than a raw stat threshold?
- Does Hidden Power matter? If so, total IV sum is the wrong objective because
  Hidden Power depends on IV bit patterns.
- Is Synchronize available as the lead ability?
- Should capture failures, low catch rates, encounter-slot odds, and time spent
  walking be included?
- What does confidence mean here: probability of finding one by a stopping
  point, or uncertainty in an estimated probability?
- Is the ROM the published v2.1 build, the later 1.1 patch, or another repack?

## Current Decision Model

The baseline treats one successfully obtained target Pokemon as one candidate.
Each of six IVs is independent and uniform on 0..31. This gives
`32^6 = 1,073,741,824` possible IV vectors.

For a success probability `p` under independent candidates:

- Expected candidates: `1 / p`.
- Probability of success by `n`: `1 - (1 - p)^n`.
- Candidates for confidence `1 - alpha`:
  `ceil(log(alpha) / log(1 - p))`.

The exact geometric result is preferable to a concentration bound when `p` is
known under the model. Clopper-Pearson is used only for a finite Monte Carlo
estimate of `p`.

## Results So Far

Under the baseline model:

| Acceptance rule | Probability | Expected candidates | 95% confidence |
|---|---:|---:|---:|
| One chosen nature | 4% | 25 | 74 |
| Either of two role-appropriate natures | 8% | 12.5 | 36 |
| Two relevant IVs >= 16 | 25% | 4 | 11 |
| Two relevant IVs >= 16 plus either of two natures | 2% | 50 | 149 |
| Previous rule plus one preferred ability of two | 1% | 100 | 299 |
| All six IVs >= 16 | 1.5625% | 64 | 191 |

The all-six rule is not automatically a sensible target. It can be easier than
a role-aware rule that also requires a useful nature and ability, while still
selecting the wrong battle profile.

## Research Findings

- Gen III IVs range from 0 through 31.
- There are 25 natures. One exact nature has marginal probability 1/25.
- A non-neutral nature raises one non-HP stat by 10% and lowers another by 10%.
- For a two-ability species, one desired normal ability is usually a 1/2
  marginal event, but Kaizo changes species data and must be checked per
  species.
- In Emerald and later Gen III games, a lead Pokemon with Synchronize gives a
  50% chance to force its nature on a wild encounter. For one desired nature,
  the marginal chance is therefore 52% before frame effects.
- Gen III Hidden Power type and power depend on IV bit patterns.
- The base Emerald wild-generation source passes `USE_RANDOM_IVS` for ordinary
  land and fishing Pokemon.
- Emerald uses a deterministic 32-bit linear-congruential RNG. Grass entry,
  movement, NPCs, lead abilities, and encounter timing change the RNG state.
- Repeating a soft reset with the same timing can repeat the same candidate.
  The independent geometric model is not a frame-accurate model.
- The Kaizo creator documents removed EV gains, but the local ROM has not been
  disassembled enough to prove that behavior for this specific build.
- The published ROMHacking.net entry describes Emerald Kaizo v2.1, released on
  14 March 2020.
- The public SHF-Kaizo-Patches repository contains an `Emerald Kaizo 1.1.bps`
  patch described as a later update.
- RomHackDex exposes species-specific Emerald Kaizo location, method, rate, and
  level rows. This is more useful for a target-species analysis than a generic
  route summary.
- Nuzlocke Tracker summarizes encounters and boss planning, but its readable
  page did not expose complete raw slot tables or percentages for the early
  routes during this research pass.

## Local ROM Evidence

The local file is recorded in `emerald_kaizo_rom_fingerprint.md`.

- Local filename: `kaizo-emerald.gba`.
- Size: 16 MiB.
- Header: `POKEMON EMER`, game code `BPEE01`.
- SHA-256: `ceb218cf629343dff7fc284f8c2de246fbc7f3e817a24f8b096b8d7102cbac4b`.
- The adjacent vanilla ROM matches the published Emerald base checksums.
- Applying the public 1.1 BPS patch to that base produces a different ROM.
- The local file has no readable version marker in the GBA header.
- The official v2.1 download endpoint was exposed by ROMHacking.net but
  returned HTTP 403 when fetched directly, so the official patched output hash
  is not yet available.

## Validation Log

The analysis lives in `emerald_kaizo_iv_analysis.py`, the parameterized role
query lives in `emerald_kaizo_role_query.py`, and the generated report is
`emerald_kaizo_iv_analysis.md`.

- Exact IV-sum distribution mass equals `32^6`.
- Exact IV-sum distribution is symmetric around sum 93.
- Exact mean six-IV sum is 93.
- Known threshold: all six IVs >= 16 has probability `1/64`.
- Known 95% geometric catch count for `p = 1/64` is 191.
- Clopper-Pearson beta inversion passes edge-case tests.
- Seeded simulation is reproducible.
- Eleven focused pytest tests pass after adding the Gen III stat checks.
- Thirteen focused pytest tests pass after adding encounter-step validation.
- One million-candidate simulation produces a two-sided 95% exact interval for
  each displayed estimate.
- A separate 1,000-replicate coverage check for `p = 1/64` covered the true
  probability in 964 intervals, or 96.4%.

## Lab Events

### Baseline IV model

Question: Is catching several copies useful if the only target is high IVs?

Action: Enumerate the six-IV sum distribution and derive geometric waiting
counts.

Result: Exact marginal probabilities are easy to compute, but all-six and total
sum criteria are poor proxies for a battle role.

### Nature and role review

Question: Should nature and ability be part of the acceptance criterion?

Action: Add illustrative filters for acceptable nature sets, relevant IVs, and
one preferred ability.

Result: Yes. Two relevant IVs plus two acceptable natures has probability 2%;
adding one preferred ability of two reduces it to 1%.

### Concentration-bound review

Question: Can a tighter generic bound replace the simulation?

Action: Compare Hoeffding with exact geometric arithmetic and
Clopper-Pearson intervals.

Result: Exact combinatorics is the right method for the marginal model.
Clopper-Pearson is the conservative finite-sample interval for Monte Carlo.
Hoeffding is too loose for rare targets.

### ROM identity review

Question: Can the local file be treated as verified Emerald Kaizo v2.1?

Action: Hash the file, verify the adjacent vanilla base, compare against the
public 1.1 patch, and inspect the GBA header.

Result: No. The local file is a real Emerald-based Kaizo candidate, but its
specific build remains unresolved.

### Parameterized role query

Question: Can the analysis answer a real acceptance rule without pretending
that all six IVs matter equally?

Action: Add `emerald_kaizo_role_query.py` with explicit inputs for acceptable
natures, relevant IV count and threshold, ability requirement, target-species
encounter share, encounter-check probability, capture probability, Synchronize,
and confidence.

Result: The default example of two acceptable natures, two relevant IVs at
least 16, and one preferred ability of two returns probability 1%, expected
100 attempts, and 299 attempts for 95% confidence.

### Early-level stat sanity

Question: Does a high IV produce a meaningful stat difference at the level
where the Pokemon is caught?

Action: Add `emerald_kaizo_gen3_stats.py` with the zero-EV Generation III stat
formulas and test Sandshrew's Attack and Defense at levels 6 and 22.

Result for Kaizo Sandshrew's base Attack 75 and Defense 85:

| Stat scenario | IV 0 | IV 31 | Difference |
|---|---:|---:|---:|
| Attack, level 6, boosted nature | 15 | 16 | 1 |
| Defense, level 6, boosted nature | 16 | 18 | 2 |
| Attack, level 22, boosted nature | 41 | 48 | 7 |
| Defense, level 22, boosted nature | 46 | 53 | 7 |

This is why early normal-play IV fishing is usually a poor trade. The
statistical difference exists, but the immediate level-6 payoff is tiny; a
useful nature, ability, moveset, and species role matter more.

### Capture-adjusted Sandshrew case

Question: How much does the actual capture process change the grass-encounter
estimate?

Action: Add `emerald_kaizo_capture.py` using the Gen III source routine's
integer rounding, ball bonuses, status multipliers, and four shake checks.

Scenario: Sandshrew with catch rate 255, maximum HP 22, an ordinary Poké Ball,
and the quality rule above.

| Capture state | Modified value | Per-ball probability | Expected balls |
|---|---:|---:|---:|
| Full HP, no status | 85 | 33.6947% | 2.97 |
| Half HP, no status | 170 | 78.4617% | 1.27 |
| 1 HP, no status | 247 | 99.9939% | 1.00 |
| 1 HP, paralysis | 370, automatic | 100% | 1.00 |

Combining the full-HP per-ball probability with the 2% quality rule and 20%
Route 102 species share gives:

- Acceptance probability per grass encounter: `0.1347787%`.
- Expected grass encounters: `741.96`.
- Grass encounters for 95% confidence: `2,222`.

At half HP, the corresponding values are approximately 0.3138468%, 318.63,
and 954. These are deliberately scenario-specific. A real playthrough can
weaken the target, inflict status, use a different ball, or lose encounters to
failed throws and battle risk.

### Grass steps versus encounter events

Question: How many steps, rather than encounter events, does this represent?

Evidence: The base Emerald source uses `MAX_ENCOUNTER_RATE = 2880` and
multiplies a map's land encounter rate by 16 before comparing it with a random
value. A base Route 102 encounter rate of 20 therefore gives `20 * 16 / 2880 =
1/9` per eligible step before bike, repel, flute, ability, and metatile effects.
The local Kaizo map rate is not yet verified.

Using that base-mechanics scenario with the full-HP Poké Ball probability:

- Acceptance probability per eligible step: `0.0149754%`.
- Expected eligible steps: `6,677.61`.
- Eligible steps for 95% confidence: `20,003`.

At half HP, the corresponding values are approximately 0.0348719%, 2,867.64,
and 8,590. These are not claims about the local Kaizo ROM's exact step rate;
they show why encounter events and walking time must remain separate units.

### Worked early target: Sandshrew

Question: What does a sensible normal-play target look like in the early game?

Evidence: RomHackDex lists Sandshrew in Route 102 grass at 20%, level 6, with
one ability, Sand Veil, and Kaizo base stats of 75 Attack and 85 Defense. It
also lists a 255 catch rate, but this case does not yet model ball failure.

The Route 102 grass table has 12 slots: Meowth 20%, Sandshrew 20%, Nidoran
male 10%, Electrike 10%, Hoothoot 10%, Gulpin 10%, Spinarak 5%, Spoink 5%,
Farfetch'd 4%, Ralts 4%, Minun 1%, and Pikachu 1%. All are listed at level 6
except Spinarak at level 7, Ralts at level 4, and Minun/Pikachu at level 5.

Rule: Accept Adamant or Impish, require Attack and Defense IVs >= 16, and do
not filter ability. This is a physical and defensive role example, not a claim
that those are the only good choices.

Result among successfully obtained Sandshrew:

- Acceptance probability: `2/25 * (16/32)^2 = 2%`.
- Expected Sandshrew catches: 50.
- Sandshrew catches for 95% confidence: 149.

Result among all Route 102 grass encounters, using the listed 20% species
share:

- Acceptance probability: `20% * 2% = 0.4%`.
- Expected grass encounters: 250.
- Grass encounters for 95% confidence: 748.

This is a useful sanity check, not a recommendation to hunt until 95%
confidence. In a normal playthrough, spending hundreds of encounters before
Roxanne is irrational. The sensible policy is likely to keep the first
serviceable Sandshrew, or filter only for a non-harmful nature and accept its
role-relevant IVs opportunistically.

## Next Experiments

- Obtain or verify the official v2.1 patched output, then compare its hash with
  the local ROM.
- Extract the local ROM's wild encounter tables and map-specific slot weights.
- Add a target configuration for species, location, encounter method, nature
  set, ability, relevant IV thresholds, and capture policy.
- Implement Gen III stat formulas with zero EVs and compare candidates against
  battle thresholds rather than generic percentiles.
- Add Hidden Power predicates when a role depends on that move.
- Model target-species slot odds and encounter rates when counting grass steps.
- Use RomHackDex or the local ROM to verify the target species' actual route,
  method, rate, and level before applying an encounter-share multiplier.
- Add catch-rate and ball/status calculations when counting throws rather than
  successfully obtained candidates.
- Add a deterministic Emerald LCG sequence model only after the ROM build and
  encounter timing are identified.

## Sources

- [Bulbapedia: Individual values](https://bulbapedia.bulbagarden.net/wiki/Individual_values)
- [Bulbapedia: Stat](https://bulbapedia.bulbagarden.net/wiki/Stat)
- [Bulbapedia: Nature](https://bulbapedia.bulbagarden.net/wiki/Nature)
- [Bulbapedia: Catch rate](https://bulbapedia.bulbagarden.net/wiki/Catch_rate)
- [Bulbapedia: Personality value](https://bulbapedia.bulbagarden.net/wiki/Personality_value)
- [Bulbapedia: Synchronize](https://bulbapedia.bulbagarden.net/wiki/Synchronize_(Ability))
- [Bulbapedia: Hidden Power calculation](https://bulbapedia.bulbagarden.net/wiki/Hidden_Power_(move)/Calculation)
- [ROMHacking.net: Emerald Kaizo v2.1](https://www.romhacking.net/hacks/4291/)
- [PokéCommunity: creator thread](https://www.pokecommunity.com/threads/pokemon-emerald-kaizo.395830/)
- [pret/pokeemerald: wild encounter generation](https://raw.githubusercontent.com/pret/pokeemerald/master/src/wild_encounter.c)
- [pret/pokeemerald: capture routine](https://raw.githubusercontent.com/pret/pokeemerald/master/src/battle_script_commands.c)
- [pret/pokeemerald: RNG implementation](https://raw.githubusercontent.com/pret/pokeemerald/master/src/random.c)
- [TASVideos: Gen 3 RNG mechanics](https://tasvideos.org/GameResources/GBA/PokemonGen3/RNG)
- [SHF-Kaizo-Patches](https://github.com/CreamElDudJafar/SHF-Kaizo-Patches)
- [RomHackDex: Emerald Kaizo Pokedex](https://romhackdex.net/emerald-kaizo/pokedex/)
- [RomHackDex: Sandshrew](https://romhackdex.net/emerald-kaizo/pokedex/sandshrew/)
- [RomHackDex: Route 102](https://romhackdex.net/emerald-kaizo/locations/route-102/)
- [Nuzlocke Tracker: Emerald Kaizo guide](https://nuzlocketracker.org/guides/emerald-kaizo)
- [Third-party itch reupload metadata](https://pokemongba.itch.io/pokemon-kaizo-emerald)
