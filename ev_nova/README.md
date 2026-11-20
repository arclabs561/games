# EV Nova and Naev

Which loop of planets earns a trader the most per jump, and can the routes players recommend be beaten? In Naev, Ore and Medicine between Burnan and Lora-Mor, one jump apart, earn 394 credits per ton per jump. Counting jumps, with one landing per system visited, no loop does better. In EV Nova the best is 217, and the routes on the EV Nova wiki include the best one for every landing cost tried. Exact, with a certificate of optimality for every best route.

![A schematic map of EV Nova systems as circles with their trading stops: Sol, Kania, Altair, Tuatha, Dani Fort, Dani, Dani Evera and Ver'ashan. Lines join the stops of the best routes, labeled with jumps per leg.](figures/evnova_routes.svg)

*The EV Nova routes that are best for some cost of a landing, on a schematic map; positions are not the game's, because EV Nova's data are Ambrosia Software's copyright and are not in the repository (see Data). Blue legs are on the wiki's route pages, and the orange one is an equal best round trip that they do not list.*

In *Escape Velocity Nova* (Ambrosia Software, 2002) and in [Naev](https://naev.org), an open-source game in the same style, a trader buys goods on one planet, flies to another and sells them. This folder treats the choice of where to trade as an optimization problem: which loop of planets earns the most per jump, how cargo space, credits and fuel change that loop, and whether the routes players recommend can be beaten.

## Rules as modeled

- A **stop** is a planet or station with a commodity exchange. Each commodity it trades has one price there. In both games the price is the same whether the player buys or sells.
- A **leg** goes from one stop to the next by the fewest jumps. It carries the one commodity whose price gains the most between the two stops, or nothing if none gains. Profit is linear in the load, so a full hold of the best commodity is optimal unless credits run short.
- A **route** is a cycle of stops, flown over and over. Its **rate** is the gain per ton of one pass divided by its **cost**: the jumps, plus a fixed cost for each landing, counted in jumps.

The cost of a landing matters. Both games have two stops in one system with a price gap, so with free landings the gain per jump has no upper bound. Two ways around this are used below. One counts jumps only, but lets the ship land once per visit to a system. The other charges each landing a cost x, and follows the best route as x changes.

The best route is a cycle with the largest ratio of gain to cost. `routes.py` finds it with Howard's policy iteration (Cochet-Terrasson et al., 1998), first in floating point and then exactly with fractions. It returns a potential for each stop. For every leg from a to b, the potential of a is at least the leg's gain minus the rate times its cost, plus the potential of b. Adding these up around any cycle shows that no cycle has a higher rate. `check_certificate` checks those inequalities leg by leg.

## Results

Every number is exact unless it says otherwise. Rates are in credits per ton at the mean prices.

| Result | Value | How it is known |
|---|---|---|
| Naev: best round trip | Burnan and Lora-Mor, 394 per jump | exact; Howard's method with a checked certificate, and Karp's algorithm on a second model |
| Naev: best route, one landing per system visited | no route beats 394 per jump | same |
| Naev: best route as the landing cost x goes from 1/10 to 10 | changes at x = 29/168 and x = 111/86; 197 at x = 1 | exact sweep, linear in x between changes |
| Naev: hidden jumps, fuel, cargo space | do not change the best route at x = 1 | exact |
| Naev: price drift on the best route | one pass gains 748 to 828, against 788 at mean prices | sampled every half period over 20,000 periods |
| EV Nova: best round trip | 217 per jump, reached by three round trips | exact, as for Naev |
| EV Nova: published routes | every route that is best somewhere is on the wiki's pages | exact scoring of the 33 published routes |
| Naev, counting game time: best route for a Llama | Burnan and Samell, 168.36 per period; Burnan and Lora-Mor 160.00 | exact under a flight model read from Naev's source; Howard's method with a checked certificate ([per period](#naev-what-does-a-route-earn-per-period)) |
| Same, as speed and thrust are scaled together | the best route changes near 0.36 and 16.8 times the Llama's | exact at each scale; changes located by bisection |
| Naev's prices | from Naev's price model, read from its source | not compared with a running game |

## How the results are checked

- Howard's method is compared with brute force, which lists every simple cycle, on 400 random galaxies, each at four landing costs. The best route is always a simple cycle: a cycle that visits a stop twice splits into two shorter cycles, and its rate lies between theirs.
- Karp's algorithm (1978) finds a maximum-mean cycle in a graph where every edge costs one unit. It runs on a second model of the same problem, with one node for each system and cargo held. Its answer matches Howard's on the random galaxies, and on both games when counting jumps only and when a landing costs one jump. Karp's method is also compared with brute force on 300 random graphs.
- Every best route on the real maps passes `check_certificate`.
- The sweep over landing costs rests on one fact: whether one route beats another at a landing cost x is a linear inequality in x. A random test compares its answer with brute force at up to 1,000 costs.
- Betweenness (Brandes, 2001, in exact fractions) and chokepoints are compared with brute force on 200 random graphs each.
- The Naev price model is tested on hand-computed cases. It has not been compared with prices in a running game, which is the largest uncertainty here.
- The EV Nova reader is tested on a small hand-written ConText file. Earth's flags decode to the levels the wiki lists, and the 80% and 125% rule reproduces the wiki's table for the Federation.
- A test checks that the committed SVGs are what `figures.py` draws, and that the Naev routes it draws tie exactly where the best route changes.

CI runs the tests on a small made-up galaxy and on Naev's committed data. The EV Nova data are not in the repository, so the test that reads them runs only where `EV_NOVA_CONTEXT` is set.

## Naev: what is the best route?

The data are Naev's XML files at commit `65cf3a39b892` (November 13, 2026). Prices come from Naev's price model, described under Data. Every rate here is in credits per ton at the mean prices, and every number is exact unless it says otherwise.

- The map has 554 systems and 331 stops. All of them trade the same eight commodities. With hidden jumps left out, it has 689 two-way links in 12 connected parts. The largest part has 542 systems, and its diameter is 44 jumps.
- The best round trip between two systems is Burnan (Scarlet) and Lora-Mor (Druss), one jump apart. Ore gains 117 one way and Medicine 671 the other, for 394 per jump.
- Counting jumps only, with one landing per visit to a system, no route beats 394 per jump. Howard's method and Karp's algorithm agree on this, as described under checks below.
- With a landing cost x, the best route is Dvaer Prime and Eist Shipyard (both in Dvaer, 116 per pass, no jumps) up to x = 29/168. From there to x = 111/86 it is Burnan and Lora-Mor. From there to x = 10 it is Modus and Tankard Cloister, two jumps apart, with 1,132 per pass. At x = 1 the rate is 197.
- Adding hidden jumps does not change the best route at x = 1. Nor does a fuel tank that holds only one or two jumps, when each refueling costs a landing. The best legs are one or two jumps long, and every system on them sells fuel.
- Cargo space does not change the best route: the gain of every route grows in proportion to it. Credits do, when they cannot fill the hold. At x = 1 with 50 tons, a trader with 2,000 credits does best carrying Water and Ore between Eenerim and Modus. With 10,000 credits, Ore and Medicine between Modus and Tankard Cloister are best. With 100,000 credits, the best route is Burnan and Lora-Mor again.
- Prices drift around their means. One pass of Burnan and Lora-Mor gains 788 at mean prices, and between 748 and 828 at a single moment. This range is sampled every half period over 20,000 periods, ignoring the time the flight takes.
- The systems with the highest betweenness are Iris, Holly and Tau Ceti. Betweenness counts, for each pair of systems, the share of their shortest paths that pass through a system; Iris has 27.2% of the largest possible total. Of the 542 systems, 108 are chokepoints, whose removal splits the map, and 83 are dead ends.

No published Naev trade guide was found to compare with. Two searches turned up only guides for other games.

![A map of Naev's 554 systems as dots joined by jump lines, taller than wide. Three routes are drawn thick and labeled in the lower right part of the map: Dvaer alone, Scarlet and Druss one jump apart, and Modus Manis and An'ku two jumps apart by two paths. Holly, Tau Ceti and Iris, near the middle, are circled.](figures/naev_map.svg)

*Naev's map, drawn from the system positions in its data files. As in Naev's own map, y points up: its view matrix puts y = 0 at the bottom (read from the code). The three routes that are best at some cost of a landing lie close together in the south. The systems with the highest betweenness are near the middle. Under CC BY-SA 4.0, like the data it is drawn from.*

![A log-log plot of credits per ton per unit of cost against the cost of a landing, from 1/10 to 10 jumps, for three routes; the best rate is drawn thick, from Dvaer Prime and Eist Shipyard at the lowest costs, to Burnan and Lora-Mor, to Modus and Tankard Cloister above 111/86.](figures/naev_landing.svg)

*Naev: the rate of each route that is best for some landing cost. Each rate is the gain per pass over jumps plus x per landing; the thick line is the best of the three. Where two lines cross, the best route changes.*

## Naev: what does a route earn per period?

Counted in game time, the best route for a Llama is Ore and Medicine between Burnan (Scarlet) and Samell (Druss): 785 credits per ton per pass in 4.66 periods, or 168.36 per period. Burnan and Lora-Mor, the best route per jump, earn 160.00 per period. Both are round trips of one jump each way. Samell gains 3 credits less per pass, but it lies 13,159 units from the jump point to Scarlet, where Lora-Mor lies 29,336. Exact under the time model below, with a certificate. The model is a reading of Naev's source and has not been compared with a running game.

The question is a maximum cost-to-time ratio cycle, the problem of G. B. Dantzig, W. O. Blattner and M. R. Rao (1967), whose motivating example is a ship for hire choosing its cycle of ports. Here the gain of a leg is its best commodity's price gain per ton, and its time is the game time from take-off to the next take-off. `flight.py` computes each leg's time, the least over all paths, by Dijkstra's method. Howard's method and `check_certificate` from `routes.py` then do the rest. They work in exact fractions, each equal to the floating-point time of its leg.

**The time model**, from Naev's source at the commit of `naev.json`:

- A period is 10,000 time increments, and game time runs at 30 increments per second of flight (`dat/constants.lua`; `src/naev.c` passes the game's time step to `ntime_update`). So 333 seconds of flight cost a period.
- A jump costs 10,000 increments times the ship's jump delay (`pilot_hyperspaceDelay`, `src/pilot.c`), plus 3 seconds of engine start and 5 of acceleration. A landing costs 10,000 increments times its landing delay (`src/land.c`). Both delays are 1 for the Llama.
- A ship arrives at the jump point that leads back to the system it came from. A jump point marked `autopos` lies at the system's radius, toward its target (`system_reconstructJumps`, `src/space.c`).
- Each flight inside a system starts and ends at rest; a ship must slow to 25 units per second before it jumps (`src/space.h`). It thrusts at its acceleration up to its speed, above which a drag of 4 × (v − speed) per second holds it near speed + accel / 4 (`src/physics.c`). To stop, it turns half a circle with the engine off, then thrusts against its motion (`pilot_brake`, `src/pilot.c`).
- Left out: the offset at which a ship arrives from a jump, the radius within which it may jump, the first turn after take-off, and the landing and take-off animations.

The Llama, with its default outfits (`dat/ships/neutral/llama.xml` and the outfit files it names), has speed 295, acceleration 299 and turn rate 181 degrees per second, so its top speed is 369.75. Its hold is 30 tons, and full it weighs 128 tons, below its engine's limit of 160, so cargo does not slow it.

| Llama, round trips | Stops | Gain per pass | Periods per pass | Per period |
|---|---|---:|---:|---:|
| Best per period | Burnan and Samell | 785 | 4.6625 | 168.36 |
| Second | Manis and Tankard Cloister (two jumps each way) | 1,123 | 6.8258 | 164.52 |
| Best per jump | Burnan and Lora-Mor | 788 | 4.9250 | 160.00 |

No route of three or more stops beats Burnan and Samell; the certificate proves this for every cycle. Of the 4.66 periods, jumps and landings take 4.048 and flight 0.61, or about 205 seconds. Naev's `src/economy.c` has a comment that puts a journey of one jump at about 3 periods. The model here gives 2.33 for each leg of the best route.

**Faster ships.** With speed and thrust scaled together by a factor k, the best route changes twice. Below k ≈ 0.36 (the Llama's speed of 295 scaled to about 106), it is Khelim and Nonuri. From there to k ≈ 16.8, it is Burnan and Samell. Above it, Burnan and Lora-Mor are best again, as they are per jump, with 194.66 per period when flight takes no time. The rates are exact at k = 1/4, 1/2, 1, 2, ..., 64 and with no flight time. Each change is located by bisection on log k, to within a 1% band.

**How these are checked.** The closed form for one flight matches a step-by-step integration of the same motion, in steps of a millisecond, to 0.01 seconds, and long flights cost 1 / (top speed) per unit of distance, as they must. The leg times match a brute force over every walk of up to seven jumps on 12 random small galaxies. Howard's method matches a list of every simple cycle on 40 more. A test recomputes the Llama's results from the committed data and compares them with `flight.json`.

## EV Nova: what is the best route?

EV Nova's data are not in this repository (see Data). These results come from a ConText export of EV Nova 1.1.1, read at the start of a game, when 398 of the 545 system records are visible.

- The map has 398 systems, 582 links and one connected part. Its diameter is 24 jumps, and the center is Kania. There are 141 stops with a commodity exchange. Of these, 128 trade between one and six of the six standard commodities, and 13 trade none.
- Sol has the highest betweenness, 52.8% of the largest possible total, and Tichel the next, 39.9%. Both are chokepoints, as are 58 other systems.
- The best round trip is 217 per jump. Three round trips reach it: Port Kane (Kania) and Altia (Altair), Forticus (Dani Fort) and Dani (Dani), and Forticus and Palein (Dani Evera). Each carries Equipment from low to high (+247) and Medical Supplies from medium to high (+187), one jump each way.
- Counting jumps only, with one landing per visit, no route beats 217.
- With a landing cost x, the best route is Ver'ar Shan and Ver'ar Sho, two stations in Ver'ashan, up to x = 167/77. It gains 334 per pass with no jumps. From there to x = 943/304, the best route is Earth, then Port Kane, then Europa: 809 per pass in four jumps. Above that it is Earth and New Ireland (Tuatha), 742 per pass in six jumps.
- Fuel does not change the answer at x = 1, for the same reason as in Naev. Neither do the special commodities, if they cost 80% of their base price where sold and fetch 125% where bought (an assumption; see below).
- A new player has a 10-ton Shuttle and 25,000 credits, which fill the hold; the best rate at x = 1 is 1,670 credits per unit of cost. A 125-ton Terrapin with the same credits can fill only part of its hold. The best route is the same, but its rate is 10,004.5, against 20,875 with 100,000 credits.

## Do the published EV Nova routes hold up?

The EV Nova wiki lists recommended routes on its [Trade Routes](https://evnova.miraheze.org/wiki/Trade_Routes) and [Making Money](https://evnova.miraheze.org/wiki/Nova:Making_Money) pages. `guides.py` holds the 33 that use only standard commodities and no hypergate. `study.py` checks the price levels and jump counts the pages give, and scores each route.

- Every route that is best somewhere in this study is on those pages. Kania/Altair and Dani/Dani Fort are best per jump. Ver'ashan is best for landing costs up to 167/77. The Kania/Sol triangle and Sol/Tuatha are best for higher costs. No better route was found, though one equal round trip is not listed: Forticus and Palein.
- Every price level on the pages matches the data except two, both on the Tekel Kymon/Eguit route. At the start of a game Kymonthia trades no Medical Supplies and Eguit no Industrial. That route gains 15 per ton per pass in the data. The page may describe a later state of the galaxy; this was not checked.
- The jump counts agree, except Aria to Dani Fort, which the page gives as five jumps and the data as four.
- The wiki's commodity table gives Polaris equipment as 600, 750 and 825, and some Auroran prices that do not follow the 80% and 125% rule. The data have one base price per commodity (STR# 4004). Ranks carry a price modifier (`PriceMod` in the rank resources), which might explain the difference; this was not checked.

## Data

| Source | What | License | Committed |
|---|---|---|---|
| [naev/naev](https://github.com/naev/naev) XML data, commit `65cf3a39b892` | systems with their map positions, jumps, spobs, commodities | CC BY-SA 4.0 | yes, as `naev.json` and `figures/naev_map.svg` (CC BY-SA 4.0) |
| The same XML data: systems' radii, jump points, spob positions, ships and outfits | the geometry and the Llama's flight numbers | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), by the Naev developers | yes, as `naev_flight.json`, adapted: autopos jump points resolved, outfit values summed (CC BY-SA 4.0) |
| Naev's `src/*.c`, `src/core/src/*.rs` and `dat/constants.lua` | the time model's constants and rules | GPL-3 (code), CC BY-SA 4.0 (data) | no; read, and restated in `flight.py` |
| Naev's `src/economy.c` | the price model | GPL-3 | no; reimplemented in `naev.py` |
| EV Nova 1.1.1, ConText export from [vasi/evnova-utils](https://github.com/vasi/evnova-utils) | systems, stops, prices | Ambrosia Software's copyright | no |
| EV Nova wiki | recommended routes, price levels, the 80% and 125% rule | not checked | the routes' stops, jumps and price levels, in `guides.py` |

**Naev.** Naev's `LICENSE` puts its source code under GPL-3 and, except where noted, every other file, XML files included, under CC BY-SA 4.0 or later. The system positions are in the XML files, so they fall under CC BY-SA, not the GPL. That license allows copying and adapting them, with credit and under the same license. So `naev.json` keeps the positions, and the map above, drawn from them, is under CC BY-SA 4.0 with credit to the Naev developers. Rebuilding `naev.json` with the positions added leaves every other line as it was.

`naev.json` is derived from the XML files in [naev/naev](https://github.com/naev/naev) at commit `65cf3a39b892614c003ac13d24e43855e98fa36b`. Naev's `LICENSE` puts its XML data under the Creative Commons Attribution-ShareAlike 4.0 license, so `naev.json` is under CC BY-SA 4.0 too, with credit to the Naev developers. `uv run python -m ev_nova.naev --checkout PATH` rebuilds it from a checkout.

The prices follow `economy_initialiseCommodityPrices` in Naev's `src/economy.c`, which is under GPL-3. `naev.py` is a separate implementation of the same arithmetic. Each spob's price starts from the commodity's base price, times modifiers for the spob's class, population, faction and presence range. The system's radius, nebula volatility and interference raise it. Then each price is averaged with the others in its system, and each system's average with its neighbors'. Over time a price swings around this mean as the sum of two sine waves. One wave, of 3% to 8% of the price, has a period of 2,000/(jumps + 1) periods and is shared by all stops in a system. The other wave's period comes from the length of the spob's landscape image name minus 19. Naev computes that length in unsigned arithmetic, so names shorter than 19 characters (all of them) give a period near 10^17 periods, and that wave never moves. This is read from the code, not observed in the game. Only the eight standard commodities are modeled; one spob also trades Neblaze, which no other spob trades.

**EV Nova.** The game's data files are Ambrosia Software's copyright and are not committed. `evnova.py` reads a ConText export, a tab-separated dump of the game's resources made by the ConText/ResStore tool. Its docstring gives the sections and fields it uses. [vasi/evnova-utils](https://github.com/vasi/evnova-utils) publishes such an export of EV Nova 1.1.1 (`Context/NovaConText.txt`). Its license states that the file was generated from Ambrosia's data, which remain Ambrosia's copyright. The numbers above come from that file. Field meanings come from the Nova Bible, Ambrosia's resource documentation, as transcribed on the [EV Nova wiki](https://evnova.miraheze.org/wiki/Sp%C3%B6b). The base prices (75, 350, 750, 900, 200 and 550) are in the data. The rule that low is 80% and high is 125% of base, rounded down, comes from the wiki's [commodity table](https://evnova.miraheze.org/wiki/Commodities); the Nova Bible does not state it.

The tests use a small made-up galaxy. Set `EV_NOVA_CONTEXT` to the path of an export to run the test that reads the real data too.

## Sources

- J. Cochet-Terrasson, G. Cohen, S. Gaubert, M. McGettrick and J.-P. Quadrat, Numerical computation of spectral elements in max-plus algebra (1998): Howard's policy iteration for the best ratio of gain to cost.
- R. M. Karp, A characterization of the minimum cycle mean in a digraph, *Discrete Mathematics* 23 (1978).
- U. Brandes, A faster algorithm for betweenness centrality, *Journal of Mathematical Sociology* 25 (2001).
- The EV Nova wiki's [Trade Routes](https://evnova.miraheze.org/wiki/Trade_Routes), [Making Money](https://evnova.miraheze.org/wiki/Nova:Making_Money) and [Commodities](https://evnova.miraheze.org/wiki/Commodities) pages, read 2026-11-18.

## Open questions

- Whether Naev's mean prices match the game. A running copy of Naev could check a few spobs.
- Time, checked in the game. The rates per period above rest on a flight model read from the source. Timing one leg in a running copy of Naev, from take-off at Burnan to take-off at Samell, would test it; the model gives 2.33 periods for it.
- Missions against trading. Cargo missions in both games pay for delivery and need no capital. They are not modeled.
- Price drift in Naev, used on purpose. Prices in a system rise and fall together, with periods of a few hundred periods, so waiting for a good phase may beat flying at once.
- Story changes. EV Nova's galaxy changes with control bits, and Naev's with its campaign. Only the start of an EV Nova game and Naev's base data are studied. EV Nova's disasters (`öops`), which shift one price for a while, are left out, as are hypergates and wormholes.
- The prices of EV Nova's special commodities, and whether a player's rank changes commodity prices.
- Legal limits. Some special commodities are contraband, and a player without enough standing with a Naev spob's faction cannot land there.

## Usage

```bash
uv run --group ev_nova python -m ev_nova.study                                   # Naev, a few minutes
uv run --group ev_nova python -m ev_nova.study --evnova path/to/NovaConText.txt  # both games
uv run python -m ev_nova.naev --checkout path/to/naev                            # rebuild naev.json
uv run pytest tests/test_ev_nova.py
uv run python -m ev_nova.figures                                                 # the figures, the Naev map included (about a minute)
uv run python -m ev_nova.flight --out ev_nova/flight.json                         # routes per period, about two minutes
uv run python -m ev_nova.flight --checkout path/to/naev                          # rebuild naev_flight.json
```

## Files

| File | Contents |
|---|---|
| `routes.py` | The model, Howard's method with its certificate, Karp's algorithm, brute force, the landing-cost sweep, fuel, credits, and the graph measures |
| `naev.py` | Naev's XML reader and price model; writes `naev.json` |
| `naev.json` | Naev's systems with their map positions, jumps, trading spobs and mean prices (CC BY-SA 4.0) |
| `flight.py`, `naev_flight.json`, `flight.json` | Routes per period: the time model, the geometry and ship numbers it reads (CC BY-SA 4.0), and the results |
| `evnova.py` | Reader for a ConText export of EV Nova, with the control-bit tests that hide and show systems |
| `guides.py` | The published EV Nova routes, with their sources |
| `study.py` | Prints every result above |
| `figures.py`, `figures/` | This README's figures, as SVG; a test checks they are current |
