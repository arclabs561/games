# Archived Naev trade routes

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

**Naev.** Naev's `LICENSE` puts its source code under GPL-3 and, except where noted, every other file, XML files included, under CC BY-SA 4.0 or later. The system positions are in the XML files, so they fall under CC BY-SA, not the GPL. That license allows copying and adapting them, with credit and under the same license. So `naev.json` keeps the positions, and the map above, drawn from them, is under CC BY-SA 4.0 with credit to the Naev developers. Rebuilding `naev.json` with the positions added leaves every other line as it was.

`naev.json` is derived from the XML files in [naev/naev](https://github.com/naev/naev) at commit `65cf3a39b892614c003ac13d24e43855e98fa36b`. Naev's `LICENSE` puts its XML data under the Creative Commons Attribution-ShareAlike 4.0 license, so `naev.json` is under CC BY-SA 4.0 too, with credit to the Naev developers. `uv run python -m ev_nova.naev --checkout PATH` rebuilds it from a checkout.

The prices follow `economy_initialiseCommodityPrices` in Naev's `src/economy.c`, which is under GPL-3. `naev.py` is a separate implementation of the same arithmetic. Each spob's price starts from the commodity's base price, times modifiers for the spob's class, population, faction and presence range. The system's radius, nebula volatility and interference raise it. Then each price is averaged with the others in its system, and each system's average with its neighbors'. Over time a price swings around this mean as the sum of two sine waves. One wave, of 3% to 8% of the price, has a period of 2,000/(jumps + 1) periods and is shared by all stops in a system. The other wave's period comes from the length of the spob's landscape image name minus 19. Naev computes that length in unsigned arithmetic, so names shorter than 19 characters (all of them) give a period near 10^17 periods, and that wave never moves. This is read from the code, not observed in the game. Only the eight standard commodities are modeled; one spob also trades Neblaze, which no other spob trades.


## Open questions

- Whether Naev's mean prices match the game. A running copy of Naev could check a few spobs.
- Time. A jump in Naev takes flying time that depends on the system, and the source comments put a jump at about three periods and a landing at about one. Rates per period would need a model of flight time. The map positions in `naev.json` are a first step: a jump point marked `autopos` lies at the system's radius, in the direction of the target system (`src/space.c`), so the distance flown inside a system between two such points can be computed. Ship speeds and the time of the jump itself are still needed.
- Missions against trading. Cargo missions in both games pay for delivery and need no capital. They are not modeled.
- Price drift in Naev, used on purpose. Prices in a system rise and fall together, with periods of a few hundred periods, so waiting for a good phase may beat flying at once.
- Story changes. EV Nova's galaxy changes with control bits, and Naev's with its campaign. Only the start of an EV Nova game and Naev's base data are studied. EV Nova's disasters (`öops`), which shift one price for a while, are left out, as are hypergates and wormholes.
- The prices of EV Nova's special commodities, and whether a player's rank changes commodity prices.
- Legal limits. Some special commodities are contraband, and a player without enough standing with a Naev spob's faction cannot land there.

## Run the archived analysis

`uv run --group ev_nova python -m ev_nova.study --naev` reproduces the comparison.
