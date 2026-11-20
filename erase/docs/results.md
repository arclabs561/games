| Result | Value | How it is known |
|---|---|---|
| Deals stuck under the printed rules (pairs that sum to 10) | {{pairs.zero.percent:.2f}}% | proved by hand<!--raw-typst  (@prop-pairs)-->, then counted over every deal (`count.py --pairs`) |
| Deals stuck with addition only | {{addition.zero.percent:.2f}}% | exact count (`count.py --addition`) |
| **Deals stuck with +, − and ×** | **{{deals.stuck}} of {{deals.total}}, about 1 in {{deals.one_in}}** | brute-force count (`count.py`); the classification<!--raw-typst  (@thm-class)--> is proved using the theorem on big hands, which is checked case by case |
| Groups needed to clear a deal | one: the whole hand plus at most one center card | proved<!--raw-typst  (@thm-single)--> using the theorem on big hands<!--raw-typst  (@thm-big)-->, whose {{certificates.cases}} cases are checked with certificates |
| Stuck hands without a 0 | caught by a remainder argument, mod 3 or mod 4 | proved<!--raw-typst  (@thm-class)--> using the theorem on big hands |
| Cards 1 to 10 only (40 cards) | {{no_zero.stuck}} of {{no_zero.total}} deals stuck, about 1 in {{no_zero.one_in:million}} | brute-force count (`count.py --no-zero`), and proved<!--raw-typst  (@prop-oneten)--> using the theorem on big hands |
| With the four kings as wild cards | {{wild.stuck}} of {{wild.total}} deals stuck, about 1 in {{wild.one_in}} | proved by hand: a deal with a wild card clears in one group<!--raw-typst  (@prop-wild)--> |
| Targets 1 to 30: the hand size from which every hand that cannot make t is blocked by a remainder | {{targets.thresholds:or}}; 8 for {{targets.threshold_counts.8}} of the targets | every multiset of 1 to {{targets.max_cards}} cards checked, with certificates and a separate checker<!--raw-typst  (@sec-targets)--> |
| Countdown problems with an exact solution | {{countdown.exact}} of {{countdown.problems}} ({{countdown.exact_percent}}%) | recomputed; agrees with Alliot (2015)<!--raw-typst  (@sec-relatives)--> |
| 24 game hands that make 24 | {{twenty_four.make.fraction}} of {{twenty_four.hands}} | exact, checked against a brute-force listing; agrees with published counts<!--raw-typst  (@sec-relatives)--> |
