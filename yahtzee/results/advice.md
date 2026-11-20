# Common advice, priced

Seventeen pieces of advice from strategy pages, forums and rule booklets, each priced exactly; `advice.json` holds the numbers. The [Yahtzee README](../README.md) gives the short answers. "Exact" means computed by the dynamic program in double precision.


Each piece of advice below was taken from a strategy page, forum or rule booklet and made precise. A player follows it wherever it applies and plays optimally everywhere else, so the rest of the game adapts to it. Its **cost** is 254.5896 minus that player's exact expected score. The last two columns say how often optimal play does what the advice says anyway, and how often per game the advice applies under optimal play. Both come from the exact chances of reaching each decision. Quotes are short excerpts; "Advice, as tested" is the reading that was priced.

| Advice, as tested | Source and words | Cost | Optimal play obeys | Applies per game |
|---|---|---:|---:|---:|
| A full house goes in Full House, not Three of a Kind | [Straight Dope message board, 2020](https://boards.straightdope.com/t/is-there-an-optimum-strategy-to-yahtzee/855164): “Take the full house over a high scoring 3-of-a-kind.” | 0.05 | 98% | 0.91 |
| With two pair before the last six turns, do not keep both pairs | [Board Game Arena forum, 2026](https://forum.boardgamearena.com/viewtopic.php?t=50222): “Don't chase full houses unless we are late in the game” | 0.09 | 94% | 2.90 |
| Four 4s, 5s or 6s go in their upper box | [Cornell math department, Yahtzee strategy page](https://pi.math.cornell.edu/~mec/2006-2007/Probability/Yahtzee.htm): “take the points in the upper section if they are available” | 0.10 | 97% | 0.96 |
| Four of a kind goes in its upper box while the bonus is open | [Board Game Arena forum, 2026](https://forum.boardgamearena.com/viewtopic.php?t=50222): “Always score 4 of a kind on the top if we need it there” | 0.11 | 97% | 1.53 |
| With a lone pair of 1s or 2s early on, do not keep the pair | [Cornell math department, Yahtzee strategy page](https://pi.math.cornell.edu/~mec/2006-2007/Probability/Yahtzee.htm): “will not get you many points” | 0.22 | 64% | 0.77 |
| With three or four of a kind and its upper box open, go for the Yahtzee | [Cornell math department, Yahtzee strategy page](https://pi.math.cornell.edu/~mec/2006-2007/Probability/Yahtzee.htm): “you should definitely go for the Yahtzee” | 0.33 | 96% | 3.79 |
| With three of a kind, keep them, except in the last three turns | [Board Game Arena forum, 2026](https://forum.boardgamearena.com/viewtopic.php?t=50222): “If we throw 3 of something, always continue with it” | 0.40 | 97% | 5.39 |
| Chance takes 21 or more, except in the last three turns | [Yahtzee Manifesto, strategy cheat sheet](https://www.yahtzeemanifesto.com/yahtzee-strategy.php): “Save it for rolls of 21+ points” | 0.44 | 96% | 6.11 |
| With a small straight after the first roll, chase the large straight | [Yahtzee Manifesto, strategy cheat sheet](https://www.yahtzeemanifesto.com/yahtzee-strategy.php): “Use your remaining rolls to chase a Large Straight instead.” | 0.50 | 65% | 0.94 |
| A forced zero goes in the upper section | [Yahtzee Manifesto, strategy cheat sheet](https://www.yahtzeemanifesto.com/yahtzee-strategy.php): “Scratch it in the Upper Section (e.g. Ones)” | 1.10 | 36% | 0.23 |
| Three or four 1s, 2s or 3s go in their upper box | [1967 rule booklet, as quoted by the Yahtzee Manifesto](https://www.yahtzeemanifesto.com/yahtzee-strategy.php): “it is better to score them in the UPPER SECTION” | 1.55 | 83% | 1.76 |
| Chance stays open for the first six turns | [Hasbro rule booklet, tips and tactics](https://www.hasbro.com/common/documents/dad2af551c4311ddbd0b0800200c9a66/8302f43150569047f57eb8d746ba9d86.pdf): “Leave your CHANCE roll until later in the game” | 2.08 | 92% | 6.00 |
| With two pair and Full House open, keep both pairs | [yahtzee.org.uk strategy page](http://yahtzee.org.uk/strategy.html): “hold the two's and sixes” | 3.82 | 18% | 2.67 |
| The Yahtzee box is not scratched before the last turn | [Yahtzee Manifesto, strategy cheat sheet](https://www.yahtzeemanifesto.com/yahtzee-strategy.php): “Keep it open as long as possible” | 4.69 | 93% | 8.40 |
| A roll that fits a lower box goes there, not in the upper section | [Straight Dope message board, 2024](https://boards.straightdope.com/t/whats-the-best-yahtzee-strategy/1006177): “I generally choose the bottom” | 9.74 | 61% | 7.68 |
| Upper boxes take three or more of their face, or a zero | [Yahtzee Manifesto, strategy cheat sheet](https://www.yahtzeemanifesto.com/yahtzee-strategy.php): “Aim for at least 3-of-a-kind in every Upper box.” | 9.79 | 86% | 11.17 |
| With three of a kind and the bonus not made, stop rolling | [dice-war.com strategy page](https://dice-war.com/yahtzee/strategy.html): “bank the sure points in Four of a Kind or Three of a Kind” | 36.18 | 6% | 5.95 |

- Most advice that names one situation is cheap: nine pieces cost under half a point, and optimal play already follows seven of them over 90% of the time (the other two 64% and 65%).
- The upper-section rules cost the most when read strictly. Never writing fewer than three of a face in an upper box, except as a zero, costs 9.79; optimal play writes fewer about 1.6 times per game. Always preferring a lower box that scores (Chance aside) costs 9.74.
- Keeping two pair for the full house costs 3.82; optimal play keeps both pairs only 18% of the time. The opposite advice, from the Board Game Arena forum, costs 0.09.
- "Bank the sure points" on three of a kind, read as stopping the roll, costs 36.18, about as much as playing greedily (37.28).
