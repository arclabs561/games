# Pokémon

Two separate analyses share this folder: the type chart, and catch odds for the Emerald Kaizo ROM hack.

## The type chart

`pokemon_types.py` holds the Generation 6 chart: 18 types, and the multiplier when each attacks each.

- No set of six attacking types is super-effective against every type, and ten sets of seven are. Fighting and Ground are in all ten. Exact, by checking every combination.
- Among the type combinations that Pokémon in `pokemon_gen7.csv` have, Fairy/Steel takes the least damage summed over the 18 attacking types: 13.25, against 18 for a type with no weaknesses or resistances.

`graph.py` writes the chart as a directed graph, with 18 types and the 120 links that are not neutral, to `display/graph.json`, and serves `display/index.html`, which draws it. `graph_analysis.py` ranks the types by PageRank and betweenness with networkx. `pokemon.ipynb` scores types, type pairs, Pokémon and random teams.

## Emerald Kaizo

`emerald_kaizo/` models IVs, stats and catch odds, to decide how many wild Pokémon of a species are worth catching for a given role. For example, a nature chosen in advance turns up 4% of the time, so 74 catches give a 95% chance of at least one. `iv_analysis.md` is the generated report, `lab_notes.md` the working notes, and `rom_fingerprint.md` identifies the ROM that was analyzed, which is not included.

## Data

- `Pokemon.csv`: 800 Pokémon and forms through Generation 6 (number 721), with types and base stats. It is the Kaggle dataset [Pokemon with stats](https://www.kaggle.com/datasets/abcsds/pokemon); its terms are on that page.
- `pokemon_gen7.csv`: 1,061 Pokémon and forms up to National Pokédex number 802, partway into Generation 7, with types, abilities and base stats. The same table appears in several public repositories, compiled from pokemon.com, pokemondb.net and Bulbapedia. Its license is unknown.

Type icon designs: NovaWarp's [Poketypes + Faketypes](https://www.deviantart.com/novawarp/art/Poketypes-%2B-Faketypes-2.1-286933205) and DrewLinne's [Pokemon Types](https://www.deviantart.com/drewlinne/art/Pokemon-Types-374262033).

```bash
uv run python -m pokemon.graph                       # writes display/graph.json, serves the page on port 8000
uv run --group pokemon python -m pokemon.graph_analysis
uv run python -m pokemon.emerald_kaizo.iv_analysis   # regenerates emerald_kaizo/iv_analysis.md
uv run --group pokemon --with jupyter jupyter lab pokemon/pokemon.ipynb
```
