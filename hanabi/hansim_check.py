"""Check `hanabi.information` against HanSim, the C++ simulator of Cox et al., game by game.

    uv run python -m hanabi.hansim_check            # 2,000 games (about a minute)
    uv run python -m hanabi.hansim_check -n 10000

It clones github.com/rjtobin/HanSim into a temporary directory, compiles it with g++ and a
small driver written here that prints each game's score, and plays the same decks in Python
(`hansim_decks` reproduces the C library's drand48 shuffle after srand48(100)). HanSim is
under the GPL and is not copied into this repository; it needs git and g++.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from hanabi.information import play_info_game, summarize
from hanabi.rules import Rules, hansim_decks

DRIVER = """
#include <cstdlib>
#include <iostream>
#include "info_player.cpp"
#include "hansim.cpp"
int main(int argc, char** argv) {
  int n = atoi(argv[1]);
  srand48(100);
  for (int i = 0; i < n; i++) { HanSim h; std::cout << h.play(false, false) << "\\n"; }
}
"""


def hansim_scores(n: int) -> list[int]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "HanSim"
        url = "https://github.com/rjtobin/HanSim"
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(src)], check=True)
        (src / "driver.cpp").write_text(DRIVER)
        exe = src / "driver"
        subprocess.run(["g++", "-O2", "-o", str(exe), str(src / "driver.cpp")], check=True)
        out = subprocess.run([str(exe), str(n)], capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=2000)
    args = parser.parse_args()
    theirs = hansim_scores(args.n)
    rules = Rules(players=5, discard_when_full=True)
    decks = hansim_decks(100)
    ours = [play_info_game(rules, next(decks)).score for _ in range(args.n)]
    differ = [i for i in range(args.n) if ours[i] != theirs[i]]
    print(f"HanSim: {summarize(theirs)}")
    print(f"here:   {summarize(ours)}")
    print(f"games with different scores: {len(differ)} of {args.n}", differ[:10])
    print("score counts:", dict(sorted(Counter(theirs).items())))


if __name__ == "__main__":
    main()
