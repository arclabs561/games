"""Level 5: whole battles between teams, in the Pokémon Showdown simulator.

Pokémon Showdown (https://github.com/smogon/pokemon-showdown, MIT license) is the simulator
the Smogon ladder plays on. This module clones it at the commit ``sources.py`` pins, builds
it, and plays battles through its own battle stream (``teams_showdown.js``), which takes a
seed for each battle, so every battle can be replayed.

**Format.** ``gen9customgame``: Generation 9 rules with team preview and no team checks. Each
Pokémon has its fixed set from ``teams_model.py`` (level 100, IVs 31, 84 EVs each, Serious
nature, no item). Nobody Terastallizes. Everything else is Showdown's: random damage rolls,
critical hits, misses, secondary effects, every ability, weather and terrain from abilities.

**Policy, the same for both players.**
- Team preview: lead with the Pokémon whose mean level-4 matchup against the six opposing
  Pokémon is highest.
- Each turn: use the move with the most expected damage at level 4 against the opposing
  Pokémon, given both sides' current stat stages (the first such move in set order). Never
  switch by choice.
- After a faint: send in the Pokémon with the best level-4 matchup against the opposing
  Pokémon in play; if that one has fainted too, against the opposing team's survivors.
- Anything else Showdown asks (a move locked in by Outrage, a recharge): Showdown's default.

**Checks.** For every hit that lands with no critical hit, no stat stages, weather, terrain,
status or ability activation on either side, no Blaze-like boost at low HP, and one damage
line, the damage Showdown dealt
must be one of the 16 rolls of ``teams_model.damage_rolls``. Each battle's exact stats from
Showdown's requests must equal ``Pokemon.stats``.

    uv run --group pokemon python -m pokemon.teams_showdown setup --dir /tmp/showdown
    uv run --group pokemon python -m pokemon.teams_showdown play --teams teams.json \\
        --games 40 --slice 1 --slices 8 --showdown /tmp/showdown --out OUT

The battles run on GitHub Actions (``experiment.yml``); ``teams.py`` reads the outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from pathlib import Path

from pokemon import teams_model as tm
from pokemon.sources import SHOWDOWN_COMMIT

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "teams_showdown.js"
REPO = "https://github.com/smogon/pokemon-showdown.git"
FORMAT = "gen9customgame"
CONCURRENT = 24  # battles in flight per simulator process
PINCH = ("blaze", "torrent", "overgrow", "swarm")
SD_STAT = {"atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}


# Setting up the simulator.


def setup(root: Path) -> Path:
    """Clone Showdown at the pinned commit and build it; returns its dist/sim directory."""
    repo = root / "pokemon-showdown"
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", REPO, str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", SHOWDOWN_COMMIT], cwd=repo, check=True)
    sim = repo / "dist" / "sim"
    if not (sim / "battle-stream.js").exists():
        subprocess.run(["npm", "install", "--omit=dev", "--no-audit", "--no-fund", "-s"],
                       cwd=repo, check=True)  # fmt: skip
        subprocess.run(["node", "build"], cwd=repo, check=True)
    return sim


def nickname(i: int) -> str:
    """The name a Pokémon goes by in battle: 'P' and its index in the pool. Showdown would
    shorten a forme's own name ('Slowking-Galar' to 'Slowking'), so species names are not used."""
    return f"P{i}"


def showdown_set(s: tm.Set, i: int) -> dict:
    """A set in Showdown's JSON form."""
    return {
        "name": nickname(i),
        "species": s.pokemon.showdown,
        "item": "",
        "ability": s.ability,
        "moves": [m.name.replace("-", "") for m in s.moves],
        "nature": "Serious",
        "evs": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), tm.EV),
        "ivs": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), tm.IV),
        "level": tm.LEVEL,
    }


def seed(label: str) -> str:
    """A Showdown PRNG seed from a label: 'sodium,' and 64 hex digits of its SHA-256."""
    return "sodium," + hashlib.sha256(label.encode()).hexdigest()


# The policy.


@cache
def pool_sets(n: int) -> tuple[tm.Set, ...]:
    return tm.sets(n)


@cache
def level4_matrix(n: int) -> list[list[Fraction]]:
    return tm.matrix(pool_sets(n), 4)


@cache
def pair(a: tm.Set, b: tm.Set) -> tuple[tm.Fighter, tm.Fighter]:
    return tm.fighters(a, b)


def species(details: str) -> str:
    """'Kingambit, F' -> 'Kingambit'."""
    return details.split(",")[0]


class Battle:
    """One battle's state, read from Showdown's full (omniscient) log, and both players."""

    def __init__(self, key: str, teams: tuple[Sequence[int], Sequence[int]], n: int):
        self.key, self.n = key, n
        ss = pool_sets(n)
        self.sets: dict[str, list[tm.Set]] = {
            side: [ss[i] for i in t] for side, t in zip(("p1", "p2"), teams, strict=True)
        }
        self.index = {
            side: {nickname(i): i for i in t} for side, t in zip(("p1", "p2"), teams, strict=True)
        }
        self.active: dict[str, str | None] = {"p1": None, "p2": None}
        self.boosts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.hp: dict[tuple[str, str], tuple[int, int]] = {}
        self.fainted: dict[str, set[str]] = {"p1": set(), "p2": set()}
        self.dirty: dict[str, bool] = {"p1": False, "p2": False}
        self.field = False  # weather or terrain in play
        self.winner: str | None = None
        self.turns = 0
        self.errors = 0
        self.checks: Counter = Counter()
        self.stat_problems: list[str] = []
        self.pending: dict | None = None

    @staticmethod
    def other(side: str) -> str:
        return "p2" if side == "p1" else "p1"

    def set_of(self, side: str, name: str) -> tm.Set:
        return pool_sets(self.n)[self.index[side][name]]

    # Reading the log.

    def read(self, chunk: str) -> list[str]:
        """Apply one chunk; returns the choices to send, as stream lines."""
        kind, _, rest = chunk.partition("\n")
        if kind == "update":
            lines = rest.split("\n")
            i = 0
            while i < len(lines):
                if lines[i].startswith("|split|"):
                    # The secret line (exact HP) comes first, then its public copy.
                    self.line(lines[i + 1])
                    i += 3
                    continue
                self.line(lines[i])
                i += 1
            return []
        if kind == "sideupdate":
            side, _, body = rest.partition("\n")
            if body.startswith("|request|"):
                return self.request(side, json.loads(body.removeprefix("|request|")))
            if body.startswith("|error|"):
                self.errors += 1
                return [f">{side} default"]
        if kind == "end":
            return []
        return []

    def line(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) < 2:
            return
        tag = parts[1]
        if tag in ("switch", "drag", "replace"):
            side = parts[2][:2]
            self.active[side] = parts[2][5:]
            self.boosts[side] = {}
            self.dirty[side] = False
            self.set_hp(side, parts[4])
        elif tag in ("-damage", "-heal", "-sethp"):
            side, name = parts[2][:2], parts[2][5:]
            before = self.hp.get((side, name))
            self.set_hp(side, parts[3], name)
            if tag == "-damage":
                self.check(side, before, parts)
        elif tag in ("-boost", "-unboost"):
            side = parts[2][:2]
            k = parts[3]
            if k in SD_STAT:
                sign = 1 if tag == "-boost" else -1
                b = self.boosts[side]
                b[k] = max(-6, min(6, b.get(k, 0) + sign * int(parts[4])))
        elif tag in ("-clearboost", "-clearallboost", "-clearnegativeboost", "-setboost",
                     "-swapboost", "-copyboost", "-invertboost"):  # fmt: skip
            for side in ("p1", "p2"):
                self.dirty[side] = True
        elif tag in ("-weather", "-fieldstart"):
            self.field = not (tag == "-weather" and parts[2] == "none")
        elif tag in ("-activate", "-start", "-status", "-ability", "-item", "-enditem"):
            if len(parts) > 2 and parts[2][:2] in ("p1", "p2"):
                self.dirty[parts[2][:2]] = True
        elif tag == "faint":
            self.fainted[parts[2][:2]].add(parts[2][5:])
        elif tag == "move":
            self.pending = {"side": parts[2][:2], "move": parts[3], "target": parts[4] if len(parts) > 4 else "",
                            "crit": False, "hits": 0, "stages": (dict(self.boosts["p1"]), dict(self.boosts["p2"])),
                            "dirty": dict(self.dirty), "field": self.field}  # fmt: skip
        elif tag == "-crit" and self.pending:
            self.pending["crit"] = True
        elif tag == "turn":
            self.turns = int(parts[2])
            self.pending = None
        elif tag == "win":
            self.winner = parts[2]
        elif tag == "tie":
            self.winner = "tie"

    def set_hp(self, side: str, cond: str, name: str | None = None) -> None:
        name = name or self.active[side]
        if name is None:
            return
        hp = cond.split()[0]
        if hp.endswith("fnt") or hp == "0":
            self.hp[(side, name)] = (0, self.hp.get((side, name), (0, 1))[1])
            return
        if "/" in hp:
            a, b = hp.split("/")
            self.hp[(side, name)] = (int(a), int(b))

    def check(self, side: str, before: tuple[int, int] | None, parts: list[str]) -> None:
        """Compare one hit's damage with the 16 rolls of level 4, when nothing else applies."""
        p = self.pending
        if p is None or len(parts) > 4 or before is None or p["side"] == side:
            return
        p["hits"] += 1
        after = self.hp[(side, parts[2][5:])]
        attacker = self.other(side)
        if p["hits"] > 1 or p["crit"] or p["field"] or self.field:
            return
        if (
            p["dirty"][attacker]
            or p["dirty"][side]
            or any(p["stages"][0].values())
            or any(p["stages"][1].values())
        ):
            return
        a_name, d_name = self.active[attacker], parts[2][5:]
        if a_name is None or after[0] == 0:
            return
        a, d = self.set_of(attacker, a_name), self.set_of(side, d_name)
        move = next((m for m in a.moves if _title(m.name) == _id(p["move"])), None)
        if move is None or move.hits != 1:
            return
        a_hp = self.hp.get((attacker, a_name), (1, 1))
        if a.ability in PINCH and 3 * a_hp[0] <= a_hp[1]:
            return  # Blaze, Torrent, Overgrow and Swarm, which level 4 does not model
        fa, fd = pair(a, d)
        rolls = tm.damage_rolls(move, fa, fd, {}, {}, before[0] == before[1])
        self.checks["agree" if before[0] - after[0] in rolls else "differ"] += 1
        if before[0] - after[0] not in rolls:
            self.checks[f"differ:{a.name}:{move.name}:{d.name}"] += 1

    def winner_hp(self) -> list[int] | None:
        """[hp, max hp] of the winner's Pokémon in play at the end, or None without a winner."""
        if self.winner not in ("p1", "p2"):
            return None
        name = self.active[self.winner]
        return list(self.hp.get((self.winner, name), (0, 1))) if name else None

    # The players.

    def request(self, side: str, req: dict) -> list[str]:
        if req.get("wait"):
            return []
        for p in req["side"]["pokemon"]:
            s = self.set_of(side, p["ident"][4:])
            stats = dict(s.pokemon.stats)
            got = {k: p["stats"][k] for k in SD_STAT}
            got["hp"] = (
                int(p["condition"].split("/")[1].split()[0])
                if "/" in p["condition"]
                else stats["hp"]
            )
            if any(got[k] != stats[k] for k in got) and s.name not in self.stat_problems:
                self.stat_problems.append(s.name)
        foe = self.other(side)
        if req.get("teamPreview"):
            own = self.sets[side]
            m4 = level4_matrix(self.n)
            ss = pool_sets(self.n)
            idx = [ss.index(x) for x in own]
            opp = [ss.index(x) for x in self.sets[foe]]
            size = len(own)
            lead = max(range(size), key=lambda k: (sum(m4[idx[k]][j] for j in opp), -k))
            order = [lead + 1] + [k + 1 for k in range(size) if k != lead]
            return [f">{side} team {''.join(map(str, order))}"]
        if req.get("forceSwitch"):
            return [f">{side} switch {self.switch_choice(side, req)}"]
        active = req["active"][0]
        mine = self.active[side]
        if mine is None:
            return [f">{side} default"]
        me = self.set_of(side, mine)
        opp_name = self.active[foe]
        if opp_name is None:
            return [f">{side} default"]
        foe_set = self.set_of(foe, opp_name)
        fa, fd = pair(me, foe_set)
        hp = self.hp.get((foe, opp_name), (1, 1))
        best, best_d = None, None
        for k, mv in enumerate(active["moves"], start=1):
            if mv.get("disabled"):
                continue
            move = next((m for m in me.moves if m.name.replace("-", "") == mv["id"]), None)
            if move is None:
                continue
            d = tm.expected_damage(
                move, fa, fd, self.boosts[side], self.boosts[foe], hp[0] == hp[1]
            )
            if best_d is None or d > best_d:
                best, best_d = k, d
        return [f">{side} move {best}" if best else f">{side} default"]

    def switch_choice(self, side: str, req: dict) -> int:
        """The bench Pokémon with the best level-4 matchup against what it will face."""
        foe = self.other(side)
        ss = pool_sets(self.n)
        m = level4_matrix(self.n)
        opp = self.active[foe]
        if opp is None or opp in self.fainted[foe]:
            targets = [self.set_of(foe, k) for k in self.index[foe] if k not in self.fainted[foe]]
        else:
            targets = [self.set_of(foe, opp)]
        best, best_v = 1, None
        for k, p in enumerate(req["side"]["pokemon"], start=1):
            if p["active"] or p["condition"].endswith("fnt"):
                continue
            i = self.index[side][p["ident"][4:]]
            v = sum((m[i][ss.index(t)] for t in targets), Fraction(0))
            if best_v is None or v > best_v:
                best, best_v = k, v
        return best


def _id(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _title(identifier: str) -> str:
    return identifier.replace("-", "")


# Running battles.


def play(jobs: Sequence[tuple[str, Sequence[int], Sequence[int]]], sim: Path, n: int) -> list[dict]:
    """Play battles in one simulator process, CONCURRENT at a time; one result per job."""
    proc = subprocess.Popen(["node", str(DRIVER), str(sim)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)  # fmt: skip
    assert proc.stdin and proc.stdout
    stdin, stdout = proc.stdin, proc.stdout
    battles: dict[int, Battle] = {}
    results: list[dict] = []
    queue = list(enumerate(jobs))

    def start() -> None:
        k, (key, a, b) = queue.pop(0)
        battles[k] = Battle(key, (a, b), n)
        msg = {"id": k, "start": {"formatid": FORMAT, "seed": seed(key)},
               "p1": [showdown_set(pool_sets(n)[i], i) for i in a],
               "p2": [showdown_set(pool_sets(n)[i], i) for i in b]}  # fmt: skip
        stdin.write(json.dumps(msg) + "\n")

    while queue and len(battles) < CONCURRENT:
        start()
    proc.stdin.flush()
    while battles:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the simulator stopped")
        msg = json.loads(line)
        b = battles.get(msg["id"])
        if b is None:
            continue
        if msg["chunk"] is None or msg["chunk"].startswith("end"):
            results.append(
                {
                    "key": b.key,
                    "winner": b.winner,
                    "turns": b.turns,
                    "errors": b.errors,
                    "hp": b.winner_hp(),
                    "checks": dict(b.checks),
                    "stat_problems": b.stat_problems,
                }  # fmt: skip
            )
            del battles[msg["id"]]
            if queue:
                start()
            proc.stdin.flush()
            continue
        for out in b.read(msg["chunk"]):
            proc.stdin.write(json.dumps({"id": msg["id"], "write": out}) + "\n")
        if b.turns > 500:  # a battle that does not end
            b.winner = "limit"
            proc.stdin.write(json.dumps({"id": msg["id"], "write": ">forcetie"}) + "\n")
        proc.stdin.flush()
    proc.stdin.close()
    proc.wait()
    return results


def schedule(teams: Sequence[Sequence[int]], games: int, base: int) -> list[tuple[str, list, list]]:
    """Every unordered pair of teams plays ``games`` battles, sides alternating. A team may
    be a single Pokémon, for one-on-one battles."""
    jobs = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            for g in range(games):
                a, b = (teams[i], teams[j]) if g % 2 == 0 else (teams[j], teams[i])
                jobs.append((f"{base}-{i}-{j}-{g}", list(a), list(b)))
    return jobs


def _worker(args: tuple) -> list[dict]:
    jobs, sim, n = args
    return play(jobs, Path(sim), n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--dir", type=Path, required=True)
    p = sub.add_parser("play")
    p.add_argument("--teams", type=Path, help="JSON: {'pool': n, 'teams': [[i, ...]]}")
    p.add_argument(
        "--duels", type=int, help="instead of teams: every Pokémon of a pool of N, alone"
    )
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slice", type=int, default=1)
    p.add_argument("--slices", type=int, default=1)
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count()))
    p.add_argument("--showdown", type=Path, required=True, help="where to clone and build")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="play only the first N battles of the slice")
    args = parser.parse_args()
    if args.command == "setup":
        print(setup(args.dir))
        return
    if args.duels:
        n, teams = args.duels, [[i] for i in range(args.duels)]
    else:
        spec = json.loads(args.teams.read_text())
        n, teams = spec["pool"], spec["teams"]
    sim = setup(args.showdown)
    pool_sets(n)
    level4_matrix(n)
    jobs = schedule(teams, args.games, args.seed)
    mine = jobs[args.slice - 1 :: args.slices][: args.limit]
    t0 = time.time()
    chunks = [(mine[k :: args.workers], str(sim), n) for k in range(args.workers)]
    with multiprocessing.Pool(args.workers) as pool:
        results = [r for part in pool.map(_worker, chunks) for r in part]
    results.sort(key=lambda r: r["key"])
    out = (
        args.out / f"battles-{args.slice}.json"
        if args.out.is_dir() or not args.out.suffix
        else args.out
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pool": n, "teams": teams, "games": args.games, "seed": args.seed,
                               "format": FORMAT, "showdown_commit": SHOWDOWN_COMMIT,
                               "slice": args.slice, "slices": args.slices,
                               "seconds": time.time() - t0, "battles": results}) + "\n")  # fmt: skip
    print(f"{len(results)} battles in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
