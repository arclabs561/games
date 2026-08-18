#!/usr/bin/env python3
"""Calculate Generation III stat values for an IV and nature scenario.

This uses the zero-EV stat formulas relevant to Emerald Kaizo's documented
EV-disabled player model. It does not calculate damage, critical hits, weather,
abilities, or battle-specific modifiers.
"""

from __future__ import annotations

import argparse


def gen3_stat(base: int, iv: int, level: int, kind: str, nature: str) -> int:
    """Return one Gen III stat with zero EVs."""
    if not 0 <= base <= 255 or not 0 <= iv <= 31 or not 1 <= level <= 100:
        raise ValueError("base, IV, and level are outside the Gen III range")
    if kind not in {"hp", "other"}:
        raise ValueError("kind must be hp or other")
    if nature not in {"neutral", "boosted", "hindered"}:
        raise ValueError("nature must be neutral, boosted, or hindered")

    core = ((2 * base + iv) * level) // 100
    if kind == "hp":
        return core + level + 10

    unmodified = core + 5
    if nature == "boosted":
        return unmodified * 11 // 10
    if nature == "hindered":
        return unmodified * 9 // 10
    return unmodified


def stat_range(base: int, level: int, kind: str, nature: str) -> tuple[int, int]:
    """Return the stat values at IV 0 and IV 31."""
    return (
        gen3_stat(base, 0, level, kind, nature),
        gen3_stat(base, 31, level, kind, nature),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--kind", choices=("hp", "other"), default="other")
    parser.add_argument("--nature", choices=("neutral", "boosted", "hindered"), default="neutral")
    args = parser.parse_args()
    low, high = stat_range(args.base, args.level, args.kind, args.nature)
    print(f"IV 0: {low}")
    print(f"IV 31: {high}")
    print(f"difference: {high - low}")


if __name__ == "__main__":
    main()
