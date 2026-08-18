#!/usr/bin/env python3
"""Calculate Generation III capture odds with the game's integer rounding."""

from __future__ import annotations

import argparse
import json
import math

BALL_BONUSES = {
    "poke": 10,
    "great": 15,
    "ultra": 20,
}
STATUS_MULTIPLIERS = {
    "none": (1, 1),
    "sleep": (2, 1),
    "freeze": (2, 1),
    "paralysis": (15, 10),
    "poison": (15, 10),
    "burn": (15, 10),
}


def modified_catch_odds(
    *,
    catch_rate: int,
    max_hp: int,
    current_hp: int,
    ball_bonus: int,
    status: str,
) -> int:
    """Return Gen III's integer modified catch value, called ``odds`` in code."""
    if not 1 <= catch_rate <= 255:
        raise ValueError("catch rate must be between 1 and 255")
    if not 1 <= current_hp <= max_hp:
        raise ValueError("current HP must be between 1 and max HP")
    if ball_bonus <= 0:
        raise ValueError("ball bonus must be positive")
    if status not in STATUS_MULTIPLIERS:
        raise ValueError(f"unknown status: {status}")

    odds = (catch_rate * ball_bonus // 10) * (3 * max_hp - 2 * current_hp)
    odds //= 3 * max_hp
    numerator, denominator = STATUS_MULTIPLIERS[status]
    return odds * numerator // denominator


def shake_threshold(modified_odds: int) -> int | None:
    """Return the four-shake threshold, or None for an automatic catch."""
    if modified_odds > 254:
        return None
    if modified_odds <= 0:
        return 0
    inner = math.isqrt(16_711_680 // modified_odds)
    denominator = math.isqrt(inner)
    return 1_048_560 // denominator


def capture_probability(modified_odds: int) -> float:
    """Return the exact four-shake probability for a modified catch value."""
    threshold = shake_threshold(modified_odds)
    if threshold is None:
        return 1.0
    return (min(threshold, 65_536) / 65_536) ** 4


def result(
    *,
    catch_rate: int,
    max_hp: int,
    current_hp: int,
    ball: str,
    status: str,
) -> dict[str, float | int | str | None]:
    if ball not in BALL_BONUSES:
        raise ValueError(f"unknown ball: {ball}")
    modified_odds_value = modified_catch_odds(
        catch_rate=catch_rate,
        max_hp=max_hp,
        current_hp=current_hp,
        ball_bonus=BALL_BONUSES[ball],
        status=status,
    )
    threshold = shake_threshold(modified_odds_value)
    probability = capture_probability(modified_odds_value)
    return {
        "catch_rate": catch_rate,
        "max_hp": max_hp,
        "current_hp": current_hp,
        "ball": ball,
        "status": status,
        "modified_odds": modified_odds_value,
        "shake_threshold": threshold,
        "probability": probability,
        "expected_balls": 1 / probability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catch-rate", type=int, required=True)
    parser.add_argument("--max-hp", type=int, required=True)
    parser.add_argument("--current-hp", type=int, required=True)
    parser.add_argument("--ball", choices=tuple(BALL_BONUSES), default="poke")
    parser.add_argument("--status", choices=tuple(STATUS_MULTIPLIERS), default="none")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    output = result(
        catch_rate=args.catch_rate,
        max_hp=args.max_hp,
        current_hp=args.current_hp,
        ball=args.ball,
        status=args.status,
    )
    if args.as_json:
        print(json.dumps(output, sort_keys=True))
        return
    print(f"modified catch value: {output['modified_odds']}")
    print(f"single-ball probability: {output['probability']:.6%}")
    print(f"expected balls: {output['expected_balls']:.2f}")


if __name__ == "__main__":
    main()
