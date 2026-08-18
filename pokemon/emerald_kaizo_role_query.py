#!/usr/bin/env python3
"""Query an explicit Emerald Kaizo catch-quality acceptance rule.

This is an independent-candidate model. It is useful for comparing a few
reasonable acceptance policies before adding species- and battle-specific
data. It does not model deterministic Emerald RNG frames.
"""

from __future__ import annotations

import argparse
import json

from emerald_kaizo_iv_analysis import (
    catches_for_confidence,
    expected_catches,
    probability_nature_set,
    probability_relevant_ivs_at_least,
)


def role_probability(
    *,
    acceptable_natures: int,
    relevant_stats: int,
    minimum_iv: int,
    ability_slots: int,
    preferred_abilities: int,
    encounter_share: float,
    encounter_check_probability: float,
    capture_probability: float,
    synchronize: bool,
) -> float:
    """Return the independent-model probability for one accepted attempt."""
    if not 1 <= ability_slots <= 2:
        raise ValueError("ability slots must be 1 or 2")
    if not 1 <= preferred_abilities <= ability_slots:
        raise ValueError("preferred abilities must be between 1 and ability slots")
    if not 0.0 < encounter_share <= 1.0:
        raise ValueError("encounter share must be in (0, 1]")
    if not 0.0 < encounter_check_probability <= 1.0:
        raise ValueError("encounter check probability must be in (0, 1]")
    if not 0.0 < capture_probability <= 1.0:
        raise ValueError("capture probability must be in (0, 1]")

    nature_probability = probability_nature_set(acceptable_natures)
    if synchronize:
        nature_probability = 0.5 + 0.5 * nature_probability

    return (
        nature_probability
        * probability_relevant_ivs_at_least(minimum_iv, relevant_stats)
        * (preferred_abilities / ability_slots)
        * encounter_share
        * encounter_check_probability
        * capture_probability
    )


def query_result(args: argparse.Namespace) -> dict[str, float | int | bool]:
    probability = role_probability(
        acceptable_natures=args.acceptable_natures,
        relevant_stats=args.relevant_stats,
        minimum_iv=args.minimum_iv,
        ability_slots=args.ability_slots,
        preferred_abilities=args.preferred_abilities,
        encounter_share=args.encounter_share,
        encounter_check_probability=args.encounter_check_probability,
        capture_probability=args.capture_probability,
        synchronize=args.synchronize,
    )
    return {
        "acceptable_natures": args.acceptable_natures,
        "relevant_stats": args.relevant_stats,
        "minimum_iv": args.minimum_iv,
        "ability_slots": args.ability_slots,
        "preferred_abilities": args.preferred_abilities,
        "encounter_share": args.encounter_share,
        "encounter_check_probability": args.encounter_check_probability,
        "capture_probability": args.capture_probability,
        "synchronize": args.synchronize,
        "probability": probability,
        "expected_attempts": expected_catches(probability),
        "confidence": args.confidence,
        "attempts_for_confidence": catches_for_confidence(probability, args.confidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptable-natures", type=int, default=2)
    parser.add_argument("--relevant-stats", type=int, default=2)
    parser.add_argument("--minimum-iv", type=int, default=16)
    parser.add_argument("--ability-slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--preferred-abilities", type=int, choices=(1, 2), default=1)
    parser.add_argument("--encounter-share", type=float, default=1.0)
    parser.add_argument("--encounter-check-probability", type=float, default=1.0)
    parser.add_argument("--capture-probability", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--synchronize", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = query_result(args)
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
        return

    print(f"acceptance probability: {result['probability']:.8g}")
    print(f"expected attempts: {result['expected_attempts']:,.2f}")
    print(
        f"attempts for {result['confidence']:.1%} confidence: {result['attempts_for_confidence']:,}"
    )


if __name__ == "__main__":
    main()
