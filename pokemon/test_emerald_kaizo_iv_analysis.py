import math

from pokemon.emerald_kaizo_iv_analysis import (
    TOTAL_VECTORS,
    all_iv_sum_distribution,
    beta_quantile,
    catches_for_confidence,
    clopper_pearson_interval,
    probability_all_at_least,
    probability_sum_at_least,
    regularized_beta,
    simulate_all_at_least_count,
    sum_rows,
)
from pokemon.emerald_kaizo_role_query import role_probability


def test_sum_distribution_has_exact_mass_and_symmetry():
    distribution = all_iv_sum_distribution()

    assert sum(distribution.values()) == TOTAL_VECTORS
    assert min(distribution) == 0
    assert max(distribution) == 186
    for value, count in distribution.items():
        assert distribution[186 - value] == count


def test_sum_distribution_has_expected_mean():
    distribution = all_iv_sum_distribution()
    mean = sum(value * count for value, count in distribution.items()) / TOTAL_VECTORS

    assert math.isclose(mean, 6 * 15.5)


def test_closed_form_and_sum_probability_agree_at_known_threshold():
    distribution = all_iv_sum_distribution()

    assert probability_all_at_least(16) == 1 / 64
    assert probability_sum_at_least(distribution, 0, TOTAL_VECTORS) == 1.0
    assert probability_sum_at_least(distribution, 187, TOTAL_VECTORS) == 0.0


def test_geometric_confidence_is_monotone_and_exact_at_boundary():
    probability = 1 / 64
    catches = catches_for_confidence(probability, 0.95)

    assert catches == 191
    assert 1 - (1 - probability) ** catches >= 0.95
    assert 1 - (1 - probability) ** (catches - 1) < 0.95


def test_clopper_pearson_edges_and_beta_inverse():
    assert math.isclose(regularized_beta(0.5, 1, 1), 0.5)
    assert math.isclose(beta_quantile(0.5, 1, 1), 0.5)

    lower_zero, upper_zero = clopper_pearson_interval(0, 100)
    lower_all, upper_all = clopper_pearson_interval(100, 100)
    assert lower_zero == 0.0
    assert upper_all == 1.0
    assert 0.0 < upper_zero < 0.1
    assert 0.9 < lower_all < 1.0


def test_simulation_is_seeded_and_reports_integer_successes():
    first = simulate_all_at_least_count(16, 10_000, 20260817)
    second = simulate_all_at_least_count(16, 10_000, 20260817)

    assert first == second
    assert 0 < first < 10_000


def test_sum_rows_are_ordered_by_percentile():
    rows = sum_rows(all_iv_sum_distribution())

    assert [row["percentile"] for row in rows] == sorted(row["percentile"] for row in rows)
    assert all(rows[index]["threshold"] < rows[index + 1]["threshold"] for index in range(4))


def test_role_query_combines_explicit_filters():
    probability = role_probability(
        acceptable_natures=2,
        relevant_stats=2,
        minimum_iv=16,
        ability_slots=2,
        preferred_abilities=1,
        encounter_share=1.0,
        capture_probability=1.0,
        synchronize=False,
    )

    assert probability == 0.01


def test_synchronize_changes_nature_probability_only():
    without_synchronize = role_probability(
        acceptable_natures=1,
        relevant_stats=0,
        minimum_iv=0,
        ability_slots=1,
        preferred_abilities=1,
        encounter_share=1.0,
        capture_probability=1.0,
        synchronize=False,
    )
    with_synchronize = role_probability(
        acceptable_natures=1,
        relevant_stats=0,
        minimum_iv=0,
        ability_slots=1,
        preferred_abilities=1,
        encounter_share=1.0,
        capture_probability=1.0,
        synchronize=True,
    )

    assert without_synchronize == 1 / 25
    assert with_synchronize == 0.52
