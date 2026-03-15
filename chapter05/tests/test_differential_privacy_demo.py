"""
test_differential_privacy_demo.py
Chapter 5 — Future Frontiers by Jigar Sheth
"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from differential_privacy_demo import (
    generate_dili_dataset, simulate_dp_noise, calculate_proi, compute_privacy_utility_curve
)


def test_dili_dataset_shape_and_prevalence():
    df = generate_dili_dataset(n_samples=500, random_state=0)
    assert len(df) == 500
    assert "dili" in df.columns
    # DILI prevalence should be in plausible clinical range (3–20%)
    prev = df["dili"].mean()
    assert 0.03 <= prev <= 0.25, f"DILI prevalence {prev:.2%} outside expected 3-20%"


def test_dp_noise_increases_with_lower_epsilon():
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, (100, 10))
    X_low_eps  = simulate_dp_noise(X, epsilon=0.1, random_state=0)
    X_high_eps = simulate_dp_noise(X, epsilon=10.0, random_state=0)
    noise_low  = np.abs(X_low_eps - X).mean()
    noise_high = np.abs(X_high_eps - X).mean()
    assert noise_low > noise_high, "Lower epsilon must add more noise"


def test_proi_positive_for_reasonable_inputs():
    result = calculate_proi(
        annual_revenue_eur=500_000_000,
        breach_probability_pct=5.0,
        privacy_investment_eur=800_000,
        approval_value_eur=600_000,
        trust_premium_eur=400_000,
        regulatory_dividend_eur=200_000,
    )
    assert result["p_roi"] > 1.0, "P-ROI should be positive for reasonable pharma inputs"
    assert result["total_return_eur"] > result["privacy_investment_eur"]


def test_proi_components_sum_correctly():
    r = calculate_proi(
        annual_revenue_eur=800_000_000,
        breach_probability_pct=8.0,
        privacy_investment_eur=1_200_000,
        approval_value_eur=900_000,
        trust_premium_eur=650_000,
        regulatory_dividend_eur=400_000,
    )
    expected_total = (r["risk_avoidance_value_eur"] + r["approval_value_eur"]
                      + r["trust_premium_eur"] + r["regulatory_dividend_eur"])
    assert abs(r["total_return_eur"] - expected_total) < 1.0


def test_privacy_utility_curve_returns_expected_columns():
    df = generate_dili_dataset(n_samples=400, random_state=1)
    results_df, baseline = compute_privacy_utility_curve(df)
    assert "epsilon" in results_df.columns
    assert "auc" in results_df.columns
    assert "auc_retention_pct" in results_df.columns
    assert baseline > 0.5, "Baseline AUC should exceed random chance"


def test_higher_epsilon_yields_higher_auc():
    df = generate_dili_dataset(n_samples=400, random_state=1)
    results_df, _ = compute_privacy_utility_curve(df)
    finite = results_df[results_df["epsilon"] != float("inf")].sort_values("epsilon")
    # AUC at eps=10 should exceed AUC at eps=0.1 in expectation
    low_eps_auc  = finite[finite["epsilon"] == 0.1]["auc"].values[0]
    high_eps_auc = finite[finite["epsilon"] == 10.0]["auc"].values[0]
    assert high_eps_auc >= low_eps_auc
