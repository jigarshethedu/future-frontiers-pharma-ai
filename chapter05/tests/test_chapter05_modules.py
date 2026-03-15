"""
test_differential_privacy_demo.py
Chapter 5: Building a Privacy-First AI Strategy
Future Frontiers by Jigar Sheth
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from differential_privacy_demo import (
    gaussian_mechanism,
    laplace_mechanism,
    PrivacyBudget,
    generate_synthetic_clinical_dataset,
    compute_privacy_utility_curve,
)


def test_gaussian_mechanism_adds_noise():
    """DP mechanisms must perturb the true value (not return it exactly)."""
    result = gaussian_mechanism(10.0, sensitivity=1.0, epsilon=0.5, delta=1e-5)
    # Should be perturbed (virtually impossible to equal exactly)
    assert result != 10.0


def test_laplace_mechanism_adds_noise():
    result = laplace_mechanism(5.0, sensitivity=1.0, epsilon=1.0)
    assert result != 5.0


def test_privacy_budget_tracking():
    """Budget must block operations once epsilon is exhausted."""
    budget = PrivacyBudget(total_epsilon=1.0)
    assert budget.consume("op1", 0.6) is True
    assert budget.consumed_epsilon == pytest.approx(0.6)
    assert budget.consume("op2", 0.5) is False  # Would exceed budget
    assert budget.exhausted is False  # Only 0.6 consumed


def test_budget_exhaustion_flag():
    budget = PrivacyBudget(total_epsilon=1.0)
    budget.consume("op1", 1.0)
    assert budget.exhausted is True


def test_synthetic_dataset_shape():
    X, y = generate_synthetic_clinical_dataset(n_patients=500)
    assert X.shape == (500, 8)
    assert y.shape == (500,)
    assert set(np.unique(y)).issubset({0, 1})


def test_privacy_utility_curve_returns_all_epsilons():
    X, y = generate_synthetic_clinical_dataset(n_patients=300)
    epsilons = [0.5, 1.0, 2.0]
    curve = compute_privacy_utility_curve(X, y, epsilons, n_runs=2)
    assert len(curve) == 3
    returned_eps = [r["epsilon"] for r in curve]
    assert returned_eps == epsilons


def test_higher_epsilon_generally_better_auc():
    """Higher epsilon (weaker privacy) should on average produce higher AUC."""
    X, y = generate_synthetic_clinical_dataset(n_patients=400, seed=7)
    curve = compute_privacy_utility_curve(X, y, epsilons=[0.1, 5.0], n_runs=3)
    # AUC at epsilon=5.0 should be >= AUC at epsilon=0.1
    auc_low_eps = curve[0]["mean_auc"]
    auc_high_eps = curve[1]["mean_auc"]
    assert auc_high_eps >= auc_low_eps - 0.05  # allow small variance


"""
test_synthetic_data_generator.py
"""
from synthetic_data_generator import (
    generate_synthetic_ehr,
    generate_synthetic_adverse_events,
    evaluate_tstr,
)


def test_ehr_dataframe_shape():
    df = generate_synthetic_ehr(n_patients=100)
    assert len(df) == 100
    assert "readmission_30d" in df.columns
    assert df["readmission_30d"].isin([0, 1]).all()


def test_ehr_no_nulls():
    df = generate_synthetic_ehr(n_patients=200)
    assert df.isnull().sum().sum() == 0


def test_adverse_events_columns():
    df = generate_synthetic_adverse_events(n_reports=100)
    required = ["report_id", "drug_name", "ae_term", "outcome"]
    for col in required:
        assert col in df.columns


def test_tstr_scores_in_range():
    real_df = generate_synthetic_ehr(n_patients=400, seed=42)
    syn_df = generate_synthetic_ehr(n_patients=400, seed=999)
    feature_cols = ["age", "bmi", "alt_u_per_l", "ast_u_per_l",
                    "creatinine_mg_dl", "hba1c_pct", "diabetes", "hypertension", "ckd"]
    result = evaluate_tstr(real_df, syn_df, "readmission_30d", feature_cols)
    assert 0 <= result.fidelity_score <= 100
    assert 0 <= result.utility_score <= 100
    assert 0 <= result.mia_resistance_score <= 100
    assert 0 <= result.overall_tstr_score <= 100


"""
test_consent_ledger_stub.py
"""
from consent_ledger_stub import ConsentLedger


def test_consent_grant_and_check():
    ledger = ConsentLedger()
    ledger.register_use_case("TEST_UC", "Test use case")
    ledger.grant_consent("patient_001", "TEST_UC")
    assert ledger.is_consented("patient_001", "TEST_UC") is True


def test_consent_withdrawal():
    ledger = ConsentLedger()
    ledger.register_use_case("TEST_UC", "Test use case")
    ledger.grant_consent("patient_002", "TEST_UC")
    ledger.withdraw_consent("patient_002", "TEST_UC")
    assert ledger.is_consented("patient_002", "TEST_UC") is False


def test_chain_integrity():
    ledger = ConsentLedger()
    ledger.register_use_case("UC1", "Use case 1")
    ledger.grant_consent("p1", "UC1")
    ledger.grant_consent("p2", "UC1")
    assert ledger.verify_chain_integrity() is True


def test_tamper_detection():
    ledger = ConsentLedger()
    ledger.register_use_case("UC1", "Use case 1")
    ledger.grant_consent("p1", "UC1")
    # Tamper
    ledger.records[0].action = "WITHDRAW"
    assert ledger.verify_chain_integrity() is False


def test_unregistered_use_case_raises():
    ledger = ConsentLedger()
    with pytest.raises(ValueError):
        ledger.grant_consent("p1", "UNREGISTERED_UC")
