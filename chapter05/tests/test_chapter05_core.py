"""
test_chapter05_core.py
========================
Chapter 5: Building a Privacy-First AI Strategy
Tests for: privacy_by_design_checklist.py, differential_privacy_demo.py,
           data_minimization_analyzer.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from privacy_by_design_checklist import (
    build_pharma_pbd_checklist, PbDChecker, PbDReport
)
from differential_privacy_demo import (
    generate_dili_dataset, calculate_proi, simulate_dp_noise
)
from data_minimization_analyzer import (
    generate_full_feature_dili_dataset
)


# ─── PbD Checklist Tests ─────────────────────────────────────────────────────

class TestPbDChecklist:

    def test_seven_principles_generated(self):
        """PbD checklist must contain exactly 7 principles."""
        principles = build_pharma_pbd_checklist()
        assert len(principles) == 7, "Expected exactly 7 PbD principles"

    def test_all_principles_have_criteria(self):
        """Every principle must have at least 2 auditable criteria."""
        principles = build_pharma_pbd_checklist()
        for p in principles:
            assert len(p.criteria) >= 2, \
                f"Principle {p.principle_id} has fewer than 2 criteria"

    def test_overall_score_all_pass(self):
        """A project with all criteria passing should score 100%."""
        checker = PbDChecker("Test Project", "Test Assessor")
        all_ids = [c.criterion_id
                   for p in checker.report.principles
                   for c in p.criteria]
        responses = {cid: True for cid in all_ids}
        report = checker.run_assessment(responses)
        assert report.overall_score == 1.0, "All-pass should yield 100% score"

    def test_overall_score_all_fail(self):
        """A project with all criteria failing should score 0%."""
        checker = PbDChecker("Test Project", "Test Assessor")
        all_ids = [c.criterion_id
                   for p in checker.report.principles
                   for c in p.criteria]
        responses = {cid: False for cid in all_ids}
        report = checker.run_assessment(responses)
        assert report.overall_score == 0.0, "All-fail should yield 0% score"

    def test_failed_criteria_list(self):
        """failed_criteria property should list only criteria that failed."""
        checker = PbDChecker("Test Project", "Test Assessor")
        responses = {"1.1": False, "1.2": True, "1.3": False}
        checker.run_assessment(responses)
        failed_ids = [c.criterion_id for c in checker.report.failed_criteria]
        assert "1.1" in failed_ids
        assert "1.3" in failed_ids
        assert "1.2" not in failed_ids

    def test_json_output_valid_structure(self):
        """JSON export must contain project_name, assessment_date, principles."""
        import json
        checker = PbDChecker("JSON Test Project", "Pytest")
        checker.run_assessment({"1.1": True})
        json_str = checker.to_json()
        data = json.loads(json_str)
        assert "project_name" in data
        assert "assessment_date" in data
        assert "principles" in data

    def test_notes_recorded(self):
        """Notes should be stored on the correct criterion after assessment."""
        checker = PbDChecker("Notes Test", "Pytest")
        checker.run_assessment({"1.2": False}, notes={"1.2": "Deferred by PM"})
        for p in checker.report.principles:
            for c in p.criteria:
                if c.criterion_id == "1.2":
                    assert "Deferred" in c.notes


# ─── P-ROI Calculator Tests ──────────────────────────────────────────────────

class TestPROI:

    def test_proi_positive_for_standard_inputs(self):
        """P-ROI should be > 1.0 for the documented worked example."""
        result = calculate_proi(
            annual_revenue_eur=800_000_000,
            breach_probability_pct=8.0,
            privacy_investment_eur=1_200_000,
            approval_value_eur=900_000,
            trust_premium_eur=650_000,
            regulatory_dividend_eur=400_000,
            control_effectiveness=0.8125,
        )
        assert result["p_roi"] > 1.0, "P-ROI should exceed 1.0 for this scenario"
        assert abs(result["p_roi"] - 3.36) < 0.1, \
            f"P-ROI should be ~3.36, got {result['p_roi']}"

    def test_proi_invest_recommendation(self):
        """Positive P-ROI should produce INVEST recommendation."""
        result = calculate_proi(800e6, 8.0, 1_200_000, 900_000, 650_000, 400_000)
        assert "INVEST" in result["recommendation"]

    def test_proi_total_return_components(self):
        """Total return should equal sum of four components."""
        result = calculate_proi(800e6, 8.0, 1_200_000, 900_000, 650_000, 400_000)
        expected_total = (result["risk_avoidance_value_eur"] +
                          result["approval_value_eur"] +
                          result["trust_premium_eur"] +
                          result["regulatory_dividend_eur"])
        assert abs(result["total_return_eur"] - expected_total) < 10, \
            "Total return should equal sum of four components"

    def test_gdpr_fine_calculation(self):
        """GDPR max fine should be 4% of annual revenue."""
        result = calculate_proi(500_000_000, 5.0, 100_000, 0, 0, 0)
        assert result["gdpr_max_fine_eur"] == 500_000_000 * 0.04


# ─── DILI Dataset Tests ──────────────────────────────────────────────────────

class TestDILIDataset:

    def test_dataset_shape(self):
        """Generated dataset should have expected number of rows and columns."""
        df = generate_dili_dataset(n_samples=500)
        assert len(df) == 500
        assert "dili" in df.columns

    def test_dili_prevalence_reasonable(self):
        """DILI prevalence should be between 5% and 15%."""
        df = generate_dili_dataset(n_samples=2000, random_state=42)
        prevalence = df["dili"].mean()
        assert 0.05 <= prevalence <= 0.15, \
            f"DILI prevalence {prevalence:.1%} outside expected 5-15% range"

    def test_dp_noise_increases_with_privacy(self):
        """Higher privacy (lower epsilon) should add more noise to features."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 10))
        X_high_privacy = simulate_dp_noise(X, epsilon=0.1, random_state=42)
        X_low_privacy  = simulate_dp_noise(X, epsilon=10.0, random_state=42)
        noise_high = np.abs(X_high_privacy - X).mean()
        noise_low  = np.abs(X_low_privacy - X).mean()
        assert noise_high > noise_low, \
            "Lower epsilon should produce higher noise magnitude"

    def test_47_feature_dataset(self):
        """Full feature dataset should have exactly 47 features plus target."""
        df = generate_full_feature_dili_dataset(n=200)
        feature_cols = [c for c in df.columns if c != "dili"]
        assert len(feature_cols) == 47, \
            f"Expected 47 features, got {len(feature_cols)}"

    def test_no_missing_values_in_dataset(self):
        """Synthetic dataset should contain no NaN values."""
        df = generate_dili_dataset(n_samples=500)
        assert df.isna().sum().sum() == 0, "Synthetic dataset should have no NaN values"


# ─── Integration: Checklist → Report → JSON roundtrip ───────────────────────

class TestIntegration:

    def test_full_assessment_roundtrip(self):
        """Full assessment should serialize to JSON and preserve score."""
        import json
        checker = PbDChecker("Integration Test", "Pytest Suite")
        all_ids = [c.criterion_id
                   for p in checker.report.principles
                   for c in p.criteria]
        # Pass half, fail half
        responses = {cid: (i % 2 == 0) for i, cid in enumerate(all_ids)}
        report = checker.run_assessment(responses)
        json_str = checker.to_json()
        data = json.loads(json_str)
        assert data["project_name"] == "Integration Test"
        # Score should be approximately 0.5
        assert 0.4 <= report.overall_score <= 0.6

    def test_text_report_contains_status(self):
        """Generated text report must contain STATUS line."""
        checker = PbDChecker("Report Test", "Pytest")
        all_ids = [c.criterion_id
                   for p in checker.report.principles
                   for c in p.criteria]
        checker.run_assessment({cid: True for cid in all_ids})
        report_text = checker.generate_text_report()
        assert "STATUS" in report_text
        assert "DPO REVIEW" in report_text
