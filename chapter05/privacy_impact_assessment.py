"""
privacy_impact_assessment.py
==============================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Implements the 8-Stage Pharma AI PIA Framework [ORIGINAL FRAMEWORK] — a
    STRIDE-adapted privacy impact assessment designed for AI systems processing
    clinical, genomic, trial, or pharmacovigilance data.

    Stages:
        1. Data Asset Inventory
        2. Processing Purpose Specification
        3. Legal Basis Mapping
        4. STRIDE Threat Analysis (adapted for ML pipelines)
        5. Re-identification Risk Scoring
        6. Privacy Control Gap Analysis
        7. Residual Risk Assessment
        8. Sign-Off and Registry Entry

Dependencies:
    - Python 3.10+
    - No external dependencies (stdlib only)

GitHub: chapter05/privacy_impact_assessment.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import date


# ---------------------------------------------------------------------------
# ENUMERATIONS
# ---------------------------------------------------------------------------

class DataCategory(Enum):
    GENOMIC = "Genomic / Biomarker"
    EHR = "Electronic Health Record"
    ADVERSE_EVENT = "Adverse Event / PV"
    WEARABLE = "Wearable / Digital Biomarker"
    IMAGING = "Medical Imaging"
    TRIAL = "Clinical Trial"
    SYNTHETIC = "Synthetic"


class LegalBasis(Enum):
    CONSENT = "Informed Consent (Art. 9(2)(a))"
    VITAL_INTEREST = "Vital Interests (Art. 9(2)(c))"
    PUBLIC_INTEREST = "Public Interest / Research (Art. 9(2)(i))"
    LEGITIMATE_INTEREST = "Legitimate Interest (Art. 6(1)(f))"
    LEGAL_OBLIGATION = "Legal Obligation (Art. 6(1)(c))"
    CONTRACT = "Contractual Necessity (Art. 6(1)(b))"


class STRIDEThreat(Enum):
    SPOOFING = "Spoofing (identity impersonation in training pipeline)"
    TAMPERING = "Tampering (data poisoning or label manipulation)"
    REPUDIATION = "Repudiation (no audit trail for model decisions)"
    INFORMATION_DISCLOSURE = "Information Disclosure (model inversion / membership inference)"
    DENIAL_OF_SERVICE = "Denial of Service (privacy budget exhaustion attack)"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege (unauthorized access to raw training data)"


class RiskLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NEGLIGIBLE = "Negligible"


# ---------------------------------------------------------------------------
# DATA STRUCTURES — STAGES 1–8
# ---------------------------------------------------------------------------

@dataclass
class DataAsset:
    """Stage 1: One data asset included in the AI system's training or inference pipeline."""
    asset_name: str
    category: DataCategory
    n_records_approx: int
    contains_direct_identifiers: bool
    contains_quasi_identifiers: bool
    cross_border_transfer: bool
    retention_years: float


@dataclass
class ProcessingPurpose:
    """Stage 2: One legitimate processing purpose for the AI system."""
    purpose_id: str
    description: str
    is_compatible_with_original_consent: bool
    requires_new_consent: bool


@dataclass
class LegalBasisEntry:
    """Stage 3: Legal basis mapped to a specific data asset and purpose."""
    asset_name: str
    purpose_id: str
    legal_basis: LegalBasis
    jurisdiction: str
    documentation_location: str


@dataclass
class STRIDEEntry:
    """Stage 4: One STRIDE threat applicable to this ML pipeline."""
    threat: STRIDEThreat
    pipeline_component: str     # e.g., "training data store", "model API endpoint"
    likelihood: RiskLevel
    impact: RiskLevel
    current_controls: str
    residual_risk: RiskLevel


@dataclass
class ReidentificationRisk:
    """Stage 5: Re-identification risk assessment for a data asset."""
    asset_name: str
    k_anonymity_k: int          # Minimum equivalence class size (0 = not assessed)
    linkage_attack_risk: RiskLevel
    model_inversion_risk: RiskLevel
    membership_inference_risk: RiskLevel
    overall_reid_risk: RiskLevel
    mitigation_required: bool


@dataclass
class PrivacyControlGap:
    """Stage 6: A gap between required and implemented privacy controls."""
    control_area: str           # e.g., "Data Minimization", "Access Control"
    required_control: str
    current_state: str
    gap_severity: RiskLevel
    remediation_action: str
    target_completion_date: str


@dataclass
class PIAResult:
    """
    Stage 7–8: Full PIA result including residual risk and sign-off.
    This object feeds the Privacy Architecture Canvas and the P-ROI Model.
    """
    project_name: str
    assessment_date: str
    assessor: str
    dpo_reviewer: str
    data_assets: list[DataAsset]
    purposes: list[ProcessingPurpose]
    legal_bases: list[LegalBasisEntry]
    stride_threats: list[STRIDEEntry]
    reid_risks: list[ReidentificationRisk]
    control_gaps: list[PrivacyControlGap]
    overall_residual_risk: RiskLevel
    approved_for_deployment: bool
    conditions_for_approval: list[str]
    next_review_date: str


# ---------------------------------------------------------------------------
# SYNTHETIC EXAMPLE BUILDER
# ---------------------------------------------------------------------------

def build_synthetic_pia_example() -> PIAResult:
    """
    Constructs a fully worked synthetic PIA for a hypothetical
    AI-powered DILI risk prediction system at a mid-size pharma company.
    No real patient data. All organizations, names, and values are fictional.
    """

    assets = [
        DataAsset("EHR_DILI_Cohort_Synthetic", DataCategory.EHR,
                  n_records_approx=18000, contains_direct_identifiers=False,
                  contains_quasi_identifiers=True, cross_border_transfer=True,
                  retention_years=10),
        DataAsset("Genomic_Panel_Synthetic", DataCategory.GENOMIC,
                  n_records_approx=4200, contains_direct_identifiers=False,
                  contains_quasi_identifiers=True, cross_border_transfer=False,
                  retention_years=25),
    ]

    purposes = [
        ProcessingPurpose("P-01", "Train DILI risk prediction model for clinical decision support",
                          is_compatible_with_original_consent=True, requires_new_consent=False),
        ProcessingPurpose("P-02", "Generate model explanations for clinical audit trail",
                          is_compatible_with_original_consent=True, requires_new_consent=False),
    ]

    legal_bases = [
        LegalBasisEntry("EHR_DILI_Cohort_Synthetic", "P-01",
                        LegalBasis.PUBLIC_INTEREST, "EU/GDPR",
                        "DPO_Register/Entry_2024_017"),
        LegalBasisEntry("Genomic_Panel_Synthetic", "P-01",
                        LegalBasis.CONSENT, "EU/GDPR",
                        "Consent_Management_Platform/Study_441"),
    ]

    stride = [
        STRIDEEntry(STRIDEThreat.INFORMATION_DISCLOSURE,
                    "Model inference API endpoint",
                    RiskLevel.HIGH, RiskLevel.CRITICAL,
                    "TLS 1.3, authentication required",
                    RiskLevel.HIGH),
        STRIDEEntry(STRIDEThreat.TAMPERING,
                    "Training data ingestion pipeline",
                    RiskLevel.MEDIUM, RiskLevel.HIGH,
                    "Input validation, schema enforcement",
                    RiskLevel.MEDIUM),
        STRIDEEntry(STRIDEThreat.REPUDIATION,
                    "Model training and retraining events",
                    RiskLevel.MEDIUM, RiskLevel.MEDIUM,
                    "Training run logs in S3 — not tamper-evident",
                    RiskLevel.HIGH),
    ]

    reid = [
        ReidentificationRisk("EHR_DILI_Cohort_Synthetic",
                             k_anonymity_k=8,
                             linkage_attack_risk=RiskLevel.MEDIUM,
                             model_inversion_risk=RiskLevel.HIGH,
                             membership_inference_risk=RiskLevel.HIGH,
                             overall_reid_risk=RiskLevel.HIGH,
                             mitigation_required=True),
    ]

    gaps = [
        PrivacyControlGap("Audit Trail Integrity",
                          "Tamper-evident, cryptographically chained audit log",
                          "Flat text logs in S3 with no integrity verification",
                          RiskLevel.HIGH,
                          "Implement append-only audit store with hash chaining (see ai_audit_trail_generator.py, Chapter 19)",
                          "2025-Q1"),
        PrivacyControlGap("Model Inversion Protection",
                          "DP noise injection or prediction confidence suppression",
                          "Raw probability scores returned via API",
                          RiskLevel.HIGH,
                          "Enable DP-SGD training (differential_privacy_demo.py) and cap API confidence to 2 decimal places",
                          "2024-Q4"),
    ]

    return PIAResult(
        project_name="Meridian Pharma — DILI Risk Prediction AI (Synthetic)",
        assessment_date=str(date.today()),
        assessor="Privacy Engineering Team (Synthetic Name)",
        dpo_reviewer="DPO Office (Synthetic Name)",
        data_assets=assets,
        purposes=purposes,
        legal_bases=legal_bases,
        stride_threats=stride,
        reid_risks=reid,
        control_gaps=gaps,
        overall_residual_risk=RiskLevel.HIGH,
        approved_for_deployment=False,
        conditions_for_approval=[
            "Implement tamper-evident audit logging before production deployment",
            "Complete model inversion risk mitigation (DP-SGD or confidence capping)",
            "Obtain DPO sign-off on cross-border transfer mechanism for EHR dataset",
        ],
        next_review_date="2025-06-30",
    )


# ---------------------------------------------------------------------------
# REPORT GENERATOR
# ---------------------------------------------------------------------------

def print_pia_report(pia: PIAResult) -> None:
    """Prints a human-readable summary of the PIA result."""
    print("=" * 70)
    print("8-STAGE PHARMA AI PIA FRAMEWORK — ASSESSMENT REPORT")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("=" * 70)
    print(f"Project:         {pia.project_name}")
    print(f"Assessment Date: {pia.assessment_date}")
    print(f"Assessor:        {pia.assessor}")
    print(f"DPO Reviewer:    {pia.dpo_reviewer}")
    print(f"Overall Risk:    {pia.overall_residual_risk.value}")
    print(f"Approved:        {'YES' if pia.approved_for_deployment else 'NO — CONDITIONS REQUIRED'}")

    print("\n--- DATA ASSETS ---")
    for a in pia.data_assets:
        xb = "YES" if a.cross_border_transfer else "No"
        print(f"  {a.asset_name}: {a.category.value}, n≈{a.n_records_approx:,}, "
              f"Cross-border: {xb}, Retention: {a.retention_years}yr")

    print("\n--- STRIDE THREATS (Top Residual Risks) ---")
    critical_threats = [t for t in pia.stride_threats
                        if t.residual_risk in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
    for t in critical_threats:
        print(f"  [{t.residual_risk.value}] {t.threat.value}")
        print(f"    Component: {t.pipeline_component}")

    print("\n--- CONTROL GAPS ---")
    for g in pia.control_gaps:
        print(f"  [{g.gap_severity.value}] {g.control_area}")
        print(f"    Action: {g.remediation_action}")
        print(f"    By: {g.target_completion_date}")

    print("\n--- CONDITIONS FOR APPROVAL ---")
    for i, cond in enumerate(pia.conditions_for_approval, 1):
        print(f"  {i}. {cond}")

    print(f"\nNext Review: {pia.next_review_date}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pia = build_synthetic_pia_example()
    print_pia_report(pia)
    print("\nPIA JSON export (first 500 chars):")
    pia_dict = {
        "project_name": pia.project_name,
        "assessment_date": pia.assessment_date,
        "overall_residual_risk": pia.overall_residual_risk.value,
        "approved_for_deployment": pia.approved_for_deployment,
        "conditions_count": len(pia.conditions_for_approval),
        "control_gaps_count": len(pia.control_gaps),
    }
    print(json.dumps(pia_dict, indent=2)[:500])
