"""
privacy_by_design_checklist.py
================================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Implements Cavoukian's seven Privacy by Design (PbD) principles as a
    structured, weighted assessment checklist for pharma AI projects.
    Returns a per-principle score, a total PbD readiness score (0–100),
    and a prioritized remediation plan — inputs to the P-ROI Model (Chapter 5).

Dependencies:
    - Python 3.10+
    - No external dependencies (stdlib only)

GitHub: chapter05/privacy_by_design_checklist.py
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class PbDQuestion:
    """A single yes/no/partial assessment question tied to one PbD principle."""
    principle_id: int        # 1–7 matching the seven PbD principles
    question_id: str         # e.g. "P1-Q1"
    text: str                # The question text presented to the assessor
    weight: float            # Contribution weight within that principle (sum to 1.0)
    answer: Optional[float] = None   # 0.0 = No, 0.5 = Partial, 1.0 = Yes


@dataclass
class PbDPrinciple:
    """One of Cavoukian's seven Privacy by Design principles."""
    principle_id: int
    name: str
    description: str
    questions: list[PbDQuestion] = field(default_factory=list)
    principle_weight: float = 1.0 / 7   # Equal weighting by default


@dataclass
class PbDAssessmentResult:
    """Full result object returned after scoring a project against the checklist."""
    project_name: str
    total_score: float                        # 0–100
    grade: str                                # A / B / C / D / F
    principle_scores: dict[str, float]        # Principle name → 0–100
    weakest_principles: list[str]             # Ordered worst-first
    remediation_priorities: list[str]         # Actionable steps, worst principle first
    passed: bool                              # Score >= 70 considered PbD-ready


# ---------------------------------------------------------------------------
# CHECKLIST DEFINITION
# ---------------------------------------------------------------------------

def build_pharma_pbd_checklist() -> list[PbDPrinciple]:
    """
    Constructs the seven PbD principles with pharma-specific assessment questions.
    Each question carries a weight summing to 1.0 within its principle.
    These questions operationalize Cavoukian's foundational framework for the
    specific context of AI systems processing clinical, genomic, or trial data.
    """

    principles = [
        PbDPrinciple(
            principle_id=1,
            name="Proactive not Reactive; Preventative not Remedial",
            description="Privacy risks are anticipated and prevented before they materialize.",
            questions=[
                PbDQuestion(1, "P1-Q1",
                    "Has a Privacy Impact Assessment (PIA) been completed before model development began?",
                    weight=0.40),
                PbDQuestion(1, "P1-Q2",
                    "Are threat models (e.g., STRIDE) applied to the AI pipeline before training?",
                    weight=0.30),
                PbDQuestion(1, "P1-Q3",
                    "Does the project have a documented privacy budget allocated prior to deployment?",
                    weight=0.30),
            ]
        ),
        PbDPrinciple(
            principle_id=2,
            name="Privacy as the Default Setting",
            description="Without action by the user, privacy protection is the automatic outcome.",
            questions=[
                PbDQuestion(2, "P2-Q1",
                    "Is the minimum necessary data collected by default (no opt-out required to limit scope)?",
                    weight=0.35),
                PbDQuestion(2, "P2-Q2",
                    "Are model outputs de-identified or aggregated by default before any downstream sharing?",
                    weight=0.35),
                PbDQuestion(2, "P2-Q3",
                    "Do all API endpoints return only the fields required for the requesting role by default?",
                    weight=0.30),
            ]
        ),
        PbDPrinciple(
            principle_id=3,
            name="Privacy Embedded into Design",
            description="Privacy is integral to system architecture, not added as a later layer.",
            questions=[
                PbDQuestion(3, "P3-Q1",
                    "Is differential privacy or federated learning embedded in the training pipeline (not a post-hoc wrapper)?",
                    weight=0.40),
                PbDQuestion(3, "P3-Q2",
                    "Are encryption and key management integrated at the data layer, not the application layer?",
                    weight=0.30),
                PbDQuestion(3, "P3-Q3",
                    "Has the architecture been reviewed by a privacy engineer (not only a security engineer) before build?",
                    weight=0.30),
            ]
        ),
        PbDPrinciple(
            principle_id=4,
            name="Full Functionality — Positive-Sum, not Zero-Sum",
            description="Both privacy and clinical utility are achieved; trade-offs are quantified, not assumed.",
            questions=[
                PbDQuestion(4, "P4-Q1",
                    "Has the privacy-utility tradeoff been formally measured (e.g., epsilon vs. AUC curve)?",
                    weight=0.40),
                PbDQuestion(4, "P4-Q2",
                    "Is there documented evidence that privacy controls did not reduce clinical model performance below clinical acceptance thresholds?",
                    weight=0.35),
                PbDQuestion(4, "P4-Q3",
                    "Are privacy-preserving alternatives evaluated before any de-identification decision?",
                    weight=0.25),
            ]
        ),
        PbDPrinciple(
            principle_id=5,
            name="End-to-End Security — Full Lifecycle Protection",
            description="Data is securely managed from collection through destruction.",
            questions=[
                PbDQuestion(5, "P5-Q1",
                    "Is data encrypted in transit (TLS 1.3+) and at rest (AES-256) across all pipeline stages?",
                    weight=0.30),
                PbDQuestion(5, "P5-Q2",
                    "Are audit logs generated and tamper-evident for every model training and inference event?",
                    weight=0.30),
                PbDQuestion(5, "P5-Q3",
                    "Is a data retention and secure destruction schedule documented and enforced programmatically?",
                    weight=0.25),
                PbDQuestion(5, "P5-Q4",
                    "Has a model inversion or membership inference risk assessment been conducted?",
                    weight=0.15),
            ]
        ),
        PbDPrinciple(
            principle_id=6,
            name="Visibility and Transparency",
            description="Independent verification that the system operates as stated.",
            questions=[
                PbDQuestion(6, "P6-Q1",
                    "Is a model card published (internally at minimum) documenting data sources, training procedures, and privacy controls?",
                    weight=0.35),
                PbDQuestion(6, "P6-Q2",
                    "Are patients or research participants notified, in plain language, that their data trains or is processed by an AI system?",
                    weight=0.35),
                PbDQuestion(6, "P6-Q3",
                    "Is the privacy impact assessment accessible to the compliance team and DPO without requesting special access?",
                    weight=0.30),
            ]
        ),
        PbDPrinciple(
            principle_id=7,
            name="Respect for User Privacy — Keep it User-Centric",
            description="The interests of individuals are protected above system or organizational convenience.",
            questions=[
                PbDQuestion(7, "P7-Q1",
                    "Do patients have a documented mechanism to withdraw consent and have their data removed from future training runs?",
                    weight=0.40),
                PbDQuestion(7, "P7-Q2",
                    "Is consent granular — allowing participation in specific use cases without blanket consent to all AI uses?",
                    weight=0.35),
                PbDQuestion(7, "P7-Q3",
                    "Has the consent process been reviewed for understandability by non-expert readers (plain language audit)?",
                    weight=0.25),
            ]
        ),
    ]
    return principles


# ---------------------------------------------------------------------------
# SCORING ENGINE
# ---------------------------------------------------------------------------

def score_checklist(principles: list[PbDPrinciple]) -> PbDAssessmentResult:
    """
    Scores all answered questions and aggregates to principle-level and total scores.
    Unanswered questions (answer=None) are treated as 0.0 (conservative default —
    in pharma, absence of evidence is not evidence of compliance).

    Returns a PbDAssessmentResult with scores, grade, and remediation priorities.
    """
    principle_scores: dict[str, float] = {}

    # Score each of the seven principles
    for principle in principles:
        weighted_score = 0.0
        for q in principle.questions:
            answer = q.answer if q.answer is not None else 0.0
            weighted_score += answer * q.weight
        # Convert to 0–100
        principle_scores[principle.name] = round(weighted_score * 100, 1)

    # Total score: equal weighting across the seven principles
    total_score = round(sum(principle_scores.values()) / 7, 1)

    # Grade assignment
    if total_score >= 90:
        grade = "A"
    elif total_score >= 80:
        grade = "B"
    elif total_score >= 70:
        grade = "C"
    elif total_score >= 60:
        grade = "D"
    else:
        grade = "F"

    # Rank principles worst-first for remediation prioritization
    ranked = sorted(principle_scores.items(), key=lambda x: x[1])
    weakest = [name for name, _ in ranked[:3]]

    # Generate remediation recommendations for the three weakest principles
    remediation_map = {
        "Proactive not Reactive; Preventative not Remedial":
            "Complete a STRIDE threat model and a Privacy Impact Assessment before the next sprint. Establish a privacy budget before training begins.",
        "Privacy as the Default Setting":
            "Audit all API endpoints for over-sharing. Enforce data minimization at the ingestion layer through schema validation.",
        "Privacy Embedded into Design":
            "Refactor training pipeline to embed DP-SGD (via Opacus) or federated training natively. Remove bolt-on anonymization wrappers.",
        "Full Functionality — Positive-Sum, not Zero-Sum":
            "Run privacy-utility tradeoff curves (epsilon vs. AUC) and document clinical acceptance thresholds before any epsilon selection.",
        "End-to-End Security — Full Lifecycle Protection":
            "Implement immutable audit logging (append-only store). Commission a model inversion risk assessment with an external red team.",
        "Visibility and Transparency":
            "Publish an internal model card this quarter. Draft patient-facing plain-language notice describing AI use, co-authored with patient advocates.",
        "Respect for User Privacy — Keep it User-Centric":
            "Implement a consent ledger (see consent_ledger_stub.py) supporting per-use-case granular consent and cryptographic proof of withdrawal.",
    }

    remediation_priorities = [
        f"[{score:.0f}/100] {name}: {remediation_map.get(name, 'Review and improve.')}"
        for name, score in ranked[:3]
    ]

    return PbDAssessmentResult(
        project_name="",  # filled by caller
        total_score=total_score,
        grade=grade,
        principle_scores=principle_scores,
        weakest_principles=weakest,
        remediation_priorities=remediation_priorities,
        passed=total_score >= 70,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC DEMO DATASET
# ---------------------------------------------------------------------------

def generate_synthetic_project_responses() -> list[dict]:
    """
    Generates two synthetic pharma AI projects for demonstration.
    Project A: a well-governed federated drug discovery pipeline.
    Project B: a hastily deployed EHR-based readmission predictor.
    No real patient data is referenced or embedded anywhere in this module.
    """
    return [
        {
            "project_name": "Meridian Pharma — Federated Drug Discovery Pipeline (Synthetic)",
            "answers": {
                "P1-Q1": 1.0, "P1-Q2": 1.0, "P1-Q3": 0.5,
                "P2-Q1": 1.0, "P2-Q2": 1.0, "P2-Q3": 0.5,
                "P3-Q1": 1.0, "P3-Q2": 1.0, "P3-Q3": 1.0,
                "P4-Q1": 1.0, "P4-Q2": 0.5, "P4-Q3": 1.0,
                "P5-Q1": 1.0, "P5-Q2": 1.0, "P5-Q3": 1.0, "P5-Q4": 0.5,
                "P6-Q1": 1.0, "P6-Q2": 0.5, "P6-Q3": 1.0,
                "P7-Q1": 1.0, "P7-Q2": 1.0, "P7-Q3": 0.5,
            }
        },
        {
            "project_name": "ClearPath Health — EHR Readmission Risk Predictor (Synthetic)",
            "answers": {
                "P1-Q1": 0.0, "P1-Q2": 0.0, "P1-Q3": 0.0,
                "P2-Q1": 0.5, "P2-Q2": 0.0, "P2-Q3": 0.5,
                "P3-Q1": 0.0, "P3-Q2": 0.5, "P3-Q3": 0.0,
                "P4-Q1": 0.0, "P4-Q2": 0.5, "P4-Q3": 0.0,
                "P5-Q1": 1.0, "P5-Q2": 0.5, "P5-Q3": 0.0, "P5-Q4": 0.0,
                "P6-Q1": 0.0, "P6-Q2": 0.0, "P6-Q3": 0.0,
                "P7-Q1": 0.5, "P7-Q2": 0.0, "P7-Q3": 0.0,
            }
        }
    ]


def apply_answers(principles: list[PbDPrinciple], answers: dict[str, float]) -> list[PbDPrinciple]:
    """Applies a dictionary of question_id → answer to the checklist."""
    for principle in principles:
        for q in principle.questions:
            if q.question_id in answers:
                q.answer = answers[q.question_id]
    return principles


# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    projects = generate_synthetic_project_responses()

    print("=" * 70)
    print("PRIVACY BY DESIGN CHECKLIST — PHARMA AI ASSESSMENT")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    for project_data in projects:
        # Build fresh checklist for each project
        principles = build_pharma_pbd_checklist()
        principles = apply_answers(principles, project_data["answers"])
        result = score_checklist(principles)
        result.project_name = project_data["project_name"]

        print(f"\nPROJECT: {result.project_name}")
        print(f"Total PbD Score: {result.total_score}/100  |  Grade: {result.grade}  |  PbD-Ready: {result.passed}")
        print("\nPrinciple Breakdown:")
        for principle_name, score in result.principle_scores.items():
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            short_name = principle_name.split("—")[0].strip()[:45]
            print(f"  {short_name:<45} [{bar}] {score:5.1f}")

        print("\nTop 3 Remediation Priorities:")
        for i, action in enumerate(result.remediation_priorities, 1):
            print(f"  {i}. {action}")
        print("-" * 70)

    print("\nModule complete. Save outputs and cross-reference with")
    print("privacy_impact_assessment.py and privacy_budget_optimizer.py.")


# ---------------------------------------------------------------------------
# COMPATIBILITY LAYER — matches test_chapter05_core.py expected API
# ---------------------------------------------------------------------------

import datetime
from dataclasses import dataclass as _dc, field as _field


@_dc
class _Criterion:
    criterion_id: str
    text: str
    weight: float
    passed: bool = False
    notes: str = ""


@_dc
class _PrincipleResult:
    principle_id: int
    name: str
    criteria: list = _field(default_factory=list)


@_dc
class PbDReport:
    project_name: str
    assessor: str
    assessment_date: str = ""
    principles: list = _field(default_factory=list)
    overall_score: float = 0.0

    @property
    def failed_criteria(self):
        return [c for p in self.principles for c in p.criteria if not c.passed]


class PbDChecker:
    """
    High-level wrapper around the PbD checklist — provides the API expected
    by test_chapter05_core.py: instantiate, run_assessment(), to_json(),
    generate_text_report().
    """

    def __init__(self, project_name: str, assessor: str):
        self.project_name = project_name
        self.assessor = assessor
        self.report = PbDReport(
            project_name=project_name,
            assessor=assessor,
            assessment_date=datetime.date.today().isoformat(),
            principles=self._build_principles(),
        )

    def _build_principles(self) -> list:
        raw = build_pharma_pbd_checklist()
        result = []
        for p in raw:
            pr = _PrincipleResult(principle_id=p.principle_id, name=p.name)
            for q in p.questions:
                pr.criteria.append(_Criterion(
                    criterion_id=q.question_id,
                    text=q.text,
                    weight=q.weight,
                ))
            result.append(pr)
        return result

    def run_assessment(self, responses: dict, notes: dict = None) -> PbDReport:
        """
        responses: {criterion_id: True/False}
        notes:     {criterion_id: str}  (optional)
        Updates the report's criteria, recomputes overall_score, returns report.
        """
        notes = notes or {}
        total_weight = 0.0
        total_score = 0.0

        for p in self.report.principles:
            for c in p.criteria:
                if c.criterion_id in responses:
                    c.passed = bool(responses[c.criterion_id])
                if c.criterion_id in notes:
                    c.notes = notes[c.criterion_id]
                total_weight += c.weight
                total_score += c.weight if c.passed else 0.0

        self.report.overall_score = (total_score / total_weight) if total_weight > 0 else 0.0
        return self.report

    def to_json(self) -> str:
        import json
        data = {
            "project_name": self.report.project_name,
            "assessor": self.report.assessor,
            "assessment_date": self.report.assessment_date,
            "overall_score": self.report.overall_score,
            "principles": [
                {
                    "principle_id": p.principle_id,
                    "name": p.name,
                    "criteria": [
                        {"criterion_id": c.criterion_id, "passed": c.passed,
                         "notes": c.notes}
                        for c in p.criteria
                    ]
                }
                for p in self.report.principles
            ]
        }
        return json.dumps(data, indent=2)

    def generate_text_report(self) -> str:
        lines = [
            f"PbD ASSESSMENT REPORT",
            f"Project  : {self.report.project_name}",
            f"Assessor : {self.report.assessor}",
            f"Date     : {self.report.assessment_date}",
            f"STATUS   : {'PASS' if self.report.overall_score >= 0.7 else 'FAIL'}",
            f"Score    : {self.report.overall_score:.1%}",
            "",
            "DPO REVIEW REQUIRED: " + (
                "No — score above threshold"
                if self.report.overall_score >= 0.7
                else "Yes — score below 70%"
            ),
        ]
        return "\n".join(lines)
