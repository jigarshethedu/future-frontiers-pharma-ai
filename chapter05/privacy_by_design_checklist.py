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

Dependencies: Python 3.10+, stdlib only.
"""

from __future__ import annotations
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PbDQuestion:
    principle_id: int
    question_id:  str
    text:         str
    weight:       float
    answer:       Optional[float] = None
    notes:        str = ""


@dataclass
class PbDPrinciple:
    principle_id:     int
    name:             str
    description:      str
    questions:        list[PbDQuestion] = field(default_factory=list)
    principle_weight: float = 1.0 / 7

    @property
    def criteria(self) -> list[PbDQuestion]:
        """Alias — tests reference .criteria; same list as .questions."""
        return self.questions


@dataclass
class PbDAssessmentResult:
    project_name:           str
    total_score:            float
    grade:                  str
    principle_scores:       dict[str, float]
    weakest_principles:     list[str]
    remediation_priorities: list[str]
    passed:                 bool


def build_pharma_pbd_checklist() -> list[PbDPrinciple]:
    return [
        PbDPrinciple(1, "Proactive not Reactive; Preventative not Remedial",
            "Privacy risks are anticipated and prevented before they materialize.",
            questions=[
                PbDQuestion(1,"P1-Q1","PIA completed before model development?",0.40),
                PbDQuestion(1,"P1-Q2","STRIDE threat models applied before training?",0.30),
                PbDQuestion(1,"P1-Q3","Documented privacy budget allocated prior to deployment?",0.30),
            ]),
        PbDPrinciple(2, "Privacy as the Default Setting",
            "Maximum privacy protection is the automatic outcome.",
            questions=[
                PbDQuestion(2,"P2-Q1","Minimum necessary data collected by default?",0.35),
                PbDQuestion(2,"P2-Q2","Model outputs de-identified by default before sharing?",0.35),
                PbDQuestion(2,"P2-Q3","API endpoints return only required fields by role?",0.30),
            ]),
        PbDPrinciple(3, "Privacy Embedded into Design",
            "Privacy is integral to system architecture.",
            questions=[
                PbDQuestion(3,"P3-Q1","DP or FL embedded natively in training pipeline?",0.40),
                PbDQuestion(3,"P3-Q2","Encryption integrated at the data layer?",0.30),
                PbDQuestion(3,"P3-Q3","Privacy engineer reviewed architecture before build?",0.30),
            ]),
        PbDPrinciple(4, "Full Functionality — Positive-Sum, not Zero-Sum",
            "Privacy and clinical utility are both achieved.",
            questions=[
                PbDQuestion(4,"P4-Q1","Privacy-utility tradeoff formally measured?",0.40),
                PbDQuestion(4,"P4-Q2","Privacy controls do not reduce performance below threshold?",0.35),
                PbDQuestion(4,"P4-Q3","Privacy-preserving alternatives evaluated before de-ID?",0.25),
            ]),
        PbDPrinciple(5, "End-to-End Security — Full Lifecycle Protection",
            "Data securely managed from collection through destruction.",
            questions=[
                PbDQuestion(5,"P5-Q1","Data encrypted in transit (TLS 1.3+) and at rest?",0.30),
                PbDQuestion(5,"P5-Q2","Tamper-evident audit logs for training and inference?",0.30),
                PbDQuestion(5,"P5-Q3","Retention and secure destruction schedule enforced?",0.25),
                PbDQuestion(5,"P5-Q4","Model inversion / membership inference assessment done?",0.15),
            ]),
        PbDPrinciple(6, "Visibility and Transparency",
            "Independent verification that the system operates as stated.",
            questions=[
                PbDQuestion(6,"P6-Q1","Model card published internally?",0.35),
                PbDQuestion(6,"P6-Q2","Participants notified in plain language of AI use?",0.35),
                PbDQuestion(6,"P6-Q3","PIA accessible to compliance and DPO without special access?",0.30),
            ]),
        PbDPrinciple(7, "Respect for User Privacy — Keep it User-Centric",
            "Interests of individuals protected above organizational convenience.",
            questions=[
                PbDQuestion(7,"P7-Q1","Mechanism to withdraw consent and remove data from training?",0.40),
                PbDQuestion(7,"P7-Q2","Consent granular — per use case?",0.35),
                PbDQuestion(7,"P7-Q3","Consent reviewed for plain-language understandability?",0.25),
            ]),
    ]


_REMEDIATION_MAP = {
    "Proactive not Reactive; Preventative not Remedial":
        "Complete STRIDE threat model and PIA before next sprint. Establish privacy budget before training.",
    "Privacy as the Default Setting":
        "Audit all API endpoints for over-sharing. Enforce data minimization at ingestion layer.",
    "Privacy Embedded into Design":
        "Refactor pipeline to embed DP-SGD (Opacus) or FL natively. Remove bolt-on anonymization.",
    "Full Functionality — Positive-Sum, not Zero-Sum":
        "Run epsilon vs AUC curves. Document clinical acceptance thresholds before epsilon selection.",
    "End-to-End Security — Full Lifecycle Protection":
        "Implement immutable audit logging. Commission model inversion risk assessment.",
    "Visibility and Transparency":
        "Publish model card this quarter. Draft patient-facing plain-language AI notice.",
    "Respect for User Privacy — Keep it User-Centric":
        "Implement consent ledger (consent_ledger_stub.py) with granular per-use-case consent.",
}


def score_checklist(principles: list[PbDPrinciple]) -> PbDAssessmentResult:
    scores = {}
    for p in principles:
        w = sum((q.answer or 0.0) * q.weight for q in p.questions)
        scores[p.name] = round(w * 100, 1)
    total = round(sum(scores.values()) / 7, 1)
    if   total >= 90: grade = "A"
    elif total >= 80: grade = "B"
    elif total >= 70: grade = "C"
    elif total >= 60: grade = "D"
    else:             grade = "F"
    ranked  = sorted(scores.items(), key=lambda x: x[1])
    weakest = [n for n, _ in ranked[:3]]
    remed   = [f"[{s:.0f}/100] {n}: {_REMEDIATION_MAP.get(n,'Review.')}" for n,s in ranked[:3]]
    return PbDAssessmentResult("", total, grade, scores, weakest, remed, total >= 70)


def apply_answers(principles: list[PbDPrinciple],
                  answers: dict[str, float]) -> list[PbDPrinciple]:
    for p in principles:
        for q in p.questions:
            if q.question_id in answers:
                q.answer = answers[q.question_id]
    return principles


def generate_synthetic_project_responses() -> list[dict]:
    return [
        {"project_name": "Meridian Pharma — Federated Drug Discovery Pipeline (Synthetic)",
         "answers": {
             "P1-Q1":1.0,"P1-Q2":1.0,"P1-Q3":0.5,
             "P2-Q1":1.0,"P2-Q2":1.0,"P2-Q3":0.5,
             "P3-Q1":1.0,"P3-Q2":1.0,"P3-Q3":1.0,
             "P4-Q1":1.0,"P4-Q2":0.5,"P4-Q3":1.0,
             "P5-Q1":1.0,"P5-Q2":1.0,"P5-Q3":1.0,"P5-Q4":0.5,
             "P6-Q1":1.0,"P6-Q2":0.5,"P6-Q3":1.0,
             "P7-Q1":1.0,"P7-Q2":1.0,"P7-Q3":0.5,
         }},
        {"project_name": "ClearPath Health — EHR Readmission Risk Predictor (Synthetic)",
         "answers": {
             "P1-Q1":0.0,"P1-Q2":0.0,"P1-Q3":0.0,
             "P2-Q1":0.5,"P2-Q2":0.0,"P2-Q3":0.5,
             "P3-Q1":0.0,"P3-Q2":0.5,"P3-Q3":0.0,
             "P4-Q1":0.0,"P4-Q2":0.5,"P4-Q3":0.0,
             "P5-Q1":1.0,"P5-Q2":0.5,"P5-Q3":0.0,"P5-Q4":0.0,
             "P6-Q1":0.0,"P6-Q2":0.0,"P6-Q3":0.0,
             "P7-Q1":0.5,"P7-Q2":0.0,"P7-Q3":0.0,
         }},
    ]


# ── PbDChecker — high-level API used by test_chapter05_core.py ────────────────

@dataclass
class _Crit:
    criterion_id: str
    text:         str
    weight:       float
    passed:       bool = False
    notes:        str  = ""


@dataclass
class _PrinRec:
    principle_id: int
    name:         str
    criteria:     list = field(default_factory=list)


@dataclass
class PbDReport:
    project_name:    str
    assessor:        str
    assessment_date: str   = ""
    principles:      list  = field(default_factory=list)
    overall_score:   float = 0.0   # 0.0–1.0

    @property
    def failed_criteria(self):
        return [c for p in self.principles for c in p.criteria if not c.passed]


class PbDChecker:
    def __init__(self, project_name: str, assessor: str):
        self.project_name = project_name
        self.assessor     = assessor
        self.report = PbDReport(
            project_name    = project_name,
            assessor        = assessor,
            assessment_date = datetime.date.today().isoformat(),
            principles      = self._build(),
        )

    def _build(self):
        recs = []
        for p in build_pharma_pbd_checklist():
            pr = _PrinRec(p.principle_id, p.name)
            for q in p.questions:
                pr.criteria.append(_Crit(q.question_id, q.text, q.weight))
            recs.append(pr)
        return recs

    def _norm(self, cid: str) -> str:
        if "-" in cid: return cid
        if "." in cid:
            a, b = cid.split(".", 1)
            return f"P{a}-Q{b}"
        return cid

    def run_assessment(self, responses: dict, notes: dict = None) -> PbDReport:
        notes = {self._norm(k): v for k, v in (notes or {}).items()}
        # Map canonical -> original key so failed_criteria preserves caller's ID format
        canon_to_orig = {self._norm(k): k for k in responses}
        norm_responses = {self._norm(k): v for k, v in responses.items()}
        tw = ts = 0.0
        for p in self.report.principles:
            for c in p.criteria:
                if c.criterion_id in norm_responses:
                    c.passed = bool(norm_responses[c.criterion_id])
                    # Store caller's original ID (e.g. "1.1") for failed_criteria output
                    c.criterion_id = canon_to_orig.get(c.criterion_id, c.criterion_id)
                if c.criterion_id in notes or self._norm(c.criterion_id) in notes:
                    c.notes = notes.get(c.criterion_id, notes.get(self._norm(c.criterion_id), ""))
                tw += c.weight
                ts += c.weight if c.passed else 0.0
        self.report.overall_score = ts / tw if tw else 0.0
        return self.report

    def to_json(self) -> str:
        return json.dumps({
            "project_name":    self.report.project_name,
            "assessor":        self.report.assessor,
            "assessment_date": self.report.assessment_date,
            "overall_score":   self.report.overall_score,
            "principles": [
                {"principle_id": p.principle_id, "name": p.name,
                 "criteria": [{"criterion_id": c.criterion_id,
                               "passed": c.passed, "notes": c.notes}
                              for c in p.criteria]}
                for p in self.report.principles
            ],
        }, indent=2)

    def generate_text_report(self) -> str:
        s = self.report.overall_score
        return "\n".join([
            "PbD ASSESSMENT REPORT",
            f"Project  : {self.report.project_name}",
            f"Assessor : {self.report.assessor}",
            f"Date     : {self.report.assessment_date}",
            f"STATUS   : {'PASS' if s >= 0.7 else 'FAIL'}",
            f"Score    : {s:.1%}",
            "",
            "DPO REVIEW REQUIRED: " + ("No" if s >= 0.7 else "Yes — score below 70%"),
        ])


if __name__ == "__main__":
    print("=" * 70)
    print("PRIVACY BY DESIGN CHECKLIST — PHARMA AI ASSESSMENT")
    print("Chapter 5 | Future Frontiers by Jigar Sheth")
    print("=" * 70)
    for pd in generate_synthetic_project_responses():
        p = apply_answers(build_pharma_pbd_checklist(), pd["answers"])
        r = score_checklist(p); r.project_name = pd["project_name"]
        print(f"\nPROJECT: {r.project_name}")
        print(f"Score: {r.total_score}/100  Grade: {r.grade}  PbD-Ready: {r.passed}")
        for n, s in r.principle_scores.items():
            print(f"  {n[:45]:<45} {s:5.1f}")
        print("Top 3 remediation:")
        for i, a in enumerate(r.remediation_priorities, 1):
            print(f"  {i}. {a}")
