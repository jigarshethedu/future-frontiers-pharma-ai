"""
test_privacy_by_design_checklist.py
Chapter 5: Building a Privacy-First AI Strategy
Future Frontiers by Jigar Sheth
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from privacy_by_design_checklist import (
    build_pharma_pbd_checklist,
    score_checklist,
    apply_answers,
    generate_synthetic_project_responses,
    PbDAssessmentResult,
)


def test_checklist_has_seven_principles():
    """The PbD checklist must contain exactly 7 principles (Cavoukian's framework)."""
    principles = build_pharma_pbd_checklist()
    assert len(principles) == 7


def test_perfect_score_yields_grade_a():
    """Answering Yes (1.0) to every question should yield a score of 100 and grade A."""
    principles = build_pharma_pbd_checklist()
    # Answer 1.0 to every question
    all_yes = {q.question_id: 1.0 for p in principles for q in p.questions}
    principles = apply_answers(principles, all_yes)
    result = score_checklist(principles)
    assert result.total_score == 100.0
    assert result.grade == "A"
    assert result.passed is True


def test_zero_score_yields_grade_f():
    """Answering No (0.0) to every question should yield 0.0 and grade F."""
    principles = build_pharma_pbd_checklist()
    all_no = {q.question_id: 0.0 for p in principles for q in p.questions}
    principles = apply_answers(principles, all_no)
    result = score_checklist(principles)
    assert result.total_score == 0.0
    assert result.grade == "F"
    assert result.passed is False


def test_synthetic_projects_return_results():
    """Both synthetic projects should return valid PbDAssessmentResult objects."""
    projects = generate_synthetic_project_responses()
    assert len(projects) == 2
    for project_data in projects:
        principles = build_pharma_pbd_checklist()
        principles = apply_answers(principles, project_data["answers"])
        result = score_checklist(principles)
        result.project_name = project_data["project_name"]
        assert isinstance(result, PbDAssessmentResult)
        assert 0 <= result.total_score <= 100
        assert result.grade in ("A", "B", "C", "D", "F")
        assert len(result.weakest_principles) == 3
        assert len(result.remediation_priorities) == 3


def test_principle_weights_sum_to_one():
    """Within each principle, question weights must sum to approximately 1.0."""
    principles = build_pharma_pbd_checklist()
    for p in principles:
        total_weight = sum(q.weight for q in p.questions)
        assert abs(total_weight - 1.0) < 0.001, (
            f"Principle '{p.name}' weights sum to {total_weight}, expected 1.0"
        )
