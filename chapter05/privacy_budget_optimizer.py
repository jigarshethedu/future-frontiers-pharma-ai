"""
privacy_budget_optimizer.py
==============================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Implements epsilon budget allocation optimization across multiple DP-protected
    operations in a pharma AI pipeline. Given a total epsilon budget and a set of
    pipeline operations (each with a utility value and a required epsilon), finds
    the allocation that maximizes total utility without exceeding the budget.

    Uses a greedy knapsack approach suitable for real-time pipeline gating.
    In production, couples with the PrivacyBudget tracker in differential_privacy_demo.py.

Dependencies:
    - Python 3.10+
    - numpy >= 1.26.0

GitHub: chapter05/privacy_budget_optimizer.py
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PipelineOperation:
    """One DP-protected operation in the pharma AI pipeline."""
    op_id: str
    description: str
    epsilon_required: float      # Minimum epsilon needed for this operation
    utility_value: float         # 0–10 scale of clinical/business value
    priority_tier: int           # 1 = must-have, 2 = high-value, 3 = optional
    can_be_deferred: bool        # Whether operation can run in a future budget period


@dataclass
class BudgetAllocationResult:
    """Output of one budget optimization run."""
    total_budget: float
    allocated_operations: list[PipelineOperation]
    deferred_operations: list[PipelineOperation]
    blocked_operations: list[PipelineOperation]
    total_epsilon_used: float
    total_utility_achieved: float
    budget_utilization_pct: float


def optimize_budget_allocation(operations: list[PipelineOperation],
                                total_budget: float) -> BudgetAllocationResult:
    """
    Greedy budget allocation: prioritize by tier first, then utility/epsilon ratio.
    Tier-1 operations are always attempted. Tier-2 and 3 are ranked by utility density.
    """
    tier1 = [op for op in operations if op.priority_tier == 1]
    tier2_3 = sorted(
        [op for op in operations if op.priority_tier > 1],
        key=lambda op: op.utility_value / op.epsilon_required,
        reverse=True
    )

    allocated, deferred, blocked = [], [], []
    budget_remaining = total_budget

    for op in tier1 + tier2_3:
        if op.epsilon_required <= budget_remaining:
            allocated.append(op)
            budget_remaining -= op.epsilon_required
        elif op.can_be_deferred:
            deferred.append(op)
        else:
            blocked.append(op)

    total_used = total_budget - budget_remaining
    total_utility = sum(op.utility_value for op in allocated)

    return BudgetAllocationResult(
        total_budget=total_budget,
        allocated_operations=allocated,
        deferred_operations=deferred,
        blocked_operations=blocked,
        total_epsilon_used=round(total_used, 4),
        total_utility_achieved=round(total_utility, 2),
        budget_utilization_pct=round(100 * total_used / total_budget, 1),
    )


def build_synthetic_pharma_pipeline() -> list[PipelineOperation]:
    """Synthetic representation of a pharma AI pipeline's DP-protected operations."""
    return [
        PipelineOperation("OP-01", "EHR cohort query: aggregate biomarker statistics",
                          epsilon_required=0.5, utility_value=9.0, priority_tier=1,
                          can_be_deferred=False),
        PipelineOperation("OP-02", "Training data summary statistics for audit report",
                          epsilon_required=0.3, utility_value=6.0, priority_tier=1,
                          can_be_deferred=False),
        PipelineOperation("OP-03", "DP-SGD training — 5 epochs (0.4 each)",
                          epsilon_required=2.0, utility_value=10.0, priority_tier=1,
                          can_be_deferred=False),
        PipelineOperation("OP-04", "Subgroup performance analysis (age / ethnicity)",
                          epsilon_required=0.8, utility_value=8.0, priority_tier=2,
                          can_be_deferred=True),
        PipelineOperation("OP-05", "External benchmarking query against consortium",
                          epsilon_required=0.6, utility_value=5.0, priority_tier=2,
                          can_be_deferred=True),
        PipelineOperation("OP-06", "Ad-hoc exploratory biomarker correlation query",
                          epsilon_required=0.4, utility_value=3.0, priority_tier=3,
                          can_be_deferred=True),
        PipelineOperation("OP-07", "Model retraining after regulatory feedback",
                          epsilon_required=1.5, utility_value=9.5, priority_tier=1,
                          can_be_deferred=False),
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("PRIVACY BUDGET OPTIMIZER")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    ops = build_synthetic_pharma_pipeline()
    total_eps = 4.0

    print(f"\nTotal epsilon budget: {total_eps}")
    print(f"Pipeline operations:  {len(ops)}")
    print(f"Total epsilon requested: {sum(op.epsilon_required for op in ops):.1f}\n")

    result = optimize_budget_allocation(ops, total_eps)

    print(f"ALLOCATED ({len(result.allocated_operations)} operations):")
    for op in result.allocated_operations:
        print(f"  ✅ [{op.epsilon_required:.1f}ε] {op.op_id}: {op.description}")

    if result.deferred_operations:
        print(f"\nDEFERRED — next budget period ({len(result.deferred_operations)}):")
        for op in result.deferred_operations:
            print(f"  ⏳ [{op.epsilon_required:.1f}ε] {op.op_id}: {op.description}")

    if result.blocked_operations:
        print(f"\nBLOCKED — cannot defer ({len(result.blocked_operations)}):")
        for op in result.blocked_operations:
            print(f"  🚫 [{op.epsilon_required:.1f}ε] {op.op_id}: {op.description}")

    print(f"\nBudget used: {result.total_epsilon_used:.2f} / {result.total_budget} "
          f"({result.budget_utilization_pct:.1f}%)")
    print(f"Utility achieved: {result.total_utility_achieved:.1f} / "
          f"{sum(op.utility_value for op in ops):.1f}")
