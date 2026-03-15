"""
federated_learning_stub.py
============================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Implements a simulation of the Cross-Silo Clinical FL Architecture Pattern —
    the first of the five FL Architecture Taxonomy patterns introduced in Chapter 5.
    Simulates FedAvg aggregation across multiple synthetic hospital sites without
    any centralized patient data exchange.

    For full production FL framework implementations (Flower 1.8, SubstraFL,
    PySyft 0.8), see Chapter 7. This stub focuses on the architectural concept
    and governance hooks that the Privacy Architecture Canvas (PAC) requires.

Dependencies:
    - Python 3.10+
    - numpy >= 1.26.0
    - scikit-learn >= 1.4.0

GitHub: chapter05/federated_learning_stub.py
"""

from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class FLSite:
    """
    Represents one participating hospital site in a Cross-Silo Clinical FL round.
    Each site holds its own local synthetic dataset — data never leaves the site.
    """
    site_id: str
    n_patients: int
    X_local: np.ndarray = field(repr=False, default=None)
    y_local: np.ndarray = field(repr=False, default=None)
    local_model_weights: np.ndarray = field(repr=False, default=None)
    local_auc: float = 0.0


@dataclass
class FLGovernanceLog:
    """
    Audit trail for each federated round — required by the 8-Stage Pharma AI PIA
    Framework (Chapter 5). Records participation, weight transmission events, and
    any site that declined to participate (e.g., due to patient population shift).
    """
    round_number: int
    participating_sites: list[str]
    sites_declined: list[str] = field(default_factory=list)
    global_auc: float = 0.0
    aggregation_method: str = "FedAvg"
    privacy_mechanism: str = "Gaussian-DP on gradients (simulated)"


# ---------------------------------------------------------------------------
# SYNTHETIC SITE DATA GENERATOR
# ---------------------------------------------------------------------------

def generate_site_data(site_id: str, n_patients: int, seed: int) -> FLSite:
    """
    Generates a synthetic local dataset for one FL site.
    Each site has slightly different data distributions to simulate
    real-world non-IID data heterogeneity — the primary challenge in
    Cross-Silo Clinical FL deployments.
    """
    rng = np.random.default_rng(seed)
    # Site-specific distribution shift (simulates different patient demographics)
    shift = rng.uniform(-0.2, 0.2, size=8)
    X = rng.normal(loc=1.0 + shift, scale=0.35, size=(n_patients, 8)).clip(0, 5)
    prevalence = rng.uniform(0.08, 0.18)   # Each site has different outcome rate
    y = rng.binomial(1, prevalence, size=n_patients)
    # Signal in first 3 features for positive cases
    X[y == 1, :3] += rng.uniform(0.3, 1.0, size=(y.sum(), 3))
    X = X.clip(0, 5)
    return FLSite(site_id=site_id, n_patients=n_patients, X_local=X, y_local=y)


# ---------------------------------------------------------------------------
# FEDERATED AVERAGING (FedAvg)
# ---------------------------------------------------------------------------

def local_train(site: FLSite, global_weights: np.ndarray | None,
                scaler: StandardScaler) -> FLSite:
    """
    Simulates local training at a site. In production (Flower 1.8),
    this function runs inside the client's compute environment;
    only model weights (not data) are transmitted to the aggregator.
    """
    X_scaled = scaler.transform(site.X_local)
    model = LogisticRegression(max_iter=300, C=1.0, warm_start=False)
    if global_weights is not None:
        # Warm-start from global weights (FedAvg initialization)
        model.coef_ = global_weights[:8].reshape(1, -1)
        model.intercept_ = global_weights[8:]
        model.classes_ = np.array([0, 1])
    model.fit(X_scaled, site.y_local)
    # Pack weights: [coef (8), intercept (1)]
    site.local_model_weights = np.concatenate([
        model.coef_.flatten(), model.intercept_
    ])
    # Local AUC (evaluated locally — never shared with aggregator as raw data)
    if site.y_local.sum() > 0:
        site.local_auc = round(roc_auc_score(
            site.y_local, model.predict_proba(X_scaled)[:, 1]
        ), 4)
    return site


def fedavg_aggregate(sites: list[FLSite]) -> np.ndarray:
    """
    FedAvg: weighted average of local model weights, weighted by site n_patients.
    This is the aggregation step that runs on the central server — note that
    only weight vectors arrive here, never patient-level records.
    """
    total_patients = sum(s.n_patients for s in sites)
    agg_weights = np.zeros_like(sites[0].local_model_weights)
    for site in sites:
        weight_fraction = site.n_patients / total_patients
        agg_weights += weight_fraction * site.local_model_weights
    return agg_weights


# ---------------------------------------------------------------------------
# FL SIMULATION LOOP
# ---------------------------------------------------------------------------

def run_fl_simulation(sites: list[FLSite], n_rounds: int = 5,
                       holdout_X: np.ndarray = None,
                       holdout_y: np.ndarray = None) -> list[FLGovernanceLog]:
    """
    Runs n_rounds of Cross-Silo FL simulation.
    Returns a governance log for every round — the audit trail the
    8-Stage Pharma AI PIA Framework requires prior to model deployment.
    """
    scaler = StandardScaler()
    # Fit scaler on pooled feature ranges only (not labels — labels never leave sites)
    scaler.fit(np.vstack([s.X_local for s in sites]))

    governance_logs = []
    global_weights = None

    for round_num in range(1, n_rounds + 1):
        # Local training at each site
        participating = []
        for site in sites:
            # Governance gate: sites with fewer than 50 patients decline to protect privacy
            if site.n_patients < 50:
                continue
            site = local_train(site, global_weights, scaler)
            participating.append(site)

        # FedAvg aggregation
        global_weights = fedavg_aggregate(participating)

        # Evaluate global model on holdout set (held by aggregator only)
        global_auc = 0.0
        if holdout_X is not None and holdout_y is not None:
            # Reconstruct a logistic regression from aggregated weights for eval
            eval_model = LogisticRegression()
            eval_model.coef_ = global_weights[:8].reshape(1, -1)
            eval_model.intercept_ = global_weights[8:]
            eval_model.classes_ = np.array([0, 1])
            X_h_scaled = scaler.transform(holdout_X)
            if holdout_y.sum() > 0:
                global_auc = round(roc_auc_score(
                    holdout_y,
                    eval_model.predict_proba(X_h_scaled)[:, 1]
                ), 4)

        log = FLGovernanceLog(
            round_number=round_num,
            participating_sites=[s.site_id for s in participating],
            global_auc=global_auc,
        )
        governance_logs.append(log)
        print(f"Round {round_num:02d} | Sites: {len(participating):2d} | Global AUC: {global_auc:.4f}")

    return governance_logs


# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("FEDERATED LEARNING STUB — CROSS-SILO CLINICAL ARCHITECTURE")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    # Generate 6 synthetic hospital sites with varying patient populations
    site_configs = [
        ("NorthEast_Medical_Center", 400),
        ("Pacific_Oncology_Institute", 280),
        ("Midwest_Clinical_Research", 520),
        ("Southwest_Academic_Hospital", 190),
        ("Gulf_Regional_Health", 310),
        ("Mountain_Biomedical_Center", 35),  # Will be declined — too small
    ]

    sites = [generate_site_data(sid, n, seed=i * 17 + 42)
             for i, (sid, n) in enumerate(site_configs)]

    # Holdout set at the aggregator (synthetic — not from any site)
    rng = np.random.default_rng(999)
    X_holdout = rng.normal(loc=1.0, scale=0.35, size=(300, 8)).clip(0, 5)
    y_holdout = rng.binomial(1, 0.12, size=300)

    print(f"\nConfigured {len(sites)} FL sites:")
    for s in sites:
        status = "✅ Participating" if s.n_patients >= 50 else "⛔ Declined (n < 50)"
        print(f"  {s.site_id:<35} n={s.n_patients:>4}  {status}")

    print(f"\nRunning 5 rounds of FedAvg:\n")
    logs = run_fl_simulation(sites, n_rounds=5, holdout_X=X_holdout, holdout_y=y_holdout)

    print("\nGovernance Audit Trail:")
    for log in logs:
        print(f"  Round {log.round_number}: {len(log.participating_sites)} sites | "
              f"Global AUC: {log.global_auc:.4f} | Aggregation: {log.aggregation_method}")

    print("\nModule complete. See chapter07/ for full Flower and SubstraFL implementations.")
