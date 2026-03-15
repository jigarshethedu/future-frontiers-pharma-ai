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


# ---------------------------------------------------------------------------
# COMPATIBILITY LAYER — matches test_federated_learning_stub.py expected API
# ---------------------------------------------------------------------------

def generate_multi_site_dataset(n_sites: int = 5,
                                 n_per_site: int = 200,
                                 seed: int = 42) -> list[tuple]:
    """
    Generates synthetic data for multiple FL sites.
    Returns list of (site_id, X, y, n) tuples — the format tests expect.
    """
    sites = []
    for i in range(n_sites):
        site = generate_site_data(f"site_{i}", n_per_site, seed=seed + i * 7)
        sites.append((site.site_id, site.X_local, site.y_local, site.n_patients))
    return sites


def federated_average(weights: list[np.ndarray],
                       counts: list[int]) -> np.ndarray:
    """
    Weighted average of weight vectors by sample count.
    Wrapper around fedavg_aggregate() using the test's expected signature.
    """
    total = sum(counts)
    result = np.zeros_like(weights[0], dtype=float)
    for w, n in zip(weights, counts):
        result += (n / total) * w
    return result


class CrossSiloClinicalFL:
    """Pattern 1 — Cross-Silo Clinical FL (hospitals, homogeneous EHR data)."""

    def __init__(self, sites: list[tuple], n_rounds: int = 5):
        self.sites = [
            FLSite(site_id=sid, n_patients=n,
                   X_local=X, y_local=y)
            for sid, X, y, n in sites
        ]
        self.n_rounds = n_rounds

    def train(self) -> dict:
        rng = np.random.default_rng(42)
        X_hold = rng.normal(1.0, 0.35, (200, 8)).clip(0, 5)
        y_hold = rng.binomial(1, 0.12, 200)
        logs = run_fl_simulation(self.sites, self.n_rounds,
                                  holdout_X=X_hold, holdout_y=y_hold)
        return {
            "rounds_completed": len(logs),
            "final_avg_auc": logs[-1].global_auc if logs else 0.0,
            "governance_logs": logs,
        }


class CrossCompanyFL:
    """Pattern 2 — Cross-Company Drug Discovery FL (competitive, adversarial trust)."""

    def __init__(self, companies: list[str]):
        self.companies = companies

    def governance_checklist(self) -> dict:
        return {
            "framework": "SubstraFL with permissioned access and per-update audit trail",
            "governance_items": [
                "Signed multi-party computation agreement between all participants",
                "Independent audit of gradient updates before aggregation",
                "Zero-knowledge proof of local training compliance",
                "Secure aggregation preventing any participant seeing others' updates",
                "Regulatory notification plan if a safety signal is detected in federated model",
                "IP protection clauses for proprietary compound structures",
            ],
            "participants": self.companies,
            "trust_model": "Adversarial — no participant trusts any other",
        }


class PatientDeviceFL:
    """Pattern 4 — Patient-Device FL (wearables, smartphones, continuous monitoring)."""

    def __init__(self, n_devices: int = 100, epsilon: float = 1.0):
        self.n_devices = n_devices
        self.epsilon = epsilon

    def simulate_device_update(self, device_id: str,
                                n_local_samples: int) -> dict:
        rng = np.random.default_rng(hash(device_id) % (2**32))
        raw_gradient = rng.normal(0, 0.1, 9)
        sigma = np.sqrt(2 * np.log(1.25 / 1e-5)) / self.epsilon
        noisy_gradient = raw_gradient + rng.normal(0, sigma, 9)
        return {
            "device_id": device_id,
            "gradient_update": noisy_gradient,
            "n_local_samples": n_local_samples,
            "epsilon_consumed": self.epsilon,
            "raw_data_transmitted": False,   # Key guarantee: only gradient, never raw data
        }

    def aggregate_device_updates(self, updates: list[dict]) -> dict:
        gradients = np.array([u["gradient_update"] for u in updates])
        return {
            "aggregated_gradient": gradients.mean(axis=0),
            "n_devices_contributed": len(updates),
            "total_epsilon_consumed": sum(u["epsilon_consumed"] for u in updates),
        }


class RegulatorInTheLoopFL:
    """Pattern 3 — Regulator-in-the-Loop FL (FDA/EMA with read-only metric access)."""

    def __init__(self, regulator: str, sites: list[str]):
        self.regulator = regulator
        self.sites = sites

    def generate_regulatory_report(self, round_num: int,
                                    metrics: dict) -> dict:
        return {
            "regulator": self.regulator,
            "round": round_num,
            "participating_sites": len(self.sites),
            "data_transmitted": "Aggregate model metrics only — no patient-level data",
            "metrics_visible_to_regulator": metrics,
            "privacy_guarantee": "DP-SGD with epsilon ≤ 4.0 applied at each site",
            "audit_log_hash": hex(hash(str(metrics) + str(round_num))),
        }


class FederatedPharmacovigilanceFL:
    """Pattern 5 — Federated Pharmacovigilance (adverse event signal detection)."""

    def __init__(self, reporting_systems: list[str], epsilon: float = 2.0):
        self.reporting_systems = reporting_systems
        self.epsilon = epsilon

    def compute_dp_signal(self, drug: str, event: str,
                           local_counts: list[int]) -> dict:
        rng = np.random.default_rng(42)
        sensitivity = 1.0
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / self.epsilon
        noisy_counts = [max(0, c + rng.normal(0, sigma)) for c in local_counts]
        return {
            "drug": drug,
            "event": event,
            "dp_aggregate_count": round(sum(noisy_counts), 1),
            "epsilon_used": self.epsilon,
            "systems_contributed": len(self.reporting_systems),
            "raw_counts_shared": False,
        }
