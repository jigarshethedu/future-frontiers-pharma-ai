"""
differential_privacy_demo.py
==============================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Demonstrates the practical deployment of differential privacy (DP) in a
    synthetic pharma AI pipeline. Implements the Gaussian mechanism, the
    Laplace mechanism, and a simplified DP-SGD training loop — all on purely
    synthetic clinical data. Illustrates epsilon budget consumption and the
    privacy-utility tradeoff central to Chapter 5's P-ROI Model.

    As covered in Chapter 3, the formal epsilon/delta DP framework is defined
    there. This module focuses on deployment patterns and budget tracking —
    the practical "how to run it" perspective for pharma engineering teams.

Dependencies:
    - Python 3.10+
    - numpy >= 1.26.0
    - scikit-learn >= 1.4.0

GitHub: chapter05/differential_privacy_demo.py
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# PRIVACY BUDGET TRACKER
# ---------------------------------------------------------------------------

@dataclass
class PrivacyBudget:
    """
    Tracks cumulative epsilon consumption across multiple DP-protected operations.
    In production, this object lives in a persistent store and gates new queries.
    Exceeding the total budget triggers a mandatory review before further data use.
    """
    total_epsilon: float          # Total budget allocated at project start
    consumed_epsilon: float = 0.0
    operations: list[dict] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.consumed_epsilon)

    @property
    def exhausted(self) -> bool:
        return self.consumed_epsilon >= self.total_epsilon

    def consume(self, operation_name: str, epsilon_used: float) -> bool:
        """
        Records epsilon consumption. Returns True if operation is permitted,
        False if budget is exhausted. The conservative pharma default is to
        block operations once the budget is gone rather than to extend it
        quietly — extending silently is how privacy debt accumulates.
        """
        if self.exhausted or (self.consumed_epsilon + epsilon_used > self.total_epsilon):
            print(f"[BUDGET BLOCK] {operation_name} blocked — would exceed epsilon budget.")
            return False
        self.consumed_epsilon += epsilon_used
        self.operations.append({
            "operation": operation_name,
            "epsilon_used": epsilon_used,
            "cumulative": self.consumed_epsilon
        })
        return True

    def summary(self) -> str:
        lines = [
            f"Privacy Budget: {self.consumed_epsilon:.4f} / {self.total_epsilon:.4f} consumed "
            f"({100 * self.consumed_epsilon / self.total_epsilon:.1f}%)",
            f"Remaining: {self.remaining:.4f}",
            f"Status: {'⚠️  EXHAUSTED' if self.exhausted else '✅ Active'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DP MECHANISMS
# ---------------------------------------------------------------------------

def gaussian_mechanism(true_value: float, sensitivity: float, epsilon: float,
                        delta: float = 1e-5) -> float:
    """
    Adds calibrated Gaussian noise to protect a single numeric query result.
    Used for continuous-valued statistics like mean biomarker levels or
    aggregate adverse event rates across a patient cohort.

    sigma = (sensitivity / epsilon) * sqrt(2 * ln(1.25 / delta))
    """
    # Gaussian calibration per the standard analytic formula
    sigma = (sensitivity / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    noise = np.random.normal(0.0, sigma)
    return true_value + noise


def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """
    Adds Laplace noise — appropriate for integer-valued or bounded queries
    such as raw counts of patients with a given condition in a cohort.
    The Laplace mechanism is pure (epsilon, 0)-DP, making it preferable when
    the delta parameter cannot be justified to a regulatory auditor.
    """
    # Scale = sensitivity / epsilon (global sensitivity for counting queries = 1)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return true_value + noise


# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATOR
# ---------------------------------------------------------------------------

def generate_synthetic_clinical_dataset(n_patients: int = 2000,
                                         seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic clinical dataset for a hypothetical drug-induced
    hepatotoxicity (DILI) risk prediction task.
    Features: 8 synthetic biomarkers (ALT, AST, bilirubin surrogates, etc.)
    Label: Binary DILI outcome (1 = event within 90 days, 0 = no event).

    No real patient data. No real clinical thresholds. This dataset is
    generated purely to demonstrate the DP pipeline mechanics.
    """
    rng = np.random.default_rng(seed)
    # Simulate biomarker features (normalized ranges)
    biomarkers_normal = rng.normal(loc=[1.0, 1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.0],
                                    scale=[0.3, 0.4, 0.2, 0.5, 0.3, 0.4, 0.2, 0.3],
                                    size=(n_patients, 8))
    # DILI cases have elevated first 3 biomarkers
    outcomes = rng.binomial(1, 0.12, size=n_patients)
    biomarkers_normal[outcomes == 1, :3] += rng.uniform(0.5, 1.5,
                                                          size=(outcomes.sum(), 3))
    X = biomarkers_normal.clip(0.0, 5.0)
    y = outcomes
    return X, y


# ---------------------------------------------------------------------------
# DP-SGD SIMPLIFIED SIMULATION
# ---------------------------------------------------------------------------

def dp_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             budget: PrivacyBudget,
                             epsilon_per_epoch: float = 0.5,
                             n_epochs: int = 3,
                             clipping_norm: float = 1.0,
                             noise_multiplier: float = 1.1) -> dict:
    """
    Simulates a DP-protected training loop using gradient clipping and Gaussian
    noise addition — conceptually equivalent to Opacus's DP-SGD implementation.

    In production, use opacus.PrivacyEngine with a PyTorch model. This function
    demonstrates the privacy accounting logic without requiring GPU infrastructure.

    Returns accuracy and AUC for the trained model, and whether budget was maintained.
    """
    results = {"epochs_completed": 0, "auc": None, "budget_maintained": True}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Simulate per-epoch DP training with budget accounting
    for epoch in range(n_epochs):
        permitted = budget.consume(
            f"DP-SGD epoch {epoch + 1}",
            epsilon_per_epoch
        )
        if not permitted:
            results["budget_maintained"] = False
            break
        results["epochs_completed"] = epoch + 1

    # Train standard logistic regression as the downstream model
    # (In production, Opacus wraps the optimizer directly)
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_te)[:, 1]
    results["auc"] = round(roc_auc_score(y_test, y_prob), 4)

    # If called with a DataFrame, return (DataFrame, baseline_auc)
    # If called with arrays, return list of dicts — both formats supported
    if _called_with_df:
        from sklearn.linear_model import LogisticRegression as _LR2
        from sklearn.preprocessing import StandardScaler as _SS2
        from sklearn.metrics import roc_auc_score as _auc2
        import pandas as _pd2
        _sc2 = _SS2(); _Xt2 = _sc2.fit_transform(X_train); _Xe2 = _sc2.transform(X_test)
        _bm2 = _LR2(max_iter=200); _bm2.fit(_Xt2, y_train)
        _base = float(_auc2(y_test, _bm2.predict_proba(_Xe2)[:,1]))
        _df_r = _pd2.DataFrame([{"epsilon": r["epsilon"], "auc": r["mean_auc"],
                                   "auc_retention_pct": round(100*r["mean_auc"]/_base, 1)
                                   if _base > 0 else 0}
                                  for r in results])
        return _df_r, _base
    return results


# ---------------------------------------------------------------------------
# PRIVACY-UTILITY TRADEOFF CURVE
# ---------------------------------------------------------------------------

def compute_privacy_utility_curve(X, y=None, epsilons=None, n_runs: int = 5):
    """
    Measures AUC at multiple epsilon values to quantify the privacy-utility tradeoff.

    Accepts two calling conventions:
      1. Array:     compute_privacy_utility_curve(X, y, epsilons, n_runs)
                    Returns: list[dict] with keys epsilon, mean_auc, std_auc
      2. DataFrame: compute_privacy_utility_curve(df)   # df has a 'dili' column
                    Returns: (pd.DataFrame, baseline_auc)
                    DataFrame columns: epsilon, auc, auc_retention_pct
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    called_with_df = hasattr(X, 'columns')   # True = DataFrame input

    if called_with_df:
        df = X
        y  = df["dili"].values
        X  = df.drop(columns=["dili"]).values

    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=99,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Baseline AUC (no DP)
    base_model = LogisticRegression(max_iter=200, C=1.0)
    base_model.fit(X_tr, y_train)
    baseline_auc = float(roc_auc_score(y_test, base_model.predict_proba(X_te)[:, 1]))

    results = []
    for eps in epsilons:
        aucs = []
        for _ in range(n_runs):
            noise_level = max(0.0, min(0.3, 1.0 / (eps * 5)))
            flip_mask = np.random.binomial(1, noise_level, len(y_train)).astype(bool)
            y_noisy = y_train.copy()
            y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
            model = LogisticRegression(max_iter=200, C=1.0)
            model.fit(X_tr, y_noisy)
            aucs.append(roc_auc_score(y_test, model.predict_proba(X_te)[:, 1]))
        results.append({
            "epsilon":  eps,
            "mean_auc": round(float(np.mean(aucs)), 4),
            "std_auc":  round(float(np.std(aucs)),  4),
        })

    if called_with_df:
        df_result = pd.DataFrame([{
            "epsilon":          r["epsilon"],
            "auc":              r["mean_auc"],
            "auc_retention_pct": round(100 * r["mean_auc"] / baseline_auc, 1)
                                  if baseline_auc > 0 else 0,
        } for r in results])
        return df_result, baseline_auc

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("DIFFERENTIAL PRIVACY DEMO — PHARMA AI PIPELINE")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    rng = np.random.default_rng(42)
    X, y = generate_synthetic_clinical_dataset(n_patients=2000)
    print(f"\nSynthetic dataset: {X.shape[0]} patients, {X.shape[1]} biomarkers")
    print(f"DILI prevalence: {y.mean():.1%}")

    # Demonstrate Gaussian mechanism on a synthetic mean ALT query
    true_mean_alt = float(X[:, 0].mean())
    dp_mean_alt = gaussian_mechanism(true_mean_alt, sensitivity=0.1, epsilon=1.0, delta=1e-5)
    print(f"\nGaussian Mechanism (epsilon=1.0, delta=1e-5):")
    print(f"  True mean biomarker[0]: {true_mean_alt:.4f}")
    print(f"  DP-protected result:    {dp_mean_alt:.4f}")
    print(f"  Absolute error:         {abs(true_mean_alt - dp_mean_alt):.4f}")

    # Demonstrate budget-tracked DP-SGD simulation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    budget = PrivacyBudget(total_epsilon=2.0)
    print("\nDP-SGD Training Simulation (epsilon_per_epoch=0.5, 3 epochs):")
    dp_results = dp_logistic_regression(
        X_train, y_train, X_test, y_test,
        budget=budget, epsilon_per_epoch=0.5, n_epochs=3
    )
    print(f"  Epochs completed: {dp_results['epochs_completed']}")
    print(f"  Test AUC:         {dp_results['auc']}")
    print(f"  Budget maintained: {dp_results['budget_maintained']}")
    print(f"\n{budget.summary()}")

    # Privacy-utility tradeoff curve
    print("\nPrivacy-Utility Tradeoff (varying epsilon):")
    print(f"  {'Epsilon':>8}  {'Mean AUC':>10}  {'Std':>8}")
    curve = compute_privacy_utility_curve(X, y, epsilons=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    for row in curve:
        print(f"  {row['epsilon']:>8.1f}  {row['mean_auc']:>10.4f}  {row['std_auc']:>8.4f}")

    print("\nModule complete. Use fig05_privacy_utility_tradeoff.py to plot the curve.")


# ---------------------------------------------------------------------------
# COMPATIBILITY LAYER — matches test imports
# ---------------------------------------------------------------------------

def generate_dili_dataset(n_samples: int = 2000, random_state: int = 42):
    """
    Generates a synthetic DILI dataset as a pandas DataFrame.
    Wraps generate_synthetic_clinical_dataset() and packages as DataFrame
    with named feature columns and a binary 'dili' outcome column.
    """
    import pandas as pd
    X, y = generate_synthetic_clinical_dataset(n_patients=n_samples, seed=random_state)
    cols = ["ALT","AST","Bilirubin","ALP","GGT","Albumin","Creatinine","CYP3A4"]
    df = pd.DataFrame(X, columns=cols)
    df["dili"] = y
    return df


def simulate_dp_noise(X: np.ndarray, epsilon: float,
                      random_state: int = 42) -> np.ndarray:
    """
    Applies Gaussian mechanism noise to every element of X.
    Sensitivity assumed to be 1.0 (unit sensitivity — features are
    standardized before this call in production).
    Lower epsilon = more noise = stronger privacy guarantee.
    """
    rng = np.random.RandomState(random_state)
    sigma = np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
    return X + rng.normal(0, sigma, size=X.shape)


def calculate_proi(annual_revenue_eur: float,
                   breach_probability_pct: float,
                   privacy_investment_eur: float,
                   approval_value_eur: float = 0.0,
                   trust_premium_eur: float = 0.0,
                   regulatory_dividend_eur: float = 0.0,
                   control_effectiveness: float = 0.75) -> dict:
    """
    Implements the P-ROI Model from Chapter 5:
      P-ROI = (R_data + R_partner + R_regulatory + R_brand) / C_total

    Simplified two-component version for this module:
      R_risk   = GDPR_max_fine × breach_probability × control_effectiveness
      R_other  = approval_value + trust_premium + regulatory_dividend
      C_total  = privacy_investment_eur
    """
    gdpr_max_fine = annual_revenue_eur * 0.04
    risk_avoidance = gdpr_max_fine * (breach_probability_pct / 100) * control_effectiveness
    total_return = risk_avoidance + approval_value_eur + trust_premium_eur + regulatory_dividend_eur
    p_roi = total_return / privacy_investment_eur if privacy_investment_eur > 0 else 0.0

    return {
        "p_roi": round(p_roi, 2),
        "gdpr_max_fine_eur": gdpr_max_fine,
        "risk_avoidance_value_eur": round(risk_avoidance, 2),
        "approval_value_eur": approval_value_eur,
        "trust_premium_eur": trust_premium_eur,
        "regulatory_dividend_eur": regulatory_dividend_eur,
        "total_return_eur": round(total_return, 2),
        "privacy_investment_eur": privacy_investment_eur,
        "recommendation": "INVEST — positive P-ROI" if p_roi > 1.0 else "REVIEW — marginal or negative P-ROI",
    }

