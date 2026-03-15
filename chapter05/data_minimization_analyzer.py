"""
data_minimization_analyzer.py
================================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Implements DM-SHAP [ORIGINAL TOOL] — a SHAP-based data minimization framework
    for pharma AI. DM-SHAP uses model-agnostic feature importance (SHAP values)
    to identify the minimum feature set required to maintain clinical model
    performance above a defined threshold, removing features that contribute
    negligible predictive value but carry disproportionate privacy risk.

    The output feeds directly into the Privacy Architecture Canvas (PAC)
    to justify the "data minimization" design decision with quantitative evidence.

Dependencies:
    - Python 3.10+
    - numpy >= 1.26.0
    - pandas >= 2.1.0
    - scikit-learn >= 1.4.0
    - shap >= 0.45.0

GitHub: chapter05/data_minimization_analyzer.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DMSHAPResult:
    """Result of one DM-SHAP feature elimination round."""
    feature: str
    shap_importance: float
    auc_without_feature: float
    auc_delta: float             # Baseline AUC - AUC without this feature
    privacy_risk_tier: str       # "Direct ID", "Quasi-ID", "Benign"
    recommendation: str          # "Retain", "Remove", "Review"


def run_dm_shap_analysis(X: pd.DataFrame, y: np.ndarray,
                          feature_privacy_tiers: dict[str, str],
                          auc_tolerance: float = 0.02) -> list[DMSHAPResult]:
    """
    DM-SHAP: Runs SHAP-based feature importance analysis and cross-references
    privacy risk tiers. Features with SHAP importance below the tolerance
    threshold AND classified as quasi-identifiers are flagged for removal.

    auc_tolerance: maximum acceptable AUC degradation from removing a feature.
    Features whose removal costs less than this threshold are candidates for removal.
    """
    try:
        import shap
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score

        feature_names = X.columns.tolist()
        X_arr = X.values.astype(float)

        # Train baseline model
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_arr, y)
        baseline_auc = float(np.mean(cross_val_score(model, X_arr, y, cv=5, scoring="roc_auc")))

        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_arr)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        results = []
        for i, feature in enumerate(feature_names):
            # AUC without this feature
            X_reduced = np.delete(X_arr, i, axis=1)
            m_reduced = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            auc_reduced = float(np.mean(
                cross_val_score(m_reduced, X_reduced, y, cv=5, scoring="roc_auc")
            ))
            delta = round(baseline_auc - auc_reduced, 4)
            shap_imp = round(float(mean_abs_shap[i]), 4)
            tier = feature_privacy_tiers.get(feature, "Benign")

            if delta < auc_tolerance and tier in ("Quasi-ID", "Direct ID"):
                rec = "Remove"
            elif delta < auc_tolerance / 2 and tier == "Benign":
                rec = "Review"
            else:
                rec = "Retain"

            results.append(DMSHAPResult(
                feature=feature,
                shap_importance=shap_imp,
                auc_without_feature=round(auc_reduced, 4),
                auc_delta=delta,
                privacy_risk_tier=tier,
                recommendation=rec,
            ))

        return sorted(results, key=lambda r: r.shap_importance)

    except ImportError:
        # Fallback if shap is not installed: use sklearn feature_importances_
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        feature_names = X.columns.tolist()
        X_arr = X.values.astype(float)
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_arr, y)
        baseline_auc = float(np.mean(cross_val_score(model, X_arr, y, cv=5, scoring="roc_auc")))
        importances = model.feature_importances_

        results = []
        for i, feature in enumerate(feature_names):
            X_reduced = np.delete(X_arr, i, axis=1)
            m2 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            auc_r = float(np.mean(cross_val_score(m2, X_reduced, y, cv=5, scoring="roc_auc")))
            delta = round(baseline_auc - auc_r, 4)
            tier = feature_privacy_tiers.get(feature, "Benign")
            if delta < auc_tolerance and tier in ("Quasi-ID", "Direct ID"):
                rec = "Remove"
            elif delta < auc_tolerance / 2 and tier == "Benign":
                rec = "Review"
            else:
                rec = "Retain"
            results.append(DMSHAPResult(
                feature=feature,
                shap_importance=round(float(importances[i]), 4),
                auc_without_feature=round(auc_r, 4),
                auc_delta=delta,
                privacy_risk_tier=tier,
                recommendation=rec,
            ))
        return sorted(results, key=lambda r: r.shap_importance)


if __name__ == "__main__":
    from synthetic_data_generator import generate_synthetic_ehr

    print("=" * 70)
    print("DM-SHAP: DATA MINIMIZATION ANALYZER")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    df = generate_synthetic_ehr(n_patients=800, seed=42)
    feature_cols = ["age", "sex", "bmi", "alt_u_per_l", "ast_u_per_l",
                    "creatinine_mg_dl", "hba1c_pct", "diabetes", "hypertension", "ckd"]
    X = df[feature_cols]
    y = df["readmission_30d"].values

    # Privacy tier classification (would come from data catalog in production)
    tiers = {
        "age": "Quasi-ID",
        "sex": "Quasi-ID",
        "bmi": "Quasi-ID",
        "alt_u_per_l": "Benign",
        "ast_u_per_l": "Benign",
        "creatinine_mg_dl": "Benign",
        "hba1c_pct": "Quasi-ID",
        "diabetes": "Benign",
        "hypertension": "Benign",
        "ckd": "Benign",
    }

    print("\nRunning DM-SHAP analysis (auc_tolerance=0.02)...\n")
    results = run_dm_shap_analysis(X, y, tiers, auc_tolerance=0.02)

    print(f"{'Feature':<22} {'SHAP Imp':>9} {'AUC Δ':>8} {'Privacy Tier':<14} {'Action'}")
    print("-" * 70)
    for r in results:
        flag = "⚠️ " if r.recommendation == "Remove" else "   "
        print(f"  {r.feature:<20} {r.shap_importance:>9.4f} {r.auc_delta:>8.4f} "
              f"{r.privacy_risk_tier:<14} {flag}{r.recommendation}")

    removable = [r for r in results if r.recommendation == "Remove"]
    print(f"\nDM-SHAP recommends removing {len(removable)} feature(s):")
    for r in removable:
        print(f"  - {r.feature} (SHAP={r.shap_importance:.4f}, Δ AUC={r.auc_delta:.4f}, Tier={r.privacy_risk_tier})")
    print("\nModule complete.")


def generate_full_feature_dili_dataset(n: int = 500, random_state: int = 42):
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(random_state)
    cols = (
        ["MolWeight","LogP","HBD","HBA","RotBonds","TPSA","AromaticRings","HeavyAtomCount"] +
        ["MTT_IC50","LDH_release","ATP_depletion","Caspase3_activity","ROS_production","MitoMembranePotential"] +
        ["ALT","AST","ALP","TotalBilirubin","GGT"] +
        ["Cmax","AUC_0inf","HalfLife","CL_F"] +
        ["Age","Sex","BMI","eGFR","AlbuminLevel","ConcomitantMeds"] +
        ["CYP1A2","CYP3A4","UGT1A1","ABCB1","ABCG2","SLC22A1","HMOX1","NRF2"] +
        ["Dose_mg","DurationDays","RouteOral","Polypharmacy_score"] +
        ["random_1","random_2","random_3","random_4","random_5","random_6"]
    )
    X = rng.randn(n, 47)
    y = rng.binomial(1, 0.10, n)
    X[y == 1, :10] += rng.uniform(0.4, 0.9, size=(y.sum(), 10))
    df = pd.DataFrame(X, columns=cols)
    df["dili"] = y
    return df
