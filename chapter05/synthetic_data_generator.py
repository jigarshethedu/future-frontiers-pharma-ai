"""
synthetic_data_generator.py
=============================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Generates synthetic clinical, genomic, and adverse-event datasets for use
    across Chapter 5 module demonstrations and for the TSTR Evaluation Framework
    [ORIGINAL FRAMEWORK] — Train on Synthetic, Test on Real — which assesses
    synthetic data on three dimensions: fidelity, utility, and membership
    inference attack (MIA) resistance.

    Uses statistical sampling approaches (no external GAN libraries required
    for the stub — production deployment uses CTGAN/DP-CTGAN, covered in Ch8).

Dependencies:
    - Python 3.10+
    - numpy >= 1.26.0
    - pandas >= 2.1.0
    - scikit-learn >= 1.4.0

GitHub: chapter05/synthetic_data_generator.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# SYNTHETIC EHR GENERATOR
# ---------------------------------------------------------------------------

def generate_synthetic_ehr(n_patients: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic EHR cohort for pharma AI demonstrations.
    All values are statistically plausible but not derived from any real patient.
    Covers demographics, biomarkers, comorbidities, and a binary outcome.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 85, size=n_patients)
    # Sex coded as 0/1 — no real patients referenced
    sex = rng.binomial(1, 0.48, size=n_patients)
    bmi = rng.normal(28.5, 5.2, size=n_patients).clip(16.0, 55.0)

    # Synthetic biomarkers
    alt = rng.lognormal(mean=3.6, sigma=0.4, size=n_patients).clip(5.0, 200.0)
    ast = alt * rng.uniform(0.7, 1.3, size=n_patients)
    creatinine = rng.lognormal(mean=0.1, sigma=0.35, size=n_patients).clip(0.4, 6.0)
    hba1c = rng.normal(6.5, 1.2, size=n_patients).clip(4.5, 14.0)

    # Comorbidities (binary flags)
    diabetes = rng.binomial(1, 0.18, size=n_patients)
    hypertension = rng.binomial(1, 0.30, size=n_patients)
    ckd = (creatinine > 1.8).astype(int)

    # Binary outcome: 30-day readmission (purely synthetic)
    log_odds = (
        -3.0
        + 0.03 * (age - 50)
        + 0.8 * diabetes
        + 0.6 * ckd
        + 0.4 * hypertension
        + 0.01 * (alt - 40)
        + rng.normal(0, 0.5, size=n_patients)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    readmission_30d = rng.binomial(1, prob)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi.round(1),
        "alt_u_per_l": alt.round(1),
        "ast_u_per_l": ast.round(1),
        "creatinine_mg_dl": creatinine.round(2),
        "hba1c_pct": hba1c.round(1),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "ckd": ckd,
        "readmission_30d": readmission_30d,
    })


def generate_synthetic_adverse_events(n_reports: int = 500, seed: int = 7) -> pd.DataFrame:
    """
    Generates synthetic adverse event (AE) reports for pharmacovigilance demos.
    Mimics the structure of FAERS-style data but contains no real cases.
    For full PV AI pipeline, see chapter12/pv_signal_detector.py.
    """
    rng = np.random.default_rng(seed)

    drugs = ["DrugAlpha (synthetic)", "DrugBeta (synthetic)", "DrugGamma (synthetic)",
             "DrugDelta (synthetic)", "DrugEpsilon (synthetic)"]
    ae_terms = ["hepatotoxicity", "rash", "nausea", "fatigue", "elevated ALT",
                "elevated AST", "pruritus", "jaundice", "abdominal pain", "dizziness"]
    outcomes = ["recovered", "recovering", "not_recovered", "fatal", "unknown"]

    return pd.DataFrame({
        "report_id": [f"SYN-{i:06d}" for i in range(n_reports)],
        "drug_name": rng.choice(drugs, size=n_reports),
        "ae_term": rng.choice(ae_terms, size=n_reports),
        "patient_age_group": rng.choice(["<18", "18-44", "45-64", "65-74", "75+"],
                                          size=n_reports),
        "outcome": rng.choice(outcomes, size=n_reports,
                               p=[0.50, 0.25, 0.10, 0.05, 0.10]),
        "days_to_onset": rng.integers(1, 180, size=n_reports),
        "reporter_type": rng.choice(["physician", "pharmacist", "consumer", "lawyer"],
                                     size=n_reports, p=[0.45, 0.30, 0.20, 0.05]),
    })


# ---------------------------------------------------------------------------
# TSTR EVALUATION FRAMEWORK
# ---------------------------------------------------------------------------

@dataclass
class TSTRResult:
    """
    Result of the Train on Synthetic, Test on Real (TSTR) evaluation.
    Three dimensions:
    - Fidelity: how closely the synthetic distribution matches the real one
    - Utility: whether a model trained on synthetic data performs on real data
    - MIA Resistance: whether the synthetic data leaks membership information
    """
    fidelity_score: float        # 0–100; marginal distribution similarity
    utility_auc_synthetic: float  # AUC: train synthetic, test synthetic
    utility_auc_tstr: float       # AUC: train synthetic, test real (key metric)
    utility_auc_real: float       # AUC: train real, test real (upper bound)
    utility_score: float          # 0–100; TSTR AUC / Real AUC
    mia_advantage: float          # Membership inference advantage (0 = perfect)
    mia_resistance_score: float   # 100 - (mia_advantage * 200) capped at 0
    overall_tstr_score: float     # Weighted composite


def evaluate_tstr(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                  target_col: str, feature_cols: list[str]) -> TSTRResult:
    """
    Evaluates synthetic data quality on fidelity, utility, and MIA resistance.
    This implements the TSTR Evaluation Framework introduced in Chapter 5.
    """

    # --- FIDELITY: mean absolute difference in marginals (0–100) ---
    fidelity_errors = []
    for col in feature_cols:
        r_mean = real_df[col].mean()
        s_mean = synthetic_df[col].mean()
        r_std = real_df[col].std() + 1e-9
        fidelity_errors.append(abs(r_mean - s_mean) / r_std)
    fidelity_score = max(0.0, 100 - (np.mean(fidelity_errors) * 50))

    # --- UTILITY ---
    X_real = real_df[feature_cols].values
    y_real = real_df[target_col].values
    X_syn = synthetic_df[feature_cols].values
    y_syn = synthetic_df[target_col].values

    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(X_real, y_real, test_size=0.3, random_state=0)

    scaler = StandardScaler()

    def train_and_eval(X_train, y_train, X_test, y_test) -> float:
        sc = StandardScaler().fit(X_train)
        model = LogisticRegression(max_iter=300, C=1.0)
        model.fit(sc.transform(X_train), y_train)
        if y_test.sum() == 0:
            return 0.5
        return roc_auc_score(y_test, model.predict_proba(sc.transform(X_test))[:, 1])

    auc_real = train_and_eval(X_r_tr, y_r_tr, X_r_te, y_r_te)
    auc_syn_only = train_and_eval(X_syn, y_syn, X_syn[:100], y_syn[:100])
    auc_tstr = train_and_eval(X_syn, y_syn, X_r_te, y_r_te)

    utility_score = round(min(100.0, 100 * auc_tstr / max(auc_real, 0.51)), 1)

    # --- MIA RESISTANCE: shadow model membership inference ---
    # A simple MIA: can we distinguish real vs. synthetic training members?
    # Advantage = P(correct classification) - 0.5 (0 = random = no leak)
    labels_real = np.ones(len(X_r_tr))
    labels_syn = np.zeros(len(X_syn))
    X_mia = np.vstack([X_r_tr, X_syn])
    y_mia = np.concatenate([labels_real, labels_syn])
    mia_model = LogisticRegression(max_iter=200)
    sc_mia = StandardScaler().fit(X_mia)
    mia_model.fit(sc_mia.transform(X_mia), y_mia)
    mia_acc = (mia_model.predict(sc_mia.transform(X_mia)) == y_mia).mean()
    mia_advantage = round(max(0.0, mia_acc - 0.5), 4)
    mia_resistance_score = round(max(0.0, 100 - mia_advantage * 200), 1)

    # Composite: 30% fidelity, 50% utility, 20% MIA resistance
    overall = round(
        0.30 * fidelity_score + 0.50 * utility_score + 0.20 * mia_resistance_score, 1
    )

    return TSTRResult(
        fidelity_score=round(fidelity_score, 1),
        utility_auc_synthetic=round(auc_syn_only, 4),
        utility_auc_tstr=round(auc_tstr, 4),
        utility_auc_real=round(auc_real, 4),
        utility_score=utility_score,
        mia_advantage=mia_advantage,
        mia_resistance_score=mia_resistance_score,
        overall_tstr_score=overall,
    )


# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC DATA GENERATOR + TSTR EVALUATION FRAMEWORK")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    real_df = generate_synthetic_ehr(n_patients=1000, seed=42)
    # Generate "synthetic" data from the same process with different seed
    # (In production, this would come from CTGAN or a diffusion model — see Ch8)
    synthetic_df = generate_synthetic_ehr(n_patients=1000, seed=999)

    print(f"\nReal dataset:      {len(real_df)} patients, {len(real_df.columns)} columns")
    print(f"Synthetic dataset: {len(synthetic_df)} patients")
    print(f"Outcome prevalence — real: {real_df['readmission_30d'].mean():.1%}, "
          f"synthetic: {synthetic_df['readmission_30d'].mean():.1%}")

    feature_cols = ["age", "bmi", "alt_u_per_l", "ast_u_per_l",
                    "creatinine_mg_dl", "hba1c_pct", "diabetes", "hypertension", "ckd"]

    print("\nRunning TSTR Evaluation Framework...")
    result = evaluate_tstr(real_df, synthetic_df, "readmission_30d", feature_cols)

    print(f"\n{'TSTR EVALUATION RESULTS':^50}")
    print(f"  Fidelity Score:          {result.fidelity_score:>6.1f}/100")
    print(f"  Utility Score:           {result.utility_score:>6.1f}/100")
    print(f"    AUC (Syn→Syn):         {result.utility_auc_synthetic:>6.4f}")
    print(f"    AUC (Syn→Real):        {result.utility_auc_tstr:>6.4f}  ← TSTR key metric")
    print(f"    AUC (Real→Real):       {result.utility_auc_real:>6.4f}  ← Upper bound")
    print(f"  MIA Resistance:          {result.mia_resistance_score:>6.1f}/100")
    print(f"    MIA Advantage:         {result.mia_advantage:>6.4f}  (0 = no leak)")
    print(f"\n  OVERALL TSTR SCORE:      {result.overall_tstr_score:>6.1f}/100")

    grade = ("A" if result.overall_tstr_score >= 80
             else "B" if result.overall_tstr_score >= 70
             else "C" if result.overall_tstr_score >= 60 else "F")
    print(f"  Synthetic Data Grade:    {grade}")

    print("\nAdverse event dataset preview:")
    ae_df = generate_synthetic_adverse_events(n_reports=500)
    print(ae_df.head(3).to_string(index=False))

    print("\nModule complete. For production synthetic data, see chapter08/gan_synthetic_ehr.py")
