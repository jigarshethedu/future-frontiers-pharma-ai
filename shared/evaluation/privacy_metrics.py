"""
Privacy evaluation metrics for PET validation.
Used in: Ch05 (TSTR), Ch08, Ch12, Ch19.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def membership_inference_attack_score(real_data, synthetic_data,
                                      n_shadow=200, seed=42):
    """MIA AUC: 0.5 = perfect privacy, 1.0 = no protection."""
    rng = np.random.default_rng(seed)
    n   = min(len(real_data), len(synthetic_data), n_shadow)
    real_s = real_data[rng.choice(len(real_data), n, replace=False)]
    syn_s  = synthetic_data[rng.choice(len(synthetic_data), n, replace=False)]
    X  = np.vstack([real_s, syn_s])
    y  = np.hstack([np.ones(n), np.zeros(n)])
    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())

def k_anonymity_score(data, quasi_identifier_indices):
    """Minimum group size. Target >= 5 standard pharma, >= 11 genomics."""
    import pandas as pd
    df      = pd.DataFrame(data)
    qi_cols = [df.columns[i] for i in quasi_identifier_indices]
    return int(df.groupby(qi_cols).size().min())

def reidentification_risk_score(mia_auc, k_anon, max_k=50):
    """Composite re-ID risk 0–100. Target < 30 for pharma AI."""
    mia_risk = max(0, (mia_auc - 0.5) / 0.5) * 60
    k_risk   = max(0, 1 - min(k_anon, max_k) / max_k) * 40
    return round(mia_risk + k_risk, 1)
