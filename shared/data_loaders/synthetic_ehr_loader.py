"""
Synthetic EHR data loader — MIMIC-III-like format.
Used across all chapters. ALL data is purely synthetic.
No real patient data is ever used or referenced.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_ehr(n_patients=1000, seed=42):
    rng = np.random.default_rng(seed)
    age        = rng.integers(18, 90, n_patients)
    sex        = rng.choice([0, 1], n_patients)
    ethnicity  = rng.choice([0,1,2,3,4], n_patients, p=[0.60,0.18,0.13,0.06,0.03])
    ldl        = rng.normal(130, 35, n_patients).clip(40, 300)
    sbp        = rng.normal(130, 20, n_patients).clip(80, 220)
    hba1c      = rng.normal(6.5, 1.5, n_patients).clip(4.0, 14.0)
    creatinine = rng.lognormal(0.1, 0.3, n_patients).clip(0.5, 10.0)
    log_odds   = (-3.0 + 0.02*(age-50) + 0.01*(ldl-130) + 0.015*(sbp-130)
                  + 0.3*(hba1c-6.5) + rng.normal(0, 0.5, n_patients))
    prob       = 1 / (1 + np.exp(-log_odds))
    dili_risk  = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "patient_id":        [f"SYN{i:05d}" for i in range(n_patients)],
        "age":               age,
        "sex":               sex,
        "ethnicity":         ethnicity,
        "ldl_mg_dl":         ldl.round(1),
        "systolic_bp_mmhg":  sbp.round(1),
        "hba1c_pct":         hba1c.round(2),
        "creatinine_mg_dl":  creatinine.round(3),
        "dili_risk":         dili_risk,
    })

def load_train_test(n_patients=1000, test_size=0.2, seed=42):
    df = generate_synthetic_ehr(n_patients, seed)
    X  = df[["age","sex","ethnicity","ldl_mg_dl",
              "systolic_bp_mmhg","hba1c_pct","creatinine_mg_dl"]].values
    y  = df["dili_risk"].values
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)

if __name__ == "__main__":
    df = generate_synthetic_ehr(500)
    print(f"Generated {len(df)} synthetic patients")
    print(f"DILI risk prevalence: {df['dili_risk'].mean():.1%}")
    print(df.describe().round(2))
