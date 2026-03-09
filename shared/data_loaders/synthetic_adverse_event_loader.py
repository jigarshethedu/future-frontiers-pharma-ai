"""
Synthetic adverse event data loader — FAERS-like format.
Used in Chapter 12 (Pharmacovigilance AI).
ALL data is purely synthetic. No real patient data ever used.
"""
import numpy as np
import pandas as pd

DRUGS     = ["DrugA","DrugB","DrugC","DrugD","DrugE"]
AE_TYPES  = ["Hepatotoxicity","Cardiotoxicity","Nephrotoxicity",
             "Neurotoxicity","Dermatitis","Nausea","Fatigue"]
OUTCOMES  = ["Recovered","Not_Recovered","Death","Unknown"]
REPORTERS = ["Physician","Pharmacist","Consumer","Lawyer","Unknown"]

def generate_synthetic_faers(n_cases=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "case_id":            [f"AE{i:06d}" for i in range(n_cases)],
        "drug":               rng.choice(DRUGS, n_cases),
        "ae_type":            rng.choice(AE_TYPES, n_cases),
        "age":                rng.integers(18, 85, n_cases),
        "sex":                rng.choice(["M","F","U"], n_cases, p=[0.45,0.45,0.1]),
        "outcome":            rng.choice(OUTCOMES, n_cases, p=[0.55,0.25,0.08,0.12]),
        "reporter":           rng.choice(REPORTERS, n_cases),
        "serious":            rng.choice([0,1], n_cases, p=[0.6,0.4]),
        "days_to_onset":      rng.integers(1, 180, n_cases),
        "concomitant_drugs":  rng.integers(0, 5, n_cases),
    })

if __name__ == "__main__":
    df = generate_synthetic_faers(200)
    print(f"Generated {len(df)} synthetic adverse event cases")
    print(f"Seriousness rate: {df['serious'].mean():.1%}")
    print(df["ae_type"].value_counts())
