#!/usr/bin/env bash
# Future Frontiers by Jigar Sheth — Repository Bootstrap
# Run once after creating the repo on github.com
set -e

echo "Creating chapter folders (Chapters 1–21)..."
for i in $(seq -w 1 21); do
  mkdir -p "chapter${i}/tests" "chapter${i}/figures" "chapter${i}/notebooks"
done

echo "Creating shared utilities and dataset folders..."
mkdir -p shared/pet_wrappers shared/data_loaders shared/evaluation shared/logging shared/templates shared/tests
mkdir -p datasets/synthetic_ehr datasets/synthetic_genomic datasets/synthetic_adverse_events
mkdir -p docs .github/workflows

echo "Writing requirements.txt..."
cat > requirements.txt << 'REQS'
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
google-dp>=2.1.0
opacus>=1.4.0
flwr>=1.8.0
tenseal>=0.3.14
sdv>=1.10.0
shap>=0.45.0
fairlearn>=0.10.0
aif360>=0.6.0
web3>=6.15.0
evidently>=0.4.0
codecarbon>=2.3.0
transformers>=4.38.0
spacy>=3.7.0
pytest>=8.0.0
pytest-cov>=5.0.0
tqdm>=4.66.0
pyyaml>=6.0.1
REQS

echo "Writing CI workflow..."
cat > .github/workflows/test.yml << 'CI'
name: Test All Modules
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "${{ matrix.python-version }}" }
    - run: pip install -r requirements.txt
    - run: pytest chapter*/tests/ shared/tests/ -v --tb=short
CI

echo "Writing shared utility modules..."
cat > shared/data_loaders/synthetic_ehr_loader.py << 'PY'
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
PY

cat > shared/data_loaders/synthetic_adverse_event_loader.py << 'PY'
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
PY

cat > shared/evaluation/privacy_metrics.py << 'PY'
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
PY

cat > shared/evaluation/fairness_metrics.py << 'PY'
"""
Unified fairness metrics. Used in: Ch04, Ch11, Ch15, Ch19.
"""
import numpy as np
from sklearn.metrics import confusion_matrix

def demographic_parity_difference(y_pred, sensitive):
    """Target < 0.05 for high-stakes clinical decision support."""
    groups = np.unique(sensitive)
    rates  = [y_pred[sensitive == g].mean() for g in groups]
    return abs(rates[0] - rates[1])

def equalized_odds_difference(y_true, y_pred, sensitive):
    groups = np.unique(sensitive)
    res    = {}
    for g in groups:
        m           = sensitive == g
        tn,fp,fn,tp = confusion_matrix(y_true[m], y_pred[m]).ravel()
        res[g]      = {"tpr": tp/(tp+fn+1e-10), "fpr": fp/(fp+tn+1e-10)}
    g0, g1 = groups
    return {
        "tpr_difference": abs(res[g0]["tpr"] - res[g1]["tpr"]),
        "fpr_difference": abs(res[g0]["fpr"] - res[g1]["fpr"]),
    }

def disparate_impact_ratio(y_pred, sensitive):
    """Four-fifths rule: target >= 0.80. Below 0.80 requires investigation."""
    groups = np.unique(sensitive)
    rates  = [y_pred[sensitive == g].mean() for g in groups]
    return min(rates) / (max(rates) + 1e-10)
PY

cat > shared/logging/audit_logger.py << 'PY'
"""
Structured audit logging for pharma AI regulatory inspection readiness.
Required for FDA SaMD, EU AI Act Art. 13, and GVP audit trails.
See: Ch10 (governance), Ch13 (security), Ch19 (monitoring).
"""
import json, datetime
from pathlib import Path

class AuditLogger:
    def __init__(self, log_dir="audit_logs", chapter=""):
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.chapter    = chapter
        self.session_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def log(self, event_type, details):
        entry = {
            "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "chapter":    self.chapter,
            "event_type": event_type,
            "details":    details,
        }
        with open(self.log_dir / f"audit_{self.session_id}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def log_model_training(self, model_name, n_samples,
                           privacy_protected, epsilon=None, metrics=None):
        return self.log("MODEL_TRAINING", {
            "model":             model_name,
            "n_samples":         n_samples,
            "privacy_protected": privacy_protected,
            "epsilon":           epsilon,
            "metrics":           metrics or {},
        })

    def log_data_access(self, accessor, data_type, purpose,
                        consent_verified, n_records=None):
        return self.log("DATA_ACCESS", {
            "accessor":         accessor,
            "data_type":        data_type,
            "purpose":          purpose,
            "consent_verified": consent_verified,
            "n_records":        n_records,
        })

    def log_model_decision(self, model_name, input_hash,
                           output, confidence, human_override=False):
        return self.log("MODEL_DECISION", {
            "model":          model_name,
            "input_hash":     input_hash,
            "output":         output,
            "confidence":     round(confidence, 4),
            "human_override": human_override,
        })
PY

echo "Writing README files for each chapter..."
for i in $(seq -w 1 21); do
cat > "chapter${i}/README.md" << README
# Chapter ${i} — Code Modules

All modules use synthetic data only. No real patient data is ever used.

## Quick Start
\`\`\`bash
pip install -r ../requirements.txt
python <module_name>.py      # runs built-in demo
pytest tests/ -v             # runs all tests
\`\`\`
README
done

echo "Writing project-level files..."
cat > CHANGELOG.md << 'CL'
# Changelog
## Format: [module path] — [what changed] — [YYYY-MM-DD]

## [Unreleased]
- Initial repository structure — Future Frontiers by Jigar Sheth
- Chapter folders 01–21 created
- Shared utilities: synthetic EHR loader, synthetic AE loader,
  privacy metrics, fairness metrics, audit logger
CL

cat > CONTRIBUTING.md << 'CON'
# Contributing

Report errors via GitHub Issues:
  - Label `errata`    → text/content errors
  - Label `code-bug`  → Python module errors

Include: chapter number, section or module name,
the incorrect content, and your suggested correction.

Pull requests welcome for bug fixes only.
Do not alter module logic — code must match book text exactly.
CON

cat > ERRATA.md << 'ERR'
# Errata
| Date | Chapter | Section/Module | Original | Corrected | Verified |
|------|---------|----------------|----------|-----------|----------|
ERR

cat > .gitignore << 'IGNORE'
__pycache__/
*.py[cod]
.eggs/
dist/
build/
*.egg-info/
.ipynb_checkpoints/
venv/
.venv/
*.log
audit_logs/
*.png
*.pdf
!docs/**
.DS_Store
Thumbs.db
IGNORE

cat > README.md << 'MAINREADME'
# Future Frontiers — Code Repository

**Book:** Future Frontiers: Harnessing AI, Safeguarding Privacy,
and Shaping Ethics in Pharma and Healthcare

**Author:** Jigar Sheth

## Structure
- `chapter01/` through `chapter21/` — one folder per chapter
- `shared/` — reusable utilities (data loaders, metrics, logging)
- `datasets/` — synthetic datasets (no real patient data ever)

## Quick Start
```bash
pip install -r requirements.txt
pytest chapter*/tests/ shared/tests/ -v
```

## License
Apache 2.0 — see LICENSE file.
MAINREADME

echo ""
echo "======================================================"
echo "  Repository structure created successfully."
echo "======================================================"
echo ""
echo "Next steps:"
echo "  git add ."
echo "  git commit -m 'Initial repository structure — Future Frontiers by Jigar Sheth'"
echo "  git push origin main"
echo ""
echo "Then on GitHub:"
echo "  Settings > Pages > Source: main branch /docs folder"
echo "  Settings > Features > Enable Discussions"
echo "  Issues > Labels > Create:"
echo "    'errata'    — color #CC0000 (red)"
echo "    'code-bug'  — color #FFA500 (yellow)"
echo "======================================================"