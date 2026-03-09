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
