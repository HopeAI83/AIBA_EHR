
"""Evaluation utilities and statistical tests."""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2


def classification_metrics_from_confusion(tn:int, fp:int, fn:int, tp:int) -> dict:
    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'accuracy':acc, 'precision':precision, 'recall':recall, 'f1':f1}


def mcnemar_test(b:int, c:int) -> tuple[float, float]:
    """McNemar chi-square test with continuity correction."""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b-c)-1)**2/(b+c)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)
