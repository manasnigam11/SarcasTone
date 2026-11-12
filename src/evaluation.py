"""
Evaluation, Paired Bootstrap Confidence Intervals, and Hypothesis Testing Module (Phase 10).

Implements:
  - Primary metric: Macro-F1
  - Secondary metrics: Accuracy, Macro Precision, Macro Recall, Weighted F1, Confusion Matrix
  - Fast Vectorized Standard and Group-Aware Paired Bootstrap Confidence Intervals (95% CI)
  - Formal Decision Rule: A difference is statistically supported only when the 95% paired
    bootstrap CI of (Macro-F1_A - Macro-F1_B) excludes zero.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute all mandated classification metrics on a set of predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _vectorized_macro_f1_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
) -> np.ndarray:
    """Compute exact Macro-F1 across B bootstrap resamples from confusion matrix counts."""
    denom_pos = 2.0 * tp + fp + fn
    f1_pos = np.where(denom_pos > 0, (2.0 * tp) / np.maximum(denom_pos, 1e-12), 0.0)
    denom_neg = 2.0 * tn + fn + fp
    f1_neg = np.where(denom_neg > 0, (2.0 * tn) / np.maximum(denom_neg, 1e-12), 0.0)
    return 0.5 * (f1_pos + f1_neg)


def _bootstrap_macro_f1_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_resamples: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """
    Fast vectorized bootstrap resampling of Macro-F1.
    Because each unit i contributes (tp_i, fp_i, fn_i, tn_i) additively to the resampled confusion
    matrix, group-level or instance-level resampling reduces to a single matrix-vector product!
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp_i = ((y_true == 1) & (y_pred == 1)).astype(np.float64)
    fp_i = ((y_true == 0) & (y_pred == 1)).astype(np.float64)
    fn_i = ((y_true == 1) & (y_pred == 0)).astype(np.float64)
    tn_i = ((y_true == 0) & (y_pred == 0)).astype(np.float64)

    if groups is None:
        n = len(y_true)
        # Multinomial weights of shape (n_resamples, n)
        idx = rng.integers(0, n, size=(n_resamples, n))
        # Aggregate per resample
        tp_b = tp_i[idx].sum(axis=1)
        fp_b = fp_i[idx].sum(axis=1)
        fn_b = fn_i[idx].sum(axis=1)
        tn_b = tn_i[idx].sum(axis=1)
    else:
        groups = np.asarray(groups)
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        n_g = len(unique_groups)
        # Sum tp, fp, fn, tn per group first -> shape (n_g,)
        tp_g = np.bincount(inverse, weights=tp_i, minlength=n_g)
        fp_g = np.bincount(inverse, weights=fp_i, minlength=n_g)
        fn_g = np.bincount(inverse, weights=fn_i, minlength=n_g)
        tn_g = np.bincount(inverse, weights=tn_i, minlength=n_g)

        sampled_g = rng.integers(0, n_g, size=(n_resamples, n_g))
        tp_b = tp_g[sampled_g].sum(axis=1)
        fp_b = fp_g[sampled_g].sum(axis=1)
        fn_b = fn_g[sampled_g].sum(axis=1)
        tn_b = tn_g[sampled_g].sum(axis=1)

    return _vectorized_macro_f1_from_counts(tp_b, fp_b, fn_b, tn_b)


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute 95% bootstrap confidence interval for Macro-F1.
    If `groups` is provided (e.g. `scene_group_id`), performs cluster/group-aware bootstrap
    resampling at the group level so correlated utterances from the same scene are resampled together.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    point_est = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    scores = _bootstrap_macro_f1_samples(
        y_true, y_pred, groups=groups, n_resamples=n_resamples, seed=seed
    )

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(scores, 100.0 * alpha))
    ci_high = float(np.percentile(scores, 100.0 * (1.0 - alpha)))
    return {
        "point": point_est,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def paired_bootstrap_macro_f1_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap comparison between Model A and Model B on identical held-out instances.
    Uses identical bootstrap resamples (same RNG seed) for both models to preserve pairing.
    Tests Δ = Macro-F1(Model A) - Macro-F1(Model B).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    f1_a = float(f1_score(y_true, y_pred_a, average="macro", zero_division=0))
    f1_b = float(f1_score(y_true, y_pred_b, average="macro", zero_division=0))
    obs_diff = f1_a - f1_b

    scores_a = _bootstrap_macro_f1_samples(
        y_true, y_pred_a, groups=groups, n_resamples=n_resamples, seed=seed
    )
    scores_b = _bootstrap_macro_f1_samples(
        y_true, y_pred_b, groups=groups, n_resamples=n_resamples, seed=seed
    )
    diffs = scores_a - scores_b

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(diffs, 100.0 * alpha))
    ci_high = float(np.percentile(diffs, 100.0 * (1.0 - alpha)))
    p_le_zero = float(np.mean(diffs <= 0.0))
    p_ge_zero = float(np.mean(diffs >= 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_le_zero, p_ge_zero)))

    excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    if ci_low > 0.0:
        verdict = "Statistically supported improvement (A > B; 95% CI > 0)."
    elif ci_high < 0.0:
        verdict = "Statistically supported decrease (A < B; 95% CI < 0)."
    else:
        verdict = "No reliable difference was observed."

    return {
        "f1_a": f1_a,
        "f1_b": f1_b,
        "diff_point": obs_diff,
        "diff_mean": float(np.mean(diffs)),
        "diff_std": float(np.std(diffs)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_two_sided": p_two_sided,
        "excludes_zero": bool(excludes_zero),
        "verdict": verdict,
    }
