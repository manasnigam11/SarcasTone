"""
Model Training, Cross-Validation, Multimodal Fusion, Control Experiments, and Registry Module.

Implements all models required by Phases 4–8:
  1. `tfidf_lr` / `tfidf_svm`: TF-IDF (unigrams+bigrams, fit strictly inside each fold) + LogisticRegression / LinearSVC
  2. `roberta_target_lr`: Frozen `roberta-base` target-utterance embeddings + inside-fold StandardScaler + PCA/LR
  3. `roberta_context_lr`: Context-aware `roberta-base` (`context </s> target`) + inside-fold StandardScaler + LR
  4. `prosody_explicit_lr` / `prosody_explicit_svm`: 41 Interpretable Prosodic Features (4 groups) + inside-fold MedianImputer + StandardScaler + LR / RBF-SVM
  5. `egemaps_lr`: 88-d `openSMILE` eGeMAPSv02 functionals + inside-fold MedianImputer + StandardScaler + LR
  6. `wav2vec2_lr`: Frozen `facebook/wav2vec2-base` speech representations + inside-fold StandardScaler + LR
  7. `fusion_roberta_prosody_early`: Early concatenation of scaled RoBERTa + Explicit Prosody + LR
  8. `fusion_roberta_prosody_late`: Late probability-level stacking/gating fit strictly on inner training CV + LR
  9. `fusion_roberta_wav2vec2_early` / `fusion_roberta_wav2vec2_late`: Multimodal fusion of RoBERTa + wav2vec 2.0
  10. `fusion_trimodal_text_prosody_w2v2`: Full Text + Explicit Prosody + wav2vec 2.0 fusion
  11. Controls & Ablations (Phase 8):
      - `control_speaker_only`: Diagnostic Speaker-ID-Only model (quantifies speaker base-rate shortcut)
      - `control_shuffled_prosody`: Multimodal RoBERTa + randomly permuted Prosody
      - `control_shuffled_wav2vec2`: Multimodal RoBERTa + randomly permuted wav2vec 2.0
      - Prosodic Cue Ablations (`ablation_no_f0`, `ablation_no_energy`, `ablation_no_timing`, `ablation_no_voice_quality`, plus single-group additions `prosody_only_f0`, `prosody_only_energy`, `prosody_only_timing`, `prosody_only_voice_quality`)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

from src.evaluation import bootstrap_macro_f1_ci, compute_classification_metrics
from src.representations import clean_transcript

logger = logging.getLogger(__name__)


def _select_best_c_inner_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: List[float],
    seed: int = 42,
) -> float:
    """Tune L2 regularization strength `C` strictly inside the training fold via 3-fold inner CV."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c = c_grid[0]
    best_score = -1.0
    for c_val in c_grid:
        scores = []
        for tr_i, val_i in skf.split(X_train, y_train):
            clf = LogisticRegression(
                C=c_val,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(X_train[tr_i], y_train[tr_i])
            preds = clf.predict(X_train[val_i])
            tp = np.sum((preds == 1) & (y_train[val_i] == 1))
            fp = np.sum((preds == 1) & (y_train[val_i] == 0))
            fn = np.sum((preds == 0) & (y_train[val_i] == 1))
            tn = np.sum((preds == 0) & (y_train[val_i] == 0))
            f1_1 = (2 * tp) / max(2 * tp + fp + fn, 1)
            f1_0 = (2 * tn) / max(2 * tn + fn + fp, 1)
            scores.append(0.5 * (f1_1 + f1_0))
        mean_sc = float(np.mean(scores))
        if mean_sc > best_score:
            best_score = mean_sc
            best_c = c_val
    return best_c


def fit_predict_fold_tabular(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str = "lr",
    c_grid: Optional[List[float]] = None,
    use_pca_n: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leakage-safe tabular fold trainer:
      1. Fits `SimpleImputer(strategy="median")` on `X_train` ONLY
      2. Fits `RobustScaler` + `StandardScaler` on `X_train` ONLY
      3. Optionally fits `PCA` on `X_train` ONLY
      4. Tunes `C` via inner CV on `X_train` ONLY and returns `(test_preds, test_probs)`
    """
    if c_grid is None:
        c_grid = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    X_tr = imputer.fit_transform(X_train)
    X_te = imputer.transform(X_test)

    # Clip extreme outliers (e.g. jitter/shimmer spikes) strictly using training 1st/99th percentiles
    low_q = np.percentile(X_tr, 1.0, axis=0)
    high_q = np.percentile(X_tr, 99.0, axis=0)
    X_tr = np.clip(X_tr, low_q, high_q)
    X_te = np.clip(X_te, low_q, high_q)

    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if use_pca_n is not None and X_tr.shape[1] > use_pca_n:
        n_comp = min(use_pca_n, X_tr.shape[0] - 1, X_tr.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    if model_type == "lr":
        best_c = _select_best_c_inner_cv(X_tr, y_train, c_grid=c_grid, seed=seed)
        clf = LogisticRegression(
            C=best_c,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    elif model_type == "svm_rbf":
        best_c = _select_best_c_inner_cv(X_tr, y_train, c_grid=[0.1, 0.5, 1.0, 2.0], seed=seed)
        clf = SVC(C=best_c, kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def fit_predict_fold_late_fusion(
    X_text_train: np.ndarray,
    X_audio_train: np.ndarray,
    y_train: np.ndarray,
    X_text_test: np.ndarray,
    X_audio_test: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leakage-safe Late / Stacking Multimodal Fusion:
      1. Generates out-of-fold inner CV probability estimates for the text branch and audio branch
         strictly within `X_train`.
      2. Fits a meta-logistic combiner on `(logit_text, logit_audio, interaction)` inside `X_train`.
      3. Fits the full text branch and audio branch on `X_train` and predicts on `X_test`.
    """
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    oof_p_text = np.zeros(len(y_train), dtype=np.float64)
    oof_p_audio = np.zeros(len(y_train), dtype=np.float64)

    for tr_i, val_i in skf.split(X_text_train, y_train):
        _, p_t = fit_predict_fold_tabular(
            X_text_train[tr_i],
            y_train[tr_i],
            X_text_train[val_i],
            model_type="lr",
            c_grid=[0.005, 0.01, 0.05, 0.1, 0.5],
            seed=seed,
        )
        _, p_a = fit_predict_fold_tabular(
            X_audio_train[tr_i],
            y_train[tr_i],
            X_audio_train[val_i],
            model_type="lr",
            c_grid=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            seed=seed,
        )
        oof_p_text[val_i] = p_t
        oof_p_audio[val_i] = p_a

    # Fit full base branches on all training data
    _, test_p_text = fit_predict_fold_tabular(
        X_text_train,
        y_train,
        X_text_test,
        model_type="lr",
        c_grid=[0.005, 0.01, 0.05, 0.1, 0.5],
        seed=seed,
    )
    _, test_p_audio = fit_predict_fold_tabular(
        X_audio_train,
        y_train,
        X_audio_test,
        model_type="lr",
        c_grid=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        seed=seed,
    )

    meta_X_tr = np.column_stack([oof_p_text, oof_p_audio])
    meta_X_te = np.column_stack([test_p_text, test_p_audio])

    meta_clf = LogisticRegression(C=0.5, class_weight="balanced", random_state=seed)
    meta_clf.fit(meta_X_tr, y_train)
    probs = meta_clf.predict_proba(meta_X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def fit_predict_fold_early_concat(
    X_text_train: np.ndarray,
    X_audio_train: np.ndarray,
    y_train: np.ndarray,
    X_text_test: np.ndarray,
    X_audio_test: np.ndarray,
    text_pca_dim: int = 64,
    audio_pca_dim: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leakage-safe Early Fusion (Feature Concatenation):
      1. Independently imputes and scales `X_text` and `X_audio` inside the training fold.
      2. Compresses high-dimensional 768-d embeddings via training-only PCA (`text_pca_dim`) so that
         768-d text features do not numerically overwhelm 41-d explicit prosodic features, and also
         concatenates the full block-normalized vectors.
      3. Fits L2-regularized LogisticRegression tuned via inner CV.
    """
    imp_t = SimpleImputer(strategy="median", keep_empty_features=True)
    scl_t = StandardScaler()
    Xt_tr = scl_t.fit_transform(imp_t.fit_transform(X_text_train))
    Xt_te = scl_t.transform(imp_t.transform(X_text_test))

    imp_a = SimpleImputer(strategy="median", keep_empty_features=True)
    scl_a = StandardScaler()
    Xa_tr = imp_a.fit_transform(X_audio_train)
    Xa_te = imp_a.transform(X_audio_test)
    low_q = np.percentile(Xa_tr, 1.0, axis=0)
    high_q = np.percentile(Xa_tr, 99.0, axis=0)
    Xa_tr = scl_a.fit_transform(np.clip(Xa_tr, low_q, high_q))
    Xa_te = scl_a.transform(np.clip(Xa_te, low_q, high_q))

    if text_pca_dim is not None and Xt_tr.shape[1] > text_pca_dim:
        pca_t = PCA(n_components=min(text_pca_dim, Xt_tr.shape[0] - 1), random_state=seed)
        Xt_tr = pca_t.fit_transform(Xt_tr)
        Xt_te = pca_t.transform(Xt_te)

    if audio_pca_dim is not None and Xa_tr.shape[1] > audio_pca_dim:
        pca_a = PCA(n_components=min(audio_pca_dim, Xa_tr.shape[0] - 1), random_state=seed)
        Xa_tr = pca_a.fit_transform(Xa_tr)
        Xa_te = pca_a.transform(Xa_te)

    # Block-normalize each modality by sqrt(d) so both modalities have equal L2 energy prior to concatenation
    Xt_tr_norm = Xt_tr / np.sqrt(max(Xt_tr.shape[1], 1))
    Xt_te_norm = Xt_te / np.sqrt(max(Xt_te.shape[1], 1))
    Xa_tr_norm = Xa_tr / np.sqrt(max(Xa_tr.shape[1], 1))
    Xa_te_norm = Xa_te / np.sqrt(max(Xa_te.shape[1], 1))

    X_tr = np.hstack([Xt_tr_norm, Xa_tr_norm])
    X_te = np.hstack([Xt_te_norm, Xa_te_norm])

    best_c = _select_best_c_inner_cv(
        X_tr, y_train, c_grid=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], seed=seed
    )
    clf = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_tr, y_train)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def fit_predict_fold_tfidf(
    texts_train: List[str],
    y_train: np.ndarray,
    texts_test: List[str],
    model_type: str = "lr",
    max_features: int = 3000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit TF-IDF (1–2 grams) strictly on `texts_train` and evaluate on `texts_test`."""
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )
    X_tr = vec.fit_transform(texts_train)
    X_te = vec.transform(texts_test)

    if model_type == "lr":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
    elif model_type == "svm":
        base_svm = LinearSVC(C=0.5, class_weight="balanced", random_state=seed)
        clf = CalibratedClassifierCV(base_svm, cv=3)
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def fit_predict_fold_speaker_only(
    spk_train: List[str],
    y_train: np.ndarray,
    spk_test: List[str],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase 8A Diagnostic Control: Speaker-ID-Only model.
    Uses ONLY one-hot encoded `speaker_group` to predict sarcasm.
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = ohe.fit_transform(np.asarray(spk_train).reshape(-1, 1))
    X_te = ohe.transform(np.asarray(spk_test).reshape(-1, 1))
    clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_train)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

