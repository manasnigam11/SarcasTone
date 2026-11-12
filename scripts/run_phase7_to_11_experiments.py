"""
Phase 7–11 Master Experiment Runner:
  - Evaluates all Unimodal (Text, Prosody, eGeMAPS, wav2vec 2.0), Multimodal Fusion (Early, Late, Trimodal),
    Diagnostic Controls (Speaker-Only, Shuffled Prosody, Shuffled wav2vec 2.0), and Prosodic Group Ablations
    across:
      * Protocol A: Stratified Random 5-Fold CV (seeds 42, 1337, 2026)
      * Protocol B: Scene-Grouped 5-Fold CV (seeds 42, 1337, 2026) — PRIMARY LEAKAGE-AWARE BENCHMARK
      * Protocol C1: Leave-One-Speaker-Out (LOSO, 13 major speakers, N=614)
      * Protocol C2: Speaker-Grouped 5-Fold CV (all 23 speaker groups, N=690)
  - Computes 5,000-iteration group-aware paired bootstrap 95% confidence intervals
  - Executes Phase 9 Disagreement & Sarcasm-Type Subgroup Analysis
  - Tests Hypotheses H1, H2, H3, and H4 under the strict CI decision rule
  - Logs every experiment to `experiments/experiment_registry.csv` and saves all out-of-fold predictions
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.disagreement import (
    compute_label_free_disagreement_annotations,
    evaluate_disagreement_and_subgroups,
)
from src.evaluation import (
    bootstrap_macro_f1_ci,
    compute_classification_metrics,
    paired_bootstrap_macro_f1_diff,
)
from src.models import (
    fit_predict_fold_early_concat,
    fit_predict_fold_late_fusion,
    fit_predict_fold_speaker_only,
    fit_predict_fold_tabular,
    fit_predict_fold_tfidf,
)
from src.prosody import ALL_EXPLICIT_PROSODY_FEATURES, PROSODY_FEATURE_GROUPS
from src.representations import clean_transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_cv_protocol(
    df_splits: pd.DataFrame,
    fold_col: str,
    fit_pred_fn: Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a 5-fold cross-validation protocol and collect fold-level + pooled out-of-fold predictions."""
    y_true = df_splits["label"].to_numpy(dtype=int)
    folds = sorted(df_splits[fold_col].unique())
    oof_preds = np.zeros(len(df_splits), dtype=int)
    oof_probs = np.zeros(len(df_splits), dtype=np.float64)
    fold_metrics: List[Dict[str, Any]] = []

    t0 = time.time()
    for f_idx in folds:
        test_mask = (df_splits[fold_col] == f_idx).to_numpy()
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        preds, probs = fit_pred_fn(train_idx, test_idx, seed + int(f_idx))
        oof_preds[test_idx] = preds
        oof_probs[test_idx] = probs
        fm = compute_classification_metrics(y_true[test_idx], preds)
        fm["fold"] = int(f_idx)
        fold_metrics.append(fm)

    elapsed = time.time() - t0
    pooled = compute_classification_metrics(y_true, oof_preds)
    fold_f1s = [fm["macro_f1"] for fm in fold_metrics]
    fold_accs = [fm["accuracy"] for fm in fold_metrics]

    return {
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "pooled": pooled,
        "fold_mean_macro_f1": float(np.mean(fold_f1s)),
        "fold_std_macro_f1": float(np.std(fold_f1s, ddof=1)) if len(fold_f1s) > 1 else 0.0,
        "fold_mean_accuracy": float(np.mean(fold_accs)),
        "fold_std_accuracy": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
        "fold_metrics": fold_metrics,
        "runtime_sec": round(elapsed, 4),
    }


def run_loso_protocol(
    df_splits: pd.DataFrame,
    fit_pred_fn: Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Leave-One-Speaker-Out (LOSO) across all 13 eligible major named speakers (N=614)."""
    y_true = df_splits["label"].to_numpy(dtype=int)
    speakers = [
        s for s in df_splits["loso_target_speaker"].unique() if s != "TRAIN_ONLY"
    ]
    # Keep deterministic order sorted by speaker sample count descending
    spk_counts = df_splits["speaker_group"].value_counts()
    speakers = sorted(speakers, key=lambda s: -int(spk_counts.get(s, 0)))

    oof_preds = np.full(len(df_splits), -1, dtype=int)
    oof_probs = np.full(len(df_splits), np.nan, dtype=np.float64)
    speaker_metrics: List[Dict[str, Any]] = []

    t0 = time.time()
    for s_idx, spk in enumerate(speakers):
        test_mask = (df_splits["speaker_group"] == spk).to_numpy()
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        preds, probs = fit_pred_fn(train_idx, test_idx, seed + s_idx)
        oof_preds[test_idx] = preds
        oof_probs[test_idx] = probs
        sm = compute_classification_metrics(y_true[test_idx], preds)
        sm["speaker_group"] = spk
        sm["sarcasm_base_rate"] = round(float(np.mean(y_true[test_idx])), 4)
        speaker_metrics.append(sm)

    elapsed = time.time() - t0
    eval_mask = oof_preds >= 0
    pooled = compute_classification_metrics(y_true[eval_mask], oof_preds[eval_mask])
    spk_f1s = [sm["macro_f1"] for sm in speaker_metrics]
    spk_accs = [sm["accuracy"] for sm in speaker_metrics]

    return {
        "eval_mask": eval_mask,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "pooled": pooled,
        "speaker_mean_macro_f1": float(np.mean(spk_f1s)),
        "speaker_std_macro_f1": float(np.std(spk_f1s, ddof=1)),
        "speaker_mean_accuracy": float(np.mean(spk_accs)),
        "speaker_std_accuracy": float(np.std(spk_accs, ddof=1)),
        "speaker_metrics": speaker_metrics,
        "runtime_sec": round(elapsed, 4),
    }


def main() -> None:
    df_master = pd.read_parquet(ROOT / "data" / "processed" / "mustard_master.parquet")
    df_splits = pd.read_parquet(ROOT / "data" / "splits" / "frozen_splits.parquet")
    df_prosody = pd.read_parquet(ROOT / "data" / "features" / "explicit_prosody.parquet")
    df_egemaps = pd.read_parquet(ROOT / "data" / "features" / "egemaps_v02.parquet")
    df_rob_utt = pd.read_parquet(ROOT / "data" / "features" / "roberta_target_embeddings.parquet")
    df_rob_ctx = pd.read_parquet(ROOT / "data" / "features" / "roberta_context_target_embeddings.parquet")
    df_sent = pd.read_parquet(ROOT / "data" / "features" / "text_sentiment_polarity.parquet")
    df_w2v = pd.read_parquet(ROOT / "data" / "features" / "wav2vec2_base_embeddings.parquet")

    # Ensure exact alignment of utterance_id across all feature tables
    uids = df_master["utterance_id"].astype(str).tolist()
    assert df_splits["utterance_id"].astype(str).tolist() == uids
    assert df_prosody["utterance_id"].astype(str).tolist() == uids
    assert df_egemaps["utterance_id"].astype(str).tolist() == uids
    assert df_rob_utt["utterance_id"].astype(str).tolist() == uids
    assert df_rob_ctx["utterance_id"].astype(str).tolist() == uids
    assert df_sent["utterance_id"].astype(str).tolist() == uids
    assert df_w2v["utterance_id"].astype(str).tolist() == uids

    y_all = df_master["label"].to_numpy(dtype=int)
    scene_groups = df_master["scene_group_id"].to_numpy()
    speaker_groups = df_master["speaker_group"].astype(str).tolist()

    target_texts = [clean_transcript(t) for t in df_master["utterance"].tolist()]
    ctx_target_texts = [
        f"{clean_transcript(c)} {clean_transcript(u)}".strip()
        for c, u in zip(df_master["context_text"].tolist(), df_master["utterance"].tolist())
    ]

    X_rob_utt = df_rob_utt.drop(columns=["utterance_id"]).to_numpy(dtype=np.float32)
    X_rob_ctx = df_rob_ctx.drop(columns=["utterance_id"]).to_numpy(dtype=np.float32)
    X_prosody_full = df_prosody[ALL_EXPLICIT_PROSODY_FEATURES].to_numpy(dtype=np.float32)
    X_egemaps = df_egemaps.drop(columns=["utterance_id"]).to_numpy(dtype=np.float32)
    X_w2v = df_w2v.drop(columns=["utterance_id"]).to_numpy(dtype=np.float32)

    # Shuffled audio matrices (controlled seed=42 permutation for Phase 8B Shuffled-Audio Control)
    rng_shuf = np.random.default_rng(42)
    shuf_perm = rng_shuf.permutation(len(uids))
    X_prosody_shuffled = X_prosody_full[shuf_perm]
    X_w2v_shuffled = X_w2v[shuf_perm]

    # Prosodic group matrices for Phase 8C Ablation
    prosody_group_matrices: Dict[str, np.ndarray] = {}
    prosody_drop_matrices: Dict[str, np.ndarray] = {}
    for g_name, g_cols in PROSODY_FEATURE_GROUPS.items():
        prosody_group_matrices[g_name] = df_prosody[g_cols].to_numpy(dtype=np.float32)
        keep_cols = [c for c in ALL_EXPLICIT_PROSODY_FEATURES if c not in g_cols]
        prosody_drop_matrices[g_name] = df_prosody[keep_cols].to_numpy(dtype=np.float32)

    # Define complete catalog of models (Phases 4, 5, 6, 7, 8)
    model_specs: List[Tuple[str, str, str, Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]]]] = [
        # Text-only models (Phase 4)
        (
            "tfidf_target_lr",
            "Text-Only",
            "TF-IDF (Target Utterance, 1-2g) + LR",
            lambda tr, te, s: fit_predict_fold_tfidf(
                [target_texts[i] for i in tr], y_all[tr], [target_texts[i] for i in te], model_type="lr", seed=s
            ),
        ),
        (
            "tfidf_target_svm",
            "Text-Only",
            "TF-IDF (Target Utterance, 1-2g) + LinearSVC",
            lambda tr, te, s: fit_predict_fold_tfidf(
                [target_texts[i] for i in tr], y_all[tr], [target_texts[i] for i in te], model_type="svm", seed=s
            ),
        ),
        (
            "tfidf_context_lr",
            "Text-Only (Context)",
            "TF-IDF (Context + Target) + LR",
            lambda tr, te, s: fit_predict_fold_tfidf(
                [ctx_target_texts[i] for i in tr], y_all[tr], [ctx_target_texts[i] for i in te], model_type="lr", seed=s
            ),
        ),
        (
            "roberta_target_lr",
            "Text-Only",
            "Frozen RoBERTa-base (Target Only) + LR",
            lambda tr, te, s: fit_predict_fold_tabular(
                X_rob_utt[tr], y_all[tr], X_rob_utt[te], model_type="lr", c_grid=[0.005, 0.01, 0.05, 0.1, 0.5], seed=s
            ),
        ),
        (
            "roberta_context_lr",
            "Text-Only (Context)",
            "Frozen RoBERTa-base (Context + Target) + LR",
            lambda tr, te, s: fit_predict_fold_tabular(
                X_rob_ctx[tr], y_all[tr], X_rob_ctx[te], model_type="lr", c_grid=[0.005, 0.01, 0.05, 0.1, 0.5], seed=s
            ),
        ),
        # Audio-only models (Phases 5 & 6)
        (
            "prosody_explicit_lr",
            "Audio-Only (Prosody)",
            "Explicit Prosody (41 features, 4 groups) + LR",
            lambda tr, te, s: fit_predict_fold_tabular(
                X_prosody_full[tr], y_all[tr], X_prosody_full[te], model_type="lr", c_grid=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0], seed=s
            ),
        ),
        (
            "prosody_explicit_svm",
            "Audio-Only (Prosody)",
            "Explicit Prosody (41 features, 4 groups) + RBF-SVM",
            lambda tr, te, s: fit_predict_fold_tabular(
                X_prosody_full[tr], y_all[tr], X_prosody_full[te], model_type="svm_rbf", seed=s
            ),
        ),
        (
            "egemaps_lr",
            "Audio-Only (eGeMAPS)",
            "openSMILE eGeMAPSv02 (88 features) + LR",
            lambda tr, te, s: fit_predict_fold_tabular(
                X_egemaps[tr], y_all[tr], X_egemaps[te], model_type="lr", c_grid=[0.005, 0.01, 0.05, 0.1, 0.5], seed=s
            ),
        ),
        (
            "wav2vec2_lr",
            "Audio-Only (Learned)",
            "Frozen wav2vec 2.0 Base (768-d) + LR",
            lambda tr, te, s: fit_predict_fold_tabular(
                X_w2v[tr], y_all[tr], X_w2v[te], model_type="lr", c_grid=[0.005, 0.01, 0.05, 0.1, 0.5], seed=s
            ),
        ),
        # Multimodal Models (Phase 7)
        (
            "fusion_roberta_prosody_early",
            "Multimodal (Text + Prosody)",
            "RoBERTa + Explicit Prosody (Early Concat Fusion)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], X_prosody_full[tr], y_all[tr], X_rob_utt[te], X_prosody_full[te],
                text_pca_dim=64, audio_pca_dim=None, seed=s
            ),
        ),
        (
            "fusion_roberta_prosody_late",
            "Multimodal (Text + Prosody)",
            "RoBERTa + Explicit Prosody (Late Stacking Fusion)",
            lambda tr, te, s: fit_predict_fold_late_fusion(
                X_rob_utt[tr], X_prosody_full[tr], y_all[tr], X_rob_utt[te], X_prosody_full[te], seed=s
            ),
        ),
        (
            "fusion_roberta_wav2vec2_early",
            "Multimodal (Text + Learned Audio)",
            "RoBERTa + wav2vec 2.0 (Early Concat Fusion)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], X_w2v[tr], y_all[tr], X_rob_utt[te], X_w2v[te],
                text_pca_dim=64, audio_pca_dim=64, seed=s
            ),
        ),
        (
            "fusion_roberta_wav2vec2_late",
            "Multimodal (Text + Learned Audio)",
            "RoBERTa + wav2vec 2.0 (Late Stacking Fusion)",
            lambda tr, te, s: fit_predict_fold_late_fusion(
                X_rob_utt[tr], X_w2v[tr], y_all[tr], X_rob_utt[te], X_w2v[te], seed=s
            ),
        ),
        (
            "fusion_trimodal_text_prosody_w2v2",
            "Multimodal (Text + Prosody + wav2vec2)",
            "RoBERTa + Explicit Prosody + wav2vec 2.0 (Early Fusion)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], np.hstack([X_prosody_full, X_w2v])[tr], y_all[tr],
                X_rob_utt[te], np.hstack([X_prosody_full, X_w2v])[te],
                text_pca_dim=64, audio_pca_dim=64, seed=s
            ),
        ),
        # Diagnostic Controls (Phase 8A & 8B)
        (
            "control_speaker_only",
            "Control (Diagnostic)",
            "Speaker-ID Only One-Hot Diagnostic Baseline",
            lambda tr, te, s: fit_predict_fold_speaker_only(
                [speaker_groups[i] for i in tr], y_all[tr], [speaker_groups[i] for i in te], seed=s
            ),
        ),
        (
            "control_shuffled_prosody",
            "Control (Shuffled Audio)",
            "RoBERTa + Shuffled Prosody Control (Broken Alignment)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], X_prosody_shuffled[tr], y_all[tr], X_rob_utt[te], X_prosody_shuffled[te],
                text_pca_dim=64, audio_pca_dim=None, seed=s
            ),
        ),
        (
            "control_shuffled_wav2vec2",
            "Control (Shuffled Audio)",
            "RoBERTa + Shuffled wav2vec 2.0 Control (Broken Alignment)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], X_w2v_shuffled[tr], y_all[tr], X_rob_utt[te], X_w2v_shuffled[te],
                text_pca_dim=64, audio_pca_dim=64, seed=s
            ),
        ),
        # Prosodic Feature Group Ablations (Phase 8C)
        (
            "ablation_text_prosody_no_f0",
            "Ablation (Drop Group)",
            "Text + Prosody WITHOUT F0 (Group 1 Dropped)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], prosody_drop_matrices["f0"][tr], y_all[tr],
                X_rob_utt[te], prosody_drop_matrices["f0"][te],
                text_pca_dim=64, audio_pca_dim=None, seed=s
            ),
        ),
        (
            "ablation_text_prosody_no_energy",
            "Ablation (Drop Group)",
            "Text + Prosody WITHOUT Energy/Intensity (Group 2 Dropped)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], prosody_drop_matrices["energy"][tr], y_all[tr],
                X_rob_utt[te], prosody_drop_matrices["energy"][te],
                text_pca_dim=64, audio_pca_dim=None, seed=s
            ),
        ),
        (
            "ablation_text_prosody_no_timing",
            "Ablation (Drop Group)",
            "Text + Prosody WITHOUT Timing/Pauses (Group 3 Dropped)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], prosody_drop_matrices["timing"][tr], y_all[tr],
                X_rob_utt[te], prosody_drop_matrices["timing"][te],
                text_pca_dim=64, audio_pca_dim=None, seed=s
            ),
        ),
        (
            "ablation_text_prosody_no_voice_quality",
            "Ablation (Drop Group)",
            "Text + Prosody WITHOUT Voice Quality (Group 4 Dropped)",
            lambda tr, te, s: fit_predict_fold_early_concat(
                X_rob_utt[tr], prosody_drop_matrices["voice_quality"][tr], y_all[tr],
                X_rob_utt[te], prosody_drop_matrices["voice_quality"][te],
                text_pca_dim=64, audio_pca_dim=None, seed=s
            ),
        ),
        # Audio-only single-group prosody diagnostics
        (
            "prosody_only_f0_lr",
            "Ablation (Single Prosody Group)",
            "Audio-Only: F0 / Pitch Features Only (11 features)",
            lambda tr, te, s: fit_predict_fold_tabular(
                prosody_group_matrices["f0"][tr], y_all[tr], prosody_group_matrices["f0"][te], model_type="lr", seed=s
            ),
        ),
        (
            "prosody_only_energy_lr",
            "Ablation (Single Prosody Group)",
            "Audio-Only: Energy & Intensity Only (10 features)",
            lambda tr, te, s: fit_predict_fold_tabular(
                prosody_group_matrices["energy"][tr], y_all[tr], prosody_group_matrices["energy"][te], model_type="lr", seed=s
            ),
        ),
        (
            "prosody_only_timing_lr",
            "Ablation (Single Prosody Group)",
            "Audio-Only: Timing & Pauses Only (11 features)",
            lambda tr, te, s: fit_predict_fold_tabular(
                prosody_group_matrices["timing"][tr], y_all[tr], prosody_group_matrices["timing"][te], model_type="lr", seed=s
            ),
        ),
        (
            "prosody_only_voice_quality_lr",
            "Ablation (Single Prosody Group)",
            "Audio-Only: Voice Quality Only (9 features)",
            lambda tr, te, s: fit_predict_fold_tabular(
                prosody_group_matrices["voice_quality"][tr], y_all[tr], prosody_group_matrices["voice_quality"][te], model_type="lr", seed=s
            ),
        ),
    ]

    registry_rows: List[Dict[str, Any]] = []
    oof_store: Dict[str, Dict[str, np.ndarray]] = {
        "random_s42": {},
        "scene_s42": {},
        "speaker_grouped_s42": {},
        "loso": {},
    }
    probs_store: Dict[str, np.ndarray] = {}
    loso_speaker_breakdown: Dict[str, List[Dict[str, Any]]] = {}

    seeds = [42, 1337, 2026]
    exp_counter = 1

    for model_id, family, display_name, fn in model_specs:
        logger.info("Evaluating model [%s]: %s", model_id, display_name)

        # 1. Run across 3 seeds for Protocol A (Random) and Protocol B (Scene-Grouped)
        for proto_prefix, proto_label, group_arr in [
            ("random_fold", "Random_5Fold", None),
            ("scene_fold", "SceneGrouped_5Fold", scene_groups),
        ]:
            seed_f1s = []
            for seed in seeds:
                col = f"{proto_prefix}_s{seed}"
                res = run_cv_protocol(df_splits, col, fn, seed=seed)
                ci = bootstrap_macro_f1_ci(
                    y_all, res["oof_preds"], groups=group_arr, n_resamples=3000, seed=seed
                )
                seed_f1s.append(res["pooled"]["macro_f1"])

                if seed == 42:
                    key = "random_s42" if proto_prefix == "random_fold" else "scene_s42"
                    oof_store[key][model_id] = res["oof_preds"]
                    if proto_prefix == "scene_fold":
                        probs_store[model_id] = res["oof_probs"]

                registry_rows.append(
                    {
                        "experiment_id": f"EXP_{exp_counter:04d}",
                        "model_id": model_id,
                        "family": family,
                        "display_name": display_name,
                        "protocol": proto_label,
                        "seed": seed,
                        "n_evaluated": res["pooled"]["n"],
                        "pooled_macro_f1": round(res["pooled"]["macro_f1"], 4),
                        "fold_mean_macro_f1": round(res["fold_mean_macro_f1"], 4),
                        "fold_std_macro_f1": round(res["fold_std_macro_f1"], 4),
                        "ci_95_low": round(ci["ci_low"], 4),
                        "ci_95_high": round(ci["ci_high"], 4),
                        "accuracy": round(res["pooled"]["accuracy"], 4),
                        "macro_precision": round(res["pooled"]["macro_precision"], 4),
                        "macro_recall": round(res["pooled"]["macro_recall"], 4),
                        "weighted_f1": round(res["pooled"]["weighted_f1"], 4),
                        "tn": res["pooled"]["tn"],
                        "fp": res["pooled"]["fp"],
                        "fn": res["pooled"]["fn"],
                        "tp": res["pooled"]["tp"],
                        "runtime_sec": res["runtime_sec"],
                    }
                )
                exp_counter += 1

        # 2. Protocol C2: Speaker-Grouped 5-Fold CV (seed 42)
        res_spkg = run_cv_protocol(df_splits, "speaker_grouped_fold_s42", fn, seed=42)
        ci_spkg = bootstrap_macro_f1_ci(
            y_all,
            res_spkg["oof_preds"],
            groups=np.asarray(speaker_groups),
            n_resamples=3000,
            seed=42,
        )
        oof_store["speaker_grouped_s42"][model_id] = res_spkg["oof_preds"]
        registry_rows.append(
            {
                "experiment_id": f"EXP_{exp_counter:04d}",
                "model_id": model_id,
                "family": family,
                "display_name": display_name,
                "protocol": "SpeakerGrouped_5Fold",
                "seed": 42,
                "n_evaluated": res_spkg["pooled"]["n"],
                "pooled_macro_f1": round(res_spkg["pooled"]["macro_f1"], 4),
                "fold_mean_macro_f1": round(res_spkg["fold_mean_macro_f1"], 4),
                "fold_std_macro_f1": round(res_spkg["fold_std_macro_f1"], 4),
                "ci_95_low": round(ci_spkg["ci_low"], 4),
                "ci_95_high": round(ci_spkg["ci_high"], 4),
                "accuracy": round(res_spkg["pooled"]["accuracy"], 4),
                "macro_precision": round(res_spkg["pooled"]["macro_precision"], 4),
                "macro_recall": round(res_spkg["pooled"]["macro_recall"], 4),
                "weighted_f1": round(res_spkg["pooled"]["weighted_f1"], 4),
                "tn": res_spkg["pooled"]["tn"],
                "fp": res_spkg["pooled"]["fp"],
                "fn": res_spkg["pooled"]["fn"],
                "tp": res_spkg["pooled"]["tp"],
                "runtime_sec": res_spkg["runtime_sec"],
            }
        )
        exp_counter += 1

        # 3. Protocol C1: Leave-One-Speaker-Out (LOSO, 13 major speakers, N=614)
        res_loso = run_loso_protocol(df_splits, fn, seed=42)
        eval_mask = res_loso["eval_mask"]
        ci_loso = bootstrap_macro_f1_ci(
            y_all[eval_mask],
            res_loso["oof_preds"][eval_mask],
            groups=scene_groups[eval_mask],
            n_resamples=3000,
            seed=42,
        )
        oof_store["loso"][model_id] = res_loso["oof_preds"]
        loso_speaker_breakdown[model_id] = res_loso["speaker_metrics"]
        registry_rows.append(
            {
                "experiment_id": f"EXP_{exp_counter:04d}",
                "model_id": model_id,
                "family": family,
                "display_name": display_name,
                "protocol": "LOSO_13Speakers",
                "seed": 42,
                "n_evaluated": res_loso["pooled"]["n"],
                "pooled_macro_f1": round(res_loso["pooled"]["macro_f1"], 4),
                "fold_mean_macro_f1": round(res_loso["speaker_mean_macro_f1"], 4),
                "fold_std_macro_f1": round(res_loso["speaker_std_macro_f1"], 4),
                "ci_95_low": round(ci_loso["ci_low"], 4),
                "ci_95_high": round(ci_loso["ci_high"], 4),
                "accuracy": round(res_loso["pooled"]["accuracy"], 4),
                "macro_precision": round(res_loso["pooled"]["macro_precision"], 4),
                "macro_recall": round(res_loso["pooled"]["macro_recall"], 4),
                "weighted_f1": round(res_loso["pooled"]["weighted_f1"], 4),
                "tn": res_loso["pooled"]["tn"],
                "fp": res_loso["pooled"]["fp"],
                "fn": res_loso["pooled"]["fn"],
                "tp": res_loso["pooled"]["tp"],
                "runtime_sec": res_loso["runtime_sec"],
            }
        )
        exp_counter += 1

    # Save experiment registry
    df_reg = pd.DataFrame(registry_rows)
    reg_path = ROOT / "experiments" / "experiment_registry.csv"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    df_reg.to_csv(reg_path, index=False)
    df_reg.to_parquet(reg_path.with_suffix(".parquet"), index=False)

    # Save out-of-fold predictions & probabilities for traceability and error analysis
    pred_dir = ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for proto_key, pred_dict in oof_store.items():
        df_p = pd.DataFrame({"utterance_id": uids, "label": y_all})
        for m_id, arr in pred_dict.items():
            df_p[m_id] = arr
        df_p.to_csv(pred_dir / f"oof_predictions_{proto_key}.csv", index=False)

    df_probs = pd.DataFrame({"utterance_id": uids, "label": y_all})
    for m_id, arr in probs_store.items():
        df_probs[m_id] = np.round(arr, 6)
    df_probs.to_csv(pred_dir / "oof_probabilities_scene_s42.csv", index=False)

    with open(ROOT / "results" / "tables" / "loso_per_speaker_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(loso_speaker_breakdown, f, indent=2)

    # Phase 9 Disagreement & Subgroup Analysis under Primary Scene-Grouped Protocol (seed 42)
    df_disagree = compute_label_free_disagreement_annotations(df_master, df_sent, df_prosody)
    df_disagree.to_csv(ROOT / "data" / "processed" / "disagreement_annotations.csv", index=False)
    disagree_results = evaluate_disagreement_and_subgroups(
        df_disagree, oof_store["scene_s42"], n_resamples=5000, seed=42
    )
    with open(ROOT / "results" / "tables" / "disagreement_and_subgroups.json", "w", encoding="utf-8") as f:
        json.dump(disagree_results, f, indent=2)

    # Phase 10 Formal Paired Bootstrap Hypothesis Tests (5,000 resamples, Scene-Aware)
    sc_preds = oof_store["scene_s42"]
    rnd_preds = oof_store["random_s42"]
    spkg_preds = oof_store["speaker_grouped_s42"]
    loso_preds = oof_store["loso"]
    loso_mask = loso_preds["roberta_target_lr"] >= 0

    paired_tests: Dict[str, Any] = {
        # H1: Text + Explicit Prosody vs Text-Only under Scene-Grouped evaluation
        "H1_early_prosody_vs_roberta_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["fusion_roberta_prosody_early"], sc_preds["roberta_target_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        "H1_late_prosody_vs_roberta_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["fusion_roberta_prosody_late"], sc_preds["roberta_target_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        "H1_early_w2v2_vs_roberta_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["fusion_roberta_wav2vec2_early"], sc_preds["roberta_target_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        "H1_trimodal_vs_roberta_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["fusion_trimodal_text_prosody_w2v2"], sc_preds["roberta_target_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        # H1 under LOSO (Speaker-Independent)
        "H1_early_prosody_vs_roberta_loso": paired_bootstrap_macro_f1_diff(
            y_all[loso_mask],
            loso_preds["fusion_roberta_prosody_early"][loso_mask],
            loso_preds["roberta_target_lr"][loso_mask],
            groups=scene_groups[loso_mask], n_resamples=5000, seed=42
        ),
        # H3: Random vs Scene-Grouped vs Speaker-Independent
        "H3_roberta_random_vs_scene": paired_bootstrap_macro_f1_diff(
            y_all, rnd_preds["roberta_target_lr"], sc_preds["roberta_target_lr"],
            groups=None, n_resamples=5000, seed=42
        ),
        "H3_roberta_scene_vs_speakergrouped": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["roberta_target_lr"], spkg_preds["roberta_target_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        "H3_speaker_only_scene_vs_speakergrouped": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["control_speaker_only"], spkg_preds["control_speaker_only"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        # H4: Learned Audio (wav2vec 2.0) vs Explicit Prosody Alone
        "H4_wav2vec2_vs_explicit_prosody_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["wav2vec2_lr"], sc_preds["prosody_explicit_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        "H4_wav2vec2_vs_explicit_prosody_loso": paired_bootstrap_macro_f1_diff(
            y_all[loso_mask],
            loso_preds["wav2vec2_lr"][loso_mask],
            loso_preds["prosody_explicit_lr"][loso_mask],
            groups=scene_groups[loso_mask], n_resamples=5000, seed=42
        ),
        # Shuffled Audio Controls vs Real Multimodal Fusion
        "Control_real_vs_shuffled_prosody_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["fusion_roberta_prosody_early"], sc_preds["control_shuffled_prosody"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        "Control_real_vs_shuffled_w2v2_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["fusion_roberta_wav2vec2_early"], sc_preds["control_shuffled_wav2vec2"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
        # Context-aware Text vs Target-only Text
        "Context_roberta_ctx_vs_target_scene": paired_bootstrap_macro_f1_diff(
            y_all, sc_preds["roberta_context_lr"], sc_preds["roberta_target_lr"],
            groups=scene_groups, n_resamples=5000, seed=42
        ),
    }

    with open(ROOT / "results" / "tables" / "paired_bootstrap_tests.json", "w", encoding="utf-8") as f:
        json.dump(paired_tests, f, indent=2)

    logger.info("Completed all %d experiments! Saved registry to %s", len(registry_rows), reg_path)


if __name__ == "__main__":
    main()

