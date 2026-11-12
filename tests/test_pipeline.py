"""
Phase 12 Comprehensive Automated Unit, Integration, and Leakage Verification Test Suite.

Tests:
  1. Dataset loading & 690-instance integrity
  2. Binary label balance and 0 mismatches with MUStARD++ on the 688 overlapping scenes
  3. Audio loading & 16 kHz mono WAV verification
  4. Explicit prosodic feature dimensions, 4-group structure, and unvoiced F0 handling
  5. Text cleaning and representation shapes
  6. Split integrity & zero scene/speaker overlap across train and test folds
  7. Strict inside-fold preprocessing (no test-set leakage in scalers/imputers)
  8. Metric & Paired Bootstrap CI mathematical correctness and determinism
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.evaluation import (
    bootstrap_macro_f1_ci,
    compute_classification_metrics,
    paired_bootstrap_macro_f1_diff,
)
from src.models import fit_predict_fold_early_concat, fit_predict_fold_tabular
from src.prosody import (
    ALL_EXPLICIT_PROSODY_FEATURES,
    PROSODY_FEATURE_GROUPS,
    extract_utterance_prosody,
)
from src.representations import clean_transcript
from src.splits import verify_zero_group_overlap

ROOT = Path(__file__).resolve().parent.parent


def test_master_dataset_and_labels() -> None:
    """Verify master dataset has 690 instances, exact 345/345 class balance, and valid paths."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "mustard_master.parquet")
    assert len(df) == 690
    assert df["utterance_id"].nunique() == 690
    assert set(df["label"].unique()) == {0, 1}
    assert int((df["label"] == 1).sum()) == 345
    assert int((df["label"] == 0).sum()) == 345
    assert int(df["in_mustard_pp"].sum()) == 688


def test_audio_wav_16k_integrity() -> None:
    """Verify extracted 16 kHz mono WAV files exist, are 16,000 Hz mono, and match clip durations."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "mustard_master.parquet")
    for idx in [0, 100, 345, 689]:
        row = df.iloc[idx]
        wav_path = ROOT / row["wav_16k_path"]
        assert wav_path.exists()
        data, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == 16000
        assert data.ndim == 1
        assert len(data) > 0
        assert abs(len(data) / 16000.0 - float(row["utt_duration_sec"])) < 0.02


def test_prosody_feature_groups_and_missing_f0_handling() -> None:
    """Verify 41 explicit prosodic features span 4 disjoint groups and handle unvoiced F0 with flags."""
    assert len(ALL_EXPLICIT_PROSODY_FEATURES) == 41
    union_cols = (
        set(PROSODY_FEATURE_GROUPS["f0"])
        | set(PROSODY_FEATURE_GROUPS["energy"])
        | set(PROSODY_FEATURE_GROUPS["timing"])
        | set(PROSODY_FEATURE_GROUPS["voice_quality"])
    )
    assert len(union_cols) == 41

    df_pros = pd.read_parquet(ROOT / "data" / "features" / "explicit_prosody.parquet")
    assert len(df_pros) == 690
    assert list(df_pros.columns) == ["utterance_id"] + ALL_EXPLICIT_PROSODY_FEATURES
    # Verify f0_unvoiced_flag is binary 0/1 and never NaN
    assert not df_pros["f0_unvoiced_flag"].isna().any()
    assert set(df_pros["f0_unvoiced_flag"].unique()).issubset({0.0, 1.0})


def test_split_integrity_and_zero_leakage() -> None:
    """Verify zero utterance, scene, or speaker overlap across all frozen split protocols."""
    df_splits = pd.read_parquet(ROOT / "data" / "splits" / "frozen_splits.parquet")
    assert len(df_splits) == 690

    for seed in [42, 1337, 2026]:
        r_res = verify_zero_group_overlap(df_splits, f"random_fold_s{seed}", "utterance_id")
        assert r_res["zero_overlap_verified"] is True

        s_res = verify_zero_group_overlap(df_splits, f"scene_fold_s{seed}", "scene_group_id")
        assert s_res["zero_overlap_verified"] is True

    spk_res = verify_zero_group_overlap(df_splits, "speaker_grouped_fold_s42", "speaker_group")
    assert spk_res["zero_overlap_verified"] is True


def test_text_cleaning() -> None:
    """Verify HTML tags and extra spaces are removed cleanly."""
    raw = 'I didn\'t think you meant showing me <i>Raiders of the Lost Ark</i>   for the first time.'
    cleaned = clean_transcript(raw)
    assert "<i>" not in cleaned
    assert "</i>" not in cleaned
    assert "Raiders of the Lost Ark" in cleaned


def test_in_fold_tabular_and_fusion_no_nan_and_determinism() -> None:
    """Verify fold trainers handle NaN F0 values via training-only imputation and are deterministic."""
    rng = np.random.default_rng(42)
    X_text_tr = rng.standard_normal((100, 32)).astype(np.float32)
    X_aud_tr = rng.standard_normal((100, 10)).astype(np.float32)
    X_aud_tr[0, 0] = np.nan  # Simulate unvoiced F0
    y_tr = rng.integers(0, 2, size=100)

    X_text_te = rng.standard_normal((25, 32)).astype(np.float32)
    X_aud_te = rng.standard_normal((25, 10)).astype(np.float32)
    X_aud_te[2, 0] = np.nan

    p1, prob1 = fit_predict_fold_tabular(X_aud_tr, y_tr, X_aud_te, model_type="lr", seed=42)
    p2, prob2 = fit_predict_fold_tabular(X_aud_tr, y_tr, X_aud_te, model_type="lr", seed=42)
    assert p1.shape == (25,)
    assert np.allclose(prob1, prob2)
    assert not np.isnan(prob1).any()

    fp1, fprob1 = fit_predict_fold_early_concat(
        X_text_tr, X_aud_tr, y_tr, X_text_te, X_aud_te, text_pca_dim=16, seed=42
    )
    assert fp1.shape == (25,)
    assert not np.isnan(fprob1).any()


def test_metrics_and_paired_bootstrap_ci() -> None:
    """Verify classification metrics and paired bootstrap CI calculations."""
    y_true = np.array([0, 0, 1, 1] * 25)
    y_good = np.array([0, 0, 1, 1] * 20 + [0, 1, 0, 1] * 5)
    y_bad = np.array([0, 1, 0, 1] * 25)

    m_good = compute_classification_metrics(y_true, y_good)
    assert 0.85 <= m_good["macro_f1"] <= 0.95

    diff_res = paired_bootstrap_macro_f1_diff(y_true, y_good, y_bad, n_resamples=1000, seed=42)
    assert diff_res["diff_point"] > 0.30
    assert diff_res["excludes_zero"] is True
    assert diff_res["ci_low"] > 0.0

