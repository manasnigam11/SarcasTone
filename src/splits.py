"""
Leakage-Safe Experimental Split Generation and Verification Module.

Implements the three evaluation protocols required by Phase 7:
  A. Stratified Random 5-Fold CV (across seeds 42, 1337, 2026)
  B. Scene-Grouped 5-Fold CV (across seeds 42, 1337, 2026) — zero scene/dialogue cluster overlap
  C. Speaker-Independent Evaluation:
     - C1: Leave-One-Speaker-Out (LOSO) across all major speakers (>= 15 utterances, both classes present)
     - C2: Speaker-Grouped 5-Fold CV across all 690 utterances — zero speaker_group overlap
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)


def verify_zero_group_overlap(
    df: pd.DataFrame,
    fold_col: str,
    group_col: str,
) -> Dict[str, Any]:
    """Assert that no group identifier appears in more than one fold (i.e. never in both train and test)."""
    valid_rows = df[df[fold_col] >= 0]
    folds = sorted(valid_rows[fold_col].unique())
    overlaps: List[Tuple[int, int, List[str]]] = []

    for i, f1 in enumerate(folds):
        g1 = set(valid_rows[valid_rows[fold_col] == f1][group_col].astype(str))
        for f2 in folds[i + 1 :]:
            g2 = set(valid_rows[valid_rows[fold_col] == f2][group_col].astype(str))
            inter = sorted(list(g1 & g2))
            if inter:
                overlaps.append((int(f1), int(f2), inter))

    if overlaps:
        raise AssertionError(
            f"LEAKAGE DETECTED in {fold_col} over group {group_col}: {overlaps}"
        )

    return {
        "fold_col": fold_col,
        "group_col": group_col,
        "num_folds": len(folds),
        "num_unique_groups": int(valid_rows[group_col].nunique()),
        "zero_overlap_verified": True,
    }


def generate_and_freeze_splits(
    df: pd.DataFrame,
    splits_dir: str | Path = "data/splits",
    seeds: List[int] = [42, 1337, 2026],
    n_folds: int = 5,
    loso_min_utterances: int = 15,
) -> Dict[str, Any]:
    """Generate, validate, and permanently save all evaluation splits to `data/splits/`."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_df = df[
        [
            "utterance_id",
            "show",
            "speaker",
            "speaker_group",
            "scene_group_id",
            "label",
        ]
    ].copy()

    y = split_df["label"].to_numpy()
    scene_groups = split_df["scene_group_id"].to_numpy()
    speaker_groups = split_df["speaker_group"].to_numpy()

    verification_records: List[Dict[str, Any]] = []

    # Protocol A: Stratified Random 5-Fold CV (3 seeds)
    for seed in seeds:
        col = f"random_fold_s{seed}"
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_assign = np.full(len(split_df), -1, dtype=int)
        for fold_idx, (_, test_idx) in enumerate(skf.split(split_df, y)):
            fold_assign[test_idx] = fold_idx
        assert (fold_assign >= 0).all()
        split_df[col] = fold_assign
        # Verify utterance_id uniqueness across folds
        verification_records.append(verify_zero_group_overlap(split_df, col, "utterance_id"))

    # Protocol B: Scene-Grouped 5-Fold CV (3 seeds) using StratifiedGroupKFold
    for seed in seeds:
        col = f"scene_fold_s{seed}"
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_assign = np.full(len(split_df), -1, dtype=int)
        for fold_idx, (_, test_idx) in enumerate(sgkf.split(split_df, y, groups=scene_groups)):
            fold_assign[test_idx] = fold_idx
        assert (fold_assign >= 0).all()
        split_df[col] = fold_assign
        verification_records.append(verify_zero_group_overlap(split_df, col, "scene_group_id"))

    # Protocol C2: Speaker-Grouped 5-Fold CV (seed 42) across all 690 instances
    sgkf_spk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seeds[0])
    spk_fold_assign = np.full(len(split_df), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(sgkf_spk.split(split_df, y, groups=speaker_groups)):
        spk_fold_assign[test_idx] = fold_idx
    split_df["speaker_grouped_fold_s42"] = spk_fold_assign
    verification_records.append(
        verify_zero_group_overlap(split_df, "speaker_grouped_fold_s42", "speaker_group")
    )

    # Protocol C1: Leave-One-Speaker-Out (LOSO) over major named speakers
    # Exclude generic catch-all "PERSON" and speakers with < loso_min_utterances or < 2 classes
    spk_stats = (
        split_df.groupby("speaker_group")["label"]
        .agg(["count", "nunique", "mean"])
        .reset_index()
    )
    eligible_loso_speakers: List[str] = []
    for _, row in spk_stats.sort_values("count", ascending=False).iterrows():
        sg = str(row["speaker_group"])
        raw_spk = sg.split("_", 1)[1]
        if (
            raw_spk != "PERSON"
            and int(row["count"]) >= loso_min_utterances
            and int(row["nunique"]) == 2
        ):
            eligible_loso_speakers.append(sg)

    split_df["loso_target_speaker"] = np.where(
        split_df["speaker_group"].isin(eligible_loso_speakers),
        split_df["speaker_group"],
        "TRAIN_ONLY",
    )

    # Save master split table and individual protocol files
    split_df.to_csv(splits_dir / "frozen_splits.csv", index=False)
    split_df.to_parquet(splits_dir / "frozen_splits.parquet", index=False)

    # Quantify actual scene & speaker leakage that occurs inside Random Splits!
    random_scene_leakage = {}
    random_speaker_leakage = {}
    for seed in seeds:
        col = f"random_fold_s{seed}"
        leaked_scenes = 0
        for sc, grp in split_df.groupby("scene_group_id"):
            if len(grp) > 1 and grp[col].nunique() > 1:
                leaked_scenes += 1
        random_scene_leakage[f"seed_{seed}"] = leaked_scenes

        leaked_spks = 0
        for spk, grp in split_df.groupby("speaker_group"):
            if len(grp) > 1 and grp[col].nunique() > 1:
                leaked_spks += 1
        random_speaker_leakage[f"seed_{seed}"] = leaked_spks

    summary = {
        "total_instances": int(len(split_df)),
        "unique_scene_clusters": int(split_df["scene_group_id"].nunique()),
        "multi_clip_scene_clusters": int(
            sum(1 for _, g in split_df.groupby("scene_group_id") if len(g) > 1)
        ),
        "unique_speaker_groups": int(split_df["speaker_group"].nunique()),
        "seeds": seeds,
        "n_folds": n_folds,
        "loso_min_utterances": loso_min_utterances,
        "eligible_loso_speakers": eligible_loso_speakers,
        "loso_test_utterance_count": int(
            (split_df["loso_target_speaker"] != "TRAIN_ONLY").sum()
        ),
        "random_split_leaked_scene_clusters": random_scene_leakage,
        "random_split_leaked_speaker_groups": random_speaker_leakage,
        "verifications": verification_records,
    }

    with open(splits_dir / "splits_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Generated and verified frozen splits in %s (LOSO speakers=%d, N_loso=%d)",
        splits_dir,
        len(eligible_loso_speakers),
        summary["loso_test_utterance_count"],
    )
    return summary

