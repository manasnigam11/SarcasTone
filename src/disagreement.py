"""
Phase 9 Disagreement and Subgroup Analysis Module.

Investigates whether speech prosody or learned audio representations provide additional value
when literal text sentiment and vocal delivery disagree, without leaking sarcasm test labels.
Also evaluates subgroup performance across MUStARD++ Sarcasm Types (`PRO`, `ILL`, `EMB`, `LIK`)
and human explicit vs. implicit emotion annotations (`N = 688` linked scenes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation import (
    bootstrap_macro_f1_ci,
    compute_classification_metrics,
    paired_bootstrap_macro_f1_diff,
)


def compute_label_free_disagreement_annotations(
    df_master: pd.DataFrame,
    df_sent: pd.DataFrame,
    df_prosody: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct explicit, reproducible, strictly label-free Text Polarity and Vocal Delivery
    indices across the 690 utterances:
      1. `text_sentiment_score` = P(pos) - P(neg) from frozen zero-shot `twitter-roberta-base-sentiment-latest`.
      2. `vocal_delivery_score`: Unsupervised acoustic-prosodic delivery index combining
         standardized pitch level (`f0_mean_hz`), pitch expressiveness (`f0_std_hz`),
         vocal intensity (`intensity_mean_db`), and harmonicity (`hnr_mean_db`) — computed
         without any access to sarcasm labels.
         - Positive / Animated / Warm Vocal Delivery (`vocal_delivery_score > 0`): higher HNR,
           animated pitch & energy modulation.
         - Negative / Flat / Strained Vocal Delivery (`vocal_delivery_score <= 0`): flat/monotone
           or lower-harmonicity / harsh delivery.
      3. `disagreement_flag` (Rule 1 — Primary Acoustic-Text Disagreement):
         - `True` when:
           (a) Literal text sentiment is Positive (`text_sentiment_label == "positive"` or `text_sentiment_score > 0.15`)
               AND vocal delivery is Flat/Negative/Strained (`vocal_delivery_score < 0.0`), OR
           (b) Literal text sentiment is Negative (`text_sentiment_label == "negative"` or `text_sentiment_score < -0.15`)
               AND vocal delivery is Animated/Positive (`vocal_delivery_score > 0.20`).
      4. `human_emotion_disagreement_flag` (Rule 2 — Held-Out MUStARD++ Explicit vs. Implicit Emotion Disagreement):
         - `True` when MUStARD++ human `explicit_emotion` and `implicit_emotion` have opposing valence
           (e.g. Explicit `Happiness`/`Excitement` vs. Implicit `Ridicule`/`Frustration`/`Disgust`/`Anger`/`Sadness`).
    """
    merged = (
        df_master[["utterance_id", "label", "show", "speaker_group", "scene_group_id",
                   "in_mustard_pp", "sarcasm_type", "explicit_emotion", "implicit_emotion",
                   "human_valence", "human_arousal"]]
        .merge(df_sent, on="utterance_id", how="left")
        .merge(df_prosody, on="utterance_id", how="left")
    )

    def _zscore(series: pd.Series) -> np.ndarray:
        vals = series.to_numpy(dtype=np.float64)
        med = np.nanmedian(vals)
        vals = np.where(np.isfinite(vals), vals, med)
        iqr = max(np.percentile(vals, 75) - np.percentile(vals, 25), 1e-6)
        return np.clip((vals - med) / iqr, -3.0, 3.0)

    z_f0_std = _zscore(merged["f0_std_hz"])
    z_int_mean = _zscore(merged["intensity_mean_db"])
    z_hnr = _zscore(merged["hnr_mean_db"])
    z_art_rate = _zscore(merged["articulation_rate_sps"])

    # Unsupervised vocal animation/warmth index (strictly label-free)
    vocal_score = 0.35 * z_hnr + 0.30 * z_f0_std + 0.20 * z_int_mean + 0.15 * z_art_rate
    merged["vocal_delivery_score"] = vocal_score.astype(np.float32)

    t_score = merged["text_sentiment_score"].to_numpy()
    t_lbl = merged["text_sentiment_label"].astype(str).to_numpy()

    pos_text_neg_vocal = ((t_lbl == "positive") | (t_score >= 0.15)) & (vocal_score <= 0.0)
    neg_text_pos_vocal = ((t_lbl == "negative") | (t_score <= -0.15)) & (vocal_score >= 0.15)
    disagree_primary = pos_text_neg_vocal | neg_text_pos_vocal

    subtypes = np.where(
        pos_text_neg_vocal,
        "pos_text_flat_neg_vocal",
        np.where(neg_text_pos_vocal, "neg_text_animated_pos_vocal", "agreement_or_neutral"),
    )
    merged["disagreement_flag"] = disagree_primary.astype(bool)
    merged["disagreement_subtype"] = subtypes

    # Auxiliary human annotation disagreement (for descriptive check on N=688)
    pos_emotions = {"Happiness", "Excitement", "Surprise"}
    neg_emotions = {"Ridicule", "Frustration", "Disgust", "Anger", "Sadness", "Fear"}
    human_disagree = []
    for _, r in merged.iterrows():
        exp_e = str(r["explicit_emotion"])
        imp_e = str(r["implicit_emotion"])
        if (exp_e in pos_emotions and imp_e in neg_emotions) or (
            exp_e in neg_emotions and imp_e in pos_emotions
        ) or (exp_e == "Neutral" and imp_e in neg_emotions):
            human_disagree.append(True)
        else:
            human_disagree.append(False)
    merged["human_emotion_disagreement_flag"] = human_disagree

    return merged


def evaluate_disagreement_and_subgroups(
    df_disagree: pd.DataFrame,
    oof_predictions: Dict[str, np.ndarray],
    n_resamples: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate unimodal vs. multimodal models across:
      1. Text-Vocal Disagreement Subset vs. Agreement/Neutral Subset (Hypothesis H2)
      2. Disagreement Subtypes (`pos_text_flat_neg_vocal`, `neg_text_animated_pos_vocal`)
      3. MUStARD++ Sarcasm Types (`PRO`, `ILL`, `EMB`, `LIK` vs. Non-Sarcastic)
      4. Human Explicit-Implicit Emotion Disagreement Subset
    """
    y_true = df_disagree["label"].to_numpy(dtype=int)
    scenes = df_disagree["scene_group_id"].to_numpy()

    subsets = {
        "All_Instances": np.ones(len(df_disagree), dtype=bool),
        "Disagreement_Subset": df_disagree["disagreement_flag"].to_numpy(dtype=bool),
        "Agreement_or_Neutral_Subset": ~df_disagree["disagreement_flag"].to_numpy(dtype=bool),
        "Subtype_PosText_FlatNegVocal": (
            df_disagree["disagreement_subtype"] == "pos_text_flat_neg_vocal"
        ).to_numpy(dtype=bool),
        "Subtype_NegText_AnimPosVocal": (
            df_disagree["disagreement_subtype"] == "neg_text_animated_pos_vocal"
        ).to_numpy(dtype=bool),
        "Human_Explicit_Implicit_Disagreement": df_disagree[
            "human_emotion_disagreement_flag"
        ].to_numpy(dtype=bool),
    }

    subset_results: Dict[str, Any] = {}
    for sub_name, mask in subsets.items():
        n_sub = int(np.sum(mask))
        if n_sub == 0:
            continue
        y_sub = y_true[mask]
        sc_sub = scenes[mask]
        n_sarc = int(np.sum(y_sub == 1))
        n_non = int(np.sum(y_sub == 0))
        exploratory = n_sub < 50

        model_scores: Dict[str, Any] = {}
        for m_name, preds in oof_predictions.items():
            p_sub = preds[mask]
            m_metrics = compute_classification_metrics(y_sub, p_sub)
            ci = bootstrap_macro_f1_ci(
                y_sub, p_sub, groups=sc_sub, n_resamples=n_resamples, seed=seed
            )
            m_metrics["ci_low"] = ci["ci_low"]
            m_metrics["ci_high"] = ci["ci_high"]
            model_scores[m_name] = m_metrics

        # Paired comparisons inside this subset (Multimodal vs Text-Only)
        comparisons: Dict[str, Any] = {}
        if "roberta_target_lr" in oof_predictions:
            base_p = oof_predictions["roberta_target_lr"][mask]
            for cand in [
                "fusion_roberta_prosody_early",
                "fusion_roberta_prosody_late",
                "fusion_roberta_wav2vec2_early",
                "fusion_roberta_wav2vec2_late",
                "fusion_trimodal_text_prosody_w2v2",
            ]:
                if cand in oof_predictions:
                    comparisons[f"{cand}_vs_roberta_target_lr"] = paired_bootstrap_macro_f1_diff(
                        y_sub,
                        oof_predictions[cand][mask],
                        base_p,
                        groups=sc_sub,
                        n_resamples=n_resamples,
                        seed=seed,
                    )

        subset_results[sub_name] = {
            "n_total": n_sub,
            "n_sarcastic": n_sarc,
            "n_non_sarcastic": n_non,
            "sarcasm_rate": round(n_sarc / max(n_sub, 1), 4),
            "is_exploratory_under_50": bool(exploratory),
            "models": model_scores,
            "paired_comparisons": comparisons,
        }

    # Sarcasm-Type Recall / Detection Rate Analysis (since each sarcasm_type is 100% positive class,
    # the scientifically appropriate metric per sarcasm type is Sarcasm Recall / Sensitivity + CI,
    # alongside Macro-F1 when paired with all 344 non-sarcastic MUStARD++ instances).
    sarcasm_type_results: Dict[str, Any] = {}
    non_sarc_mask = (df_disagree["in_mustard_pp"].to_numpy(dtype=bool)) & (y_true == 0)
    for stype in ["PRO", "ILL", "EMB", "LIK"]:
        type_mask = (df_disagree["sarcasm_type"] == stype).to_numpy(dtype=bool)
        n_type = int(np.sum(type_mask))
        if n_type == 0:
            continue
        combined_mask = type_mask | non_sarc_mask
        per_model_type: Dict[str, Any] = {}
        for m_name, preds in oof_predictions.items():
            recall_on_type = float(np.mean(preds[type_mask] == 1))
            comb_f1 = compute_classification_metrics(
                y_true[combined_mask], preds[combined_mask]
            )["macro_f1"]
            per_model_type[m_name] = {
                "recall_sensitivity": round(recall_on_type, 4),
                "correct_count": int(np.sum(preds[type_mask] == 1)),
                "total_type_count": n_type,
                "macro_f1_vs_nonsarc": round(comb_f1, 4),
            }
        sarcasm_type_results[stype] = {
            "n_type": n_type,
            "is_exploratory_under_50": bool(n_type < 50),
            "models": per_model_type,
        }

    return {
        "subsets": subset_results,
        "sarcasm_types": sarcasm_type_results,
    }

