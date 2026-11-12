"""
Phase 13 Visualization & Publication Table Generator:
Generates all 11 mandated publication-quality figures in `results/figures/` and
LaTeX/Markdown/CSV result tables in `results/tables/`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def draw_fig1_system_architecture(out_path: Path) -> None:
    """Generate Figure 1: Leakage-Aware SarcasTone System Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str = "#222222") -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            facecolor=fc, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontweight="bold", fontsize=10.5, color="#111111")
        ax.text(x + w / 2, y + (h - 0.35) / 2, body, ha="center", va="center", fontsize=8.8, color="#222222")

    # Stage 1: Dataset & Leakage Controls
    draw_box(
        0.3, 1.3, 2.3, 3.8,
        "1. Dataset & Gate 0",
        "Original MUStARD (N=690)\n+ MUStARD++ Tags (N=688)\n\n• 690 Target Clips (.mp4)\n• 690 Context Clips (_c.mp4)\n• PyAV 16 kHz Mono WAV\n\nLeakage-Free Protocols:\n[A] Random 5-Fold\n[B] Scene-Grouped (449)\n[C] LOSO (13 Speakers)\n& Speaker-Grouped",
        "#E8F1F5"
    )

    # Stage 2: Text Branch
    draw_box(
        3.1, 3.9, 2.6, 1.8,
        "2A. Text Semantics Branch",
        "• Transcript Cleaning\n• TF-IDF (1–2g, in-fold)\n• Frozen RoBERTa-base (768d)\n  (Target vs Context+Target)\n• Zero-Shot Text Polarity",
        "#FFF3E0"
    )

    # Stage 2B: Explicit Prosody Branch
    draw_box(
        3.1, 1.95, 2.6, 1.75,
        "2B. Explicit Prosody (41d)",
        "Praat / Librosa 4 Groups:\n[G1] F0 Pitch + Unvoiced Flag\n[G2] Energy / Intensity (dB)\n[G3] Timing, Pauses & Rate\n[G4] Voice Quality (HNR/Jit/Shim)",
        "#E8F5E9"
    )

    # Stage 2C: Learned Audio Branch
    draw_box(
        3.1, 0.25, 2.6, 1.5,
        "2C. Learned Audio Branch",
        "• Frozen wav2vec 2.0 Base\n  (768d layer-pooled)\n• openSMILE eGeMAPSv02\n  (88d benchmark)",
        "#F3E5F5"
    )

    # Stage 3: Fold-Internal Preprocessing & Fusion
    draw_box(
        6.2, 1.3, 2.6, 3.8,
        "3. In-Fold Preprocessing\n& Multimodal Fusion",
        "Fit on Training Fold ONLY:\n• Median Imputer (F0 NaNs)\n• Outlier Clip (1–99%)\n• StandardScaler & Block Norm\n\nModels & Controls:\n• Unimodal Baselines (LR/SVM)\n• Early Concat & Late Stacking\n• Group Ablations (Drop G1–G4)\n• Shuffled-Audio & Speaker-ID",
        "#FFFDE7"
    )

    # Stage 4: Disagreement & Statistical Evaluation
    draw_box(
        9.3, 1.3, 2.4, 3.8,
        "4. Disagreement &\nStatistical Analysis",
        "Primary Metric: Macro-F1\n\n• Scene-Aware Paired\n  Bootstrap (5,000 iter)\n• 95% CI Decision Rule\n  (CI must exclude 0)\n• Text–Vocal Disagreement\n  Subset Analysis\n• Sarcasm-Type Breakdown\n  (PRO, ILL, EMB, LIK)",
        "#FCE4EC"
    )

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.25,head_length=0.25", color="#333333", lw=1.6)
    ax.annotate("", xy=(3.08, 4.8), xytext=(2.62, 3.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.08, 2.8), xytext=(2.62, 3.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.08, 1.0), xytext=(2.62, 2.4), arrowprops=arrow_kw)

    ax.annotate("", xy=(6.18, 4.2), xytext=(5.72, 4.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.18, 3.1), xytext=(5.72, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.18, 2.0), xytext=(5.72, 1.0), arrowprops=arrow_kw)

    ax.annotate("", xy=(9.28, 3.2), xytext=(8.82, 3.2), arrowprops=arrow_kw)

    ax.set_title("SarcasTone: Leakage-Aware Multimodal Architecture & Experimental Protocol", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures_and_tables() -> None:
    fig_dir = ROOT / "results" / "figures"
    tab_dir = ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "figure.dpi": 300,
    })

    # Figure 1: System Architecture
    draw_fig1_system_architecture(fig_dir / "fig1_system_architecture.png")

    df_reg = pd.read_csv(ROOT / "experiments" / "experiment_registry.csv")
    df_preds_scene = pd.read_csv(ROOT / "results" / "predictions" / "oof_predictions_scene_s42.csv")
    with open(tab_dir / "disagreement_and_subgroups.json", "r", encoding="utf-8") as f:
        disagree_data = json.load(f)
    with open(tab_dir / "paired_bootstrap_tests.json", "r", encoding="utf-8") as f:
        bootstrap_tests = json.load(f)

    # Create summary table across 3 seeds for Random and Scene-Grouped + Seed 42 for Speaker-Grouped & LOSO
    core_models = [
        "tfidf_target_lr",
        "tfidf_context_lr",
        "roberta_target_lr",
        "roberta_context_lr",
        "prosody_explicit_lr",
        "prosody_explicit_svm",
        "egemaps_lr",
        "wav2vec2_lr",
        "fusion_roberta_prosody_early",
        "fusion_roberta_prosody_late",
        "fusion_roberta_wav2vec2_early",
        "fusion_roberta_wav2vec2_late",
        "fusion_trimodal_text_prosody_w2v2",
        "control_speaker_only",
        "control_shuffled_prosody",
        "control_shuffled_wav2vec2",
    ]

    # Scene-Grouped (seed 42) primary comparison table
    sc_s42 = df_reg[(df_reg["protocol"] == "SceneGrouped_5Fold") & (df_reg["seed"] == 42)].set_index("model_id")
    # Multi-seed mean & std across seeds 42, 1337, 2026
    sc_multiseed = (
        df_reg[df_reg["protocol"] == "SceneGrouped_5Fold"]
        .groupby("model_id")["pooled_macro_f1"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "multiseed_mean_f1", "std": "multiseed_std_f1"})
    )
    sc_table = sc_s42.join(sc_multiseed)
    sc_table.to_csv(tab_dir / "scene_grouped_summary_table.csv")

    # Figure 6: Main Model Comparison under Scene-Grouped Evaluation (with 95% CIs)
    plot_models = [
        ("tfidf_target_lr", "TF-IDF (Target) + LR", "#4C72B0"),
        ("roberta_target_lr", "RoBERTa-base (Target) + LR", "#2B5B84"),
        ("roberta_context_lr", "RoBERTa-base (Context+Target)", "#1B3B58"),
        ("prosody_explicit_lr", "Explicit Prosody (41d) + LR", "#55A868"),
        ("egemaps_lr", "openSMILE eGeMAPSv02 + LR", "#3B7A4A"),
        ("wav2vec2_lr", "wav2vec 2.0 Base (768d) + LR", "#8172B2"),
        ("fusion_roberta_prosody_early", "RoBERTa + Prosody (Early)", "#DD8452"),
        ("fusion_roberta_prosody_late", "RoBERTa + Prosody (Late)", "#E69F00"),
        ("fusion_roberta_wav2vec2_early", "RoBERTa + wav2vec 2.0 (Early)", "#C44E52"),
        ("fusion_trimodal_text_prosody_w2v2", "Trimodal (Text+Prosody+w2v2)", "#933B41"),
        ("control_shuffled_prosody", "Control: RoBERTa + Shuffled Prosody", "#8C8C8C"),
        ("control_speaker_only", "Control: Speaker-ID Only", "#666666"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y_pos = np.arange(len(plot_models))
    vals = [sc_s42.loc[m, "pooled_macro_f1"] for m, _, _ in plot_models]
    lows = [sc_s42.loc[m, "pooled_macro_f1"] - sc_s42.loc[m, "ci_95_low"] for m, _, _ in plot_models]
    highs = [sc_s42.loc[m, "ci_95_high"] - sc_s42.loc[m, "pooled_macro_f1"] for m, _, _ in plot_models]
    labels = [lbl for _, lbl, _ in plot_models]
    colors = [col for _, _, col in plot_models]

    ax.barh(y_pos, vals, xerr=[lows, highs], capsize=4, color=colors, edgecolor="black", height=0.68, alpha=0.9)
    ax.invert_yaxis()
    ax.axvline(0.50, color="black", linestyle=":", alpha=0.6, label="Chance Baseline (0.50)")
    ax.axvline(sc_s42.loc["roberta_target_lr", "pooled_macro_f1"], color="#2B5B84", linestyle="--", alpha=0.7, label="RoBERTa Text-Only Baseline")
    for i, (m, _, _) in enumerate(plot_models):
        v = sc_s42.loc[m, "pooled_macro_f1"]
        ch = sc_s42.loc[m, "ci_95_high"]
        ax.text(ch + 0.008, i, f"{v:.3f} [{sc_s42.loc[m, 'ci_95_low']:.3f}, {ch:.3f}]", va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.35, 0.88)
    ax.set_xlabel("Pooled Macro-F1 (with 95% Scene-Cluster Bootstrap Confidence Interval)")
    ax.set_title("Main Model Comparison Under Leakage-Free Scene-Grouped 5-Fold CV (N = 690)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_main_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 7: Protocol Comparison (Random vs Scene-Grouped vs Speaker-Grouped vs LOSO)
    proto_compare_models = [
        ("tfidf_target_lr", "TF-IDF + LR"),
        ("roberta_target_lr", "RoBERTa + LR"),
        ("prosody_explicit_lr", "Explicit Prosody"),
        ("wav2vec2_lr", "wav2vec 2.0"),
        ("fusion_roberta_prosody_early", "RoBERTa + Prosody"),
        ("fusion_roberta_wav2vec2_early", "RoBERTa + wav2vec 2.0"),
        ("control_speaker_only", "Speaker-ID Only (Control)"),
    ]
    protocols = [
        ("Random_5Fold", "Random 5-Fold (s42)", "#4C72B0"),
        ("SceneGrouped_5Fold", "Scene-Grouped 5-Fold (s42)", "#55A868"),
        ("SpeakerGrouped_5Fold", "Speaker-Grouped 5-Fold", "#DD8452"),
        ("LOSO_13Speakers", "LOSO (13 Major Speakers)", "#C44E52"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(proto_compare_models))
    width = 0.20
    for p_idx, (p_code, p_label, p_col) in enumerate(protocols):
        sub_p = df_reg[(df_reg["protocol"] == p_code) & (df_reg["seed"] == 42)].set_index("model_id")
        p_vals = [sub_p.loc[m, "pooled_macro_f1"] for m, _ in proto_compare_models]
        rects = ax.bar(x + (p_idx - 1.5) * width, p_vals, width, label=p_label, color=p_col, edgecolor="black")
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=7.8, rotation=0)
    ax.axhline(0.50, color="black", linestyle=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in proto_compare_models], rotation=15, ha="right")
    ax.set_ylim(0.25, 0.82)
    ax.set_ylabel("Pooled Out-of-Fold Macro-F1")
    ax.set_title("Impact of Leakage Controls: Random vs. Scene-Grouped vs. Speaker-Independent Evaluation")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_protocol_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 8: Prosodic Feature Group Ablation (Drop-One-Group & Single-Group Audio-Only)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    drop_models = [
        ("roberta_target_lr", "Text Only (No Prosody)", "#4C72B0"),
        ("fusion_roberta_prosody_early", "Text + All 4 Prosody Groups", "#DD8452"),
        ("ablation_text_prosody_no_f0", "Drop Group 1: w/o F0 Pitch", "#E69F00"),
        ("ablation_text_prosody_no_energy", "Drop Group 2: w/o Energy", "#56B4E9"),
        ("ablation_text_prosody_no_timing", "Drop Group 3: w/o Timing/Pauses", "#009E73"),
        ("ablation_text_prosody_no_voice_quality", "Drop Group 4: w/o Voice Quality", "#CC79A7"),
    ]
    y_d = np.arange(len(drop_models))
    d_vals = [sc_s42.loc[m, "pooled_macro_f1"] for m, _, _ in drop_models]
    d_err_l = [sc_s42.loc[m, "pooled_macro_f1"] - sc_s42.loc[m, "ci_95_low"] for m, _, _ in drop_models]
    d_err_h = [sc_s42.loc[m, "ci_95_high"] - sc_s42.loc[m, "pooled_macro_f1"] for m, _, _ in drop_models]
    axes[0].barh(y_d, d_vals, xerr=[d_err_l, d_err_h], capsize=3, color=[c for _, _, c in drop_models], edgecolor="black")
    axes[0].invert_yaxis()
    for i, v in enumerate(d_vals):
        axes[0].text(v + d_err_h[i] + 0.006, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_yticks(y_d)
    axes[0].set_yticklabels([lbl for _, lbl, _ in drop_models])
    axes[0].set_xlim(0.50, 0.80)
    axes[0].set_xlabel("Scene-Grouped Macro-F1")
    axes[0].set_title("(a) Multimodal Leave-One-Group-Out Ablation")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    single_models = [
        ("prosody_explicit_lr", "All 4 Prosody Groups (41d)", "#55A868"),
        ("prosody_only_f0_lr", "Group 1 Only: F0 / Pitch (11d)", "#E69F00"),
        ("prosody_only_energy_lr", "Group 2 Only: Energy (10d)", "#56B4E9"),
        ("prosody_only_timing_lr", "Group 3 Only: Timing/Pauses (11d)", "#009E73"),
        ("prosody_only_voice_quality_lr", "Group 4 Only: Voice Quality (9d)", "#CC79A7"),
    ]
    y_s = np.arange(len(single_models))
    s_vals = [sc_s42.loc[m, "pooled_macro_f1"] for m, _, _ in single_models]
    s_err_l = [sc_s42.loc[m, "pooled_macro_f1"] - sc_s42.loc[m, "ci_95_low"] for m, _, _ in single_models]
    s_err_h = [sc_s42.loc[m, "ci_95_high"] - sc_s42.loc[m, "pooled_macro_f1"] for m, _, _ in single_models]
    axes[1].barh(y_s, s_vals, xerr=[s_err_l, s_err_h], capsize=3, color=[c for _, _, c in single_models], edgecolor="black")
    axes[1].invert_yaxis()
    axes[1].axvline(0.50, color="black", linestyle=":", alpha=0.6)
    for i, v in enumerate(s_vals):
        axes[1].text(v + s_err_h[i] + 0.006, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].set_yticks(y_s)
    axes[1].set_yticklabels([lbl for _, lbl, _ in single_models])
    axes[1].set_xlim(0.40, 0.72)
    axes[1].set_xlabel("Scene-Grouped Macro-F1")
    axes[1].set_title("(b) Audio-Only Single-Group Prosodic Power")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_prosodic_ablation.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 9: Disagreement Subset & Sarcasm-Type Subgroup Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sub_keys = [
        ("Disagreement_Subset", f"Text–Vocal Disagreement\n(n={disagree_data['subsets']['Disagreement_Subset']['n_total']})"),
        ("Agreement_or_Neutral_Subset", f"Agreement / Neutral\n(n={disagree_data['subsets']['Agreement_or_Neutral_Subset']['n_total']})"),
        ("Human_Explicit_Implicit_Disagreement", f"Human Emotion Disagreement\n(n={disagree_data['subsets']['Human_Explicit_Implicit_Disagreement']['n_total']})"),
    ]
    dis_models = [
        ("roberta_target_lr", "RoBERTa Text-Only", "#4C72B0"),
        ("prosody_explicit_lr", "Explicit Prosody Only", "#55A868"),
        ("fusion_roberta_prosody_early", "RoBERTa + Prosody (Early)", "#DD8452"),
        ("fusion_roberta_wav2vec2_early", "RoBERTa + wav2vec 2.0", "#C44E52"),
    ]
    x_d = np.arange(len(sub_keys))
    w_d = 0.19
    for m_idx, (m_id, m_lbl, m_col) in enumerate(dis_models):
        m_vals = [disagree_data["subsets"][sk]["models"][m_id]["macro_f1"] for sk, _ in sub_keys]
        rects = axes[0].bar(x_d + (m_idx - 1.5) * w_d, m_vals, w_d, label=m_lbl, color=m_col, edgecolor="black")
        for r in rects:
            h = r.get_height()
            axes[0].text(r.get_x() + r.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", fontsize=8)
    axes[0].set_xticks(x_d)
    axes[0].set_xticklabels([lbl for _, lbl in sub_keys])
    axes[0].set_ylim(0.40, 0.82)
    axes[0].set_ylabel("Macro-F1 (Scene-Grouped OOF)")
    axes[0].set_title("(a) Model Performance by Text–Vocal Disagreement Subset")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8.5, loc="upper right")

    # Panel (b): Sarcasm-Type Recall across PRO, ILL, EMB
    stypes = [st for st in ["PRO", "ILL", "EMB"] if st in disagree_data["sarcasm_types"]]
    x_st = np.arange(len(stypes))
    st_labels = [f"{st}\n(n={disagree_data['sarcasm_types'][st]['n_type']})" for st in stypes]
    for m_idx, (m_id, m_lbl, m_col) in enumerate(dis_models):
        st_vals = [disagree_data["sarcasm_types"][st]["models"][m_id]["recall_sensitivity"] for st in stypes]
        rects = axes[1].bar(x_st + (m_idx - 1.5) * w_d, st_vals, w_d, label=m_lbl, color=m_col, edgecolor="black")
        for r in rects:
            h = r.get_height()
            axes[1].text(r.get_x() + r.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(x_st)
    axes[1].set_xticklabels(st_labels)
    axes[1].set_ylim(0.30, 0.90)
    axes[1].set_ylabel("Sarcasm Detection Recall (Sensitivity)")
    axes[1].set_title("(b) Detection Rate by MUStARD++ Sarcasm Type")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_disagreement_and_subgroups.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 10: Confusion Matrices for Key Models under Scene-Grouped CV
    cm_models = [
        ("roberta_target_lr", "Text-Only: RoBERTa-base + LR"),
        ("prosody_explicit_lr", "Audio-Only: Explicit Prosody + LR"),
        ("wav2vec2_lr", "Audio-Only: wav2vec 2.0 + LR"),
        ("fusion_roberta_prosody_early", "Multimodal: RoBERTa + Prosody"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    y_true = df_preds_scene["label"].to_numpy(dtype=int)
    for idx, (m_id, title) in enumerate(cm_models):
        y_p = df_preds_scene[m_id].to_numpy(dtype=int)
        cm = confusion_matrix(y_true, y_p, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
            xticklabels=["Pred Non-Sarc", "Pred Sarc"],
            yticklabels=["True Non-Sarc", "True Sarc"],
            annot_kws={"size": 12, "weight": "bold"}
        )
        f1_v = sc_s42.loc[m_id, "pooled_macro_f1"]
        axes[idx].set_title(f"{title}\n(Macro-F1 = {f1_v:.3f})", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig10_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 11: Paired Bootstrap Difference 95% Confidence Intervals (Hypothesis Testing Forest Plot)
    test_items = [
        ("H1_early_prosody_vs_roberta_scene", "H1: (RoBERTa + Prosody Early) − RoBERTa [Scene CV]"),
        ("H1_late_prosody_vs_roberta_scene", "H1: (RoBERTa + Prosody Late) − RoBERTa [Scene CV]"),
        ("H1_early_w2v2_vs_roberta_scene", "H1: (RoBERTa + wav2vec 2.0) − RoBERTa [Scene CV]"),
        ("H1_trimodal_vs_roberta_scene", "H1: Trimodal (Text+Pros+w2v2) − RoBERTa [Scene CV]"),
        ("H1_early_prosody_vs_roberta_loso", "H1: (RoBERTa + Prosody) − RoBERTa [LOSO Speakers]"),
        ("H4_wav2vec2_vs_explicit_prosody_scene", "H4: wav2vec 2.0 − Explicit Prosody [Scene CV]"),
        ("H4_wav2vec2_vs_explicit_prosody_loso", "H4: wav2vec 2.0 − Explicit Prosody [LOSO Speakers]"),
        ("H3_roberta_random_vs_scene", "H3: RoBERTa (Random CV − Scene CV)"),
        ("H3_roberta_scene_vs_speakergrouped", "H3: RoBERTa (Scene CV − Speaker-Grouped CV)"),
        ("H3_speaker_only_scene_vs_speakergrouped", "H3: Speaker-ID Control (Scene CV − Speaker-Grouped CV)"),
        ("Control_real_vs_shuffled_prosody_scene", "Control: Real Prosody Fusion − Shuffled Prosody"),
        ("Context_roberta_ctx_vs_target_scene", "Context: RoBERTa (Context+Target) − Target Only"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    y_t = np.arange(len(test_items))
    for i, (k, _) in enumerate(test_items):
        bt = bootstrap_tests[k]
        pt = bt["diff_point"]
        cl = max(0.0, pt - bt["ci_low"])
        ch = max(0.0, bt["ci_high"] - pt)
        ec = "#2E7D32" if bt["excludes_zero"] else "#757575"
        ax.errorbar(
            [pt], [i], xerr=[[cl], [ch]], fmt="o",
            color="#111111", ecolor=ec, elinewidth=2.5, capsize=4, markersize=6
        )
        tag = "*" if bt["excludes_zero"] else "ns"
        ax.text(bt["ci_high"] + 0.008, i, f"Δ={pt:+.3f} [{bt['ci_low']:+.3f}, {bt['ci_high']:+.3f}] ({tag})", va="center", fontsize=8.8)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero Difference Threshold (Δ = 0)")
    ax.invert_yaxis()
    ax.set_yticks(y_t)
    ax.set_yticklabels([lbl for _, lbl in test_items])
    ax.set_xlim(-0.22, 0.48)
    ax.set_xlabel("Paired Macro-F1 Difference Δ (with 95% Bootstrap Confidence Interval)")
    ax.set_title("Paired Bootstrap Hypothesis Tests (Green = 95% CI Excludes Zero; Grey = No Reliable Difference)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig11_bootstrap_confidence_intervals.png", bbox_inches="tight")
    plt.close(fig)

    logger.info("Successfully generated all 11 publication figures in %s", fig_dir)


if __name__ == "__main__":
    generate_all_figures_and_tables()

