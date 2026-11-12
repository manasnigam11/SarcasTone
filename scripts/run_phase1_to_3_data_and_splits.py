"""
Phase 1–3 Execution Script: Build Master Dataset, Extract 16 kHz Mono WAVs,
Generate & Verify Frozen Splits, Run Leakage Audit, and Generate EDA Visualizations.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_master_dataset
from src.splits import generate_and_freeze_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_eda_figures(df: pd.DataFrame, splits_summary: dict, out_dir: Path) -> None:
    """Generate publication-quality EDA visualizations in `results/figures/eda/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 300,
    })

    # 1. Class Distribution & Show Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    class_counts = df["label"].map({0: "Non-Sarcastic (0)", 1: "Sarcastic (1)"}).value_counts()
    axes[0].bar(class_counts.index, class_counts.values, color=["#4C72B0", "#DD8452"], edgecolor="black", width=0.55)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 8, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 410)
    axes[0].set_ylabel("Number of Utterances")
    axes[0].set_title("(a) Overall Class Distribution (N = 690)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    show_ct = pd.crosstab(df["show"], df["label"].map({0: "Non-Sarcastic", 1: "Sarcastic"}))
    show_ct = show_ct.loc[df["show"].value_counts().index]
    show_ct.plot(kind="bar", stacked=False, ax=axes[1], color=["#4C72B0", "#DD8452"], edgecolor="black")
    axes[1].set_title("(b) Class Distribution by Television Show")
    axes[1].set_xlabel("Source Show")
    axes[1].set_ylabel("Number of Utterances")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(title="Class")
    plt.tight_layout()
    plt.savefig(out_dir / "class_and_show_distribution.png", bbox_inches="tight")
    plt.savefig(out_dir.parent / "fig2_dataset_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Speaker Distribution & Sarcasm Rate per Speaker
    fig, ax = plt.subplots(figsize=(11, 5.5))
    top_spks = df["speaker_group"].value_counts().head(14).index
    sub_spk = df[df["speaker_group"].isin(top_spks)]
    spk_ct = pd.crosstab(sub_spk["speaker_group"], sub_spk["label"].map({0: "Non-Sarcastic", 1: "Sarcastic"}))
    spk_ct = spk_ct.loc[top_spks]
    spk_ct.plot(kind="barh", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="black")
    ax.invert_yaxis()
    for i, spk in enumerate(top_spks):
        tot = spk_ct.loc[spk].sum()
        sarc_pct = spk_ct.loc[spk, "Sarcastic"] / tot * 100
        ax.text(tot + 2, i, f"n={tot} ({sarc_pct:.0f}% sarc)", va="center", fontsize=9.5)
    ax.set_xlim(0, 190)
    ax.set_xlabel("Number of Utterances")
    ax.set_ylabel("Speaker Group (Show_Speaker)")
    ax.set_title("Utterance Count and Sarcasm Base Rate Across Top 14 Speaker Groups")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "speaker_distribution.png", bbox_inches="tight")
    plt.savefig(out_dir.parent / "fig3_speaker_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Scene Cluster Size Distribution & Leakage Illustration
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    scene_sizes = df["scene_group_id"].value_counts()
    size_dist = scene_sizes.value_counts().sort_index()
    axes[0].bar(size_dist.index.astype(str), size_dist.values, color="#55A868", edgecolor="black")
    for i, (k, v) in enumerate(size_dist.items()):
        axes[0].text(i, v + 10, f"{v}", ha="center", fontweight="bold")
    axes[0].set_ylim(0, max(size_dist.values) * 1.15)
    axes[0].set_xlabel("Utterances per Connected Scene Cluster")
    axes[0].set_ylabel("Number of Scene Clusters")
    axes[0].set_title(f"(a) Scene Cluster Size Distribution ({len(scene_sizes)} Clusters)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Multi-clip scene sarcasm composition
    multi_scenes = scene_sizes[scene_sizes > 1].index
    sub_multi = df[df["scene_group_id"].isin(multi_scenes)]
    sc_ct = pd.crosstab(sub_multi["scene_group_id"], sub_multi["label"])
    sc_types = pd.Series(index=sc_ct.index, dtype=str)
    for idx, row in sc_ct.iterrows():
        if row.get(0, 0) > 0 and row.get(1, 0) > 0:
            sc_types[idx] = "Mixed Labels in Scene"
        elif row.get(1, 0) > 0:
            sc_types[idx] = "All Sarcastic"
        else:
            sc_types[idx] = "All Non-Sarcastic"
    type_counts = sc_types.value_counts()
    axes[1].bar(type_counts.index, type_counts.values, color=["#C44E52", "#DD8452", "#4C72B0"], edgecolor="black", width=0.55)
    for i, v in enumerate(type_counts.values):
        axes[1].text(i, v + 1, f"{v}", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(type_counts.values) * 1.2)
    axes[1].set_ylabel("Number of Multi-Clip Scene Clusters")
    axes[1].set_title(f"(b) Label Composition of Multi-Clip Scenes (n={len(multi_scenes)})")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "scene_distribution.png", bbox_inches="tight")
    plt.savefig(out_dir.parent / "fig4_scene_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Audio Duration Distribution by Class
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for lbl, name, col in [(0, "Non-Sarcastic", "#4C72B0"), (1, "Sarcastic", "#DD8452")]:
        durs = df[df["label"] == lbl]["utt_duration_sec"]
        axes[0].hist(durs, bins=25, alpha=0.6, label=f"{name} (mean={durs.mean():.2f}s)", color=col, edgecolor="black")
    axes[0].axvline(1.0, color="red", linestyle="--", label="Short Clip Threshold (1.0s)")
    axes[0].set_xlabel("Target Utterance Audio Duration (seconds)")
    axes[0].set_ylabel("Number of Utterances")
    axes[0].set_title("(a) Target Utterance Duration by Class")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].scatter(df["ctx_duration_sec"], df["utt_duration_sec"], c=df["label"], cmap="coolwarm", alpha=0.65, edgecolors="k", linewidth=0.3)
    axes[1].set_xlabel("Preceding Dialogue Context Duration (seconds)")
    axes[1].set_ylabel("Target Utterance Duration (seconds)")
    axes[1].set_title("(b) Target vs. Context Audio Duration")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "audio_duration_distribution.png", bbox_inches="tight")
    plt.savefig(out_dir.parent / "fig5_audio_duration_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Transcript Length & Context Turn Distribution
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for lbl, name, col in [(0, "Non-Sarcastic", "#4C72B0"), (1, "Sarcastic", "#DD8452")]:
        wc = df[df["label"] == lbl]["utterance_word_count"]
        axes[0].hist(wc, bins=25, alpha=0.6, label=f"{name} (mean={wc.mean():.1f} words)", color=col, edgecolor="black")
    axes[0].set_xlabel("Target Utterance Length (Words)")
    axes[0].set_ylabel("Number of Utterances")
    axes[0].set_title("(a) Target Transcript Word Count Distribution")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    ctx_ct = pd.crosstab(df["context_num_turns"], df["label"].map({0: "Non-Sarcastic", 1: "Sarcastic"}))
    ctx_ct.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"], edgecolor="black")
    axes[1].set_xlabel("Number of Preceding Context Turns")
    axes[1].set_ylabel("Number of Utterances")
    axes[1].set_title("(b) Preceding Dialogue Context Turn Count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "transcript_length_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Metadata & MUStARD++ Annotation Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    corr_cols = [
        "label",
        "utt_duration_sec",
        "ctx_duration_sec",
        "utterance_word_count",
        "context_num_turns",
        "audio_rms",
        "human_valence",
        "human_arousal",
    ]
    corr_df = df[corr_cols].rename(columns={
        "label": "Sarcasm Label",
        "utt_duration_sec": "Utt Duration (s)",
        "ctx_duration_sec": "Context Duration (s)",
        "utterance_word_count": "Word Count",
        "context_num_turns": "Context Turns",
        "audio_rms": "Audio RMS Energy",
        "human_valence": "Human Valence (MPP)",
        "human_arousal": "Human Arousal (MPP)",
    }).corr()
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", vmin=-0.8, vmax=0.8, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix of Dataset Metadata & Held-Out Annotations")
    plt.tight_layout()
    plt.savefig(out_dir / "metadata_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_master_dataset()
    splits_summary = generate_and_freeze_splits(df)
    generate_eda_figures(df, splits_summary, ROOT / "results" / "figures" / "eda")
    print("Master dataset shape:", df.shape)
    print("Splits summary:", json.dumps(splits_summary, indent=2))


if __name__ == "__main__":
    main()

