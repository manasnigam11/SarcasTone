# SarcasTone — Reproducibility & Team Module Ownership Guide (`REPRODUCIBILITY.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  

---

## 1. End-to-End Deterministic Reproduction Instructions

Every result, table, figure, and paper artifact in this repository is 100% reproducible from the raw dataset and pinned Python environment.

### Step 1: Environment Setup
Ensure Python 3.13 (64-bit) is active and install the pinned dependencies:
```powershell
python -m pip install -r requirements.txt
```

### Step 2: Verify Raw Media & Metadata (Phase 1–3)
Place the raw MUStARD video folders (`mmsd_raw_data/utterances_final/*.mp4` and `mmsd_raw_data/context_final/*_c.mp4`) in the workspace root (`main/`). Then execute:
```powershell
python scripts/run_phase1_to_3_data_and_splits.py
```
- **Outputs produced:**
  - `data/processed/audio_16k/*.wav` (690 16 kHz mono PCM WAV files)
  - `data/processed/mustard_master.parquet` & `.csv` (690 instances with linked MUStARD++ subgroup metadata)
  - `data/splits/frozen_splits.parquet`, `.csv`, & `splits_manifest.json` (Random, Scene-Grouped, Speaker-Grouped, and LOSO splits with zero-overlap verification)
  - `results/figures/eda/*.png` & `results/figures/fig2_*.png`–`fig5_*.png`

### Step 3: Extract Explicit Prosodic, Text, Sentiment & Learned Audio Representations (Phase 4–6)
```powershell
python scripts/run_phase4_to_6_features.py
```
- **Outputs produced:**
  - `data/features/explicit_prosody.parquet` & `.csv` (41 interpretable prosodic features across 4 groups + quality flags)
  - `data/features/egemaps_v02.parquet` (88 openSMILE eGeMAPSv02 functionals)
  - `data/features/roberta_target_embeddings.parquet` (768-d frozen `roberta-base` target embeddings)
  - `data/features/roberta_context_target_embeddings.parquet` (768-d frozen `roberta-base` context+target embeddings)
  - `data/features/text_sentiment_polarity.parquet` & `.csv` (zero-shot `cardiffnlp/twitter-roberta-base-sentiment-latest` polarity scores)
  - `data/features/wav2vec2_base_embeddings.parquet` (768-d frozen `facebook/wav2vec2-base` representations)

### Step 4: Run All Cross-Validation, LOSO, Control, Ablation & Disagreement Experiments (Phase 7–11)
```powershell
python scripts/run_phase7_to_11_experiments.py
```
- **Outputs produced:**
  - `experiments/experiment_registry.csv` & `.parquet` (200+ logged experiment runs across 25 model configurations, 4 protocols, and 3 seeds)
  - `results/predictions/oof_predictions_*.csv` & `oof_probabilities_scene_s42.csv` (exact out-of-fold predictions for every instance)
  - `results/tables/paired_bootstrap_tests.json` (5,000-iteration group-aware paired bootstrap 95% confidence intervals)
  - `results/tables/disagreement_and_subgroups.json` (Phase 9 Disagreement & Sarcasm-Type subgroup analysis)
  - `results/tables/loso_per_speaker_breakdown.json` (Per-speaker LOSO generalization scores across 13 major speakers)

### Step 5: Run Automated Unit & Leakage Test Suite (Phase 12)
```powershell
python -m pytest tests/test_pipeline.py -v
```

### Step 6: Generate Publication Figures, IEEE Research Paper (`.docx` & `.pdf`), and Final Presentation (Phase 13–17)
```powershell
python scripts/run_phase13_visualizations.py
python scripts/generate_paper_and_presentation.py
```

---

## 2. Artifact Traceability Map

| Reported Artifact | Source Script | Saved Data / Prediction Artifact | Verification Test |
| :--- | :--- | :--- | :--- |
| Dataset & Split Statistics | `scripts/run_phase1_to_3_data_and_splits.py` | `data/splits/splits_manifest.json` | `test_master_dataset_and_labels`, `test_split_integrity_and_zero_leakage` |
| Explicit Prosody (41d) | `src/prosody.py` | `data/features/explicit_prosody.parquet` | `test_prosody_feature_groups_and_missing_f0_handling` |
| Main Model Comparison | `scripts/run_phase7_to_11_experiments.py` | `experiments/experiment_registry.csv`, `results/tables/scene_grouped_summary_table.csv` | `test_in_fold_tabular_and_fusion_no_nan_and_determinism` |
| Paired Bootstrap CIs | `src/evaluation.py` | `results/tables/paired_bootstrap_tests.json` | `test_metrics_and_paired_bootstrap_ci` |
| Disagreement & Subgroups | `src/disagreement.py` | `results/tables/disagreement_and_subgroups.json` | Verified in `run_phase7_to_11_experiments.py` |
| Figures 1–11 | `scripts/run_phase13_visualizations.py` | `results/figures/fig1_*.png`–`fig11_*.png` | Verified on disk |

---

## 3. Team Organization, Module Ownership & Review Checklist (Section 19)

In accordance with Section 19 of the Master Directive and the Milestone 1 team roster (Bennett University), module ownership and independent methodology review responsibilities are structured as follows:

| Role ID | Role Title | Team Member (from Milestone 1 Roster) | Owned Modules & Artifacts | Independent Review Conducted By |
| :---: | :--- | :--- | :--- | :--- |
| **Member 1** | **Data & Evaluation Lead** | Manas Nigam (`S24CSEU2391`) | `src/data.py`, `src/splits.py`, `src/evaluation.py`, `docs/GATE_0_DATASET_AUDIT.md`, `docs/LEAKAGE_AUDIT.md`, `data/splits/` | Member 4 (Fusion & Analysis Lead) |
| **Member 2** | **Text Lead** | Sri Charan Reddy (`S24CSEU0563`) | `src/representations.py` (Text & Sentiment), `src/models.py` (TF-IDF, RoBERTa target & context models) | Member 1 (Data & Evaluation Lead) |
| **Member 3** | **Audio Lead** | Sanskar Gupta (`S24CSEU0464`) | `src/prosody.py` (4-Group Explicit Prosody, missing-$F_0$ flags, eGeMAPSv02), `src/representations.py` (`wav2vec 2.0`) | Member 2 (Text Lead) |
| **Member 4** | **Fusion & Analysis Lead** | Taanya Raawat (`S24CSEU0587`) | `src/models.py` (Early/Late/Trimodal Fusion, Shuffled & Speaker Controls, Ablations), `src/disagreement.py`, `scripts/run_phase7_to_11_experiments.py` | Member 3 (Audio Lead) |

---

## 4. Ten-Week Project Execution Timeline (Recalculated from Project Start: Nov 5, 2026)

| Week | Date Window | Milestone / Gate | Deliverables Completed |
| :---: | :--- | :---: | :--- |
| **Week 1** | Nov 5 – Nov 11, 2026 | **Gate 0** | Environment audit (`ENVIRONMENT_REPORT.md`), dataset/media validation, license verification, `GATE_0_DATASET_AUDIT.md`. |
| **Week 2** | Nov 12 – Nov 18, 2026 | Split Freeze | 16 kHz WAV extraction, connected-component scene graph (449 clusters), speaker disambiguation, `LEAKAGE_AUDIT.md`, frozen splits (`data/splits/`). |
| **Week 3** | Nov 19 – Nov 25, 2026 | **Gate 1** | TF-IDF & frozen `roberta-base` baselines, context-on vs. context-off experiments. |
| **Week 4** | Nov 26 – Dec 2, 2026 | Audio Baselines | 41-feature explicit prosody suite (4 groups + unvoiced $F_0$ flags), `eGeMAPSv02`, frozen `wav2vec 2.0` speech embeddings. |
| **Week 5** | Dec 3 – Dec 9, 2026 | **Gate 2** | Early & Late multimodal fusion, Shuffled-Audio controls, Speaker-ID diagnostic baseline. |
| **Week 6** | Dec 10 – Dec 16, 2026 | Generalization & Ablation | Leave-One-Speaker-Out (13 speakers) & Speaker-Grouped CV, 4-group prosodic cue leave-one-out and single-group ablations. |
| **Week 7** | Dec 17 – Dec 23, 2026 | **Gate 3 (Freeze)** | Label-free Text–Vocal Disagreement analysis, MUStARD++ Sarcasm-Type (`PRO`, `ILL`, `EMB`) subgroup analysis, 5,000-resample paired bootstrap CIs, error analysis. |
| **Week 8** | Dec 24 – Dec 30, 2026 | Figures & Tables | Publication figures (`fig1`–`fig11`), summary tables (`results/tables/`), initial paper draft. |
| **Week 9** | Dec 31, 2026 – Jan 6, 2027 | **Gate 4** | Automated `pytest` suite (`tests/test_pipeline.py`), full code/leakage audit, reproducibility verification. |
| **Week 10** | Jan 7 – Jan 13, 2027 | **Gate 5** | Final IEEE-style Research Paper (`.docx` & `.pdf`), Final Presentation (`.pptx`), GitHub-ready repository release. |

