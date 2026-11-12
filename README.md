# SarcasTone

**A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection**

**Official Repository:** [https://github.com/manasnigam11/SarcasTone](https://github.com/manasnigam11/SarcasTone)

---

## Overview

**SarcasTone** is a reproducible, leakage-aware experimental framework investigating multimodal sarcasm detection across text semantics, interpretable speech prosody, and self-supervised audio representations (`wav2vec 2.0`) on the **MUStARD** benchmark ($N = 690$ audiovisual utterances, enriched with $N = 688$ held-out **MUStARD++** subgroup annotations).

### Core Research Question
> *"Under scene-grouped and leave-one-speaker-out evaluation on MUStARD, which explicit prosodic cues add information beyond frozen text and learned audio representations, and does this change when text sentiment and vocal delivery disagree?"*

### Key Contributions
1. **Leakage Discovery & Protocol Design:** We demonstrate empirically that standard random 5-fold splits leak **81.7%–88.0% of multi-clip dialogue scenes** (116–125 of 142 multi-clip scene clusters) and **20 of 23 speaker groups** across train and test sets. Furthermore, recurring sitcom characters exhibit extreme sarcasm base-rate skew (e.g., Dorothy `97.4%` sarcastic, Chandler `75.9%` vs. Joey `8.3%`, Ross `13.2%`). We construct and freeze four evaluation protocols: **Stratified Random 5-Fold CV**, **Scene-Grouped 5-Fold CV** (449 disjoint connected-component scene clusters), **Speaker-Grouped 5-Fold CV**, and **Leave-One-Speaker-Out (LOSO)** across 13 major characters.
2. **4-Group Interpretable Prosodic Suite & Missing-$F_0$ Handling:** We extract 41 interpretable prosodic features using **Praat (`parselmouth`)** and **`librosa`** organized into four phonetic groups (**Group 1: $F_0$ Pitch**, **Group 2: Energy/Intensity**, **Group 3: Timing/Pauses**, **Group 4: Voice Quality/HNR/Jitter/Shimmer**) with explicit quality flags (`f0_unvoiced_flag`, `short_clip_flag`) and training-fold-only imputation, alongside an 88-dimensional **`openSMILE` `eGeMAPSv02`** benchmark and frozen **`facebook/wav2vec2-base`** speech embeddings.
3. **Diagnostic Controls & Modality Ablation:** We pair all unimodal and multimodal models with a **Speaker-ID Only Diagnostic Control** (quantifying the exact ceiling of speaker memorization), **Shuffled-Audio Controls** (testing genuine cross-modal alignment), and systematic **Leave-One-Prosody-Group-Out** and **Single-Group** ablations.
4. **Label-Free Text–Vocal Disagreement & Sarcasm-Type Subgroup Analysis:** Using zero-shot conversational sentiment polarity (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and an unsupervised acoustic delivery index, we evaluate whether multimodal fusion provides larger gains when literal text sentiment and vocal delivery disagree, and analyze sensitivity across **Propositional (`PRO`)**, **Illocutionary (`ILL`)**, and **Embedded (`EMB`)** sarcasm types.

---

## Repository Structure

```text
main/
├── configs/
│   └── default.yaml                           # Central deterministic experiment configuration
├── data/
│   ├── metadata/                              # Official MUStARD & MUStARD++ metadata + MIT License
│   ├── processed/                             # Canonical master table & 16 kHz mono WAVs (gitignored)
│   ├── splits/                                # Frozen leakage-verified split tables & manifest
│   └── features/                              # Cached Prosody, eGeMAPSv02, RoBERTa, Sentiment, wav2vec2
├── docs/
│   ├── ORIGINAL_MUSTARD_DATASET_ANALYSIS.md   # Initial raw media inspection report
│   ├── ENVIRONMENT_REPORT.md                  # Phase 0 hardware & software audit
│   ├── GATE_0_DATASET_AUDIT.md                # Phase 1 Gate 0 dataset & fallback audit
│   ├── DECISIONS.md                           # Autonomous technical & scientific decision log
│   ├── RISK_REGISTER.md                       # Risk management & mitigation log
│   ├── REFERENCES.md                          # Verified academic bibliography
│   ├── LEAKAGE_AUDIT.md                       # Phase 2–3 scene & speaker leakage audit
│   ├── METHODOLOGY.md                         # Complete technical methodology specification
│   ├── EXPERIMENT_LOG.md                      # Detailed log of all executed experiments & gates
│   ├── RESULTS_SUMMARY.md                     # Final technical report, tables & hypothesis verdicts
│   ├── REPRODUCIBILITY.md                     # Reproduction instructions & team ownership matrix
│   ├── SarcasTone_IEEE_Research_Paper.docx    # Final IEEE-style Research Paper (Word)
│   ├── SarcasTone_IEEE_Research_Paper.pdf     # Final IEEE-style Research Paper (PDF)
│   └── SarcasTone_Final_Presentation.pptx     # Final Project Presentation Slides
├── experiments/
│   └── experiment_registry.csv                # Complete registry of all 150+ evaluated runs
├── results/
│   ├── figures/                               # All 11 required publication figures + EDA plots
│   ├── tables/                                # CSV & JSON statistical and subgroup tables
│   └── predictions/                           # Out-of-fold predictions & probabilities for every model
├── scripts/
│   ├── run_phase1_to_3_data_and_splits.py     # Dataset assembly, WAV extraction, splits & EDA
│   ├── run_phase4_to_6_features.py            # Prosody, eGeMAPS, RoBERTa, Sentiment & wav2vec2 extraction
│   ├── run_phase7_to_11_experiments.py        # Full training, CV, LOSO, controls, ablations & bootstrap CIs
│   ├── run_phase13_visualizations.py          # Publication figure & summary table generation
│   └── generate_paper_and_presentation.py     # IEEE Word/PDF paper & PPTX presentation builder
├── src/
│   ├── __init__.py
│   ├── data.py                                # MP4 parser, PyAV 16 kHz decoder, scene graph builder
│   ├── splits.py                              # Split generator & zero-overlap verification assertions
│   ├── prosody.py                             # 4-group Praat/librosa prosody & openSMILE extractor
│   ├── representations.py                     # Frozen RoBERTa, zero-shot sentiment & wav2vec 2.0
│   ├── models.py                              # Leakage-safe fold trainers, early/late fusion & controls
│   ├── evaluation.py                          # Metrics & group-aware paired bootstrap CI engine
│   └── disagreement.py                        # Text–vocal disagreement & MUStARD++ subgroup evaluator
├── tests/
│   └── test_pipeline.py                       # Automated pytest unit & leakage verification suite
├── requirements.txt                           # Pinned Python dependencies
└── .gitignore                                 # Excludes raw media (*.mp4, *.wav) from Git
```

---

## Dataset Download & Attribution Instructions

Per copyright and licensing policy, raw sitcom video clips (`*.mp4`) and extracted audio (`*.wav`) are excluded from Git version control.
1. **Original MUStARD Metadata & Code (MIT License):** `https://github.com/soujanyaporia/MUStARD` (Castro et al., ACL 2019). Place `mmsd_raw_data/utterances_final/*.mp4` and `mmsd_raw_data/context_final/*_c.mp4` in the project root.
2. **MUStARD++ Subgroup Annotations:** `https://github.com/cfiltnlp/MUStARD_Plus_Plus` (Ray et al., LREC 2022). Automatically fetched to `data/metadata/mustard_plus_plus_text.csv`.

---

## Quick Start & Full Reproduction

```powershell
# 1. Install pinned dependencies
python -m pip install -r requirements.txt

# 2. Build master dataset, extract 16 kHz mono WAVs, freeze splits & generate EDA plots
python scripts/run_phase1_to_3_data_and_splits.py

# 3. Extract explicit prosody, eGeMAPSv02, frozen RoBERTa, sentiment & wav2vec 2.0 features
python scripts/run_phase4_to_6_features.py

# 4. Run all models, protocols, controls, ablations, disagreement analyses & bootstrap tests
python scripts/run_phase7_to_11_experiments.py

# 5. Run automated unit & leakage tests
python -m pytest tests/test_pipeline.py -v

# 6. Generate publication figures, IEEE research paper (.docx & .pdf), and presentation (.pptx)
python scripts/run_phase13_visualizations.py
python scripts/generate_paper_and_presentation.py
```

