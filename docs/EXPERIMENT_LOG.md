# SarcasTone — Master Experiment Log & Quality Gate Verification (`EXPERIMENT_LOG.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  
**Registry File:** [`experiments/experiment_registry.csv`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/experiments/experiment_registry.csv) (200 Completed Runs Across 25 Models, 4 Protocols, and 3 Random Seeds)  

---

## 1. Autonomous Quality Gate Audit Log

| Gate | Gate Name | Verification Evidence & Artifacts | Status |
| :---: | :--- | :--- | :---: |
| **GATE 0** | **Dataset Usable** | Verified 690 Original MUStARD video pairs (`utterances_final/` + `context_final/`), 100% stream integrity, exact 345/345 class balance (`sarcasm_data.json`), and 688/690 MUStARD++ subgroup metadata links (`0` label discrepancies). Documented in [`docs/GATE_0_DATASET_AUDIT.md`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/docs/GATE_0_DATASET_AUDIT.md). | **PASSED** |
| **GATE 1** | **Text Baselines Valid** | Implemented and evaluated inside-fold `TF-IDF + LR`, `TF-IDF + LinearSVC`, frozen `RoBERTa-base (Target Only) + LR`, and context-aware `TF-IDF` / `RoBERTa-base (Context + Target) + LR` across all 4 protocols (`EXP_0001`–`EXP_0040`). | **PASSED** |
| **GATE 2** | **Main Experiment Complete** | Evaluated all core Text-only, Explicit Prosody (41d Praat/librosa), `openSMILE` `eGeMAPSv02` (88d), frozen `wav2vec 2.0 Base` (768d), Early Concatenation Fusion, Late Stacking Fusion, Trimodal Fusion, Speaker-ID Control, and Shuffled-Audio Controls under frozen splits (`EXP_0001`–`EXP_0136`). | **PASSED** |
| **GATE 3** | **Analysis Complete (Experiments Frozen)** | Completed Leave-One-Prosody-Group-Out and Single-Group ablations (`EXP_0137`–`EXP_0200`), label-free Text–Vocal Disagreement analysis (`results/tables/disagreement_and_subgroups.json`), MUStARD++ Sarcasm-Type (`PRO`, `ILL`, `EMB`, `LIK`) sensitivity breakdown, 5,000-iteration scene-aware paired bootstrap CIs (`results/tables/paired_bootstrap_tests.json`), and qualitative error analysis. | **PASSED** |
| **GATE 4** | **Research Validation** | Executed automated `pytest` unit and leakage verification suite (`tests/test_pipeline.py`: `7 passed in 15.81s`), confirming zero group overlap, deterministic fold training, and exact metric/CI computation. | **PASSED** |
| **GATE 5** | **Final Delivery** | Generated all 11 publication figures (`results/figures/fig1_*.png`–`fig11_*.png`), complete documentation (`docs/*.md`), IEEE Research Paper (`.docx` & `.pdf`), and Final Presentation (`.pptx`). | **PASSED** |

---

## 2. Complete Summary of Executed Experiments (`EXP_0001`–`EXP_0200`)

All 200 individual experiment runs (25 model configurations $\times$ 8 protocol-seed combinations: `Random_5Fold` seeds `42, 1337, 2026`; `SceneGrouped_5Fold` seeds `42, 1337, 2026`; `SpeakerGrouped_5Fold` seed `42`; `LOSO_13Speakers` seed `42`) are permanently logged in [`experiments/experiment_registry.csv`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/experiments/experiment_registry.csv).

### 2.1 Primary Benchmark Results — Scene-Grouped 5-Fold CV ($N = 690$, 449 Disjoint Scene Clusters)

| Exp ID (s42) | Model ID | Model Family | Seed 42 Pooled Macro-F1 | 95% Scene-Bootstrap CI | Accuracy | Macro Prec | Macro Rec | 3-Seed Mean $\pm$ SD Macro-F1 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP_0004` | `tfidf_target_lr` | Text-Only | `0.6029` | `[0.5676, 0.6370]` | `0.6029` | `0.6029` | `0.6029` | `0.5961 ± 0.0066` |
| `EXP_0012` | `tfidf_target_svm` | Text-Only | `0.5927` | `[0.5569, 0.6264]` | `0.5928` | `0.5928` | `0.5928` | `0.5870 ± 0.0058` |
| `EXP_0020` | `tfidf_context_lr` | Text-Only (Context) | `0.6478` | `[0.6127, 0.6820]` | `0.6493` | `0.6517` | `0.6493` | `0.6347 ± 0.0162` |
| `EXP_0028` | `roberta_target_lr` | Text-Only | `0.6331` | `[0.5946, 0.6712]` | `0.6333` | `0.6337` | `0.6333` | `0.6384 ± 0.0050` |
| `EXP_0036` | `roberta_context_lr` | Text-Only (Context) | `0.6536` | `[0.6168, 0.6910]` | `0.6536` | `0.6537` | `0.6536` | `0.6527 ± 0.0030` |
| `EXP_0044` | `prosody_explicit_lr` | Audio-Only (Prosody) | `0.6809` | `[0.6439, 0.7177]` | `0.6812` | `0.6817` | `0.6812` | `0.6803 ± 0.0079` |
| `EXP_0052` | `prosody_explicit_svm` | Audio-Only (Prosody) | `0.6708` | `[0.6331, 0.7069]` | `0.6710` | `0.6715` | `0.6710` | `0.6809 ± 0.0150` |
| `EXP_0060` | `egemaps_lr` | Audio-Only (eGeMAPS) | `0.6536` | `[0.6160, 0.6899]` | `0.6536` | `0.6536` | `0.6536` | `0.6568 ± 0.0029` |
| `EXP_0068` | `wav2vec2_lr` | Audio-Only (Learned) | `0.7144` | `[0.6792, 0.7491]` | `0.7145` | `0.7149` | `0.7145` | `0.7025 ± 0.0113` |
| `EXP_0076` | `fusion_roberta_prosody_early` | Multimodal (Text+Prosody) | `0.6840` | `[0.6465, 0.7217]` | `0.6841` | `0.6843` | `0.6841` | `0.6833 ± 0.0039` |
| `EXP_0084` | `fusion_roberta_prosody_late` | Multimodal (Text+Prosody) | `0.6808` | `[0.6444, 0.7175]` | `0.6812` | `0.6819` | `0.6812` | `0.6905 ± 0.0121` |
| `EXP_0092` | `fusion_roberta_wav2vec2_early` | Multimodal (Text+w2v2) | `0.7143` | `[0.6745, 0.7530]` | `0.7145` | `0.7150` | `0.7145` | `0.7156 ± 0.0024` |
| `EXP_0100` | `fusion_roberta_wav2vec2_late` | Multimodal (Text+w2v2) | `0.7101` | `[0.6734, 0.7448]` | `0.7101` | `0.7103` | `0.7101` | `0.7085 ± 0.0016` |
| `EXP_0108` | **`fusion_trimodal_text_prosody_w2v2`** | **Multimodal (Trimodal)** | **`0.7217`** | **`[0.6840, 0.7593]`** | **`0.7217`** | **`0.7219`** | **`0.7217`** | **`0.7162 ± 0.0061`** |
| `EXP_0116` | `control_speaker_only` | Control (Diagnostic) | `0.6912` | `[0.6523, 0.7290]` | `0.6913` | `0.6917` | `0.6913` | `0.6805 ± 0.0094` |
| `EXP_0124` | `control_shuffled_prosody` | Control (Shuffled Audio) | `0.6517` | `[0.6149, 0.6877]` | `0.6522` | `0.6530` | `0.6522` | `0.6433 ± 0.0096` |
| `EXP_0132` | `control_shuffled_wav2vec2` | Control (Shuffled Audio) | `0.6520` | `[0.6140, 0.6899]` | `0.6522` | `0.6524` | `0.6522` | `0.6419 ± 0.0101` |
| `EXP_0140` | `ablation_text_prosody_no_f0` | Ablation (Drop G1) | `0.6913` | `[0.6529, 0.7302]` | `0.6913` | `0.6913` | `0.6913` | `0.6863 ± 0.0053` |
| `EXP_0148` | `ablation_text_prosody_no_energy` | Ablation (Drop G2) | `0.6707` | `[0.6322, 0.7093]` | `0.6710` | `0.6717` | `0.6710` | `0.6766 ± 0.0063` |
| `EXP_0156` | `ablation_text_prosody_no_timing` | Ablation (Drop G3) | `0.6794` | `[0.6434, 0.7180]` | `0.6797` | `0.6804` | `0.6797` | `0.6853 ± 0.0059` |
| `EXP_0164` | `ablation_text_prosody_no_voice_quality` | Ablation (Drop G4) | `0.6911` | `[0.6540, 0.7289]` | `0.6913` | `0.6919` | `0.6913` | `0.6847 ± 0.0066` |
| `EXP_0172` | `prosody_only_f0_lr` | Single-Group Audio (G1) | `0.6194` | `[0.5817, 0.6587]` | `0.6203` | `0.6215` | `0.6203` | `0.6180 ± 0.0014` |
| `EXP_0180` | `prosody_only_energy_lr` | Single-Group Audio (G2) | `0.5985` | `[0.5596, 0.6362]` | `0.5986` | `0.5986` | `0.5986` | `0.5961 ± 0.0096` |
| `EXP_0188` | `prosody_only_timing_lr` | Single-Group Audio (G3) | `0.6314` | `[0.5915, 0.6726]` | `0.6319` | `0.6326` | `0.6319` | `0.6358 ± 0.0072` |
| `EXP_0196` | `prosody_only_voice_quality_lr` | Single-Group Audio (G4) | `0.6782` | `[0.6409, 0.7142]` | `0.6783` | `0.6784` | `0.6783` | `0.6689 ± 0.0104` |

---

## 3. Optimization & Engineering Adjustments Logged

1. **Vectorized Group-Aware Paired Bootstrap (`src/evaluation.py`):**
   - *Initial Issue:* Calling `sklearn.metrics.f1_score` inside a Python loop 5,000 times per model across 200 experiments took ~45 seconds per model.
   - *Resolution:* Exploited the additive property of per-instance and per-scene-cluster confusion matrix counts `(tp_g, fp_g, fn_g, tn_g)`, reducing 5,000 group-aware resamples to vectorized NumPy array indexing (`<0.05 s` per 5,000-iteration test) with identical mathematical outputs (`tests/test_pipeline.py` verified).
2. **Matplotlib Per-Point Errorbar Coloring (`scripts/run_phase13_visualizations.py`):**
   - *Initial Issue:* Passing a list of hex color strings to `ecolor=` in a single `ax.errorbar()` call raised `ValueError: RGBA sequence should have length 3 or 4` in Matplotlib `3.10.5`.
   - *Resolution:* Plotted each hypothesis test row with its own `ax.errorbar()` call, producing `results/figures/fig11_bootstrap_confidence_intervals.png` cleanly.

