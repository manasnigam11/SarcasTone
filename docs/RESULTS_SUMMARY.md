# SarcasTone — Final Technical Report & Experimental Results Summary (`RESULTS_SUMMARY.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  
**Dataset:** Original MUStARD ($N = 690$ audiovisual utterances; $N = 688$ linked with held-out MUStARD++ subgroup annotations)  
**Primary Metric:** Out-of-Fold Pooled Macro-F1 with 5,000-Iteration Scene-Aware Paired Bootstrap 95% Confidence Intervals  

---

## 1. Executive Summary of Findings

### What Was Implemented
1. **Leakage-Verified Evaluation Infrastructure:** Built a connected-component scene graph isolating **449 disjoint dialogue scene clusters** across the 690 utterances (`data/splits/frozen_splits.parquet`) and evaluated 25 model configurations across four protocols: **Stratified Random 5-Fold CV**, **Scene-Grouped 5-Fold CV**, **Speaker-Grouped 5-Fold CV**, and **Leave-One-Speaker-Out (LOSO)** across 13 major characters ($N=614$).
2. **Multi-Granular Text, Prosody & Self-Supervised Speech Representations:** Extracted inside-fold `TF-IDF (1–2g)`, frozen `roberta-base` (Target Only and Context+Target), zero-shot conversational sentiment polarity (`cardiffnlp/twitter-roberta-base-sentiment-latest`), a **41-feature Interpretable Prosodic Suite** (`Praat/parselmouth` + `librosa` across 4 phonetic groups: $F_0$, Energy, Timing/Pauses, Voice Quality), 88-dimensional `openSMILE` `eGeMAPSv02` functionals, and frozen 768-dimensional `facebook/wav2vec2-base` representations.
3. **Diagnostic Controls & Subgroup Disagreement Analysis:** Evaluated Speaker-ID-Only diagnostic models, Shuffled-Audio controls, Leave-One-Prosody-Group-Out and Single-Group ablations, label-free Text–Vocal Disagreement subsets, and MUStARD++ Sarcasm-Type (`PRO`, `ILL`, `EMB`, `LIK`) sensitivity breakdowns.

---

## 2. Main Comparison Across All Four Evaluation Protocols

| Model Configuration | Family | Random 5-Fold Macro-F1 | Scene-Grouped 5-Fold Macro-F1 (95% CI) | Speaker-Grouped 5-Fold Macro-F1 | LOSO (13 Speakers, $N=614$) Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tfidf_target_lr` | Text-Only | `0.5985` | `0.6029` `[0.5676, 0.6370]` | `0.5603` | `0.5534` |
| `tfidf_context_lr` | Text (Context+Target) | `0.6501` | `0.6478` `[0.6127, 0.6820]` | `0.6186` | `0.5747` |
| `roberta_target_lr` | Text-Only | `0.6477` | `0.6331` `[0.5946, 0.6712]` | `0.5659` | `0.5945` |
| `roberta_context_lr` | Text (Context+Target) | `0.6521` | `0.6536` `[0.6168, 0.6910]` | `0.6108` | `0.6199` |
| `prosody_explicit_lr` | Audio-Only (41d Prosody) | `0.6940` | `0.6809` `[0.6439, 0.7177]` | `0.6536` | `0.6569` |
| `egemaps_lr` | Audio-Only (88d eGeMAPS) | `0.6725` | `0.6536` `[0.6160, 0.6899]` | `0.6346` | `0.6319` |
| `wav2vec2_lr` | Audio-Only (768d Learned) | `0.7057` | `0.7144` `[0.6792, 0.7491]` | `0.6637` | `0.6661` |
| `fusion_roberta_prosody_early` | Multimodal (Text+Prosody) | `0.6811` | `0.6840` `[0.6465, 0.7217]` | `0.6478` | `0.6654` |
| `fusion_roberta_prosody_late` | Multimodal (Text+Prosody) | `0.6953` | `0.6808` `[0.6444, 0.7175]` | `0.6110` | `0.6429` |
| `fusion_roberta_wav2vec2_early` | Multimodal (Text+w2v2) | `0.7114` | `0.7143` `[0.6745, 0.7530]` | `0.6665` | `0.6884` |
| `fusion_roberta_wav2vec2_late` | Multimodal (Text+w2v2) | **`0.7202`** | `0.7101` `[0.6734, 0.7448]` | `0.6652` | `0.6824` |
| **`fusion_trimodal_text_prosody_w2v2`** | **Multimodal (Text+Pros+w2v2)** | `0.7070` | **`0.7217` `[0.6840, 0.7593]`** | **`0.6695`** | **`0.6948`** |
| `control_speaker_only` | Control (Speaker-ID Only) | `0.6841` | `0.6912` `[0.6523, 0.7290]` | `0.5346` | `0.6022` *(0.3598 spk-mean)* |
| `control_shuffled_prosody` | Control (Shuffled Audio) | `0.6500` | `0.6517` `[0.6149, 0.6877]` | `0.6184` | `0.6295` |
| `control_shuffled_wav2vec2` | Control (Shuffled Audio) | `0.6157` | `0.6520` `[0.6140, 0.6899]` | `0.6087` | `0.6106` |

> **Note on Trimodal vs. `wav2vec 2.0` (`IMP-01`) and LOSO `control_speaker_only` (`IMP-02`):**
> 1. **Trimodal vs. `wav2vec 2.0`:** While `fusion_trimodal_text_prosody_w2v2` achieves the highest point-estimate Macro-F1 across Scene-Grouped (`0.7217`), Speaker-Grouped (`0.6695`), and LOSO (`0.6948`) evaluations and achieves a statistically supported improvement over `RoBERTa` text-only ($\Delta = +0.0886$, 95% CI `[+0.0466, +0.1311]`, $p < 0.0002$), its direct paired bootstrap comparison against unimodal `wav2vec2_lr` yields $\Delta = +0.0073$ (95% CI `[-0.0232, +0.0384]`, $p = 0.6472$) in Scene-Grouped CV and $\Delta = +0.0287$ (95% CI `[-0.0053, +0.0623]`, $p = 0.1016$) in LOSO. Because both 95% CIs span zero, the gain of Trimodal Fusion over unimodal `wav2vec 2.0` is reported strictly as a numerical point-estimate improvement rather than a statistically supported difference.
> 2. **LOSO `control_speaker_only` (`0.6022` Pooled vs. `0.3598` Per-Speaker Mean):** Under LOSO, `OneHotEncoder(handle_unknown="ignore")` maps the held-out test speaker to an all-zero vector, so `control_speaker_only` outputs a single constant class for every utterance of that speaker, yielding a per-speaker mean Macro-F1 (`fold_mean_macro_f1`) of **`0.3598`**. Specifically, it predicts constant `0` on 12 of the 13 speakers and constant `1` on `FRIENDS_CHANDLER` ($n = 158$, 120 sarcastic / 38 non-sarcastic) because removing Chandler's 120 positive clips leaves the remaining training set with a negative majority, causing `class_weight="balanced"` to flip the learned intercept positive. When pooled across all 614 utterances, this single-speaker label-complement artifact produces `0.6022`, whereas `SpeakerGrouped_5Fold` (`0.5346` pooled, `0.3011` fold-mean) eliminates this effect.

---

## 3. Formal Hypothesis Verdicts (Paired Bootstrap 95% CI Decision Rule)

Under Section 14.2 of the Master Directive, a performance gain counts as statistically supported **only when the 95% paired bootstrap confidence interval of $\Delta = \text{Macro-F1}(A) - \text{Macro-F1}(B)$ excludes zero**.

| Hypothesis | Tested Comparison | Observed $\Delta$ Macro-F1 | 95% Paired Bootstrap CI | Two-Sided $p$-Value | Verdict |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **H1** | **Text + Explicit Prosody vs. Text-Only (`roberta_target_lr`)** | | | | **SUPPORTED** |
| — H1a | Early Fusion (`RoBERTa + Prosody`) vs. `RoBERTa` (Scene-Grouped CV) | **`+0.0509`** | **`[+0.0149, +0.0873]`** | `p = 0.0064` | **Statistically Supported ($\text{CI} > 0$)** |
| — H1b | Late Fusion (`RoBERTa + Prosody`) vs. `RoBERTa` (Scene-Grouped CV) | **`+0.0477`** | **`[+0.0110, +0.0839]`** | `p = 0.0096` | **Statistically Supported ($\text{CI} > 0$)** |
| — H1c | Early Fusion (`RoBERTa + Prosody`) vs. `RoBERTa` (LOSO 13 Speakers) | **`+0.0709`** | **`[+0.0308, +0.1125]`** | `p = 0.0008` | **Statistically Supported ($\text{CI} > 0$)** |
| — H1d | Trimodal (`RoBERTa + Prosody + wav2vec2`) vs. `RoBERTa` (Scene CV) | **`+0.0886`** | **`[+0.0466, +0.1311]`** | `p < 0.0002` | **Statistically Supported ($\text{CI} > 0$)** |
| **H2** | **Is the Audio Gain Larger in the Text–Vocal Disagreement Subset?** | | | | **PARTIALLY SUPPORTED / NUANCED** |
| — H2a | Unimodal `Explicit Prosody` vs. `RoBERTa` inside Disagreement Subset ($n=200$) vs. Agreement Subset ($n=490$) | Disagreement: **`+0.0872`** (`0.7148` vs `0.6276`)<br>Agreement: **`+0.0324`** (`0.6671` vs `0.6347`) | — | — | **Supported for Unimodal Audio & Late/Trimodal Fusion on NegText+AnimVocal ($n=112$, $\Delta=+0.1459$, $\text{CI}=[+0.0482, +0.2407]$)** |
| — H2b | Early Concat (`RoBERTa + Prosody`) vs. `RoBERTa` on Full Disagreement Subset ($n=200$) vs. Agreement ($n=490$) | Disagreement: `+0.0421`<br>Agreement: `+0.0551` | Disagreement: `[-0.0275, +0.1112]`<br>Agreement: `[+0.0127, +0.0981]` | `p = 0.236` vs `p = 0.011` | **No reliable difference for Early Prosody Fusion on the 200-sample subset (`95% CI` spans 0); Late `w2v2` & Trimodal Fusion ARE statistically supported ($\Delta=+0.0973, \text{CI}=[+0.0243, +0.1718]$).** |
| **H3** | **Does Random Evaluation Produce Higher Scores Than Grouped or Speaker-Independent Evaluation?** | | | | **SUPPORTED FOR SPEAKER GROUPING; NOT SIGNIFICANT FOR SCENE GROUPING ALONE** |
| — H3a | `RoBERTa` Random 5-Fold (`0.6477`) vs. Scene-Grouped 5-Fold (`0.6331`) | `+0.0146` | `[-0.0160, +0.0449]` | `p = 0.3440` | **No reliable difference between Random and Scene-Grouped CV alone.** |
| — H3b | `RoBERTa` Scene-Grouped (`0.6331`) vs. Speaker-Grouped (`0.5659`) | **`+0.0672`** | **`[+0.0306, +0.1040]`** | `p = 0.0004` | **Statistically Supported Drop under Speaker-Independent CV ($\text{CI} > 0$).** |
| — H3c | `Speaker-ID Control` Scene-Grouped (`0.6912`) vs. Speaker-Grouped (`0.5346`) | **`+0.1565`** | **`[+0.1105, +0.2044]`** | `p < 0.0002` | **Statistically Supported Collapse of Speaker Shortcut ($\Delta = +15.65\text{ pp}$).** |
| **H4** | **Do Learned Audio Representations (`wav2vec 2.0`) Outperform Explicit Prosodic Features Alone?** | | | | **INCONCLUSIVE / NO RELIABLE DIFFERENCE** |
| — H4a | `wav2vec 2.0` (`0.7144`) vs. `Explicit Prosody` (`0.6809`) [Scene-Grouped CV] | `+0.0334` | `[-0.0016, +0.0673]` | `p = 0.0684` | **No reliable difference was observed (95% CI slightly overlaps zero at `-0.0016`).** |
| — H4b | `wav2vec 2.0` (`0.6661`) vs. `Explicit Prosody` (`0.6569`) [LOSO 13 Speakers] | `+0.0092` | `[-0.0305, +0.0504]` | `p = 0.6404` | **No reliable difference was observed under unseen speakers.** |

---

## 4. Key Insights from Control, Ablation, Disagreement & Error Analyses

### 4.1 Diagnostic Controls (Phase 8A & 8B)
1. **The Speaker-ID Shortcut (`control_speaker_only`):** Under Random (`0.6841`) and Scene-Grouped (`0.6912`) splits, a model given **nothing except the one-hot speaker identifier** outperforms `RoBERTa-base` text (`0.6331`)! However, when evaluated under **Speaker-Grouped 5-Fold CV**, `control_speaker_only` collapses by **$-15.65\text{ percentage points}$** to `0.5346` (near random guessing), whereas **Explicit Prosody (`0.6536`)**, **`wav2vec 2.0` (`0.6637`)**, and **Trimodal Fusion (`0.6695` in Speaker-Grouped; `0.6948` in LOSO)** retain strong predictive power on unseen speakers.
2. **Shuffled-Audio Controls (`control_shuffled_prosody`, `control_shuffled_wav2vec2`):** Permuting the audio rows across utterances drops `RoBERTa + Prosody` from `0.6840` to `0.6517` ($\Delta = +0.0323, \text{95\% CI} = [+0.0045, +0.0604], p = 0.0224$) and drops `RoBERTa + wav2vec 2.0` from `0.7143` to `0.6520` ($\Delta = +0.0623, \text{95\% CI} = [+0.0219, +0.1047], p = 0.0040$). Because both 95% CIs exclude zero, the multimodal improvement is verified to arise from **genuine utterance-level text–audio alignment**, not mere regularization noise.

### 4.2 Prosodic Feature Group Ablations (Phase 8C)
- **Single-Group Predictive Power (Scene-Grouped / LOSO):**
  - **Group 4 — Voice Quality & Spectral Balance (`hnr_mean_db`, `jitter`, `shimmer`, `spectral_centroid`):** Strongest standalone prosodic group at **`0.6782`** (Scene CV) and **`0.6573`** (LOSO). Sarcastic utterances in MUStARD exhibit lower harmonicity (`HNR = 4.66 dB` vs. `6.20 dB` for non-sarcastic) indicating strained, creaky, or deadpan delivery.
  - **Group 3 — Timing, Rhythm & Pauses (`utt_duration_sec`, `speech_ratio`, `articulation_rate_sps`):** Second strongest standalone group at **`0.6314`** (Scene CV) and **`0.6457`** (LOSO). Sarcastic utterances are systematically longer (`5.85 s` vs. `4.57 s`) and articulated more slowly (`3.95` vs. `4.54` syllables/sec).
  - **Group 1 — Fundamental Frequency ($F_0$ Pitch):** Achieves **`0.6194`** (Scene CV) and **`0.6093`** (LOSO); sarcastic speech shows lower mean $F_0$ (`192.98 Hz` vs. `217.20 Hz`), consistent with Cheang & Pell (2008).
  - **Group 2 — Energy & Intensity:** Achieves **`0.5985`** (Scene CV) and **`0.5862`** (LOSO) alone, **but** in Leave-One-Group-Out multimodal ablation (`ablation_text_prosody_no_energy`), dropping Energy causes the largest drop in multimodal F1 (from `0.6840` down to `0.6707`), showing that vocal intensity modulation provides complementary cross-modal signal when combined with text.

### 4.3 Sarcasm-Type Subgroup Analysis ($N = 688$ MUStARD++ Linked Scenes)
- **Propositional Sarcasm (`PRO`, $n = 196$):** Because propositional sarcasm relies on non-sentiment-bearing propositions that require contextual or prosodic framing, `RoBERTa` text-only detects only **62.8%** (`123/196`), whereas adding Explicit Prosody boosts recall to **70.4%** (`138/196`) and `RoBERTa + wav2vec 2.0` reaches **73.5%** (`144/196`).
- **Illocutionary Sarcasm (`ILL`, $n = 100$):** Sincere lexical expressions contradicted by non-verbal prosody; `wav2vec 2.0` achieves **73.0%** recall (`73/100`) vs. **68.0%** for `RoBERTa` text-only.
- **Embedded Sarcasm (`EMB`, $n = 46$, *exploratory $n < 50$*):** Multimodal `RoBERTa + wav2vec 2.0` detects **84.8%** (`39/46`) vs. **73.9%** (`34/46`) for text-only.

### 4.4 Qualitative Error Analysis (`results/predictions/oof_predictions_scene_s42.csv`)
- **84 / 690 instances (12.2%)** where `RoBERTa` text-only failed were **corrected** by Multimodal Prosody & Trimodal Fusion:
  - Example [`1_3707`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/mmsd_raw_data/utterances_final/1_3707.mp4) (*BBT*, Leonard): *"Oh, I'm so glad you asked it like that. You."* — Literal text sentiment is strongly positive ($S_{\text{text}} = +0.96$), fooling `RoBERTa` into predicting non-sarcastic (`0`). However, Leonard's low pitch ($F_0 = 137.2\text{ Hz}$) and flat/negative vocal delivery ($S_{\text{vocal}} = -0.53$) enable both `RoBERTa + Prosody` and `Trimodal` models to correctly classify it as sarcastic (`1`).
  - Example [`1_3259`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/mmsd_raw_data/utterances_final/1_3259.mp4) (*BBT*): *"No, I mean Princess Leia. Of course I mean Bernadette. She's a wonderful girl..."* — Positive literal sentiment ($S_{\text{text}} = +0.94$) misleads text-only models, whereas prosodic timing and intonation correctly recover the sarcastic label.
- **107 / 690 instances (15.5%)** failed across **both** unimodal text and multimodal models:
  - These errors fall into two primary categories: (1) **World-knowledge / Character-lore deadpan statements** (e.g., [`1_90`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/mmsd_raw_data/utterances_final/1_90.mp4), where Sheldon seriously calculates Lois Lane's free-fall acceleration in a pedantic tone that mimics his sarcastic cadence, causing false positives), and (2) **Visual-only / facial-expression sarcasm** where vocal delivery remains cheerful ($S_{\text{vocal}} > 0$) and sarcasm is signaled by an eye-roll or deadpan facial expression (e.g., [`1_1973`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/mmsd_raw_data/utterances_final/1_1973.mp4), Penny: *"I hope I'm a waitress at the Cheesecake Factory for my whole life."*).

---

## 5. Limitations & Future Work

1. **Sample Size ($N=690$) & Sitcom Domain:** Because the local raw archive contains the 690 Original MUStARD clips from four sitcoms, subgroup analyses on smaller slices (such as `EMB` with $n=46$ or `PosText_FlatNegVocal` with $n=88$) have wider confidence intervals ($\pm 9\text{–}10\text{ pp}$).
2. **Acted Broadcast Speech & Studio Laugh Tracks:** Sitcom actors project vocal prosody more theatrically than spontaneous everyday speakers, and residual studio audience laughter in sitcom clips can affect utterance-final acoustic statistics despite silence trimming.
3. **Absence of Visual Modality by Design:** SarcasTone intentionally isolates text and speech prosody; cases where sarcasm relies exclusively on visual gestures (eye-rolling, smirking) remain unresolvable from text and audio alone.
4. **Future Work:** Extend the leakage-aware evaluation pipeline to spontaneous conversational corpora, incorporate word-level forced alignment (prosodic stress on specific sentiment words), and evaluate cross-lingual prosodic transfer (e.g., on MuSaG).

