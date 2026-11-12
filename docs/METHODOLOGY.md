# SarcasTone — Methodology & Technical Specification (`METHODOLOGY.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  

---

## 1. Research Problem & Scope

Spoken sarcasm frequently relies on an incongruity between literal lexical semantics and vocal delivery. However, empirical progress in multimodal sarcasm detection on sitcom benchmarks (such as MUStARD and MUStARD++) is complicated by two structural forms of evaluation leakage:
1. **Scene & Context Overlap Leakage:** Multiple utterances extracted from the same continuous multi-turn dialogue scene share verbatim context sentences and acoustic room environments. Under standard random $K$-fold cross-validation, 81.7%–88.0% of multi-clip scenes in MUStARD are split across both training and test sets.
2. **Speaker & Show Base-Rate Confounding:** Recurring sitcom characters exhibit heavily skewed sarcasm base rates (e.g., Dorothy in *The Golden Girls* is 97.4% sarcastic; Chandler in *Friends* is 75.9% sarcastic; Joey is 8.3% sarcastic; Ross is 13.2% sarcastic). Because both voice quality/pitch and character catchphrases identify the speaker, models evaluated on speaker-overlapping splits can achieve artificially inflated scores by memorizing speaker identities rather than learning generalizable sarcasm cues.

**Core Research Question:**
> *"Under scene-grouped and leave-one-speaker-out evaluation on MUStARD, which explicit prosodic cues add information beyond frozen text and learned audio representations, and does this change when text sentiment and vocal delivery disagree?"*

---

## 2. Dataset & Evaluation Protocols

### 2.1 Canonical Dataset ($N = 690$, Enriched with $N = 688$ MUStARD++ Subgroup Tags)
- **Media:** 690 isolated target utterance clips (`mmsd_raw_data/utterances_final/<id>.mp4`) and 690 preceding context clips (`mmsd_raw_data/context_final/<id>_c.mp4`), decoded via `av` (PyAV `19.0.2`) to 16,000 Hz mono 16-bit PCM WAV (`data/processed/audio_16k/<id>.wav`).
- **Labels:** 345 sarcastic (`1`) and 345 non-sarcastic (`0`) instances from `sarcasm_data.json`.
- **Subgroup Metadata (`N = 688`):** Exact `SCENE` join with `mustard++_text.csv` provides held-out human annotations for **Sarcasm Type** (`PRO`: Propositional, `ILL`: Illocutionary, `EMB`: Embedded, `LIK`: Like-prefixed), **Explicit Emotion**, **Implicit Emotion**, **Valence** (1–9), and **Arousal** (1–9). These annotations are **never** used as model inputs.

### 2.2 Four Evaluation Protocols (`src/splits.py`)
1. **Protocol A — Stratified Random 5-Fold CV (`random_fold_s42`, `s1337`, `s2026`):** Conventional baseline ($N=690$) that permits scene and speaker overlap across folds.
2. **Protocol B — Scene-Grouped 5-Fold CV (`scene_fold_s42`, `s1337`, `s2026`):** Primary leakage-aware benchmark ($N=690$). Uses `StratifiedGroupKFold` over **449 connected-component scene clusters** (`scene_group_id`) so that no shared context turn or contiguous scene clip ever crosses a fold boundary.
3. **Protocol C1 — Leave-One-Speaker-Out (`LOSO_13Speakers`, $N=614$ evaluated):** Evaluates generalization to unseen speakers across all 13 named characters with $\ge 15$ utterances and both classes present (`FRIENDS_CHANDLER`, `BBT_SHELDON`, `BBT_HOWARD`, `GOLDENGIRLS_DOROTHY`, `FRIENDS_ROSS`, `FRIENDS_JOEY`, `FRIENDS_PHOEBE`, `BBT_LEONARD`, `BBT_PENNY`, `FRIENDS_RACHEL`, `FRIENDS_MONICA`, `BBT_RAJ`, `BBT_AMY`).
4. **Protocol C2 — Speaker-Grouped 5-Fold CV (`SpeakerGrouped_5Fold`, $N=690$):** Uses `StratifiedGroupKFold` over all 23 disambiguated `speaker_group` identifiers (`<show>_<speaker>`) so every instance in the 690-clip dataset receives a strictly speaker-independent prediction.

---

## 3. Feature Extraction Pipelines

### 3.1 Text Representations (`src/representations.py`, `src/models.py`)
- **TF-IDF Baseline:** Lowercase unigram + bigram `TfidfVectorizer(max_features=3000, min_df=2, sublinear_tf=True)`, fit strictly inside each training fold.
- **Frozen Contextual Embeddings (`roberta-base`):** 768-dimensional attention-masked mean-pooled embeddings from the final hidden layer of `roberta-base` (evaluated both on **Target Utterance Only** and **Dialogue Context + Target Utterance** `context </s></s> target`).
- **Zero-Shot Text Sentiment Polarity (`cardiffnlp/twitter-roberta-base-sentiment-latest`):** Extracts $P(\text{neg}), P(\text{neu}), P(\text{pos})$ and continuous polarity $S_{\text{text}} = P(\text{pos}) - P(\text{neg}) \in [-1, +1]$ without training on sarcasm labels.

### 3.2 Explicit Prosodic Feature Suite (41 Features Across 4 Interpretable Groups, `src/prosody.py`)
Extracted from 16 kHz mono WAV audio using **Praat (`praat-parselmouth 0.4.7`)** and **`librosa 1.0.0`**:
- **Group 1 — Fundamental Frequency ($F_0$, 11 features):** Autocorrelation pitch tracker (`75–500 Hz`, `10 ms` step): `f0_mean_hz`, `f0_median_hz`, `f0_std_hz`, `f0_min_hz`, `f0_max_hz`, `f0_range_hz`, `f0_iqr_hz`, `f0_cv`, `f0_slope_hz_per_sec`, `voiced_fraction`, and binary quality indicator `f0_unvoiced_flag` (`1.0` if $<3$ voiced frames, keeping unvoiced $F_0$ stats as `NaN` for fold-internal median imputation).
- **Group 2 — Energy & Intensity (10 features):** Praat intensity contour (dB) and short-time RMS energy (`32 ms` window, `10 ms` hop): `intensity_mean_db`, `intensity_std_db`, `intensity_max_db`, `intensity_range_db`, `intensity_iqr_db`, `rms_mean`, `rms_std`, `rms_max`, `rms_dynamic_range`, `energy_slope_db_per_sec`.
- **Group 3 — Timing, Rhythm & Pauses (11 features):** Non-silent interval segmentation (`top_db=25`) and onset-envelope syllable nucleus detection: `utt_duration_sec`, `speech_duration_sec`, `speech_ratio`, `silence_ratio`, `pause_count` (internal silent gaps $\ge 80\text{ ms}$), `pause_mean_dur_sec`, `pause_max_dur_sec`, `pause_rate_per_sec`, `speaking_rate_wpm_approx`, `articulation_rate_sps`, and `short_clip_flag` (`1.0` if duration $< 1.0\text{ s}`).
- **Group 4 — Voice Quality & Spectral Balance (9 features):** Praat cross-correlation PointProcess perturbation and harmonicity plus spectral timbre: `jitter_local`, `jitter_rap`, `shimmer_local`, `shimmer_apq3`, `hnr_mean_db`, `hnr_std_db`, `spectral_centroid_mean_hz`, `spectral_flatness_mean`, `zero_crossing_rate_mean`.
- **Standardized Acoustic Benchmark (`openSMILE eGeMAPSv02`):** 88-dimensional functional parameter set extracted via `opensmile 2.6.0`.

### 3.3 Learned Self-Supervised Audio Representations (`facebook/wav2vec2-base`)
- 768-dimensional frame-averaged representations combining layer 7 (mid-level prosodic/phonetic representation) and layer 12 (final transformer layer) of `facebook/wav2vec2-base` on 16 kHz mono audio.

---

## 4. Leakage-Free Classification, Fusion & Control Architecture

Within every cross-validation or LOSO fold (`src/models.py`):
1. **Fold-Internal Preprocessing:** `SimpleImputer(strategy="median")`, 1st–99th percentile outlier clipping, `StandardScaler()`, and optional dimensionality balancing (`PCA(n_components=64)`) are fit **exclusively on `X_train`** and applied to `X_test`.
2. **Hyperparameter Selection:** Regularization strength $C \in \{0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0\}$ is tuned via inner 3-fold stratified cross-validation on `X_train` only.
3. **Multimodal Fusion Strategies:**
   - **Early Concatenation Fusion:** Modality-specific standardization, PCA dimension balancing (`64-d` text block + `41-d` prosody block or `64-d` wav2vec2 block), $\ell_2$ block-length scaling ($1/\sqrt{d_m}$), and regularized Logistic Regression.
   - **Trimodal Early Fusion (`fusion_trimodal_text_prosody_w2v2`):** Concatenates the `41-d` explicit prosodic features and `768-d` `wav2vec 2.0` features into an `809-d` combined acoustic representation, applies fold-internal imputation, 1st–99th percentile clipping, standardization, and PCA compression to `64-d`, and combines it with the `64-d` `RoBERTa` text PCA block via $\ell_2$ block normalization (`128-d` total input to inner-CV-tuned Logistic Regression).
   - **Late Stacking Fusion:** Out-of-fold inner-CV probability predictions from the text and audio classifiers are combined by a meta-level Logistic Regression classifier fit strictly on `X_train`.
4. **Diagnostic Controls:**
   - **Speaker-ID Only Control:** One-hot encoded `speaker_group` -> Logistic Regression.
   - **Shuffled-Audio Controls:** Permutes the audio feature rows across examples (`seed=42`) while holding text features and labels intact, testing whether multimodal fusion exploits genuine cross-modal alignment.

---

## 5. Statistical Testing & Decision Rule (`src/evaluation.py`)

- **Primary Metric:** Pooled out-of-fold **Macro-F1** (accompanied by fold mean $\pm$ SD, Accuracy, Macro Precision, Macro Recall, and Weighted F1).
- **Group-Aware Paired Bootstrap (5,000 Resamples):** Under Scene-Grouped evaluation, bootstrap resampling draws whole `scene_group_id` clusters with replacement so correlated clips from the same scene are never treated as independent observations.
- **Strict Decision Rule:** A model comparison $\Delta = \text{Macro-F1}(A) - \text{Macro-F1}(B)$ is declared statistically supported **only if the 95% paired bootstrap confidence interval excludes zero**. Otherwise, we report *"No reliable difference was observed."*

