# Phase 2 & 3 — Comprehensive Data Exploration and Leakage Audit (`LEAKAGE_AUDIT.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  
**Audit Date:** November 7, 2026  
**Status:** **VERIFIED & FROZEN** (`data/splits/frozen_splits.parquet` & `data/splits/splits_manifest.json`)  

---

## 1. Empirical Leakage Pathways Discovered in MUStARD ($N = 690$)

Before training any model, we conducted an exhaustive programmatic audit of the 690 instances in Original MUStARD (`sarcasm_data.json` + `mmsd_raw_data/`) to identify every mechanism by which information from held-out test instances could leak into training data.

### Pathway 1: Verbatim Dialogue Context Overlap & Contiguous Scene Splitting
- **Discovery:** In Original MUStARD, multiple target utterances were extracted from sequential turns of the **exact same sitcom scene**.
  - Specifically, **62 target utterances** appear **verbatim as a dialogue context turn** inside another instance in the dataset, and **29 unique non-trivial sentences ($\ge 20$ characters)** are shared across multiple clips.
  - Furthermore, **361 clip pairs** within the same TV show have clip numbers separated by $\Delta \le 5$ (e.g., `1_533`, `1_536`, `1_537` in *The Big Bang Theory*; `1_1180`, `1_1185`, `1_1189`).
  - When we construct the **Connected-Component Scene Graph** (`src/data.py:build_scene_clusters`), the 690 utterances collapse into **449 distinct scene clusters**, of which **142 are multi-clip scene clusters** containing between 2 and 7 clips from the same continuous scene.
- **Empirical Leakage Under Standard Random 5-Fold Splits:**
  - Auditing our 3 random 5-fold splits (`random_fold_s42`, `random_fold_s1337`, `random_fold_s2026`) reveals that **116 out of 142 (81.7%)**, **125 out of 142 (88.0%)**, and **123 out of 142 (86.6%)** of the multi-clip scene clusters are split across both training and test folds under random cross-validation!
- **Control Implemented (`Protocol B: Scene-Grouped 5-Fold CV`):**
  - Using `StratifiedGroupKFold(n_splits=5)` grouped by `scene_group_id`, **0 of the 449 scene clusters** ever cross a train/test fold boundary across all 3 seeds (`scene_fold_s42`, `scene_fold_s1337`, `scene_fold_s2026`).

### Pathway 2: Extreme Speaker & TV Show Base-Rate Confounding
- **Discovery:** Across the 23 disambiguated `speaker_group` identifiers (`<show>_<speaker>`):
  - `GOLDENGIRLS_DOROTHY` ($n=39$) is **97.44% sarcastic** (38 sarcastic, 1 non-sarcastic).
  - `FRIENDS_CHANDLER` ($n=158$) is **75.95% sarcastic** (120 sarcastic, 38 non-sarcastic).
  - `BBT_PENNY` ($n=34$) is **67.65% sarcastic** (23 sarcastic, 11 non-sarcastic).
  - Conversely, `FRIENDS_JOEY` ($n=36$) is **8.33% sarcastic** (3 sarcastic, 33 non-sarcastic), and `FRIENDS_ROSS` ($n=38$) is **13.16% sarcastic** (5 sarcastic, 33 non-sarcastic).
  - At the show level, `SARCASMOHOLICS` ($n=14$) is **100.0% sarcastic** and `GOLDENGIRLS` ($n=40$) is **97.5% sarcastic**.
- **Empirical Leakage Under Random and Scene-Grouped Splits:**
  - Under random 5-fold splits, **20 of the 23 speaker groups** appear in both training and test sets. Because vocal pitch ($F_0$), timbre (spectral envelope / `wav2vec 2.0`), and character catchphrases strongly identify the speaker, any model evaluated under speaker-overlapping splits can exploit speaker identity as a shortcut for sarcasm prediction.
- **Controls Implemented:**
  1. **Strict Feature Exclusion:** Neither `speaker`, `speaker_group`, nor `show` is ever included as an input feature to any text, audio, or multimodal sarcasm classifier.
  2. **Diagnostic Speaker-Only Control (Phase 8A):** We explicitly train a diagnostic model using *only* one-hot `speaker_group` indicators to quantify the exact performance ceiling attributable purely to speaker base-rate memorization.
  3. **Leave-One-Speaker-Out (`Protocol C1: LOSO`, 13 major speakers, $N=614$ evaluated) & Speaker-Grouped 5-Fold CV (`Protocol C2`, all $N=690$ evaluated):** Evaluates generalization when the test speaker's voice and character lexical habits are completely unseen during training.

### Pathway 3: Short Lexical Duplicate Utterances
- **Discovery:** Exactly **17 target utterances** share short generic transcripts across different scenes: `"Okay?"` (3 times), `"Hey."` (2), `"No!"` (2), `"Really?"` (2), `"Thank you."` (2), `"Uh-huh!"` (2), `"Yeah?"` (2), and one 2-clip scene pair (`2_161` and `2_558`: `"I can see it from right here. It'll cost you one husband."`).
- **Control Implemented:** The scene pair (`2_161`, `2_558`) is automatically merged into the same scene cluster (`SCENE_2_161`) by our connected-component graph so it never splits across folds in Protocol B. The remaining short lexical expressions (`"Okay?"`, `"Yeah?"`) come from distinct episodes/speakers with different audio recordings (`0` duplicate MD5 hashes across all 1,380 `.mp4` files).

### Pathway 4: Preprocessing, Scaling, Imputation & Vectorization Leakage
- **Risk:** Fitting `TfidfVectorizer`, `StandardScaler`, `SimpleImputer` (for unvoiced $F_0$), or dimensionality reduction on the full 690-instance dataset before cross-validation leaks test-fold vocabulary IDF weights, feature means/variances, and median pitch values.
- **Control Implemented:** Every model pipeline fits all preprocessors (`TfidfVectorizer`, `SimpleImputer(strategy="median")`, `StandardScaler`, inner hyperparameter selection) **strictly on `X_train` within each fold** and applies `.transform(X_test)` to the held-out fold.

---

## 2. Formal Leakage Verification Checklist

| Check ID | Leakage Control Requirement | Verification Method | Status |
| :---: | :--- | :--- | :---: |
| **LC-01** | No `utterance_id` appears in both train and test within any fold. | `verify_zero_group_overlap(..., "utterance_id")` | **PASSED (100%)** |
| **LC-02** | No `scene_group_id` appears in both train and test under Scene-Grouped CV (`s42`, `s1337`, `s2026`). | `verify_zero_group_overlap(..., "scene_group_id")` across 449 clusters | **PASSED (0 overlaps)** |
| **LC-03** | No `speaker_group` appears in both train and test under Speaker-Grouped CV and LOSO. | `verify_zero_group_overlap(..., "speaker_group")` across 23 speaker groups | **PASSED (0 overlaps)** |
| **LC-04** | Target `label` (`sarcasm`) is never used when computing text sentiment ($S_{\text{text}}$) or vocal delivery ($S_{\text{vocal}}$). | Zero-shot `twitter-roberta-base-sentiment-latest` + unsupervised acoustic delivery index | **PASSED** |
| **LC-05** | Held-out MUStARD++ human annotations (`Valence`, `Arousal`, `Explicit_Emotion`, `Implicit_Emotion`, `Sarcasm_Type`) are never passed as model inputs. | Feature matrix column assertions in `src/models.py` and `tests/test_pipeline.py` | **PASSED** |
| **LC-06** | `TfidfVectorizer`, `SimpleImputer`, and `StandardScaler` are fit strictly on training indices per fold. | Encapsulated inside fold loops / `sklearn.pipeline.Pipeline` | **PASSED** |
| **LC-07** | Zero duplicate audio/video files across the 690 instances. | Cryptographic MD5 check across all 1,380 `.mp4` files | **PASSED (0 duplicates)** |

