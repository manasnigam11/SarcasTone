# Architectural & Scientific Decisions Log (`DECISIONS.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  

Every major technical, methodological, and architectural decision made autonomously during the execution of SarcasTone is recorded here with its alternatives, rationale, and verification status.

---

### D-01: Primary Dataset Selection — Original MUStARD ($N=690$) with MUStARD++ Subgroup Annotation Join ($N=688$)
- **Date:** 2026-11-07
- **Context:** Phase 1 (Gate 0) required auditing MUStARD++ (`1,202` instances) first, with Original MUStARD (`690` instances) as the authorized fallback.
- **Alternatives Considered:**
  1. Attempt unauthenticated scraping of 2,404 individual `.mp4` files from the MUStARD++ Google Drive folder.
  2. Use the 100% complete, locally verified Original MUStARD video archive (`mmsd_raw_data/`, 690 target + 690 context `.mp4` files) paired with `sarcasm_data.json`, and enrich 688/690 instances with `mustard++_text.csv` subgroup metadata (`Sarcasm_Type`, `Valence`, `Arousal`, `Explicit_Emotion`, `Implicit_Emotion`).
- **Decision:** Select **Option 2**.
- **Rationale:** All 690 Original MUStARD `.mp4` pairs are present and verified 100% intact locally, whereas 514 of the 1,202 MUStARD++ clips are absent locally and throttled by Google Drive's unauthenticated folder listing/download quotas. Furthermore, 688 of the 690 Original MUStARD scenes appear in `mustard++_text.csv` with **0 label mismatches**, allowing us to perform all Phase 9 subgroup analyses (`PRO`, `ILL`, `EMB`, `LIK` sarcasm types and human valence/emotion comparisons) without ever mixing mismatched video files.

---

### D-02: MP4 Audio Extraction Backend — `av` (PyAV `19.0.2`) Resampling to 16 kHz Mono Float32
- **Date:** 2026-11-07
- **Context:** Local `.mp4` files have mixed audio sample rates (491 at 48 kHz, 199 at 44.1 kHz, all 2-channel stereo AAC) and Windows `PATH` lacks a standalone `ffmpeg.exe` binary.
- **Alternatives Considered:**
  1. System `ffmpeg` subprocess calls (unavailable on `PATH`).
  2. `torchaudio.load()` (fails with `ImportError: TorchCodec is required`).
  3. `av` (PyAV `19.0.2`) using `av.AudioResampler(format='fltp', layout='mono', rate=16000)` + caching 16 kHz mono `.wav` files via `soundfile`.
- **Decision:** Select **Option 3 (`av` + `soundfile`)**.
- **Rationale:** Self-contained C-library binding that decodes AAC (`mp4a`) and resamples to canonical 16,000 Hz mono PCM deterministically without modifying the raw `.mp4` files.

---

### D-03: Scene-Cluster Grouping Construction for Protocol B (Scene-Grouped 5-Fold CV)
- **Date:** 2026-11-07
- **Context:** Original MUStARD keys (`1_533`, `1_536`, `1_537`) do not carry explicit episode IDs, yet adjacent clips from the same sitcom scene share verbatim context sentences (62 target utterances appear verbatim inside another clip's context).
- **Alternatives Considered:**
  1. Group only by exact utterance ID (equivalent to random splitting — leaks shared scene contexts across train/test).
  2. Group by TV show (`4` groups — conflates scene grouping with domain/show grouping).
  3. Construct a **Connected-Component Scene Graph** where two clips are linked into the same scene group if they (a) share any non-trivial dialogue turn ($\ge 20$ characters) across target or context transcripts, or (b) share the same prefix/show and have numeric clip IDs within $\Delta \le 10$ for `1_` (BBT) or $\Delta \le 2$ with overlapping speaker sets for `2_` (capturing contiguous cuts from the same episode scene).
- **Decision:** Select **Option 3**, yielding **449 disjoint scene clusters** (142 multi-clip clusters covering 383 clips + 307 singleton clusters) across the 690 instances, used with stratified-aware group folding across 3 random seeds (`42`, `1337`, `2026`).
- **Rationale:** Eliminates all verbatim context-turn leakage and contiguous-scene audio/visual overlap across folds while preserving balanced 5-fold cross-validation.

---

### D-04: Leave-One-Speaker-Out (LOSO) Evaluation Protocol (Protocol C)
- **Date:** 2026-11-07
- **Context:** MUStARD contains 21 unique `speaker` tags, but several have extreme sample sizes or 100% single-class skew (e.g., `DOROTHY` has 38 sarcastic and only 1 non-sarcastic clip; minor tags like `PERSON3` have $N=1$; generic `PERSON` merges multiple unnamed characters across `BBT` and `FRIENDS`).
- **Decision:**
  - Assign every utterance a disambiguated speaker group (`<show>_<speaker>`, e.g., `FRIENDS_CHANDLER`, `BBT_SHELDON`) so `BBT_PERSON` and `FRIENDS_PERSON` (and `BBT_PERSON1` / `FRIENDS_PERSON1`) are never conflated, yielding **23 unique `speaker_group` identifiers**.
  - Evaluate **Leave-One-Speaker-Out (LOSO)** across all **13 major named speakers with $\ge 15$ utterances and both classes present** (`FRIENDS_CHANDLER`, `BBT_SHELDON`, `BBT_HOWARD`, `GOLDENGIRLS_DOROTHY`, `FRIENDS_ROSS`, `FRIENDS_JOEY`, `FRIENDS_PHOEBE`, `BBT_PENNY`, `BBT_LEONARD`, `FRIENDS_RACHEL`, `FRIENDS_MONICA`, `BBT_RAJ`, `BBT_AMY` — covering **614 / 690 utterances = 89.0%** of the dataset), while using **all 690 instances** appropriately in training folds (excluding the held-out target speaker) and reporting both per-speaker metrics and pooled out-of-fold Macro-F1 across all held-out speaker predictions. Also evaluate a **5-Fold Grouped-by-Speaker CV** over all 690 examples (`23` speaker groups) so that 100% of examples have an out-of-speaker prediction.

---

### D-05: Pretrained Frozen Encoders for Text and Speech on CPU Hardware
- **Date:** 2026-11-07
- **Context:** Hardware audit confirmed an Intel Core i5-1340P CPU (16 threads, 16 GB RAM) with no CUDA GPU and $N=690$ examples.
- **Alternatives Considered:**
  1. End-to-end fine-tuning of large models (`roberta-large`, `wav2vec2-large`) inside each fold (prohibitively slow on CPU across $3 \times 5 + 13$ folds and high risk of overfitting on $N=690$).
  2. Frozen **`roberta-base`** (768-d contextual text embeddings) and frozen **`facebook/wav2vec2-base`** (768-d self-supervised speech representations, combining layer 7 and layer 12 mean-pooling), paired with inside-fold `StandardScaler` + regularized `LogisticRegression` / `SVM`.
- **Decision:** Select **Option 2**.
- **Rationale:** Scientifically clean separation of representation quality from classifier fitting, zero test-set leakage during representation extraction, fast and reproducible execution on CPU, and direct comparability with explicit prosodic features.

---

### D-06: Explicit Prosodic Feature Extraction — Dual Praat (`parselmouth`) + `librosa` Interpretable 4-Group Suite + `openSMILE` (`eGeMAPSv02`) Benchmark
- **Date:** 2026-11-07
- **Context:** Phase 5 & Phase 8C require explicit prosodic features structured into four interpretable acoustic groups (**Group 1: $F_0$/Pitch**, **Group 2: Energy/Intensity**, **Group 3: Timing/Pauses**, **Group 4: Voice Quality**) for controlled group-ablation experiments, plus handling of unvoiced/short clips.
- **Decision:** Extract a dedicated, fully documented **41-feature Interpretable Prosodic Suite** (`11` $F_0$, `10` Energy, `11` Timing, `9` Voice Quality) grouped strictly into the 4 mandated categories using `praat-parselmouth` (`0.4.7`) and `librosa` (`1.0.0`), complete with explicit quality/missingness flags (`f0_unvoiced_flag`, `short_clip_flag`, `voiced_fraction`), **and** additionally extract the 88-dimensional **`openSMILE` `eGeMAPSv02`** functional set for diagnostic comparison.

---

### D-07: Independent Text Polarity & Vocal Delivery Disagreement Definition (Phase 9)
- **Date:** 2026-11-07
- **Context:** Phase 9 requires testing whether audio helps more when text sentiment and vocal delivery disagree, without training on sarcasm labels or leaking video human annotations.
- **Decision:**
  1. **Text Polarity ($S_{\text{text}} \in [-1, +1]$):** Computed using a pretrained conversational sentiment transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`) strictly in zero-shot inference ($S_{\text{text}} = P(\text{pos}) - P(\text{neg})$).
  2. **Vocal Delivery / Acoustic Affective Valence ($S_{\text{vocal}}$):** Computed via an **Unsupervised Acoustic-Prosodic Delivery Index** (`src/disagreement.py`), combining robust z-scored harmonicity (`hnr_mean_db`), pitch variability (`f0_std_hz`), vocal intensity (`intensity_mean_db`), and articulation rate (`articulation_rate_sps`) strictly without sarcasm labels.
  3. **Disagreement Rule:** An utterance is classified into the **Text–Vocal Disagreement Subset** ($n = 200$) when:
     - **Positive Text + Negative/Flat Vocal Delivery** ($S_{\text{text}} \ge 0.15$ or label `"positive"`, and $S_{\text{vocal}} \le 0.0$, $n = 88$), OR
     - **Negative Text + Positive/Animated Vocal Delivery** ($S_{\text{text}} \le -0.15$ or label `"negative"`, and $S_{\text{vocal}} \ge 0.15$, $n = 112$).
  4. All remaining utterances form the **Agreement / Neutral Subset** ($n = 490$), and we also report an auxiliary descriptive comparison against the **MUStARD++ Human Explicit vs. Implicit Emotion Disagreement** annotations ($n = 295$ across $N=688$ linked clips).

