# GATE 0 — Mandatory Dataset Audit & Selection Report

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  
**Date of Audit:** November 7, 2026  
**Gate Status:** **PASSED (Fallback to Original MUStARD Activated with Enriched MUStARD++ Subgroup Annotations for 688/690 Overlapping Scenes)**  

---

## 1. Audit Summary & Dataset Fallback Decision

In accordance with Phase 1 (Gate 0) of the Master Directive, we audited both the **primary candidate dataset (MUStARD++)** and the **fallback dataset (Original MUStARD)** against the local workspace media (`mmsd_raw_data/`).

| Criterion | Primary Candidate: **MUStARD++** (`cfiltnlp/MUStARD_Plus_Plus`) | Fallback Candidate: **Original MUStARD** (`soujanyaporia/MUStARD`) |
| :--- | :--- | :--- |
| **Metadata Availability** | `mustard++_text.csv` downloaded (6,041 rows; **1,202 target utterances**: 601 sarcastic, 601 non-sarcastic). | `sarcasm_data.json` downloaded (**690 target utterances**: 345 sarcastic, 345 non-sarcastic). |
| **Local Video/Audio Media Match** | **INCOMPLETE (57.2% present, 42.8% missing).** Of the 1,202 MUStARD++ target utterances, only **688** correspond to the local `mmsd_raw_data/` clips; **514 clips** (`1_S09E*`–`1_S12E*` and `3_S*` *The Office* clips) are **absent locally**. | **100.0% COMPLETE (690 / 690 present).** Every single one of the 690 keys in `sarcasm_data.json` matches the exact 690 `.mp4` files in `mmsd_raw_data/utterances_final/` and `mmsd_raw_data/context_final/`. |
| **Remote Video Download Feasibility** | The official Google Drive (`1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW`) stores unarchived individual `.mp4` files inside `final_utterance_videos` and `final_context_videos`, which limits unauthenticated listing/downloading to 50 files per folder and triggers automated download rate limits. Furthermore, some re-cut MUStARD++ clips on Drive differ in duration from the local original MUStARD clips (e.g., `1_60_u.mp4` on Drive is 8.09 s vs. 4.00 s locally). | All 1,380 `.mp4` files (690 target + 690 context, 1.315 GiB) are **already present locally on disk** in `mmsd_raw_data/` and verified 100% intact. |
| **Label Consistency** | 601 positive / 601 negative (50.0% balanced). | 345 positive / 345 negative (50.0% balanced). Across the 688 overlapping scenes between Original MUStARD and MUStARD++, **100% (688/688) have identical binary sarcasm labels** (`0` mismatches). |
| **Subgroup Annotations (`Sarcasm_Type`, `Valence`, `Arousal`, `Emotion`)** | Present for all 1,202 scenes in `mustard++_text.csv`. | Not in `sarcasm_data.json`, **but available for 688 of the 690 clips (99.71%)** via exact `SCENE` ID join with `mustard++_text.csv`! |

### Formal Gate 0 Decision

1. **Primary Benchmark Dataset Selected:** **Original MUStARD ($N = 690$)** (Castro et al., ACL 2019), using the 690 locally verified target utterance videos (`mmsd_raw_data/utterances_final/*.mp4`), 690 context videos (`mmsd_raw_data/context_final/*_c.mp4`), and official `sarcasm_data.json` metadata.
2. **No Silent Mixing:** All core experiments, cross-validation splits, model training, and primary metrics run strictly on the canonical **690 Original MUStARD instances** so that audio/video durations and transcripts match 100% with the original clips.
3. **Subgroup & Descriptive Analysis Enrichment (Phase 9):** Because **688 of the 690 Original MUStARD scenes (99.71%)** are a strict subset of MUStARD++ (with 0 label discrepancies), we link the auxiliary MUStARD++ human annotations (`Sarcasm_Type`: Propositional `PRO`, Illocutionary `ILL`, Embedded `EMB`, Like-prefixed `LIK`; plus human `Valence`, `Arousal`, `Explicit_Emotion`, and `Implicit_Emotion`) for those 688 scenes **strictly as held-out evaluation metadata for Phase 9 subgroup and descriptive analyses** (never as model inputs). Only 2 clips (`2_99` and `2_167`) lack MUStARD++ subgroup tags.

---

## 2. Dataset Provenance, Licensing & Usage Restrictions

- **Original MUStARD Repository:** `https://github.com/soujanyaporia/MUStARD` — Released under the **MIT License** (`Copyright (c) 2019 Multimodal Language Understanding Group (MLUG)`), saved locally at [`data/metadata/mustard_LICENSE`](file:///c:/Users/Manas%20Nigam/Documents/5th%20Sem%20nots/NLP%20346/project/main/data/metadata/mustard_LICENSE).
- **MUStARD++ Repository:** `https://github.com/cfiltnlp/MUStARD_Plus_Plus` — Released for academic research (*"A Multimodal Corpus for Emotion Recognition in Sarcasm"*, LREC 2022, Ray et al., IIT Bombay).
- **Media Copyright Policy:** The audiovisual clips originate from television broadcasts (*Friends*, *The Big Bang Theory*, *The Golden Girls*, *Sarcasmaholics Anonymous*) and are used strictly for non-commercial academic research. Raw `.mp4` and `.wav` files are excluded from Git version control via `.gitignore`.

---

## 3. Complete Dataset Inventory & Distributions ($N = 690$)

### 3.1 Class Distribution
- **Total Target Utterances:** `690`
- **Sarcastic (`label = 1` / `True`):** `345` (50.00%)
- **Non-Sarcastic (`label = 0` / `False`):** `345` (50.00%)

### 3.2 Distribution by TV Show (`show`)

| Show Code | Full Title | Utterance Count | % of Dataset | Sarcastic Count | Non-Sarcastic Count | Sarcasm Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `FRIENDS` | *Friends* | 356 | 51.59% | 152 | 204 | 42.70% |
| `BBT` | *The Big Bang Theory* | 280 | 40.58% | 140 | 140 | 50.00% |
| `GOLDENGIRLS` | *The Golden Girls* | 40 | 5.80% | 39 | 1 | **97.50%** |
| `SARCASMOHOLICS` | *Sarcasmaholics Anonymous* | 14 | 2.03% | 14 | 0 | **100.00%** |
| **Total** | **All 4 Shows** | **690** | **100.00%** | **345** | **345** | **50.00%** |

> **Critical Leakage Finding #1 (Show & Speaker Confounding):** While `BBT` is 50.0% sarcastic and `FRIENDS` is 42.7% sarcastic, `GOLDENGIRLS` is **97.5% sarcastic (39/40)** and `SARCASMOHOLICS` is **100.0% sarcastic (14/14)**. Similarly, individual speakers exhibit extreme base-rate skew: `DOROTHY` is 97.4% sarcastic (38/39), `CHANDLER` is 75.9% sarcastic (120/158), `PENNY` is 67.6% sarcastic (23/34), whereas `JOEY` is only 8.3% sarcastic (3/36) and `ROSS` is 13.2% sarcastic (5/38). This confirms why our **Speaker-Only Control (Phase 8A)** and **Leave-One-Speaker-Out Evaluation (Phase 7C)** are essential to separate genuine semantic/prosodic learning from speaker memorization.

### 3.3 Speaker Distribution (`speaker`, 21 Unique Labels)

| Rank | Speaker Identifier | Show(s) | Total Clips | Sarcastic | Non-Sarcastic | Sarcasm Rate |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `CHANDLER` | `FRIENDS` | 158 | 120 | 38 | 75.95% |
| 2 | `SHELDON` | `BBT` | 89 | 45 | 44 | 50.56% |
| 3 | `HOWARD` | `BBT` | 47 | 23 | 24 | 48.94% |
| 4 | `PERSON` *(generic/minor)* | `FRIENDS` (25), `BBT` (20) | 45 | 12 | 33 | 26.67% |
| 5 | `DOROTHY` | `GOLDENGIRLS` | 39 | 38 | 1 | 97.44% |
| 6 | `ROSS` | `FRIENDS` | 38 | 5 | 33 | 13.16% |
| 7 | `JOEY` | `FRIENDS` | 36 | 3 | 33 | 8.33% |
| 8 | `PHOEBE` | `FRIENDS` | 35 | 8 | 27 | 22.86% |
| 9 | `PENNY` | `BBT` | 34 | 23 | 11 | 67.65% |
| 10 | `LEONARD` | `BBT` | 34 | 19 | 15 | 55.88% |
| 11 | `RACHEL` | `FRIENDS` | 32 | 7 | 25 | 21.88% |
| 12 | `MONICA` | `FRIENDS` | 29 | 6 | 23 | 20.69% |
| 13 | `RAJ` | `BBT` | 26 | 13 | 13 | 50.00% |
| 14 | `AMY` | `BBT` | 17 | 9 | 8 | 52.94% |
| 15 | `BERNADETTE` | `BBT` | 11 | 8 | 3 | 72.73% |
| 16–21 | `MODERATOR` (7), `MEMBER-GIRL` (5), `PERSON1` (4), `MEMBER-BOY` (2), `PERSON3` (1), `BLANCHE` (1) | `SARCASMOHOLICS`, `FRIENDS`, `GOLDENGIRLS` | 20 | 16 | 4 | 80.00% |

### 3.4 Scene & Dialogue Group Identification

In Original MUStARD, multiple clips were sometimes extracted from the **same continuous sitcom scene** (e.g., `1_533`, `1_536`, `1_537` in *The Big Bang Theory*, where the target utterance of `1_533` appears verbatim inside the dialogue context of `1_536` and `1_537`).
- Our programmatic graph audit identified **29 non-trivial sentences ($\ge 20$ characters) shared verbatim across multiple clips** and **142 multi-clip scene clusters covering 383 clips** (resulting in **449 distinct scene/dialogue clusters** across the 690 instances when combining verbatim turn overlap and contiguous show clip IDs in `src/data.py:build_scene_clusters`).
- Under **Scene-Grouped Evaluation (Protocol B)**, all clips belonging to the same connected dialogue/scene cluster are strictly forced into the same cross-validation fold so that no shared conversation or overlapping context ever crosses the train/test boundary.

### 3.5 Media & Audio Segmentation Verification

- **690/690 Target Utterance Videos (`mmsd_raw_data/utterances_final/<id>.mp4`):** Verified 100% present, uncorrupted H.264 video + stereo 16-bit AAC audio (491 at 48 kHz, 199 at 44.1 kHz).
- **Audio Decoding Verified:** Direct decoding via `av` (PyAV `19.0.2`) to 16,000 Hz mono float32 waveforms succeeded and was verified with both `praat-parselmouth` (`0.4.7`) and `opensmile` (`2.6.0`).
- **Pre-Isolated Target Utterances:** Target utterances are already isolated from the preceding conversation in `utterances_final/<id>.mp4` (median duration `5.00 s`, mean `5.47 s`, range `0.38 s` to `20.02 s`).

---

## 4. Remaining Limitations Documented at Gate 0

1. **Sample Size ($N = 690$):** Because the local raw video archive provides the 690 Original MUStARD clips rather than the 514 additional unbundled MUStARD++ clips, total sample size is 690. All model complexity choices and statistical power analyses (paired bootstrap confidence intervals) are calibrated for $N = 690$.
2. **Sitcom Audience Laughter & Short Utterances:** Some clips contain canned laughter or are under 1.0 s long (minimum `0.38 s`). Our Phase 5 prosodic pipeline incorporates voiced-frame filtering, silence/energy gating, and missingness indicators (`f0_missing_flag`) to prevent unvoiced or short segments from corrupting feature distributions.

