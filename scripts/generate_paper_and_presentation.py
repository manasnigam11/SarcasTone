"""
Phase 16 & 17 Academic Research Paper (.docx & .pdf) and Final Presentation (.pptx) Generator.

Generates:
  1. `docs/SarcasTone_IEEE_Research_Paper.docx` — Complete 19-section IEEE-style conference paper
     with embedded real tables, publication figures, and verified references.
  2. `docs/SarcasTone_IEEE_Research_Paper.pdf` — Typeset PDF version of the complete IEEE paper
     with embedded tables and publication figures.
  3. `docs/SarcasTone_Final_Presentation.pptx` — 12-slide final presentation following the
     Milestone 1 structure, updated with actual experimental implementations, figures, and results.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import docx
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from pptx import Presentation
from pptx.dml.color import RGBColor as PPTXRGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PPTXInches, Pt as PPTXPt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_paper_sections() -> Tuple[str, str, str, List[Tuple[str, List[str], List[Tuple[str, List[List[str]]]], List[Tuple[str, str]]]]]:
    """
    Return the structured 19-section content of the SarcasTone IEEE research paper,
    where every number corresponds strictly to verified experimental artifacts.
    """
    title = "SarcasTone: A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection"
    authors = (
        "Manas Nigam (S24CSEU2391), Sri Charan Reddy (S24CSEU0563), Sanskar Gupta (S24CSEU0464), "
        "Taanya Raawat (S24CSEU0587), and Sriram (S24CSEU1654)"
    )
    affiliation = (
        "School of Computer Science Engineering and Technology, Bennett University, Greater Noida, India\n"
        "Project Instructor: Dr. Saurabh Kumar Srivastava (saurabh.srivastava@bennett.edu.in)\n"
        "Code & Reproducibility Repository: https://github.com/manasnigam11/SarcasTone"
    )

    sections: List[Tuple[str, List[str], List[Tuple[str, List[List[str]]]], List[Tuple[str, str]]]] = [
        (
            "Abstract",
            [
                "Spoken sarcasm frequently conveys an intended meaning that contradicts the literal polarity of the spoken words through vocal prosody and acoustic delivery. While multimodal sarcasm detection benchmarks such as MUStARD and MUStARD++ have enabled joint modeling of text, audio, and video, standard evaluation practices often overlook two critical sources of experimental leakage: (1) multi-clip scene overlap, where sequential utterances from the same continuous sitcom scene share verbatim context turns across training and test splits, and (2) extreme speaker and television-show base-rate confounding, where recurring characters exhibit heavily skewed sarcasm rates (ranging from 8.3% to 97.4%). In this paper, we present SarcasTone, a reproducible, leakage-aware comparative evaluation of textual representations, explicit phonetic prosodic cues (organized into four interpretable groups: fundamental frequency F0, energy/intensity, timing/pauses, and voice quality), and self-supervised audio representations (wav2vec 2.0) on the 690-instance MUStARD audiovisual benchmark enriched with held-out MUStARD++ subgroup annotations (N = 688). Using 5,000-iteration scene-cluster-aware paired bootstrap confidence intervals across Random, Scene-Grouped (449 disjoint scene clusters), Speaker-Grouped, and Leave-One-Speaker-Out (LOSO, 13 speakers) protocols, we demonstrate that: (i) a diagnostic model using speaker identity alone achieves 0.6912 Macro-F1 under scene-grouped splits—surpassing frozen RoBERTa-base text (0.6331)—before collapsing by 15.65 percentage points to 0.5346 under speaker-grouped evaluation; (ii) explicit prosodic features (0.6809 Scene-Grouped; 0.6569 LOSO) and frozen wav2vec 2.0 representations (0.7144 Scene-Grouped; 0.6661 LOSO) generalize substantially better across unseen speakers than text alone (0.5945 LOSO); (iii) combining frozen RoBERTa text with explicit prosody yields statistically supported gains over text-only under both Scene-Grouped CV (+0.0509 Macro-F1, 95% CI [+0.0149, +0.0873], p = 0.0064) and LOSO (+0.0709, 95% CI [+0.0308, +0.1125], p = 0.0008), while trimodal text + prosody + wav2vec 2.0 fusion achieves the highest performance across all leakage-controlled settings (0.7217 Scene-Grouped; 0.6948 LOSO); and (iv) voice quality (HNR, jitter, shimmer) and temporal rhythm/pause cues are the most generalizable explicit prosodic groups across unseen speakers. All source code, frozen evaluation splits, and reproducible experiment pipelines are publicly available at https://github.com/manasnigam11/SarcasTone."
            ],
            [],
            [],
        ),
        (
            "1. Introduction",
            [
                "Sarcasm is a pragmatic phenomenon in which a speaker's intended communicative meaning diverges from—and often inverts—the literal semantic interpretation of the spoken words [1]. In spoken human communication, listeners rarely rely on lexical semantics in isolation to resolve sarcastic intent. For example, an utterance such as 'Oh, I'm so glad you asked it like that' reads as strongly positive at the surface lexical level, yet a flat, low-pitched, or strained vocal delivery immediately signals sarcastic dissatisfaction [2]. Conventional text-only sarcasm detection systems have no access to fundamental frequency (F0), vocal intensity, speaking rate, pause structure, or glottal voice quality, rendering purely acoustic markers of sarcasm invisible to text classifiers.",
                "Despite growing interest in multimodal sarcasm detection following the introduction of the MUStARD [4] and MUStARD++ [5] corpora, evaluating the true incremental value of speech prosody remains methodologically challenging. Because multi-turn conversational datasets are harvested from episodic television sitcoms (such as Friends, The Big Bang Theory, and The Golden Girls), multiple target utterances are frequently sliced from the same continuous scene, and recurring characters display strong character-specific base rates of sarcasm. When models are evaluated under conventional random cross-validation, high-capacity acoustic or textual encoders can exploit scene-context memorization or speaker voiceprint shortcuts rather than learning generalizable prosodic indicators of sarcasm.",
                "In this study, we address the following core research question: 'Under scene-grouped and leave-one-speaker-out evaluation on MUStARD, which explicit prosodic cues add information beyond frozen text and learned audio representations, and does this change when text sentiment and vocal delivery disagree?' Rather than claiming a novel neural architecture, our contribution is a rigorous, reproducible, leakage-aware comparative evaluation that isolates textual semantics, explicit 4-group phonetic prosody, and self-supervised speech representations under strict statistical controls."
            ],
            [],
            [("results/figures/fig1_system_architecture.png", "Figure 1: SarcasTone leakage-aware multimodal system architecture and experimental evaluation pipeline.")],
        ),
        (
            "2. Related Work",
            [
                "Text-Based Sarcasm Detection: Early computational sarcasm research focused primarily on written social media and product reviews. Joshi et al. [1] synthesized this literature into three major milestones: lexical and pragmatic pattern extraction, distant supervision via user-supplied hashtags (#sarcasm), and the incorporation of author history or conversational context. Notably, Riloff et al. [11] modeled sarcasm as an explicit polarity contrast between a positive sentiment expression and a negative situation, while Bamman and Smith [12] demonstrated that incorporating author/speaker historical priors and conversational context significantly boosts accuracy—albeit at the risk of learning speaker-specific priors. Pretrained bidirectional transformer encoders such as BERT [17] and RoBERTa [6] later improved contextual semantic modeling, yet text-only models remain fundamentally blind to paralinguistic vocal cues.",
                "Acoustic and Phonetic Studies of Sarcasm: Phonetic research has established that speakers systematically modulate vocal prosody when producing sarcasm. Cheang and Pell [2] analyzed studio recordings across sarcastic, sincere, humorous, and neutral conditions, finding that English sarcasm is consistently marked by a lower mean fundamental frequency (F0), reduced F0 variation, slower speaking rate, and reduced harmonics-to-noise ratio (HNR) indicative of nasalized or laryngopharyngeal vocal quality. In spontaneous telephone dialogue, Tepperman et al. [3] studied 131 occurrences of 'yeah right' in Switchboard and Fisher, finding that spectral, durational, and contextual cues were necessary to disambiguate sarcastic usage. More broadly, paralinguistic speech emotion challenges [13] and the openSMILE toolkit [14] led Eyben et al. [8] to formalize the Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) for affective voice research, while Boersma and Weenink's Praat software [16] and Jadoul et al.'s Parselmouth interface [9] brought exact phonetic periodicity and glottal source algorithms into Python.",
                "Multimodal Sarcasm Benchmarks: Building upon context-dependent multimodal sentiment modeling in conversational videos [18], Castro et al. [4] introduced MUStARD, compiling 690 video clips (345 sarcastic, 345 non-sarcastic) with preceding dialogue context from four television sources, showing that combining BERT text [17], librosa spectral features [15], and ResNet visual features reduced error rates under 5-fold cross-validation. Ray et al. [5] subsequently expanded the corpus in MUStARD++ and annotated sarcasm subtypes (Propositional, Illocutionary, Embedded, Like-prefixed) alongside explicit/implicit emotions and valence/arousal ratings. Meanwhile, self-supervised speech models such as wav2vec 2.0 [7] have emerged as powerful general-purpose audio encoders, raising the question of whether explicit phonetic prosodic features still provide interpretable or complementary value under leakage-controlled evaluation."
            ],
            [],
            [],
        ),
        (
            "3. Research Motivation and Contributions",
            [
                "The basic comparison of text-only, audio-only, and multimodal sarcasm classification has been explored in prior literature [4], [5], [18]. However, existing evaluations rarely audit whether overlapping dialogue scenes or character-specific base rates [12] inflate reported multimodal gains, nor do they systematically ablate interpretable phonetic prosody groups against learned speech embeddings under paired bootstrap confidence intervals [20].",
                "Our specific contribution is a reproducible, leakage-aware comparative evaluation of textual representations, explicit prosodic cues, and learned audio representations, including prosodic ablation and disagreement-focused analysis. Specifically, we contribute: (1) a comprehensive leakage audit demonstrating that random 5-fold splits leak 81.7%–88.0% of multi-clip sitcom scenes across train/test boundaries and that speaker identity alone achieves 0.6912 Macro-F1 when speakers overlap; (2) frozen evaluation splits spanning Random, Scene-Grouped (449 clusters), Speaker-Grouped, and Leave-One-Speaker-Out (LOSO) protocols; (3) a 41-feature interpretable prosodic suite spanning F0, Energy, Timing/Pauses, and Voice Quality with explicit missingness/quality flags; (4) diagnostic Speaker-ID and Shuffled-Audio controls; and (5) a label-free Text–Vocal Disagreement and MUStARD++ Sarcasm-Type subgroup analysis evaluated under 5,000-iteration group-aware paired bootstrap testing."
            ],
            [],
            [],
        ),
        (
            "4. Dataset and Experimental Protocol",
            [
                "Gate 0 Dataset Audit and Selection: We first audited the official MUStARD++ repository (1,202 CSV instances) and the local raw video archive (mmsd_raw_data/, 1.315 GiB). Programmatic verification revealed that while MUStARD++ lists 1,202 rows, its additional 514 video clips are hosted as unarchived individual files on Google Drive subject to automated bulk-download throttling, whereas all 690 Original MUStARD target clips (mmsd_raw_data/utterances_final/*.mp4) and 690 context clips (mmsd_raw_data/context_final/*_c.mp4) are 100% present and uncorrupted locally. Crucially, joining Original MUStARD (sarcasm_data.json) with mustard++_text.csv by scene ID confirmed that 688 of the 690 clips (99.71%) are present in MUStARD++ with zero binary label discrepancies (688/688 exact label agreement). Following Gate 0 protocol, we therefore conduct all core experiments on the 100%-verified 690-instance Original MUStARD benchmark (345 sarcastic, 345 non-sarcastic) and link the held-out MUStARD++ annotations (Sarcasm_Type, Valence, Arousal, Explicit/Implicit Emotion) for the 688 overlapping scenes strictly for Phase 9 subgroup evaluation (never as model inputs).",
                "Media Properties and Leakage Discovery: All 1,380 MP4 files contain valid H.264 video and 2-channel 16-bit AAC audio (491 pairs at 48 kHz, 199 pairs at 44.1 kHz; target duration range 0.38 s to 20.02 s, median 5.00 s, mean 5.47 s). Our programmatic leakage audit uncovered two major vulnerabilities in the dataset: First, 62 target utterances appear verbatim as a context turn inside another clip, and constructing a connected-component scene graph (linking clips that share any sentence >= 20 characters or contiguous show clip IDs) grouped the 690 clips into 449 disjoint scene clusters (142 multi-clip clusters covering 383 clips). Under standard random 5-fold CV, 116 to 125 of these 142 multi-clip scenes leak across train and test folds. Second, sarcasm base rates vary drastically across shows (The Golden Girls is 97.5% sarcastic [39/40]; Sarcasmaholics Anonymous is 100.0% sarcastic [14/14]; BBT is 50.0% [140/280]; Friends is 42.7% [152/356]) and across speakers (Dorothy 97.4%, Chandler 75.9%, Penny 67.6% vs. Joey 8.3%, Ross 13.2%).",
                "Frozen Evaluation Protocols: To isolate these effects, we permanently froze four evaluation protocols in data/splits/frozen_splits.parquet, verified by automated zero-overlap assertions: (A) Stratified Random 5-Fold CV (seeds 42, 1337, 2026); (B) Scene-Grouped 5-Fold CV (StratifiedGroupKFold over 449 scene clusters across seeds 42, 1337, 2026, ensuring zero scene overlap); (C1) Leave-One-Speaker-Out (LOSO) across all 13 major named speakers with >= 15 utterances and both classes present (evaluating N = 614 instances); and (C2) Speaker-Grouped 5-Fold CV over all 23 disambiguated Show_Speaker groups (evaluating all N = 690 instances)."
            ],
            [
                (
                    "Table I: Dataset Composition and Sarcasm Base Rate by Television Source (N = 690)",
                    [
                        ["Show Code", "Full Television Source", "Total Clips", "% of Dataset", "Sarcastic (1)", "Non-Sarcastic (0)", "Sarcasm Rate"],
                        ["FRIENDS", "Friends", "356", "51.59%", "152", "204", "42.70%"],
                        ["BBT", "The Big Bang Theory", "280", "40.58%", "140", "140", "50.00%"],
                        ["GOLDENGIRLS", "The Golden Girls", "40", "5.80%", "39", "1", "97.50%"],
                        ["SARCASMOHOLICS", "Sarcasmaholics Anonymous", "14", "2.03%", "14", "0", "100.00%"],
                        ["TOTAL", "All 4 Sources Combined", "690", "100.00%", "345", "345", "50.00%"],
                    ],
                )
            ],
            [
                ("results/figures/fig2_dataset_distribution.png", "Figure 2: Overall binary class balance and per-show sarcasm distribution across the 690 MUStARD instances."),
                ("results/figures/fig3_speaker_distribution.png", "Figure 3: Utterance counts and extreme sarcasm base-rate skew across the top 14 speaker groups."),
                ("results/figures/fig4_scene_distribution.png", "Figure 4: Connected-component scene cluster size distribution (449 clusters) and multi-clip label composition."),
            ],
        ),
        (
            "5. Text Processing Pipeline",
            [
                "All transcripts were cleaned by stripping residual HTML formatting tags (such as <i>...</i> in BBT subtitles) and normalizing whitespace while preserving punctuation (question marks, exclamation points, and ellipses). Crucially, speaker names and show identifiers are strictly excluded from all text representations so that lexical classifiers cannot directly read speaker identity tags.",
                "We implemented three text representation paradigms: (1) TF-IDF Baselines (tfidf_target_lr and tfidf_target_svm), using unigram and bigram sublinear TF-IDF vectorization (max_features = 3,000, min_df = 2) fit strictly inside each training fold; (2) Frozen Transformer Embeddings (roberta_target_lr), extracting 768-dimensional attention-masked mean-pooled contextual representations from pretrained roberta-base [6]; and (3) Context-Aware Text Models (tfidf_context_lr and roberta_context_lr), which prepend the preceding dialogue turns to the target utterance ('context </s></s> target') without exposing speaker tags."
            ],
            [],
            [],
        ),
        (
            "6. Audio Processing and Explicit Prosodic Features",
            [
                "Audio Normalization: Using PyAV (av 19.0.2), all 690 target utterance MP4 audio streams were decoded from stereo 48 kHz / 44.1 kHz AAC into canonical 16,000 Hz mono 16-bit PCM WAV files (data/processed/audio_16k/) while preserving the original MP4 files untouched. Leading and trailing silence/non-speech segments were segmented using a 25 dB relative energy threshold (librosa.effects.split [15], 32 ms window, 10 ms hop).",
                "4-Group Interpretable Prosodic Feature Suite (41 Features): Using Praat [16] via Parselmouth (0.4.7) [9] and librosa (1.0.0) [15], we extracted 41 explicit prosodic features organized into four phonetic groups for controlled ablation: Group 1 — Fundamental Frequency F0 (11 features: mean, median, SD, min, max, range, IQR, coefficient of variation, linear slope in Hz/s, voiced frame fraction, and f0_unvoiced_flag via Praat autocorrelation pitch tracking over 75–500 Hz); Group 2 — Energy and Intensity (10 features: Praat intensity mean, SD, max, range, IQR in dB, short-time RMS mean, SD, max, 5th–100th percentile dynamic range, and energy slope in dB/s); Group 3 — Timing, Rhythm, and Pauses (11 features: total duration, active speech duration, speech ratio, silence ratio, internal pause count >= 80 ms, mean and max pause duration, pause rate per second, approximate speaking rate in WPM, articulation rate in syllables/sec via onset-strength peaks, and short_clip_flag for clips < 1.0 s); and Group 4 — Voice Quality and Spectral Balance (9 features: Praat cross-correlation PointProcess local and RAP jitter, local and APQ3 shimmer, mean and SD harmonics-to-noise ratio HNR in dB, spectral centroid, spectral flatness, and zero-crossing rate).",
                "In addition to the 41-feature interpretable suite, we extracted the standardized 88-dimensional openSMILE [14] eGeMAPSv02 functional set [8] (egemaps_lr) for diagnostic comparison. Across the 690 utterances, sarcastic speech exhibited lower mean F0 (192.98 Hz vs. 217.20 Hz for non-sarcastic), higher mean vocal intensity (70.72 dB vs. 67.57 dB), longer utterance duration (5.85 s vs. 4.57 s), slower articulation rate (3.95 vs. 4.54 syllables/sec), and lower harmonicity HNR (4.66 dB vs. 6.20 dB)."
            ],
            [
                (
                    "Table II: Descriptive Comparison of Key Explicit Prosodic Features by Class (N = 690)",
                    [
                        ["Prosodic Group", "Feature Name", "Non-Sarcastic Mean", "Sarcastic Mean", "Direction in Sarcasm"],
                        ["Group 1: F0 Pitch", "f0_mean_hz (Hz)", "217.20", "192.98", "Lower pitch (-24.2 Hz)"],
                        ["Group 1: F0 Pitch", "f0_range_hz (Hz)", "265.78", "285.81", "Wider pitch span (+20.0 Hz)"],
                        ["Group 1: F0 Pitch", "voiced_fraction", "0.509", "0.475", "Lower voiced frame ratio"],
                        ["Group 2: Energy", "intensity_mean_db (dB)", "67.57", "70.72", "Higher vocal intensity (+3.15 dB)"],
                        ["Group 2: Energy", "intensity_std_db (dB)", "8.38", "7.57", "Flatter intensity contour"],
                        ["Group 3: Timing", "utt_duration_sec (s)", "4.57", "5.85", "Longer utterance (+1.28 s)"],
                        ["Group 3: Timing", "articulation_rate_sps (syl/s)", "4.54", "3.95", "Slower articulation (-0.59 syl/s)"],
                        ["Group 4: Voice Quality", "hnr_mean_db (dB)", "6.20", "4.66", "Lower harmonicity / harsher (-1.55 dB)"],
                    ],
                )
            ],
            [("results/figures/fig5_audio_duration_distribution.png", "Figure 5: Target utterance and dialogue context audio duration distributions by class.")],
        ),
        (
            "7. Learned Audio Representations and Model Architectures",
            [
                "Learned Speech Representations (wav2vec 2.0): To compare handcrafted phonetic features against self-supervised speech representations, we extracted 768-dimensional utterance embeddings from frozen facebook/wav2vec2-base [7] (combining hidden state layer 7 and final layer 12 via temporal mean pooling across the 16 kHz waveform).",
                "Leakage-Free Fold Classification Pipeline: Because sample size is N = 690, end-to-end fine-tuning of 95M–125M parameter encoders risks severe overfitting and speaker memorization. Instead, within every training fold, we apply a strictly fold-internal scikit-learn [19] preprocessing pipeline: (1) median imputation fit on X_train only; (2) 1st–99th training-percentile outlier clipping; (3) StandardScaler fit on X_train only; and (4) L2-regularized Logistic Regression (with balanced class weights and C tuned over {0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0} via inner 3-fold stratified CV on X_train only). We also evaluate RBF-SVM on explicit prosody and LinearSVC on TF-IDF."
            ],
            [],
            [],
        ),
        (
            "8. Multimodal Fusion Strategies and Control Experiments",
            [
                "We evaluated three multimodal fusion designs: (1) Early Concatenation Fusion (fusion_roberta_prosody_early and fusion_roberta_wav2vec2_early), which standardizes each modality inside the training fold, compresses high-dimensional 768-d embeddings to 64 principal components (fit on X_train only) so 768-d vectors do not overwhelm 41-d prosodic features, normalizes each modality block by 1/sqrt(d_m) to equalize L2 energy, and trains an inner-CV-tuned Logistic Regression classifier; (2) Late Stacking Fusion (fusion_roberta_prosody_late and fusion_roberta_wav2vec2_late), which generates out-of-fold inner-CV probability predictions from the text and audio classifiers inside X_train and trains a meta-level Logistic Regression combiner; and (3) Trimodal Early Fusion (fusion_trimodal_text_prosody_w2v2), which concatenates the 41-d explicit prosodic features and 768-d wav2vec 2.0 features into an 809-d combined acoustic representation, compresses it via fold-internal PCA to 64 dimensions, and combines it with the 64-d RoBERTa text PCA block via L2 block normalization (128-d total input).",
                "To guard against spurious conclusions, we implemented two diagnostic control families: (A) Speaker-Only Diagnostic Baseline (control_speaker_only), which receives only a one-hot vector of the speaker_group identifier and zero text or audio features, measuring how much accuracy can be achieved purely by memorizing character sarcasm base rates; and (B) Shuffled-Audio Controls (control_shuffled_prosody and control_shuffled_wav2vec2), which randomly permute the audio feature rows across utterances (seed = 42) while keeping text features and labels intact, testing whether multimodal gains depend on genuine utterance-level cross-modal alignment."
            ],
            [],
            [],
        ),
        (
            "9. Experimental Setup and Statistical Methodology",
            [
                "Following Phase 10 guidelines, our primary evaluation metric is pooled out-of-fold Macro-F1, supplemented by fold mean +/- SD Macro-F1, Accuracy, Macro Precision, Macro Recall, and Weighted F1. All 25 model configurations were evaluated across all four protocols (200 total experiment runs logged in experiments/experiment_registry.csv).",
                "For statistical inference, we implemented 5,000-iteration group-aware paired bootstrap resampling [20] (src/evaluation.py). Under Scene-Grouped evaluation, bootstrap resampling draws whole scene clusters (449 clusters) with replacement so correlated clips from the same scene remain grouped together. Per our pre-specified decision rule, a performance difference Delta = Macro-F1(Model A) - Macro-F1(Model B) is declared statistically supported if and only if its 95% paired bootstrap confidence interval excludes zero; otherwise we report 'No reliable difference was observed.'"
            ],
            [],
            [],
        ),
        (
            "10. Main Experimental Results Across Protocols",
            [
                "Table III and Figures 6–7 present the primary experimental results across Random 5-Fold, Scene-Grouped 5-Fold (our primary leakage-free benchmark), Speaker-Grouped 5-Fold, and Leave-One-Speaker-Out (LOSO, 13 major speakers, N = 614).",
                "First, under Scene-Grouped 5-Fold CV, unimodal text models achieve 0.6029 Macro-F1 (TF-IDF + LR) and 0.6331 Macro-F1 (RoBERTa-base target + LR; 3-seed mean 0.6384 +/- 0.0050). Adding dialogue context modestly increases text performance to 0.6478 (TF-IDF) and 0.6536 (RoBERTa Context+Target). Notably, unimodal audio models outperform unimodal text: Explicit Prosody (41 features) achieves 0.6809 Macro-F1 (3-seed mean 0.6803 +/- 0.0079), openSMILE eGeMAPSv02 achieves 0.6536, and frozen wav2vec 2.0 achieves 0.7144 (3-seed mean 0.7025 +/- 0.0113).",
                "Second, multimodal fusion consistently outperforms text alone: RoBERTa + Explicit Prosody achieves 0.6840 (Early) and 0.6808 (Late; 3-seed mean 0.6905), RoBERTa + wav2vec 2.0 achieves 0.7143 (Early; 3-seed mean 0.7156), and Trimodal Fusion (RoBERTa + Explicit Prosody + wav2vec 2.0) achieves the highest overall point-estimate Macro-F1 of 0.7217 (95% CI [0.6840, 0.7593]). While Trimodal Fusion's improvement over RoBERTa text-only is statistically supported (+0.0886, 95% CI [+0.0466, +0.1311], p < 0.0002), its incremental gain over unimodal wav2vec 2.0 (+0.0073 in Scene CV; +0.0287 in LOSO) has a 95% bootstrap confidence interval that overlaps zero and is therefore interpreted strictly as a point-estimate improvement.",
                "Third, comparing protocols exposes the exact role of speaker confounding: under Scene-Grouped CV (where speakers still overlap across scenes), the diagnostic Speaker-ID-Only baseline achieves 0.6912 Macro-F1—higher than RoBERTa text! However, when evaluated on unseen speakers under Speaker-Grouped 5-Fold CV, the Speaker-ID baseline collapses by 15.65 percentage points to 0.5346 (0.3011 fold-mean), and RoBERTa text drops by 6.72 percentage points from 0.6331 to 0.5659 (0.5945 in LOSO). Note that in LOSO, because OneHotEncoder maps the unseen test speaker to an all-zero vector, control_speaker_only outputs a single constant class per held-out speaker, yielding a per-speaker mean Macro-F1 of only 0.3598; its pooled LOSO score of 0.6022 is a label-complement artifact caused by class_weight='balanced' flipping the learned intercept positive when Chandler (120 sarcastic / 38 non-sarcastic) is held out from training (predicting all 1s for Chandler and all 0s for the other 12 speakers). In contrast, Explicit Prosody (0.6536 Speaker-Grouped; 0.6569 LOSO), wav2vec 2.0 (0.6637 Speaker-Grouped; 0.6661 LOSO), Early RoBERTa + Prosody Fusion (0.6478 Speaker-Grouped; 0.6654 LOSO), and Trimodal Fusion (0.6695 Speaker-Grouped; 0.6948 LOSO) maintain strong generalization to completely unseen speakers."
            ],
            [
                (
                    "Table III: Main Model Comparison Across Evaluation Protocols (Pooled Out-of-Fold Macro-F1; *LOSO per-speaker mean for control_speaker_only is 0.3598)",
                    [
                        ["Model ID", "Modality / Family", "Random 5-Fold", "Scene-Grouped 5-Fold (95% CI)", "3-Seed Scene Mean±SD", "Speaker-Grouped 5-Fold", "LOSO (13 Spk, N=614)"],
                        ["tfidf_target_lr", "Text-Only (TF-IDF)", "0.5985", "0.6029 [0.568, 0.637]", "0.5961 ± 0.0066", "0.5603", "0.5534"],
                        ["tfidf_context_lr", "Text (Context+Target)", "0.6501", "0.6478 [0.613, 0.682]", "0.6347 ± 0.0162", "0.6186", "0.5747"],
                        ["roberta_target_lr", "Text-Only (RoBERTa)", "0.6477", "0.6331 [0.595, 0.671]", "0.6384 ± 0.0050", "0.5659", "0.5945"],
                        ["roberta_context_lr", "Text (Context+Target)", "0.6521", "0.6536 [0.617, 0.691]", "0.6527 ± 0.0030", "0.6108", "0.6199"],
                        ["prosody_explicit_lr", "Audio-Only (41d Prosody)", "0.6940", "0.6809 [0.644, 0.718]", "0.6803 ± 0.0079", "0.6536", "0.6569"],
                        ["egemaps_lr", "Audio-Only (88d eGeMAPS)", "0.6725", "0.6536 [0.616, 0.690]", "0.6568 ± 0.0029", "0.6346", "0.6319"],
                        ["wav2vec2_lr", "Audio-Only (wav2vec 2.0)", "0.7057", "0.7144 [0.679, 0.749]", "0.7025 ± 0.0113", "0.6637", "0.6661"],
                        ["fusion_roberta_prosody_early", "Multimodal (Text+Prosody)", "0.6811", "0.6840 [0.647, 0.722]", "0.6833 ± 0.0039", "0.6478", "0.6654"],
                        ["fusion_roberta_prosody_late", "Multimodal (Text+Prosody)", "0.6953", "0.6808 [0.644, 0.718]", "0.6905 ± 0.0121", "0.6110", "0.6429"],
                        ["fusion_roberta_wav2vec2_early", "Multimodal (Text+w2v2)", "0.7114", "0.7143 [0.675, 0.753]", "0.7156 ± 0.0024", "0.6665", "0.6884"],
                        ["fusion_roberta_wav2vec2_late", "Multimodal (Text+w2v2)", "0.7202", "0.7101 [0.673, 0.745]", "0.7085 ± 0.0016", "0.6652", "0.6824"],
                        ["fusion_trimodal_text_prosody_w2v2", "Trimodal (Text+Pros+w2v2)", "0.7070", "0.7217 [0.684, 0.759]", "0.7162 ± 0.0061", "0.6695", "0.6948"],
                        ["control_speaker_only", "Control (Speaker-ID Only)", "0.6841", "0.6912 [0.652, 0.729]", "0.6805 ± 0.0094", "0.5346", "0.6022 (0.3598*)"],
                        ["control_shuffled_prosody", "Control (Shuffled Prosody)", "0.6500", "0.6517 [0.615, 0.688]", "0.6433 ± 0.0096", "0.6184", "0.6295"],
                        ["control_shuffled_wav2vec2", "Control (Shuffled w2v2)", "0.6157", "0.6520 [0.614, 0.690]", "0.6419 ± 0.0101", "0.6087", "0.6106"],
                    ],
                )
            ],
            [
                ("results/figures/fig6_main_model_comparison.png", "Figure 6: Main model comparison under Scene-Grouped 5-Fold CV with 95% scene-cluster bootstrap confidence intervals."),
                ("results/figures/fig7_protocol_comparison.png", "Figure 7: Comparison across Random, Scene-Grouped, Speaker-Grouped, and LOSO evaluation protocols."),
                ("results/figures/fig10_confusion_matrices.png", "Figure 10: Out-of-fold confusion matrices under Scene-Grouped 5-Fold CV for key unimodal and multimodal models."),
            ],
        ),
        (
            "11. Statistical Hypothesis Analysis",
            [
                "Table IV and Figure 11 summarize the 5,000-iteration paired bootstrap tests for Hypotheses H1–H4 and our experimental controls:",
                "Hypothesis H1 (Text + Explicit Prosody outperforms Text-Only under grouped evaluation): SUPPORTED. Under Scene-Grouped CV, Early RoBERTa + Prosody Fusion beats RoBERTa text-only by Delta = +0.0509 (95% CI [+0.0149, +0.0873], p = 0.0064), and Late Fusion beats RoBERTa by Delta = +0.0477 (95% CI [+0.0110, +0.0839], p = 0.0096). Under Leave-One-Speaker-Out (LOSO), the gain of Early RoBERTa + Prosody over RoBERTa text increases to Delta = +0.0709 (95% CI [+0.0308, +0.1125], p = 0.0008). Trimodal fusion achieves a statistically supported improvement over RoBERTa text-only of Delta = +0.0886 (95% CI [+0.0466, +0.1311], p < 0.0002). However, when Trimodal Fusion is compared directly against unimodal wav2vec 2.0, the paired difference is not statistically significant in either Scene-Grouped CV (Delta = +0.0073, 95% CI [-0.0232, +0.0384], p = 0.6472) or LOSO (Delta = +0.0287, 95% CI [-0.0053, +0.0623], p = 0.1016).",
                "Shuffled-Audio Control Verification: Real RoBERTa + Prosody Fusion significantly outperforms the Shuffled-Prosody Control (Delta = +0.0323, 95% CI [+0.0045, +0.0604], p = 0.0224), and Real RoBERTa + wav2vec 2.0 significantly outperforms the Shuffled-wav2vec 2.0 Control (Delta = +0.0623, 95% CI [+0.0219, +0.1047], p = 0.0040), confirming that multimodal gains require true utterance-level audio–text alignment.",
                "Hypothesis H3 (Random vs. Scene-Grouped vs. Speaker-Independent evaluation): PARTIALLY SUPPORTED. Between Random CV (0.6477) and Scene-Grouped CV (0.6331), no reliable difference was observed for RoBERTa (Delta = +0.0146, 95% CI [-0.0160, +0.0449], p = 0.3440). However, moving from Scene-Grouped CV to Speaker-Grouped CV produces a statistically supported drop of Delta = +0.0672 for RoBERTa (95% CI [+0.0306, +0.1040], p = 0.0004) and Delta = +0.1565 for the Speaker-ID control (95% CI [+0.1105, +0.2044], p < 0.0002), demonstrating that speaker overlap—rather than scene overlap alone—is the dominant source of optimistic bias.",
                "Hypothesis H4 (Learned wav2vec 2.0 vs. Explicit Prosody alone): INCONCLUSIVE / NO RELIABLE DIFFERENCE. Although wav2vec 2.0 achieves a higher point estimate than Explicit Prosody under Scene-Grouped CV (0.7144 vs. 0.6809, Delta = +0.0334), the 95% paired bootstrap confidence interval slightly crosses zero ([-0.0016, +0.0673], p = 0.0684), and under LOSO unseen speakers the two representations perform nearly identically (0.6661 vs. 0.6569, Delta = +0.0092, 95% CI [-0.0305, +0.0504], p = 0.6404). Per our decision rule, no reliable difference was observed between learned wav2vec 2.0 and the 41 explicit prosodic features."
            ],
            [
                (
                    "Table IV: Paired Bootstrap Hypothesis Tests (5,000 Resamples, 95% Confidence Intervals)",
                    [
                        ["Hypothesis / Control Test", "Model A", "Model B", "Delta F1", "95% Bootstrap CI", "p-val", "Statistical Verdict"],
                        ["H1a: Multimodal Prosody (Scene)", "RoBERTa+Pros (Early)", "RoBERTa (Target)", "+0.0509", "[+0.0149, +0.0873]", "0.0064", "Supported (CI > 0)"],
                        ["H1b: Multimodal Prosody (Scene)", "RoBERTa+Pros (Late)", "RoBERTa (Target)", "+0.0477", "[+0.0110, +0.0839]", "0.0096", "Supported (CI > 0)"],
                        ["H1c: Multimodal Prosody (LOSO)", "RoBERTa+Pros (Early)", "RoBERTa (Target)", "+0.0709", "[+0.0308, +0.1125]", "0.0008", "Supported (CI > 0)"],
                        ["H1d: Trimodal vs Text (Scene)", "Trimodal Fusion", "RoBERTa (Target)", "+0.0886", "[+0.0466, +0.1311]", "0.0000", "Supported (CI > 0)"],
                        ["Trimodal vs w2v2 (Scene)", "Trimodal Fusion", "wav2vec 2.0 + LR", "+0.0073", "[-0.0232, +0.0384]", "0.6472", "No reliable difference"],
                        ["Trimodal vs w2v2 (LOSO)", "Trimodal Fusion", "wav2vec 2.0 + LR", "+0.0287", "[-0.0053, +0.0623]", "0.1016", "No reliable difference"],
                        ["H3a: Random vs Scene CV", "RoBERTa (Random)", "RoBERTa (Scene)", "+0.0146", "[-0.0160, +0.0449]", "0.3440", "No reliable difference"],
                        ["H3b: Scene vs Speaker CV", "RoBERTa (Scene)", "RoBERTa (SpkGrp)", "+0.0672", "[+0.0306, +0.1040]", "0.0004", "Supported (CI > 0)"],
                        ["H3c: Speaker Shortcut Collapse", "SpkControl (Scene)", "SpkControl (SpkGrp)", "+0.1565", "[+0.1105, +0.2044]", "0.0000", "Supported (CI > 0)"],
                        ["H4a: w2v2 vs Prosody (Scene)", "wav2vec 2.0 + LR", "Explicit Prosody", "+0.0334", "[-0.0016, +0.0673]", "0.0684", "No reliable difference"],
                        ["H4b: w2v2 vs Prosody (LOSO)", "wav2vec 2.0 + LR", "Explicit Prosody", "+0.0092", "[-0.0305, +0.0504]", "0.6404", "No reliable difference"],
                        ["Ctrl: Real vs Shuffled Prosody", "RoBERTa+Pros (Early)", "Shuffled Prosody", "+0.0323", "[+0.0045, +0.0604]", "0.0224", "Supported (CI > 0)"],
                        ["Ctrl: Real vs Shuffled w2v2", "RoBERTa+w2v2 (Early)", "Shuffled w2v2", "+0.0623", "[+0.0219, +0.1047]", "0.0040", "Supported (CI > 0)"],
                    ],
                )
            ],
            [("results/figures/fig11_bootstrap_confidence_intervals.png", "Figure 11: Paired bootstrap 95% confidence intervals across primary hypothesis and control comparisons.")],
        ),
        (
            "12. Prosodic Cue Ablation Study",
            [
                "Table V and Figure 8 report the results of our two-way prosodic cue ablation study across the four phonetic feature groups: Group 1 (F0 Pitch, 11 features), Group 2 (Energy/Intensity, 10 features), Group 3 (Timing/Pauses/Rate, 11 features), and Group 4 (Voice Quality/HNR/Spectral, 9 features).",
                "In Single-Group Audio-Only evaluation (Figure 8b), Group 4 (Voice Quality: HNR, jitter, shimmer, spectral balance) is the strongest standalone predictor under both Scene-Grouped CV (0.6782) and LOSO unseen speakers (0.6573), followed by Group 3 (Timing, Pauses, and Articulation Rate: 0.6314 Scene CV; 0.6457 LOSO), Group 1 (F0 Pitch: 0.6194 Scene CV; 0.6093 LOSO), and Group 2 (Energy/Intensity: 0.5985 Scene CV; 0.5862 LOSO).",
                "Interestingly, in Leave-One-Group-Out Multimodal Ablation (Figure 8a, dropping one group at a time from RoBERTa + Prosody), dropping Group 2 (Energy/Intensity) causes the largest reduction in Scene-Grouped Macro-F1 (from 0.6840 down to 0.6707), while under LOSO speaker-independent evaluation, dropping Group 3 (Timing/Pauses) causes the steepest drop (from 0.6654 down to 0.6351). This indicates that while voice quality and timing carry the strongest standalone speaker-independent acoustic signal, energy and timing provide the most complementary disambiguating information when combined with lexical semantics."
            ],
            [
                (
                    "Table V: Prosodic Feature Group Ablation Study Across Scene-Grouped and LOSO Protocols",
                    [
                        ["Model / Ablation Variant", "Active Prosody Groups", "Scene-Grouped F1 (95% CI)", "Speaker-Grouped F1", "LOSO (13 Spk) F1"],
                        ["fusion_roberta_prosody_early", "Text + All 4 Groups (41d)", "0.6840 [0.647, 0.722]", "0.6478", "0.6654"],
                        ["ablation_text_prosody_no_f0", "Text + Groups 2, 3, 4 (Drop F0)", "0.6913 [0.653, 0.730]", "0.6434", "0.6636"],
                        ["ablation_text_prosody_no_energy", "Text + Groups 1, 3, 4 (Drop Energy)", "0.6707 [0.632, 0.709]", "0.6434", "0.6683"],
                        ["ablation_text_prosody_no_timing", "Text + Groups 1, 2, 4 (Drop Timing)", "0.6794 [0.643, 0.718]", "0.6319", "0.6351"],
                        ["ablation_text_prosody_no_voice_quality", "Text + Groups 1, 2, 3 (Drop VQ)", "0.6911 [0.654, 0.729]", "0.6333", "0.6538"],
                        ["prosody_only_f0_lr", "Audio Only: Group 1 F0 (11d)", "0.6194 [0.582, 0.659]", "0.6016", "0.6093"],
                        ["prosody_only_energy_lr", "Audio Only: Group 2 Energy (10d)", "0.5985 [0.560, 0.636]", "0.5620", "0.5862"],
                        ["prosody_only_timing_lr", "Audio Only: Group 3 Timing (11d)", "0.6314 [0.592, 0.673]", "0.6354", "0.6457"],
                        ["prosody_only_voice_quality_lr", "Audio Only: Group 4 Voice Quality (9d)", "0.6782 [0.641, 0.714]", "0.6565", "0.6573"],
                    ],
                )
            ],
            [("results/figures/fig8_prosodic_ablation.png", "Figure 8: Multimodal leave-one-group-out ablation and single-group audio-only predictive power.")],
        ),
        (
            "13. Text–Vocal Disagreement and Sarcasm-Type Subgroup Analysis",
            [
                "To test Hypothesis H2 without leaking sarcasm labels, we partitioned the 690 utterances using zero-shot text sentiment polarity (cardiffnlp/twitter-roberta-base-sentiment-latest [10]) and our label-free unsupervised vocal delivery index (combining standardized HNR, F0 variability, vocal intensity, and articulation rate). This yielded n = 200 instances in the Text–Vocal Disagreement Subset (99 sarcastic, 101 non-sarcastic; subdivided into n = 88 Positive-Text + Flat/Negative-Vocal and n = 112 Negative-Text + Animated/Positive-Vocal) and n = 490 instances in the Agreement/Neutral Subset (246 sarcastic, 244 non-sarcastic).",
                "As shown in Table VI and Figure 9a, unimodal audio models achieve substantially larger gains over RoBERTa text inside the Disagreement Subset than in the Agreement Subset: on the Disagreement Subset (n = 200), Explicit Prosody reaches 0.7148 Macro-F1 vs. 0.6276 for RoBERTa (+0.0872 gain), and wav2vec 2.0 reaches 0.7247 (+0.0971 gain), compared to +0.0324 in the Agreement Subset. For multimodal fusion on the n = 200 Disagreement Subset, Late RoBERTa + wav2vec 2.0 Fusion (+0.0970, 95% CI [+0.0282, +0.1687]) and Trimodal Fusion (+0.0973, 95% CI [+0.0243, +0.1718]) achieve statistically supported improvements over text-only, particularly on the Negative-Text + Animated-Vocal subtype (n = 112, Delta = +0.1459, 95% CI [+0.0482, +0.2407]), whereas Early RoBERTa + Prosody Fusion shows a positive point gain (+0.0421, 95% CI [-0.0275, +0.1112]) whose wider confidence interval on n = 200 overlaps zero. On the held-out MUStARD++ Human Explicit-vs-Implicit Emotion Disagreement subset (n = 295), Late Prosody Fusion (+0.0351, 95% CI [+0.0076, +0.0633]) and Trimodal Fusion (+0.0413, 95% CI [+0.0128, +0.0747]) also yield statistically supported gains.",
                "Across MUStARD++ Sarcasm Types (Figure 9b), Propositional sarcasm (PRO, n = 196) shows the largest sensitivity boost from explicit prosody: detection recall jumps from 62.76% (123/196) with RoBERTa text-only to 72.96% (143/196) with Explicit Prosody and 73.47% (144/196) with RoBERTa + wav2vec 2.0. Illocutionary sarcasm (ILL, n = 100) reaches 73.00% recall with wav2vec 2.0 vs. 68.00% with text-only, and Embedded sarcasm (EMB, n = 46, flagged as exploratory since n < 50) reaches 80.43% with RoBERTa + Prosody and 84.78% with RoBERTa + wav2vec 2.0 vs. 73.91% with text-only."
            ],
            [
                (
                    "Table VI: Disagreement Subset and MUStARD++ Sarcasm-Type Performance (Scene-Grouped OOF)",
                    [
                        ["Subset / Subgroup", "Sample Count (Sarc / Non)", "RoBERTa Text-Only", "Explicit Prosody", "wav2vec 2.0", "RoBERTa + Pros (Early)", "Trimodal Fusion"],
                        ["All Instances (Macro-F1)", "690 (345 / 345)", "0.6331", "0.6809", "0.7144", "0.6840*", "0.7217*"],
                        ["Disagreement Subset (Macro-F1)", "200 (99 / 101)", "0.6276", "0.7148", "0.7247", "0.6697", "0.7249*"],
                        ["— PosText + Flat/Neg Vocal (F1)", "88 (52 / 36)", "0.6399", "0.7229", "0.6415", "0.6631", "0.6606"],
                        ["— NegText + Anim/Pos Vocal (F1)", "112 (47 / 65)", "0.6158", "0.6901", "0.7617", "0.6683", "0.7518*"],
                        ["Agreement / Neutral Subset (F1)", "490 (246 / 244)", "0.6347", "0.6671", "0.7101", "0.6897*", "0.7204*"],
                        ["PRO Sarcasm Recall (n=196)", "196 Sarcastic", "62.76%", "72.96%", "71.94%", "70.41%", "72.96%"],
                        ["ILL Sarcasm Recall (n=100)", "100 Sarcastic", "68.00%", "64.00%", "73.00%", "65.00%", "70.00%"],
                        ["EMB Sarcasm Recall (n=46, expl.)", "46 Sarcastic", "73.91%", "76.09%", "80.43%", "80.43%", "82.61%"],
                    ],
                )
            ],
            [("results/figures/fig9_disagreement_and_subgroups.png", "Figure 9: Performance across Text–Vocal Disagreement subsets and MUStARD++ sarcasm types.")],
        ),
        (
            "14. Qualitative Error Analysis",
            [
                "Inspecting the exact out-of-fold predictions under Scene-Grouped CV (results/predictions/oof_predictions_scene_s42.csv) reveals that Multimodal Prosody and Trimodal Fusion corrected 84 of the 690 instances (12.2%) where RoBERTa text-only failed. For example, in utterance 1_3707 (BBT, Leonard: 'Oh, I'm so glad you asked it like that. You.'), the literal text sentiment is overwhelmingly positive (+0.96), causing RoBERTa to predict non-sarcastic (0). However, Leonard delivers the line with a low fundamental frequency (137.2 Hz) and flat/negative vocal delivery (-0.53), enabling both RoBERTa + Prosody and Trimodal models to predict sarcastic (1). Similarly, in 1_3259 ('No, I mean Princess Leia. Of course I mean Bernadette. She's a wonderful girl...'), positive surface sentiment (+0.94) misleads text-only classification while prosodic timing and intonation recover the true sarcastic label.",
                "Conversely, 107 of the 690 instances (15.5%) were misclassified by both text-only and multimodal models. These failures cluster into two primary phenomena: (1) Character-Lore Deadpan Sincerity, such as 1_90 (Sheldon earnestly calculating Lois Lane's free-fall acceleration in a pedantic, low-pitch cadence that mimics his sarcastic prosody, causing false-positive sarcasm predictions across audio models), and (2) Visual-Only Sarcasm, such as 1_1973 (Penny: 'I hope I'm a waitress at the Cheesecake Factory for my whole life.'), delivered with an upbeat, cheerful voice (+0.48) where sarcastic intent is communicated visually via facial expression and shared world knowledge."
            ],
            [],
            [],
        ),
        (
            "15. Discussion and Limitations",
            [
                "Our findings highlight three methodological lessons for multimodal affective computing. First, evaluation protocols that permit speaker overlap across train and test splits conflate character base-rate memorization with sarcasm detection; reporting Leave-One-Speaker-Out or Speaker-Grouped metrics alongside Scene-Grouped metrics is essential for honest benchmarking. Second, explicit phonetic prosody (specifically voice quality HNR/jitter/shimmer and temporal pause/articulation structure) generalizes remarkably well to unseen speakers (0.6569 LOSO Macro-F1), matching 768-dimensional wav2vec 2.0 representations (0.6661 LOSO) at a fraction of the dimensionality and computational cost. Third, combining text, explicit prosody, and wav2vec 2.0 yields complementary gains across all protocols.",
                "Limitations: (1) Our dataset comprises N = 690 sitcom clips from four television shows; while 100% media-verified, smaller subgroup slices (such as EMB with n = 46 or PosText+FlatVocal with n = 88) have wider bootstrap confidence intervals (+/- 9–10 percentage points) and must be interpreted with caution. (2) Sitcom speech is professionally acted and occasionally contains studio audience laughter near clip boundaries, which differs from spontaneous naturalistic conversation. (3) By focusing strictly on text and speech prosody, SarcasTone intentionally omits visual facial expressions, leaving purely visual sarcasm out of scope."
            ],
            [],
            [],
        ),
        (
            "16. Conclusion",
            [
                "We presented SarcasTone, a leakage-aware comparative study of textual semantics, explicit 4-group speech prosody, and self-supervised audio representations on the MUStARD benchmark. Through rigorous scene-grouped and leave-one-speaker-out evaluations backed by 5,000-iteration paired bootstrap confidence intervals, we showed that explicit prosodic cues—led by voice quality, timing/pauses, and energy modulation—significantly improve sarcasm detection over frozen text representations (+5.09 percentage points in Scene-Grouped CV; +7.09 percentage points in LOSO). While trimodal integration of text, explicit prosody, and wav2vec 2.0 achieves the highest point-estimate Macro-F1 (0.7217 Scene-Grouped; 0.6948 LOSO) and significantly outperforms text-only (+8.86 percentage points, p < 0.0002), its incremental difference over unimodal wav2vec 2.0 (+0.73 percentage points in Scene CV; +2.87 percentage points in LOSO) is not statistically significant under paired bootstrap testing. All code, frozen split tables, cached representations, and automated verification tests are publicly available for full reproducibility at https://github.com/manasnigam11/SarcasTone."
            ],
            [],
            [],
        ),
        (
            "17. References",
            [
                "[1] A. Joshi, P. Bhattacharyya, and M. J. Carman, 'Automatic sarcasm detection: A survey,' ACM Computing Surveys, vol. 50, no. 5, Art. 73, pp. 1–22, 2017, doi: 10.1145/3124420.",
                "[2] H. S. Cheang and M. D. Pell, 'The sound of sarcasm,' Speech Communication, vol. 50, no. 5, pp. 366–381, 2008, doi: 10.1016/j.specom.2007.11.003.",
                "[3] J. Tepperman, D. Traum, and S. Narayanan, '\"Yeah right\": Sarcasm recognition for spoken dialogue systems,' in Proc. Interspeech, Pittsburgh, PA, USA, 2006, pp. 1838–1841, doi: 10.21437/Interspeech.2006-507.",
                "[4] S. Castro, D. Hazarika, V. Pérez-Rosas, R. Zimmermann, R. Mihalcea, and S. Poria, 'Towards multimodal sarcasm detection (An obviously perfect paper),' in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL), Florence, Italy, 2019, pp. 4619–4629, doi: 10.18653/v1/P19-1455.",
                "[5] A. Ray, S. Mishra, A. Nunna, and P. Bhattacharyya, 'A multimodal corpus for emotion recognition in sarcasm,' in Proc. 13th Lang. Resources Eval. Conf. (LREC), Marseille, France, 2022, pp. 6992–7000.",
                "[6] Y. Liu et al., 'RoBERTa: A robustly optimized BERT pretraining approach,' arXiv preprint arXiv:1907.11692, 2019.",
                "[7] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, 'wav2vec 2.0: A framework for self-supervised learning of speech representations,' in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 12449–12460.",
                "[8] F. Eyben et al., 'The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective computing,' IEEE Trans. Affective Comput., vol. 7, no. 2, pp. 190–202, 2016, doi: 10.1109/TAFFC.2015.2457417.",
                "[9] Y. Jadoul, B. Thompson, and B. de Boer, 'Introducing Parselmouth: A Python interface to Praat,' Journal of Phonetics, vol. 71, pp. 1–15, 2018, doi: 10.1016/j.wocn.2018.07.001.",
                "[10] D. Loureiro, F. Barbieri, L. Neves, L. Espinosa Anke, and J. Camacho-Collados, 'TimeLMs: Diachronic language models from Twitter,' in Proc. 60th Annu. Meeting Assoc. Comput. Linguistics: System Demonstrations (ACL), 2022, pp. 251–260, doi: 10.18653/v1/2022.acl-demo.25.",
                "[11] E. Riloff, A. Qadir, P. Surve, L. De Silva, N. Gilbert, and R. Huang, 'Sarcasm as contrast between a positive sentiment and negative situation,' in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), Seattle, WA, USA, 2013, pp. 704–714.",
                "[12] D. Bamman and N. A. Smith, 'Contextualized sarcasm detection on Twitter,' in Proc. 9th Int. AAAI Conf. Web Social Media (ICWSM), Oxford, UK, 2015, pp. 574–577.",
                "[13] B. Schuller, S. Steidl, and A. Batliner, 'The INTERSPEECH 2009 emotion challenge,' in Proc. Interspeech, Brighton, UK, 2009, pp. 312–315, doi: 10.21437/Interspeech.2009-103.",
                "[14] F. Eyben, M. Wöllmer, and B. Schuller, 'Opensmile: The Munich versatile and fast open-source audio feature extractor,' in Proc. 18th ACM Int. Conf. Multimedia, Firenze, Italy, 2010, pp. 1459–1462, doi: 10.1145/1873951.1874246.",
                "[15] B. McFee et al., 'librosa: Audio and music signal analysis in Python,' in Proc. 14th Python in Science Conf. (SciPy), Austin, TX, USA, 2015, pp. 18–25, doi: 10.25080/Majora-7b98e3ed-003.",
                "[16] P. Boersma and D. Weenink, 'Praat: Doing phonetics by computer,' Glot International, vol. 5, no. 9/10, pp. 341–345, 2001.",
                "[17] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, 'BERT: Pre-training of deep bidirectional transformers for language understanding,' in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT), Minneapolis, MN, USA, 2019, pp. 4171–4186, doi: 10.18653/v1/N19-1423.",
                "[18] S. Poria, E. Cambria, D. Hazarika, N. Mazumder, A. Zadeh, and L.-P. Morency, 'Context-dependent sentiment analysis in user-generated videos,' in Proc. 55th Annu. Meeting Assoc. Comput. Linguistics (ACL), Vancouver, Canada, 2017, pp. 873–883, doi: 10.18653/v1/P17-1081.",
                "[19] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
                "[20] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall/CRC, 1993."
            ],
            [],
            [],
        ),
    ]
    return title, authors, affiliation, sections


def build_word_paper(out_docx: Path) -> None:
    """Generate the editable IEEE-style Word document (`docs/SarcasTone_IEEE_Research_Paper.docx`)."""
    title, authors, affiliation, sections = get_paper_sections()
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(10)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run(title)
    r_title.bold = True
    r_title.font.size = Pt(15)
    r_title.font.name = "Times New Roman"

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_auth = p_auth.add_run(authors + "\n")
    r_auth.bold = True
    r_auth.font.size = Pt(10.5)
    r_aff = p_auth.add_run(affiliation)
    r_aff.italic = True
    r_aff.font.size = Pt(9.5)

    for sec_heading, paragraphs, tables, figures in sections:
        p_h = doc.add_paragraph()
        p_h.paragraph_format.space_before = Pt(10)
        p_h.paragraph_format.space_after = Pt(4)
        r_h = p_h.add_run(sec_heading.upper())
        r_h.bold = True
        r_h.font.size = Pt(11)
        r_h.font.name = "Times New Roman"

        for para_text in paragraphs:
            p = doc.add_paragraph(para_text)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(5)

        for tab_caption, rows in tables:
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(tab_caption)
            r_cap.bold = True
            r_cap.font.size = Pt(9.5)

            tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    cell = tbl.cell(i, j)
                    cell.text = str(val)
                    for cp in cell.paragraphs:
                        for cr in cp.runs:
                            cr.font.name = "Times New Roman"
                            cr.font.size = Pt(8.5)
                            if i == 0:
                                cr.bold = True
            doc.add_paragraph()

        for fig_rel_path, fig_caption in figures:
            fig_path = ROOT / fig_rel_path
            if fig_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(fig_path), width=Inches(6.2))
                p_fcap = doc.add_paragraph()
                p_fcap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r_fcap = p_fcap.add_run(fig_caption)
                r_fcap.italic = True
                r_fcap.font.size = Pt(9)

    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_docx))
    logger.info("Saved IEEE Word Research Paper -> %s", out_docx)


def build_pdf_paper(out_pdf: Path) -> None:
    """Generate the typeset IEEE-style PDF Research Paper (`docs/SarcasTone_IEEE_Research_Paper.pdf`)."""
    title, authors, affiliation, sections = get_paper_sections()
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=14,
        leading=17,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    author_style = ParagraphStyle(
        "PaperAuthor",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    sec_style = ParagraphStyle(
        "PaperSec",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=10.5,
        leading=13,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#111111"),
    )
    body_style = ParagraphStyle(
        "PaperBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12.5,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    cap_style = ParagraphStyle(
        "PaperCap",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=8,
    )
    tcap_style = ParagraphStyle(
        "TableCap",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=9.0,
        leading=11,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=4,
    )

    story: List[Any] = []
    story.append(Paragraph(title, title_style))
    story.append(
        Paragraph(
            f"<b>{authors}</b><br/><i>{affiliation.replace(chr(10), '<br/>')}</i>",
            author_style,
        )
    )

    for sec_heading, paragraphs, tables, figures in sections:
        story.append(Paragraph(sec_heading.upper(), sec_style))
        for p_txt in paragraphs:
            safe_txt = (
                p_txt.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(safe_txt, body_style))

        for tab_caption, rows in tables:
            story.append(Paragraph(tab_caption.replace("&", "&amp;"), tcap_style))
            tbl = Table(rows, repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                        ("LEADING", (0, 0), (-1, -1), 9.5),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("ALIGN", (0, 1), (1, -1), "LEFT"),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#666666")),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 6))

        for fig_rel_path, fig_caption in figures:
            fig_path = ROOT / fig_rel_path
            if fig_path.exists():
                img = RLImage(str(fig_path), width=6.1 * inch, height=3.1 * inch, kind="proportional")
                story.append(img)
                story.append(Paragraph(fig_caption.replace("&", "&amp;"), cap_style))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    logger.info("Saved IEEE PDF Research Paper -> %s", out_pdf)


def build_final_presentation(out_pptx: Path) -> None:
    """
    Generate the 12-slide Final Presentation (`docs/SarcasTone_Final_Presentation.pptx`)
    reusing the Milestone 1 structure and replacing planned placeholders with actual verified results.
    """
    prs = Presentation()
    prs.slide_width = PPTXInches(13.333)
    prs.slide_height = PPTXInches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide: Any, title_text: str, subtitle_text: str = "") -> None:
        tb = slide.shapes.add_textbox(PPTXInches(0.5), PPTXInches(0.3), PPTXInches(12.3), PPTXInches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = PPTXPt(24)
        p.font.bold = True
        p.font.color.rgb = PPTXRGBColor(20, 45, 90)
        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.size = PPTXPt(13)
            p2.font.color.rgb = PPTXRGBColor(90, 100, 115)

    def add_bullets(slide: Any, left: float, top: float, width: float, height: float, items: List[str], font_size: int = 14) -> None:
        tb = slide.shapes.add_textbox(PPTXInches(left), PPTXInches(top), PPTXInches(width), PPTXInches(height))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = PPTXPt(font_size)
            p.font.color.rgb = PPTXRGBColor(30, 30, 30)
            p.space_after = PPTXPt(8)

    # Slide 1: Title & Team
    s1 = prs.slides.add_slide(blank_layout)
    add_header(
        s1,
        "SarcasTone: A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection",
        "Final Project Presentation | Bennett University, Greater Noida, India | Project Instructor: Dr. Saurabh Kumar Srivastava",
    )
    add_bullets(
        s1, 0.6, 1.6, 6.2, 5.2,
        [
            "Core Research Question: Under scene-grouped and leave-one-speaker-out evaluation on MUStARD, which explicit prosodic cues add information beyond frozen text and learned audio representations, and does this change when text sentiment and vocal delivery disagree?",
            "Benchmark Dataset: 690 verified audiovisual clips from Original MUStARD (100% media intact) linked with N=688 MUStARD++ subgroup annotations (0 label discrepancies).",
            "Project Team (5 Members):",
            "  1. Manas Nigam (S24CSEU2391) — Data & Evaluation Lead",
            "  2. Sri Charan Reddy (S24CSEU0563) — Text Lead",
            "  3. Sanskar Gupta (S24CSEU0464) — Audio & Prosody Lead",
            "  4. Taanya Raawat (S24CSEU0587) — Fusion & Statistical Analysis Lead",
            "  5. Sriram (S24CSEU1654) — Writing & Reproducibility Lead",
        ],
        font_size=14,
    )
    fig1 = ROOT / "results/figures/fig1_system_architecture.png"
    if fig1.exists():
        s1.shapes.add_picture(str(fig1), PPTXInches(6.9), PPTXInches(1.6), width=PPTXInches(6.0))

    # Slide 2: Problem Domain, Motivation & Leakage Discovery
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Problem Domain, Motivation & Empirical Leakage Discovery", "Why standard random splits overestimate sarcasm detection on sitcom datasets")
    add_bullets(
        s2, 0.5, 1.4, 6.0, 5.5,
        [
            "Lexical Ambiguity: Positive literal wording ('Oh, I'm so glad you asked it like that') often masks sarcastic intent carried by low pitch, slow speaking rate, or strained voice quality.",
            "Leakage Pathway #1 (Scene Overlap): 62 target utterances appear verbatim inside another clip's context. Connected-component graph yields 449 scene clusters (142 multi-clip). Random 5-fold CV leaks 81.7%–88.0% of multi-clip scenes across train and test!",
            "Leakage Pathway #2 (Speaker Confounding): Extreme base-rate skew across characters: Dorothy (97.4% sarcastic), Chandler (75.9%), Penny (67.6%) vs. Joey (8.3%), Ross (13.2%).",
            "Four Frozen Evaluation Protocols: [A] Stratified Random 5-Fold, [B] Scene-Grouped 5-Fold (449 clusters, 0 overlap), [C1] Leave-One-Speaker-Out (LOSO, 13 speakers, N=614), [C2] Speaker-Grouped 5-Fold (N=690).",
        ],
        font_size=13,
    )
    fig3 = ROOT / "results/figures/fig3_speaker_distribution.png"
    if fig3.exists():
        s2.shapes.add_picture(str(fig3), PPTXInches(6.7), PPTXInches(1.5), width=PPTXInches(6.1))

    # Slide 3: Dataset Audit (Gate 0) & Scene Clustering
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Gate 0 Dataset Validation & Scene-Cluster Structure", "100% locally verified audio-visual clips (N = 690) + MUStARD++ subgroup metadata (N = 688)")
    fig2 = ROOT / "results/figures/fig2_dataset_distribution.png"
    fig4 = ROOT / "results/figures/fig4_scene_distribution.png"
    if fig2.exists():
        s3.shapes.add_picture(str(fig2), PPTXInches(0.5), PPTXInches(1.5), width=PPTXInches(6.1))
    if fig4.exists():
        s3.shapes.add_picture(str(fig4), PPTXInches(6.7), PPTXInches(1.5), width=PPTXInches(6.1))
    add_bullets(
        s3, 0.5, 4.9, 12.2, 2.2,
        [
            "Gate 0 Audit: MUStARD++ (1,202 rows) lacks 514 videos locally; Original MUStARD has 690/690 target & context MP4s intact with 16-bit stereo AAC audio.",
            "Exact Subgroup Enrichment: 688/690 Original MUStARD scenes match MUStARD++ with 0 label discrepancies, providing held-out Sarcasm_Type (PRO, ILL, EMB, LIK) and Explicit/Implicit Emotion tags.",
        ],
        font_size=13,
    )

    # Slide 4: Text & Explicit Prosody Feature Engineering
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Text, Explicit Prosody (4 Groups) & Learned Speech Pipeline", "Strictly fold-internal preprocessing with zero test-set leakage")
    add_bullets(
        s4, 0.5, 1.4, 6.1, 5.6,
        [
            "Text Branch: (1) In-fold TF-IDF (1–2g, 3,000 features) + LR/SVM; (2) Frozen RoBERTa-base (768d) on Target Only (0.6331 Scene F1) and Context+Target (0.6536 Scene F1).",
            "41 Interpretable Prosodic Features (Praat / Parselmouth + librosa):",
            "  • Group 1 (F0 Pitch, 11d): Sarcastic speech has lower mean F0 (193.0 Hz vs 217.2 Hz) & wider span.",
            "  • Group 2 (Energy/Intensity, 10d): Higher mean dB (70.7 vs 67.6 dB) & flatter contour.",
            "  • Group 3 (Timing & Pauses, 11d): Longer duration (5.85s vs 4.57s) & slower articulation (3.95 vs 4.54 syl/s).",
            "  • Group 4 (Voice Quality, 9d): Lower HNR (4.66 vs 6.20 dB) indicating strained/deadpan delivery.",
            "Learned Audio & Trimodal Architecture: openSMILE eGeMAPSv02 (88d) & Frozen wav2vec 2.0 (768d). In Trimodal Fusion, 41d prosody + 768d wav2vec 2.0 = 809d audio (compressed via in-fold PCA to 64d) is concatenated with 64d RoBERTa text PCA (128d total).",
        ],
        font_size=12,
    )
    fig5 = ROOT / "results/figures/fig5_audio_duration_distribution.png"
    if fig5.exists():
        s4.shapes.add_picture(str(fig5), PPTXInches(6.8), PPTXInches(1.6), width=PPTXInches(6.0))

    # Slide 5: Main Experimental Results (Scene-Grouped CV)
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Main Experimental Results (Scene-Grouped 5-Fold CV)", "5,000-iteration scene-cluster bootstrap 95% confidence intervals")
    fig6 = ROOT / "results/figures/fig6_main_model_comparison.png"
    if fig6.exists():
        s5.shapes.add_picture(str(fig6), PPTXInches(0.5), PPTXInches(1.4), width=PPTXInches(7.3))
    add_bullets(
        s5, 8.0, 1.5, 4.9, 5.5,
        [
            "Text-Only Baselines: TF-IDF achieves 0.6029; Frozen RoBERTa-base achieves 0.6331 [0.595, 0.671].",
            "Audio-Only Models Beat Text: Explicit Prosody (41d) reaches 0.6809 [0.644, 0.718]; wav2vec 2.0 reaches 0.7144 [0.679, 0.749].",
            "Multimodal Fusion Gains: Early RoBERTa + Prosody reaches 0.6840 (+5.09 pp over RoBERTa, p=0.0064). Trimodal Fusion achieves 0.7217 [0.684, 0.759] (statistically significant +8.86 pp gain over RoBERTa, p<0.0002; +0.73 pp point gain over wav2vec 2.0 is ns, 95% CI [-0.023, +0.038]).",
            "Shuffled-Audio Controls Confirm Alignment: Shuffling prosody drops F1 to 0.6517 (p=0.0224); shuffling wav2vec 2.0 drops F1 to 0.6520 (p=0.0040).",
        ],
        font_size=12,
    )

    # Slide 6: Leakage Controls Across Protocols (Random vs Scene vs Speaker/LOSO)
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Protocol Comparison: Uncovering the Speaker-ID Shortcut", "Random vs. Scene-Grouped vs. Speaker-Grouped vs. Leave-One-Speaker-Out (LOSO)")
    fig7 = ROOT / "results/figures/fig7_protocol_comparison.png"
    if fig7.exists():
        s6.shapes.add_picture(str(fig7), PPTXInches(0.5), PPTXInches(1.4), width=PPTXInches(7.4))
    add_bullets(
        s6, 8.1, 1.5, 4.8, 5.5,
        [
            "Speaker-ID Diagnostic Control: Achieves 0.6912 when speakers overlap in Scene CV, but collapses by -15.65 pp to 0.5346 under Speaker-Grouped CV! In LOSO, its per-speaker mean is 0.3598 (constant-class prediction per speaker; 0.6022 pooled is a balanced-intercept artifact when Chandler's 120 positive clips are held out).",
            "Text Drops on Unseen Speakers: RoBERTa drops from 0.6331 (Scene CV) to 0.5659 (Speaker-Grouped) and 0.5945 (LOSO, p=0.0004).",
            "Prosody & Trimodal Generalize Across Speakers: In LOSO (13 unseen speakers), Explicit Prosody achieves 0.6569, Early RoBERTa+Prosody achieves 0.6654 (+7.09 pp over text, p=0.0008), and Trimodal achieves 0.6948 (+2.87 pp over wav2vec 2.0, ns CI [-0.005, +0.062]).",
        ],
        font_size=12,
    )

    # Slide 7: Prosodic Feature Group Ablation Study
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Prosodic Cue Ablation Study (Groups 1–4)", "Which explicit prosodic cues drive performance and cross-speaker generalization?")
    fig8 = ROOT / "results/figures/fig8_prosodic_ablation.png"
    if fig8.exists():
        s7.shapes.add_picture(str(fig8), PPTXInches(0.5), PPTXInches(1.4), width=PPTXInches(12.3))
    add_bullets(
        s7, 0.5, 5.2, 12.3, 2.0,
        [
            "Standalone Audio-Only Power: Group 4 (Voice Quality: HNR, jitter, shimmer) is strongest (0.6782 Scene; 0.6573 LOSO), followed by Group 3 (Timing/Pauses: 0.6314 Scene; 0.6457 LOSO) and Group 1 (F0 Pitch: 0.6194 Scene; 0.6093 LOSO).",
            "Multimodal Complementarity: Dropping Group 2 (Energy) reduces Scene-Grouped multimodal F1 most (0.6840 -> 0.6707), while dropping Group 3 (Timing/Pauses) hurts LOSO speaker generalization most (0.6654 -> 0.6351).",
        ],
        font_size=13,
    )

    # Slide 8: Text-Vocal Disagreement & Sarcasm-Type Subgroup Analysis
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Text–Vocal Disagreement & Sarcasm-Type Analysis", "Zero-shot text polarity vs. unsupervised vocal delivery & MUStARD++ subtypes")
    fig9 = ROOT / "results/figures/fig9_disagreement_and_subgroups.png"
    if fig9.exists():
        s8.shapes.add_picture(str(fig9), PPTXInches(0.5), PPTXInches(1.4), width=PPTXInches(12.3))
    add_bullets(
        s8, 0.5, 5.2, 12.3, 2.0,
        [
            "Disagreement Subset (n=200): In Fig. 9a (bimodal bars), Explicit Prosody beats RoBERTa by +8.72 pp (0.7148 vs 0.6276) and bimodal Early RoBERTa+w2v2 reaches 0.6899. Adding full Trimodal / Late w2v2 Fusion reaches 0.7249 (+9.73 pp over RoBERTa, 95% CI [+0.024, +0.172]), led by Negative-Text + Animated-Vocal (+14.59 pp).",
            "MUStARD++ Sarcasm Types (Fig. 9b): Propositional sarcasm (PRO, n=196) recall jumps from 62.8% (RoBERTa) to 73.0% (Explicit Prosody) and 73.5% (RoBERTa+w2v2); Illocutionary (ILL, n=100) reaches 73.0% with wav2vec 2.0.",
        ],
        font_size=12,
    )

    # Slide 9: Statistical Hypothesis Testing & Confusion Matrices
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Paired Bootstrap Hypothesis Verdicts & Confusion Matrices", "Strict decision rule: 95% paired bootstrap CI of Macro-F1 difference must exclude zero")
    fig11 = ROOT / "results/figures/fig11_bootstrap_confidence_intervals.png"
    fig10 = ROOT / "results/figures/fig10_confusion_matrices.png"
    if fig11.exists():
        s9.shapes.add_picture(str(fig11), PPTXInches(0.4), PPTXInches(1.4), width=PPTXInches(6.8))
    if fig10.exists():
        s9.shapes.add_picture(str(fig10), PPTXInches(7.3), PPTXInches(1.5), width=PPTXInches(5.6))
    add_bullets(
        s9, 7.3, 3.6, 5.6, 3.5,
        [
            "H1 (Text+Prosody > Text): SUPPORTED in Scene CV (+0.0509, CI [+0.015, +0.087]) & LOSO (+0.0709, CI [+0.031, +0.113]). Trimodal > RoBERTa is SUPPORTED (+0.0886, p<0.0002), while Trimodal vs wav2vec 2.0 is ns (+0.0073 Scene; +0.0287 LOSO).",
            "H2 (Larger gain in Disagreement): PARTIALLY SUPPORTED (+8.72 pp unimodal prosody; +9.73 pp Trimodal CI > 0; Early Prosody CI spans 0 on n=200).",
            "H3 (Random > Grouped): SUPPORTED for Speaker Grouping (-6.72 pp RoBERTa, -15.65 pp Speaker Control); ns for Scene Grouping alone.",
            "H4 (wav2vec 2.0 > Explicit Prosody): NO RELIABLE DIFFERENCE (Scene CI [-0.002, +0.067]; LOSO CI [-0.031, +0.050]).",
        ],
        font_size=11,
    )

    # Slide 10: Qualitative Error Analysis & Conclusions
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Qualitative Error Analysis, Limitations & Conclusions", "Where multimodal prosody succeeds (84/690 fixed) and where it fails (107/690 hard errors)")
    add_bullets(
        s10, 0.6, 1.5, 12.1, 5.5,
        [
            "84 / 690 Errors Corrected by Multimodal Prosody+Audio: E.g., 1_3707 (Leonard: 'Oh, I'm so glad you asked it like that. You.') has +0.96 positive text sentiment (fooling RoBERTa), but low pitch (137.2 Hz) and flat delivery (-0.53) allow Prosody & Trimodal models to recover the sarcastic label.",
            "107 / 690 Shared Failures Across All Models: (1) Character-lore deadpan sincerity (e.g., 1_90 Sheldon earnestly calculating physics in a flat pedantic cadence -> false positive), and (2) Visual-only eye-roll sarcasm delivered in a cheerful voice (e.g., 1_1973 Penny).",
            "Limitations: Sample size N=690 widens CIs on small slices (n<50 exploratory); acted sitcom speech and studio laugh tracks differ from spontaneous dialogue; visual gestures intentionally excluded.",
            "Complete Deliverables Prepared: Verified codebase (src/, scripts/), 7/7 passing pytest suite, frozen splits, 200-run experiment registry, 11 publication figures, and IEEE Research Paper (.docx & .pdf).",
        ],
        font_size=14,
    )

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    logger.info("Saved Final Presentation -> %s", out_pptx)


def main() -> None:
    build_word_paper(ROOT / "docs" / "SarcasTone_IEEE_Research_Paper.docx")
    build_pdf_paper(ROOT / "docs" / "SarcasTone_IEEE_Research_Paper.pdf")
    # Note: Do not overwrite user-customized docs/SarcasTone_Final_Presentation.pptx


if __name__ == "__main__":
    main()

