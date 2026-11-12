# Verified Academic References (`REFERENCES.md`)

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  

All citations used in SarcasTone have been verified against their original publication venues, DOIs/URLs, methodologies, and relevance to this study.

---

### [1] Joshi, Bhattacharyya, and Carman (2017)
- **Title:** Automatic Sarcasm Detection: A Survey
- **Authors:** Aditya Joshi, Pushpak Bhattacharyya, Mark J. Carman
- **Year:** 2017
- **Venue:** *ACM Computing Surveys (CSUR)*, Vol. 50, No. 5, Article 73, pp. 1–22
- **DOI / URL:** `https://doi.org/10.1145/3124420`
- **Methodology:** Comprehensive taxonomy of computational sarcasm detection datasets, rule-based/statistical approaches, distant supervision (`#sarcasm`), and context integration.
- **Dataset:** Survey of Twitter, Reddit, product reviews, and TV dialogue datasets.
- **Relevant Results & Limitations:** Identifies three paradigm shifts (lexical pattern mining, hashtag distant supervision, and contextual/author modeling) while highlighting that text-only systems fail when sarcastic intent is conveyed solely through vocal prosody or visual expression.
- **Relevance to SarcasTone:** Establishes the theoretical foundation for our text-only baseline (`TF-IDF`, `RoBERTa`), context-aware text experiments, and the motivation for incorporating speech prosody.

---

### [2] Cheang and Pell (2008)
- **Title:** The Sound of Sarcasm
- **Authors:** Henry S. Cheang, Marc D. Pell
- **Year:** 2008
- **Venue:** *Speech Communication*, Vol. 50, No. 5, pp. 366–381
- **DOI / URL:** `https://doi.org/10.1016/j.specom.2007.11.003`
- **Methodology:** Controlled acoustic-phonetic study analyzing fundamental frequency ($F_0$ mean, SD, range), amplitude/intensity, speaking rate, and harmonics-to-noise ratio (HNR) across sarcastic, humorous, sincere, and neutral attitudes.
- **Dataset:** Studio recordings of 6 native English speakers producing controlled target utterances across four attitudinal conditions.
- **Relevant Results & Limitations:** Demonstrated that English sarcasm is reliably marked by reductions in mean $F_0$, decreased $F_0$ standard deviation, reduced speaking rate, and shifts in voice quality (HNR). Limited to acted studio speech rather than naturalistic multi-speaker dialogue.
- **Relevance to SarcasTone:** Directly motivates our four explicit prosodic feature groups (**Group 1: $F_0$**, **Group 2: Energy/Intensity**, **Group 3: Timing/Pauses**, **Group 4: Voice Quality / HNR / Jitter / Shimmer**).

---

### [3] Tepperman, Traum, and Narayanan (2006)
- **Title:** "Yeah Right": Sarcasm Recognition for Spoken Dialogue Systems
- **Authors:** Joseph Tepperman, David Traum, Shrikanth Narayanan
- **Year:** 2006
- **Venue:** *Proceedings of Interspeech 2006 (9th International Conference on Spoken Language Processing)*, Pittsburgh, PA, USA, pp. 1838–1841
- **DOI / URL:** `https://doi.org/10.21437/Interspeech.2006-507`
- **Methodology:** Decision tree classification using prosodic ($F_0$, energy, duration), spectral (MFCCs), and contextual (laughter, pause, dialogue act) features.
- **Dataset:** 131 occurrences of the phrase *"yeah right"* in the Switchboard and Fisher conversational telephone corpora.
- **Relevant Results & Limitations:** Found that spectral and contextual cues (including laughter and dialogue context) outperformed isolated pitch/energy on spontaneous telephone speech, but restricted analysis to a single two-word lexical expression.
- **Relevance to SarcasTone:** Demonstrates both the importance of spectral/contextual cues and the confounding role of laughter in spoken sarcasm detection.

---

### [4] Castro et al. (2019) — Original MUStARD Benchmark
- **Title:** Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)
- **Authors:** Santiago Castro, Devamanyu Hazarika, Verónica Pérez-Rosas, Roger Zimmermann, Rada Mihalcea, Soujanya Poria
- **Year:** 2019
- **Venue:** *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)*, Florence, Italy, pp. 4619–4629
- **DOI / URL:** `https://doi.org/10.18653/v1/P19-1455` | GitHub: `https://github.com/soujanyaporia/MUStARD`
- **Methodology:** Compiled the MUStARD dataset and evaluated SVM classifiers over BERT text features, `librosa` acoustic features (MFCCs, spectral centroid, chroma), and ResNet-152 visual features under speaker-dependent (5-fold CV) and speaker-independent splits.
- **Dataset:** **MUStARD** — 690 video clips (345 sarcastic, 345 non-sarcastic) with preceding dialogue context from *Friends*, *The Big Bang Theory*, *The Golden Girls*, and *Sarcasmaholics Anonymous*.
- **Relevant Results & Limitations:** Reported ~71.7% weighted F1 (Text+Audio) and ~74.5% (Text+Video) under speaker-dependent 5-fold CV, dropping to ~62.8%–64.6% under speaker-independent evaluation (holding out *Friends* as test set). Did not isolate scene-overlap leakage within random 5-fold CV or perform systematic prosodic-group ablations.
- **Relevance to SarcasTone:** Provides our primary benchmark dataset ($N=690$) and motivates our leakage-aware comparison across Random, Scene-Grouped, and Leave-One-Speaker-Out protocols.

---

### [5] Ray, Mishra, Nunna, and Bhattacharyya (2022) — MUStARD++
- **Title:** A Multimodal Corpus for Emotion Recognition in Sarcasm
- **Authors:** Anupama Ray, Shubham Mishra, Apoorva Nunna, Pushpak Bhattacharyya
- **Year:** 2022
- **Venue:** *Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022)*, Marseille, France, pp. 6992–7000
- **URL:** `https://aclanthology.org/2022.lrec-1.756/` | GitHub: `https://github.com/cfiltnlp/MUStARD_Plus_Plus`
- **Methodology:** Extended MUStARD to 1,202 instances and added fine-grained human annotations for **Sarcasm Type** (Propositional `PRO`, Embedded `EMB`, Illocutionary `ILL`, Like-prefixed `LIK`), **Explicit Emotion**, **Implicit Emotion**, **Valence** (1–9), and **Arousal** (1–9).
- **Dataset:** **MUStARD++** (1,202 utterances, including 688 of the 690 original MUStARD utterances).
- **Relevant Results & Limitations:** Showed that implicit and explicit emotions frequently diverge in sarcastic utterances (especially in Illocutionary sarcasm).
- **Relevance to SarcasTone:** Provides verified human annotations for 688/690 scenes in our dataset, enabling our Phase 9 Sarcasm-Type subgroup analysis (`PRO`, `ILL`, `EMB`) and descriptive comparison of Explicit vs. Implicit emotion disagreement.

---

### [6] Liu et al. (2019) — RoBERTa
- **Title:** RoBERTa: A Robustly Optimized BERT Pretraining Approach
- **Authors:** Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
- **Year:** 2019
- **Venue:** *arXiv preprint arXiv:1907.11692*
- **URL:** `https://arxiv.org/abs/1907.11692`
- **Relevance to SarcasTone:** Provides the pretrained contextual text encoder (`roberta-base`) and sentiment backbone (`cardiffnlp/twitter-roberta-base-sentiment-latest`) used in our text and disagreement pipelines.

---

### [7] Baevski et al. (2020) — wav2vec 2.0
- **Title:** wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- **Authors:** Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli
- **Year:** 2020
- **Venue:** *Advances in Neural Information Processing Systems (NeurIPS 2020)*, Vol. 33, pp. 12449–12460
- **URL:** `https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html`
- **Relevance to SarcasTone:** Provides the self-supervised learned audio representation backbone (`facebook/wav2vec2-base`) evaluated in Phase 6 and compared against explicit prosodic features in Hypothesis H4.

---

### [8] Eyben et al. (2016) — eGeMAPS & openSMILE
- **Title:** The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing
- **Authors:** Florian Eyben, Klaus R. Scherer, Björn W. Schuller, Johan Sundberg, Elisabeth André, Carlos Busso, Laurence Y. Devillers, Julien Epps, Petri Laukka, Shrikanth S. Narayanan, Khiet P. Truong
- **Year:** 2016
- **Venue:** *IEEE Transactions on Affective Computing*, Vol. 7, No. 2, pp. 190–202
- **DOI / URL:** `https://doi.org/10.1109/TAFFC.2015.2457417`
- **Relevance to SarcasTone:** Standardizes acoustic parameter extraction for affective computing and supports our `openSMILE` `eGeMAPSv02` feature benchmark.

---

### [9] Jadoul, Thompson, and de Boer (2018) — Parselmouth (Praat in Python)
- **Title:** Introducing Parselmouth: A Python Interface to Praat
- **Authors:** Yannick Jadoul, Bill Thompson, Bart de Boer
- **Year:** 2018
- **Venue:** *Journal of Phonetics*, Vol. 71, pp. 1–15
- **DOI / URL:** `https://doi.org/10.1016/j.wocn.2018.07.001`
- **Relevance to SarcasTone:** Enables exact, reproducible Praat autocorrelation pitch ($F_0$), intensity, jitter, shimmer, and harmonicity (HNR) extraction in Python for our 4-group explicit prosody suite.

---

### [10] Loureiro et al. (2022) — TimeLMs / Conversational Sentiment RoBERTa
- **Title:** TimeLMs: Diachronic Language Models from Twitter
- **Authors:** Daniel Loureiro, Francesco Barbieri, Leonardo Neves, Luis Espinosa Anke, Jose Camacho-Collados
- **Year:** 2022
- **Venue:** *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations (ACL 2022)*, pp. 251–260
- **DOI / URL:** `https://doi.org/10.18653/v1/2022.acl-demo.25`
- **Relevance to SarcasTone:** Provides the pretrained zero-shot conversational sentiment classifier (`cardiffnlp/twitter-roberta-base-sentiment-latest`) used to estimate literal text polarity ($S_{\text{text}}$) without training on sarcasm labels in Phase 9.

