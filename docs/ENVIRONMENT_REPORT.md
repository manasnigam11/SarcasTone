# Phase 0 — Environment and Hardware Audit Report

**Project:** SarcasTone — A Leakage-Aware Comparative Study of Textual Semantics and Speech Prosody for Multimodal Sarcasm Detection  
**Date of Audit:** November 7, 2026  
**Workspace Root:** `c:\Users\Manas Nigam\Documents\5th Sem nots\NLP 346\project\main`  

---

## 1. Hardware & System Specifications

| Component | Specification | Audit Notes & Implications for SarcasTone |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Single Language (Build 26200, 64-bit) | Native x64 Windows environment; PowerShell shell. |
| **CPU** | 13th Gen Intel® Core™ i5-1340P (12 physical cores, 16 logical threads) | Strong multi-core CPU performance for parallel audio decoding, prosodic extraction, and cross-validation. |
| **System Memory (RAM)** | 16,025 MB (~16.0 GB physical RAM; ~3.7–5.0 GB free at idle) | Moderate RAM headroom; requires batching and disk caching (`data/features/`) when running pretrained Transformer and wav2vec 2.0 inference so memory is never exhausted. |
| **GPU / Accelerator** | Intel® Iris® Xe Integrated Graphics (128 MB dedicated aperture, Driver `31.0.101.5186`) | **No discrete NVIDIA CUDA GPU available** (`torch.cuda.is_available() == False`). |
| **Storage (Drive C:)** | 475.59 GB total, 345.07 GB used, **130.52 GB free** | Ample disk space (>130 GB free) to cache 16 kHz mono WAV files (~70 MB), frozen Transformer embeddings, wav2vec 2.0 representations, and all fold predictions. |

---

## 2. Python Environment & Key Library Audit

- **Python Version:** `3.13.7` (64-bit, `[MSC v.1944 64 bit (AMD64)]`)
- **Executable Path:** `C:\Python313\python.exe`

### Verified Core Dependencies

| Category | Package | Installed Version | Role in SarcasTone |
| :--- | :--- | :--- | :--- |
| **Deep Learning & Transformers** | `torch` | `2.11.0+cpu` | Tensor operations & frozen neural encoder inference |
| | `torchaudio` | `2.11.0` | Audio utilities |
| | `transformers` | `5.4.0` | Pretrained text (`roberta-base`, sentiment) & speech (`wav2vec2-base`) encoders |
| | `tokenizers` / `safetensors` | `0.22.2` / `0.7.0` | Fast tokenization and safe weight serialization |
| **Audio Decoding & Prosody** | `av` (PyAV) | `19.0.2` | Direct in-memory AAC (`mp4a`) audio decoding and 16 kHz mono resampling from `.mp4` |
| | `praat-parselmouth` | `0.4.7` | Gold-standard phonetic pitch ($F_0$), intensity, jitter, shimmer, HNR, and voiced/unvoiced analysis |
| | `opensmile` | `2.6.0` | Standardized `eGeMAPSv02` acoustic & prosodic functionals |
| | `librosa` | `1.0.0` | Spectral features, RMS energy, and rhythm/onset analysis |
| | `soundfile` / `soxr` | `0.14.0` / `1.1.0` | Lossless WAV I/O and high-quality sample-rate conversion |
| **Machine Learning & Stats** | `scikit-learn` | `1.7.2` | TF-IDF, Logistic Regression, SVM, GroupKFold, scalers, imputers, metrics |
| | `scipy` | `1.16.1` | Statistical tests, correlations, and signal processing |
| | `xgboost` | `3.3.0` | Non-linear gradient-boosted decision tree comparisons |
| | `numpy` / `pandas` | `2.2.6` / `2.3.1` | Numerical arrays, data manipulation, and experiment registries |
| **Visualization & Reporting** | `matplotlib` / `seaborn` | `3.10.5` / `0.13.2` | Publication-quality figures and diagnostic plots |
| | `python-docx` / `reportlab` | `1.2.0` / `4.5.0` | Automated generation of IEEE-style Word (`.docx`) and PDF research papers |
| **Testing & Quality** | `pytest` | `9.1.1` | Automated unit and leakage verification test suite |

---

## 3. Hardware-Aware Architectural Adaptations

Because the system runs on a 12-core / 16-thread Intel i5-1340P CPU without a CUDA GPU:
1. **Frozen Pretrained Encoders with Disk Caching:** Rather than end-to-end backpropagating through 100M+ parameter Transformers (`roberta-base`) or speech encoders (`facebook/wav2vec2-base`) inside every cross-validation fold (which would be computationally prohibitive on CPU and prone to severe overfitting on $N=690$ examples), we extract deterministic frozen representations once with `torch.no_grad()`, cache them to `data/features/`, and train regularized classifiers (`LogisticRegression`, `LinearSVC`, `MLP`) strictly within each fold.
2. **PyAV Direct Audio Decoding:** Because system-wide `ffmpeg.exe` is not installed on Windows `PATH`, we use `av` (PyAV `19.0.2`, which bundles FFmpeg shared libraries natively) to decode all `.mp4` audio streams directly to 16 kHz mono float32 waveforms.
3. **Deterministic Execution:** All scripts enforce fixed random seeds (`42`, `1337`, `2026`) across `numpy`, `random`, `torch`, and `scikit-learn`.

