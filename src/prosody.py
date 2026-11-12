"""
Explicit Prosodic Feature Extraction Module (Phase 5).

Extracts the 4 mandated interpretable prosodic feature groups from 16 kHz mono WAV audio:
  - Group 1: Fundamental Frequency (F0) via Praat (`parselmouth`) autocorrelation pitch tracker
  - Group 2: Energy and Intensity via Praat Intensity + `librosa` RMS contours
  - Group 3: Timing, Rhythm, and Pauses via silence/speech segmentation and syllable-nucleus peaks
  - Group 4: Voice Quality via Praat PointProcess Jitter, Shimmer, and Harmonics-to-Noise Ratio (HNR)

Also includes explicit quality/missingness flags (`f0_unvoiced_flag`, `short_clip_flag`, `voiced_fraction`)
and optional `openSMILE` eGeMAPSv02 88-dimensional functional extraction.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import opensmile
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf

logger = logging.getLogger(__name__)

# Explicit feature group definitions for Phase 8C Modality & Cue Ablation
PROSODY_FEATURE_GROUPS: Dict[str, List[str]] = {
    "f0": [
        "f0_mean_hz",
        "f0_median_hz",
        "f0_std_hz",
        "f0_min_hz",
        "f0_max_hz",
        "f0_range_hz",
        "f0_iqr_hz",
        "f0_cv",
        "f0_slope_hz_per_sec",
        "voiced_fraction",
        "f0_unvoiced_flag",
    ],
    "energy": [
        "intensity_mean_db",
        "intensity_std_db",
        "intensity_max_db",
        "intensity_range_db",
        "intensity_iqr_db",
        "rms_mean",
        "rms_std",
        "rms_max",
        "rms_dynamic_range",
        "energy_slope_db_per_sec",
    ],
    "timing": [
        "utt_duration_sec",
        "speech_duration_sec",
        "speech_ratio",
        "silence_ratio",
        "pause_count",
        "pause_mean_dur_sec",
        "pause_max_dur_sec",
        "pause_rate_per_sec",
        "speaking_rate_wpm_approx",
        "articulation_rate_sps",
        "short_clip_flag",
    ],
    "voice_quality": [
        "jitter_local",
        "jitter_rap",
        "shimmer_local",
        "shimmer_apq3",
        "hnr_mean_db",
        "hnr_std_db",
        "spectral_centroid_mean_hz",
        "spectral_flatness_mean",
        "zero_crossing_rate_mean",
    ],
}

ALL_EXPLICIT_PROSODY_FEATURES: List[str] = (
    PROSODY_FEATURE_GROUPS["f0"]
    + PROSODY_FEATURE_GROUPS["energy"]
    + PROSODY_FEATURE_GROUPS["timing"]
    + PROSODY_FEATURE_GROUPS["voice_quality"]
)


def _count_text_syllables_approx(text: str) -> int:
    """Approximate English syllable count from transcript words (used only as diagnostic reference)."""
    text = text.lower()
    words = re.findall(r"[a-z']+", text)
    if not words:
        return 1
    total = 0
    for w in words:
        w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
        w = re.sub(r"^y", "", w)
        vowels = re.findall(r"[aeiouy]{1,2}", w)
        total += max(1, len(vowels))
    return max(1, total)


def extract_utterance_prosody(
    wav_path: str | Path,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    silence_top_db: float = 25.0,
) -> Dict[str, float]:
    """
    Extract all 41 explicit prosodic features across the 4 groups from a single 16 kHz mono WAV file.
    Note: Does NOT use the transcript so that the audio feature representation remains strictly audio-only!
    Unvoiced F0 or PointProcess failures remain `np.nan` so training-fold-only imputation handles them cleanly.
    """
    waveform, sr = sf.read(str(wav_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    total_dur = float(len(waveform)) / float(sr) if sr > 0 else 0.0
    short_clip_flag = 1.0 if total_dur < 1.0 else 0.0

    # Peak-normalize waveform copy for relative intensity/voice-quality stability while preserving raw RMS
    rms_frames = librosa.feature.rms(y=waveform, frame_length=512, hop_length=160)[0]
    rms_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_std = float(np.std(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_max = float(np.max(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_min = float(np.percentile(rms_frames, 5)) if len(rms_frames) > 0 else 0.0
    rms_dyn = float(rms_max - rms_min)

    # Trim leading/trailing silence and segment internal non-silent intervals via librosa.effects.split
    non_silent_intervals = librosa.effects.split(waveform, top_db=silence_top_db, frame_length=512, hop_length=160)
    if len(non_silent_intervals) > 0:
        speech_samples = sum(int(end - start) for start, end in non_silent_intervals)
        speech_dur = float(speech_samples) / float(sr)
        # Internal pauses between voiced/active intervals
        pause_durations: List[float] = []
        for i in range(len(non_silent_intervals) - 1):
            gap_sec = float(non_silent_intervals[i + 1][0] - non_silent_intervals[i][1]) / float(sr)
            if gap_sec >= 0.08:  # >= 80 ms counts as an acoustic pause
                pause_durations.append(gap_sec)
        # Active waveform trimmed to first speech onset -> last speech offset (helps suppress trailing laugh/silence)
        active_start = int(non_silent_intervals[0][0])
        active_end = int(non_silent_intervals[-1][1])
        active_wav = waveform[active_start:active_end] if (active_end - active_start) >= int(0.25 * sr) else waveform
    else:
        speech_dur = 0.0
        pause_durations = []
        active_wav = waveform

    speech_ratio = float(np.clip(speech_dur / max(total_dur, 1e-4), 0.0, 1.0))
    silence_ratio = float(1.0 - speech_ratio)
    pause_count = float(len(pause_durations))
    pause_mean_dur = float(np.mean(pause_durations)) if pause_durations else 0.0
    pause_max_dur = float(np.max(pause_durations)) if pause_durations else 0.0
    pause_rate = float(pause_count / max(total_dur, 1e-4))

    # Acoustic syllable-nucleus estimation via onset envelope peaks on the active waveform
    onset_env = librosa.onset.onset_strength(y=active_wav, sr=sr, hop_length=160)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.25, wait=8
    )
    est_syllables = float(max(len(peaks), 1))
    speaking_rate_sps = float(est_syllables / max(total_dur, 1e-4))
    articulation_rate_sps = float(est_syllables / max(speech_dur, 0.15))

    # Praat Sound object for F0, Intensity, Jitter, Shimmer, and HNR
    snd = parselmouth.Sound(waveform.astype(np.float64), sampling_frequency=float(sr))

    # Group 1: Fundamental Frequency (F0)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0_vals = pitch.selected_array["frequency"]
    voiced_mask = f0_vals > 0
    voiced_f0 = f0_vals[voiced_mask]
    voiced_fraction = float(np.mean(voiced_mask)) if len(f0_vals) > 0 else 0.0

    if len(voiced_f0) >= 3:
        f0_unvoiced_flag = 0.0
        f0_mean = float(np.mean(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_min_val = float(np.min(voiced_f0))
        f0_max_val = float(np.max(voiced_f0))
        f0_range = float(f0_max_val - f0_min_val)
        f0_iqr = float(np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25))
        f0_cv = float(f0_std / max(f0_mean, 1e-4))
        voiced_times = np.where(voiced_mask)[0] * 0.01
        if len(voiced_times) >= 3 and np.std(voiced_times) > 0:
            f0_slope = float(np.polyfit(voiced_times, voiced_f0, 1)[0])
        else:
            f0_slope = 0.0
    else:
        f0_unvoiced_flag = 1.0
        f0_mean = np.nan
        f0_median = np.nan
        f0_std = np.nan
        f0_min_val = np.nan
        f0_max_val = np.nan
        f0_range = np.nan
        f0_iqr = np.nan
        f0_cv = np.nan
        f0_slope = np.nan

    # Group 2: Intensity (dB)
    try:
        intensity = snd.to_intensity(minimum_pitch=f0_min, time_step=0.01)
        int_vals = intensity.values[0]
        int_vals = int_vals[np.isfinite(int_vals)]
    except Exception:
        int_vals = np.array([])

    if len(int_vals) >= 2:
        int_mean = float(np.mean(int_vals))
        int_std = float(np.std(int_vals))
        int_max = float(np.max(int_vals))
        int_range = float(int_max - np.min(int_vals))
        int_iqr = float(np.percentile(int_vals, 75) - np.percentile(int_vals, 25))
        t_int = np.arange(len(int_vals)) * 0.01
        energy_slope = float(np.polyfit(t_int, int_vals, 1)[0]) if len(t_int) >= 3 else 0.0
    else:
        int_mean = float(20.0 * np.log10(max(rms_mean, 1e-6)))
        int_std = 0.0
        int_max = float(20.0 * np.log10(max(rms_max, 1e-6)))
        int_range = 0.0
        int_iqr = 0.0
        energy_slope = 0.0

    # Group 4: Voice Quality (Jitter, Shimmer, HNR, Spectral descriptors)
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
        jitter_local = float(call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3))
        jitter_rap = float(call(point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3))
        shimmer_local = float(
            call([snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        )
        shimmer_apq3 = float(
            call([snd, point_process], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        )
    except Exception:
        jitter_local = np.nan
        jitter_rap = np.nan
        shimmer_local = np.nan
        shimmer_apq3 = np.nan

    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
        hnr_vals = harmonicity.values[0]
        valid_hnr = hnr_vals[hnr_vals > -100.0]
        if len(valid_hnr) >= 2:
            hnr_mean = float(np.mean(valid_hnr))
            hnr_std = float(np.std(valid_hnr))
        else:
            hnr_mean = np.nan
            hnr_std = np.nan
    except Exception:
        hnr_mean = np.nan
        hnr_std = np.nan

    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=512, hop_length=160)))
    spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=waveform, n_fft=512, hop_length=160)))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform, frame_length=512, hop_length=160)))

    return {
        # Group 1: F0
        "f0_mean_hz": f0_mean,
        "f0_median_hz": f0_median,
        "f0_std_hz": f0_std,
        "f0_min_hz": f0_min_val,
        "f0_max_hz": f0_max_val,
        "f0_range_hz": f0_range,
        "f0_iqr_hz": f0_iqr,
        "f0_cv": f0_cv,
        "f0_slope_hz_per_sec": f0_slope,
        "voiced_fraction": voiced_fraction,
        "f0_unvoiced_flag": f0_unvoiced_flag,
        # Group 2: Energy & Intensity
        "intensity_mean_db": int_mean,
        "intensity_std_db": int_std,
        "intensity_max_db": int_max,
        "intensity_range_db": int_range,
        "intensity_iqr_db": int_iqr,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "rms_dynamic_range": rms_dyn,
        "energy_slope_db_per_sec": energy_slope,
        # Group 3: Timing & Pauses
        "utt_duration_sec": total_dur,
        "speech_duration_sec": speech_dur,
        "speech_ratio": speech_ratio,
        "silence_ratio": silence_ratio,
        "pause_count": pause_count,
        "pause_mean_dur_sec": pause_mean_dur,
        "pause_max_dur_sec": pause_max_dur,
        "pause_rate_per_sec": pause_rate,
        "speaking_rate_wpm_approx": speaking_rate_sps * 60.0,
        "articulation_rate_sps": articulation_rate_sps,
        "short_clip_flag": short_clip_flag,
        # Group 4: Voice Quality
        "jitter_local": jitter_local,
        "jitter_rap": jitter_rap,
        "shimmer_local": shimmer_local,
        "shimmer_apq3": shimmer_apq3,
        "hnr_mean_db": hnr_mean,
        "hnr_std_db": hnr_std,
        "spectral_centroid_mean_hz": spec_cent,
        "spectral_flatness_mean": spec_flat,
        "zero_crossing_rate_mean": zcr_mean,
    }


def extract_all_prosody_features(
    df_master: pd.DataFrame,
    out_prosody_parquet: str | Path = "data/features/explicit_prosody.parquet",
    out_egemaps_parquet: str | Path = "data/features/egemaps_v02.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract both the 4-group Interpretable Prosody Suite and openSMILE eGeMAPSv02 across all 690 clips."""
    out_prosody_parquet = Path(out_prosody_parquet)
    out_egemaps_parquet = Path(out_egemaps_parquet)
    out_prosody_parquet.parent.mkdir(parents=True, exist_ok=True)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    prosody_rows: List[Dict[str, Any]] = []
    egemaps_rows: List[Dict[str, Any]] = []

    for _, row in df_master.iterrows():
        uid = str(row["utterance_id"])
        wav_path = str(row["wav_16k_path"])
        feats = extract_utterance_prosody(wav_path)
        feats["utterance_id"] = uid
        prosody_rows.append(feats)

        # openSMILE eGeMAPSv02
        eg_df = smile.process_file(wav_path)
        eg_dict = {f"egemaps_{col}": float(eg_df.iloc[0][col]) for col in eg_df.columns}
        eg_dict["utterance_id"] = uid
        egemaps_rows.append(eg_dict)

    df_prosody = pd.DataFrame(prosody_rows)
    # Reorder with utterance_id first
    cols_p = ["utterance_id"] + ALL_EXPLICIT_PROSODY_FEATURES
    df_prosody = df_prosody[cols_p]
    df_prosody.to_parquet(out_prosody_parquet, index=False)
    df_prosody.to_csv(out_prosody_parquet.with_suffix(".csv"), index=False)

    df_egemaps = pd.DataFrame(egemaps_rows)
    eg_cols = ["utterance_id"] + [c for c in df_egemaps.columns if c != "utterance_id"]
    df_egemaps = df_egemaps[eg_cols]
    df_egemaps.to_parquet(out_egemaps_parquet, index=False)

    logger.info(
        "Extracted explicit prosody (%s) and eGeMAPSv02 (%s)",
        df_prosody.shape,
        df_egemaps.shape,
    )
    return df_prosody, df_egemaps

