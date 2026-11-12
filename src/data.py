"""
Dataset Assembly, Media Validation, Audio Extraction, and Scene-Cluster Graph Module.

Builds the canonical 690-instance Original MUStARD dataset enriched with held-out
MUStARD++ subgroup annotations (Sarcasm_Type, Valence, Arousal, Explicit/Implicit Emotion),
extracts 16 kHz mono WAV audio files from the raw MP4 archive without modifying original files,
and constructs disjoint scene/dialogue clusters for leakage-free evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)


def parse_mp4_header(filepath: str | Path) -> Dict[str, Any]:
    """Parse MP4 container atoms (`ftyp`, `mvhd`, `trak`, `stsd`, `stsz`) to extract media metadata."""
    filepath = Path(filepath)
    info: Dict[str, Any] = {
        "file_size_bytes": filepath.stat().st_size,
        "valid_mp4": False,
        "has_video": False,
        "has_audio": False,
        "v_codec": None,
        "v_width": None,
        "v_height": None,
        "v_duration_sec": None,
        "v_frames": None,
        "v_fps": None,
        "a_codec": None,
        "a_channels": None,
        "a_sample_size": None,
        "a_sample_rate": None,
        "a_duration_sec": None,
        "a_packets": None,
    }
    with open(filepath, "rb") as f:
        end = info["file_size_bytes"]

        def walk(start: int, stop: int, cur_track: Optional[Dict[str, Any]] = None) -> None:
            pos = start
            while pos + 8 <= stop:
                f.seek(pos)
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                sz, atype_b = struct.unpack(">I4s", hdr)
                atype = atype_b.decode("latin1", errors="replace")
                hdr_len = 8
                if sz == 1:
                    sz = struct.unpack(">Q", f.read(8))[0]
                    hdr_len = 16
                elif sz == 0:
                    sz = stop - pos
                if sz < hdr_len or pos + sz > stop:
                    break
                if atype == "ftyp":
                    info["valid_mp4"] = True
                elif atype == "trak":
                    t: Dict[str, Any] = {
                        "handler": None,
                        "ts": 0,
                        "dur": 0,
                        "samples": 0,
                        "codec": None,
                        "w": None,
                        "h": None,
                        "ch": None,
                        "ss": None,
                        "sr": None,
                    }
                    walk(pos + hdr_len, pos + sz, t)
                    if t["handler"] == "vide":
                        info["has_video"] = True
                        info["v_codec"] = t["codec"]
                        info["v_width"] = t["w"]
                        info["v_height"] = t["h"]
                        info["v_frames"] = t["samples"]
                        if t["ts"] > 0:
                            info["v_duration_sec"] = t["dur"] / t["ts"]
                            if info["v_duration_sec"] > 0:
                                info["v_fps"] = round(t["samples"] / info["v_duration_sec"], 2)
                    elif t["handler"] == "soun":
                        info["has_audio"] = True
                        info["a_codec"] = t["codec"]
                        info["a_channels"] = t["ch"]
                        info["a_sample_size"] = t["ss"]
                        info["a_sample_rate"] = t["sr"]
                        info["a_packets"] = t["samples"]
                        if t["ts"] > 0:
                            info["a_duration_sec"] = t["dur"] / t["ts"]
                elif atype in ("moov", "mdia", "minf", "stbl"):
                    walk(pos + hdr_len, pos + sz, cur_track)
                elif atype == "mdhd" and cur_track is not None:
                    f.seek(pos + hdr_len)
                    ver = struct.unpack(">B", f.read(1))[0]
                    f.read(3)
                    if ver == 1:
                        f.read(16)
                        ts, dur = struct.unpack(">IQ", f.read(12))
                    else:
                        f.read(8)
                        ts, dur = struct.unpack(">II", f.read(8))
                    cur_track["ts"] = ts
                    cur_track["dur"] = dur
                elif atype == "hdlr" and cur_track is not None:
                    f.seek(pos + hdr_len + 8)
                    cur_track["handler"] = f.read(4).decode("latin1", errors="replace")
                elif atype == "stsd" and cur_track is not None:
                    f.seek(pos + hdr_len + 4)
                    n_ent = struct.unpack(">I", f.read(4))[0]
                    if n_ent > 0:
                        esize, etype_b = struct.unpack(">I4s", f.read(8))
                        etype = etype_b.decode("latin1", errors="replace")
                        cur_track["codec"] = etype
                        if etype in ("mp4a", "aac "):
                            f.seek(f.tell() + 16)
                            ch, ss, _, _ = struct.unpack(">HHHH", f.read(8))
                            sr = struct.unpack(">I", f.read(4))[0] >> 16
                            cur_track["ch"] = ch
                            cur_track["ss"] = ss
                            cur_track["sr"] = sr
                        elif etype in ("avc1", "mp4v", "hvc1"):
                            f.seek(f.tell() + 24)
                            w, h = struct.unpack(">HH", f.read(4))
                            cur_track["w"] = w
                            cur_track["h"] = h
                elif atype == "stsz" and cur_track is not None:
                    f.seek(pos + hdr_len + 8)
                    cur_track["samples"] = struct.unpack(">I", f.read(4))[0]
                pos += sz

        walk(0, end)
    return info


def decode_mp4_audio_to_mono_16k(mp4_path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Decode AAC audio stream from an MP4 file into a 1D float32 mono waveform at `target_sr` Hz."""
    container = av.open(str(mp4_path))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    chunks: List[np.ndarray] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray()[0])
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray()[0])
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    waveform = np.concatenate(chunks).astype(np.float32)
    return waveform


def build_scene_clusters(sarcasm_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Construct disjoint scene/dialogue group identifiers (`scene_group_id`) across the 690 clips.

    Two clips are merged into the same connected scene component if:
      1. They share any non-trivial dialogue sentence (>= 20 characters) across their target
         utterance or context dialogue turns, OR
      2. They belong to the same TV show and prefix (`1_` or `2_`) and their clip numbers
         are within delta <= 10 for BBT (`1_`) or delta <= 2 for `2_` with overlapping speakers.
    """
    uids = sorted(sarcasm_data.keys(), key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))
    parent = {u: u for u in uids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep lexicographically smaller root for determinism
            if (int(ra.split("_")[0]), int(ra.split("_")[1])) <= (int(rb.split("_")[0]), int(rb.split("_")[1])):
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Rule 1: Shared non-trivial sentence (>= 20 chars) across target or context
    sent_to_ids: Dict[str, List[str]] = defaultdict(list)
    for u in uids:
        entry = sarcasm_data[u]
        all_turns = [entry["utterance"].strip()] + [c.strip() for c in entry.get("context", [])]
        for turn in all_turns:
            norm = " ".join(turn.lower().split())
            if len(norm) >= 20:
                sent_to_ids[norm].append(u)

    for norm_sent, id_list in sent_to_ids.items():
        if len(id_list) > 1:
            first = id_list[0]
            for other in id_list[1:]:
                if sarcasm_data[first]["show"] == sarcasm_data[other]["show"]:
                    union(first, other)

    # Rule 2: Close numeric clip IDs in the same show with overlapping speaker cast
    for i in range(len(uids) - 1):
        u1, u2 = uids[i], uids[i + 1]
        p1, n1 = map(int, u1.split("_"))
        p2, n2 = map(int, u2.split("_"))
        if p1 != p2 or sarcasm_data[u1]["show"] != sarcasm_data[u2]["show"]:
            continue
        spk1 = {sarcasm_data[u1]["speaker"]} | set(sarcasm_data[u1].get("context_speakers", []))
        spk2 = {sarcasm_data[u2]["speaker"]} | set(sarcasm_data[u2].get("context_speakers", []))
        diff = abs(n2 - n1)
        if (p1 == 1 and diff <= 10) or (p1 == 2 and diff <= 2 and len(spk1 & spk2) > 0):
            union(u1, u2)

    return {u: f"SCENE_{find(u)}" for u in uids}


def build_master_dataset(
    raw_video_dir: str | Path = "mmsd_raw_data",
    metadata_json_path: str | Path = "data/metadata/sarcasm_data.json",
    mustard_pp_csv_path: str | Path = "data/metadata/mustard_plus_plus_text.csv",
    audio_out_dir: str | Path = "data/processed/audio_16k",
    master_parquet_path: str | Path = "data/processed/mustard_master.parquet",
    master_csv_path: str | Path = "data/processed/mustard_master.csv",
    target_sr: int = 16000,
) -> pd.DataFrame:
    """
    Validate all 690 utterance and context MP4 files, extract 16 kHz mono WAVs,
    join `sarcasm_data.json` and held-out `mustard++_text.csv` subgroup metadata,
    and save the canonical master table.
    """
    raw_video_dir = Path(raw_video_dir)
    utt_dir = raw_video_dir / "utterances_final"
    ctx_dir = raw_video_dir / "context_final"
    audio_out_dir = Path(audio_out_dir)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        sarcasm_data: Dict[str, Any] = json.load(f)

    # Load MUStARD++ target rows for subgroup metadata enrichment
    df_mpp = pd.read_csv(mustard_pp_csv_path)
    mpp_targets = df_mpp[~df_mpp["KEY"].astype(str).str.contains("_c_")].copy()
    mpp_targets["SCENE"] = mpp_targets["SCENE"].astype(str)
    mpp_lookup = mpp_targets.set_index("SCENE").to_dict(orient="index")

    scene_clusters = build_scene_clusters(sarcasm_data)
    uids = sorted(sarcasm_data.keys(), key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    records: List[Dict[str, Any]] = []
    for uid in uids:
        meta = sarcasm_data[uid]
        utt_mp4 = utt_dir / f"{uid}.mp4"
        ctx_mp4 = ctx_dir / f"{uid}_c.mp4"
        if not utt_mp4.exists() or not ctx_mp4.exists():
            raise FileNotFoundError(f"Missing video pair for {uid}: {utt_mp4} or {ctx_mp4}")

        utt_hdr = parse_mp4_header(utt_mp4)
        ctx_hdr = parse_mp4_header(ctx_mp4)

        wav_path = audio_out_dir / f"{uid}.wav"
        if not wav_path.exists():
            waveform = decode_mp4_audio_to_mono_16k(utt_mp4, target_sr=target_sr)
            sf.write(str(wav_path), waveform, target_sr, subtype="PCM_16")
        else:
            waveform, _ = sf.read(str(wav_path), dtype="float32")

        audio_dur_sec = float(len(waveform)) / float(target_sr)
        rms_global = float(np.sqrt(np.mean(np.square(waveform)))) if len(waveform) > 0 else 0.0
        peak_abs = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0

        mpp_info = mpp_lookup.get(uid, {})
        utt_text = str(meta["utterance"]).strip()
        ctx_turns = [str(c).strip() for c in meta.get("context", [])]
        ctx_speakers = [str(s).strip() for s in meta.get("context_speakers", [])]
        ctx_formatted = " | ".join(
            f"{spk}: {turn}" for spk, turn in zip(ctx_speakers, ctx_turns)
        )

        speaker_raw = str(meta["speaker"]).strip()
        show_raw = str(meta["show"]).strip()
        speaker_group = f"{show_raw}_{speaker_raw}"

        records.append(
            {
                "utterance_id": uid,
                "prefix_batch": int(uid.split("_")[0]),
                "clip_number": int(uid.split("_")[1]),
                "show": show_raw,
                "speaker": speaker_raw,
                "speaker_group": speaker_group,
                "scene_group_id": scene_clusters[uid],
                "label": int(bool(meta["sarcasm"])),
                "utterance": utt_text,
                "utterance_char_len": len(utt_text),
                "utterance_word_count": len(utt_text.split()),
                "context_turns_json": json.dumps(ctx_turns, ensure_ascii=False),
                "context_speakers_json": json.dumps(ctx_speakers, ensure_ascii=False),
                "context_num_turns": len(ctx_turns),
                "context_text": " ".join(ctx_turns),
                "context_dialogue_formatted": ctx_formatted,
                "utt_mp4_path": str(utt_mp4.as_posix()),
                "ctx_mp4_path": str(ctx_mp4.as_posix()),
                "wav_16k_path": str(wav_path.as_posix()),
                "v_width": int(utt_hdr["v_width"]),
                "v_height": int(utt_hdr["v_height"]),
                "v_fps": float(utt_hdr["v_fps"]),
                "orig_audio_sr": int(utt_hdr["a_sample_rate"]),
                "orig_audio_channels": int(utt_hdr["a_channels"]),
                "utt_duration_sec": round(audio_dur_sec, 4),
                "ctx_duration_sec": round(float(ctx_hdr["v_duration_sec"] or 0.0), 4),
                "audio_rms": round(rms_global, 6),
                "audio_peak": round(peak_abs, 6),
                "in_mustard_pp": uid in mpp_lookup,
                "sarcasm_type": str(mpp_info.get("Sarcasm_Type", "MISSING")),
                "implicit_emotion": str(mpp_info.get("Implicit_Emotion", "MISSING")),
                "explicit_emotion": str(mpp_info.get("Explicit_Emotion", "MISSING")),
                "human_valence": float(mpp_info["Valence"]) if pd.notna(mpp_info.get("Valence")) else np.nan,
                "human_arousal": float(mpp_info["Arousal"]) if pd.notna(mpp_info.get("Arousal")) else np.nan,
            }
        )

    df = pd.DataFrame(records)
    Path(master_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(master_parquet_path, index=False)
    df.to_csv(master_csv_path, index=False)
    logger.info("Built master dataset with %d rows -> %s", len(df), master_parquet_path)
    return df

