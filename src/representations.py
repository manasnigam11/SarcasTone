"""
Phase 4, 6 & 9 Neural Representation & Zero-Shot Polarity Extraction Module.

Extracts and caches deterministic frozen representations to `data/features/`:
  1. Target-only frozen RoBERTa embeddings (`roberta-base`, 768-d mean-pooled)
  2. Context-aware frozen RoBERTa embeddings (`context </s> target`, 768-d mean-pooled,
     strictly excluding target speaker identity from the target string)
  3. Zero-shot literal text sentiment polarity (`cardiffnlp/twitter-roberta-base-sentiment-latest`:
     `p_neg`, `p_neu`, `p_pos`, `text_sentiment_score = p_pos - p_neg`)
  4. Frozen `facebook/wav2vec2-base` speech representations (768-d mean-pooled over time frames)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)

logger = logging.getLogger(__name__)


def clean_transcript(text: str) -> str:
    """Strip HTML tags (e.g. `<i>...</i>`) and normalize whitespace while preserving punctuation."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_embeddings_and_sentiment(
    df_master: pd.DataFrame,
    roberta_name: str = "roberta-base",
    sentiment_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    out_dir: str | Path = "data/features",
    batch_size: int = 32,
) -> Dict[str, Path]:
    """
    Extract frozen `roberta-base` embeddings (target-only and context+target) and
    zero-shot literal text sentiment scores across all 690 utterances.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    utt_emb_path = out_dir / "roberta_target_embeddings.parquet"
    ctx_emb_path = out_dir / "roberta_context_target_embeddings.parquet"
    sent_path = out_dir / "text_sentiment_polarity.parquet"

    uids = df_master["utterance_id"].astype(str).tolist()
    target_texts = [clean_transcript(t) for t in df_master["utterance"].tolist()]
    # Construct context + target without leaking target speaker ID
    ctx_target_texts = []
    for _, row in df_master.iterrows():
        c_txt = clean_transcript(row["context_text"])
        u_txt = clean_transcript(row["utterance"])
        if c_txt:
            ctx_target_texts.append(f"{c_txt} </s></s> {u_txt}")
        else:
            ctx_target_texts.append(u_txt)

    if not (utt_emb_path.exists() and ctx_emb_path.exists()):
        logger.info("Loading frozen text encoder: %s", roberta_name)
        tokenizer = AutoTokenizer.from_pretrained(roberta_name)
        model = AutoModel.from_pretrained(roberta_name)
        model.eval()

        def encode_texts(texts: List[str], prefix: str) -> pd.DataFrame:
            all_vecs: List[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    enc = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                    out = model(**enc)
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    all_vecs.append(pooled.cpu().numpy().astype(np.float32))
            mat = np.vstack(all_vecs)
            cols = [f"{prefix}_{i:03d}" for i in range(mat.shape[1])]
            df_out = pd.DataFrame(mat, columns=cols)
            df_out.insert(0, "utterance_id", uids)
            return df_out

        df_utt_emb = encode_texts(target_texts, "roberta_utt")
        df_utt_emb.to_parquet(utt_emb_path, index=False)

        df_ctx_emb = encode_texts(ctx_target_texts, "roberta_ctx")
        df_ctx_emb.to_parquet(ctx_emb_path, index=False)
        logger.info("Saved frozen RoBERTa embeddings -> %s, %s", utt_emb_path, ctx_emb_path)

    if not sent_path.exists():
        logger.info("Loading pretrained zero-shot sentiment model: %s", sentiment_name)
        s_tok = AutoTokenizer.from_pretrained(sentiment_name)
        s_mod = AutoModelForSequenceClassification.from_pretrained(sentiment_name)
        s_mod.eval()

        probs_list: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(target_texts), batch_size):
                batch = target_texts[start : start + batch_size]
                enc = s_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                logits = s_mod(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_list.append(probs)

        probs_mat = np.vstack(probs_list)
        # cardiffnlp/twitter-roberta-base-sentiment-latest label order: 0 -> negative, 1 -> neutral, 2 -> positive
        p_neg = probs_mat[:, 0]
        p_neu = probs_mat[:, 1]
        p_pos = probs_mat[:, 2]
        polarity = p_pos - p_neg
        pred_cls = np.argmax(probs_mat, axis=1)
        label_map = {0: "negative", 1: "neutral", 2: "positive"}

        df_sent = pd.DataFrame(
            {
                "utterance_id": uids,
                "text_p_neg": p_neg.astype(np.float32),
                "text_p_neu": p_neu.astype(np.float32),
                "text_p_pos": p_pos.astype(np.float32),
                "text_sentiment_score": polarity.astype(np.float32),
                "text_sentiment_label": [label_map[int(c)] for c in pred_cls],
            }
        )
        df_sent.to_parquet(sent_path, index=False)
        df_sent.to_csv(sent_path.with_suffix(".csv"), index=False)
        logger.info("Saved zero-shot text sentiment polarity -> %s", sent_path)

    return {
        "roberta_target": utt_emb_path,
        "roberta_context": ctx_emb_path,
        "text_sentiment": sent_path,
    }


def extract_wav2vec2_embeddings(
    df_master: pd.DataFrame,
    checkpoint: str = "facebook/wav2vec2-base",
    out_dir: str | Path = "data/features",
    max_duration_sec: float = 15.0,
) -> Path:
    """
    Extract frozen `facebook/wav2vec2-base` representations (768-d mean-pooled over time)
    across all 690 16 kHz mono WAV files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w2v_path = out_dir / "wav2vec2_base_embeddings.parquet"
    if w2v_path.exists():
        logger.info("Cached wav2vec2 embeddings already exist at %s", w2v_path)
        return w2v_path

    logger.info("Loading frozen speech encoder: %s", checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = Wav2Vec2Model.from_pretrained(checkpoint)
    model.eval()

    uids: List[str] = []
    vecs: List[np.ndarray] = []
    max_samples = int(max_duration_sec * 16000)

    with torch.no_grad():
        for idx, row in df_master.iterrows():
            uid = str(row["utterance_id"])
            wav_path = str(row["wav_16k_path"])
            waveform, sr = sf.read(wav_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            elif len(waveform) < 1600:
                waveform = np.pad(waveform, (0, 1600 - len(waveform)))

            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            outputs = model(inputs.input_values, output_hidden_states=True)
            # Combine last hidden state and intermediate transformer layer (layer 7) for rich prosodic+phonetic representation
            last_hs = outputs.last_hidden_state.squeeze(0)  # (T, 768)
            mid_hs = outputs.hidden_states[7].squeeze(0)    # (T, 768)
            combined = 0.5 * (last_hs + mid_hs)
            mean_vec = combined.mean(dim=0).cpu().numpy().astype(np.float32)
            uids.append(uid)
            vecs.append(mean_vec)

    mat = np.vstack(vecs)
    cols = [f"w2v2_{i:03d}" for i in range(mat.shape[1])]
    df_w2v = pd.DataFrame(mat, columns=cols)
    df_w2v.insert(0, "utterance_id", uids)
    df_w2v.to_parquet(w2v_path, index=False)
    logger.info("Saved wav2vec2 embeddings (%s) -> %s", df_w2v.shape, w2v_path)
    return w2v_path

