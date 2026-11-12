"""
Phase 4–6 Feature Extraction Runner:
  1. Extracts 41-feature Interpretable Explicit Prosody Suite (4 groups + quality flags) + 88-d eGeMAPSv02
  2. Extracts frozen `roberta-base` target and context+target embeddings
  3. Extracts zero-shot literal text sentiment polarity (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
  4. Extracts frozen `facebook/wav2vec2-base` speech embeddings
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prosody import extract_all_prosody_features
from src.representations import (
    extract_text_embeddings_and_sentiment,
    extract_wav2vec2_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df_master = pd.read_parquet(ROOT / "data" / "processed" / "mustard_master.parquet")

    t0 = time.time()
    logger.info("Step 1/3: Extracting explicit prosodic features and eGeMAPSv02...")
    df_prosody, df_egemaps = extract_all_prosody_features(df_master)
    logger.info("Prosody extraction complete in %.1f s (missing F0 count=%d)",
                time.time() - t0, int(df_prosody["f0_unvoiced_flag"].sum()))

    t1 = time.time()
    logger.info("Step 2/3: Extracting frozen RoBERTa embeddings and zero-shot text sentiment...")
    text_paths = extract_text_embeddings_and_sentiment(df_master)
    logger.info("Text representation & sentiment extraction complete in %.1f s -> %s",
                time.time() - t1, text_paths)

    t2 = time.time()
    logger.info("Step 3/3: Extracting frozen wav2vec 2.0 speech representations...")
    w2v_path = extract_wav2vec2_embeddings(df_master)
    logger.info("wav2vec 2.0 extraction complete in %.1f s -> %s", time.time() - t2, w2v_path)


if __name__ == "__main__":
    main()

