"""
Bridge Script: Paper 1 (ParentPaper/FinBERT) -> Paper 2 (LOBFrame/DeepLOB).

Builds on ParentPaper: we use his load_finbert() and (when available) load_news_csv().
We only add hidden-state extraction (768-d vectors) for LOBFrame; we do not rewrite his code.

1) Extract high-dimensional sentiment vectors (hidden state) from FinBERT, not just labels.
2) Output format compatible with LOBFrame: (N, sentiment_dim) tensors alignable to LOB samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse ParentPaper's FinBERT loading; we only add hidden-state extraction
from parent_paper_engine1 import (
    load_finbert_and_tokenizer,
    text_to_sentiment_vectors,
    load_news_from_parent_paper,
    SENTIMENT_DIM,
)

DEFAULT_MODEL = "ProsusAI/finbert"
HIDDEN_SIZE = SENTIMENT_DIM


def load_finbert_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Load FinBERT using ParentPaper's load_finbert when available; else standalone."""
    return load_finbert_and_tokenizer(model_name, device)


def extract_hidden_states(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 128,
    pool: str = "cls",
) -> np.ndarray:
    """
    Extract 768-d sentiment vectors. Uses ParentPaper's model (model.bert for hidden state).
    """
    use_cls = pool == "cls"
    vectors = text_to_sentiment_vectors(
        texts, tokenizer, model, device, max_length=max_length, use_cls_token=use_cls
    )
    return vectors.numpy().astype(np.float32)


def run_on_csv(
    csv_path: str,
    text_column: str = "headline",
    model_name: str = DEFAULT_MODEL,
    max_length: int = 128,
    pool: str = "cls",
    output_path: str | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Read CSV, run FinBERT hidden-state extraction, return DataFrame with extra column
    of list/array (or save as .npz for LOBFrame).

    CSV should have at least a column for text (e.g. headline). Uses ParentPaper's load_news_csv when available.
    """
    try:
        df = load_news_from_parent_paper(csv_path=csv_path)
    except Exception:
        df = None
    if df is None:
        df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"CSV must have column '{text_column}'")

    tokenizer, model, device = load_finbert_encoder(model_name)

    all_vectors = []
    texts = df[text_column].astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        vecs = extract_hidden_states(
            batch_texts, tokenizer, model, device, max_length=max_length, pool=pool
        )
        all_vectors.append(vecs)

    vectors = np.concatenate(all_vectors, axis=0)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".npz"), sentiment_vectors=vectors)
        print(f"Saved sentiment vectors shape {vectors.shape} to {out.with_suffix('.npz')}")

    # Optionally attach to DataFrame (as object column of arrays)
    df["sentiment_vector"] = list(vectors)
    if output_path and str(output_path).endswith(".csv"):
        # CSV can't store arrays nicely; save vectors separately
        df.drop(columns=["sentiment_vector"], inplace=True)
        df.to_csv(output_path, index=False)
        print(f"Saved CSV (without vector column) to {output_path}")

    return df, vectors


def align_sentiment_to_lob_indices(
    sentiment_vectors: np.ndarray,
    lob_indices_or_timestamps: np.ndarray,
    news_timestamps: np.ndarray,
    news_sentiment_vectors: np.ndarray,
    strategy: str = "latest",
) -> np.ndarray:
    """
    Map each LOB sample index (or timestamp) to a sentiment vector.

    Strategies:
      - "latest": for each LOB time t, use sentiment from the most recent news with time <= t.
      - "index": lob_indices_or_timestamps is 1:1 with LOB dataset __len__; news is aligned
        by same index (e.g. pre-merged dataset). Then sentiment_vectors[i] = news_sentiment_vectors[i].

    For "latest": lob_indices_or_timestamps and news_timestamps should be numeric or datetime-like.
    """
    if strategy == "index":
        if len(lob_indices_or_timestamps) != len(news_sentiment_vectors):
            raise ValueError("For index strategy, len(lob_indices) must equal len(news_sentiment_vectors)")
        return np.array(news_sentiment_vectors, dtype=np.float32)

    if strategy != "latest":
        raise ValueError('strategy must be "latest" or "index"')

    # latest: for each LOB time, find latest news <= that time
    lob_times = np.asarray(lob_indices_or_timestamps).ravel()
    news_times = np.asarray(news_timestamps).ravel()
    out = np.zeros((len(lob_times), news_sentiment_vectors.shape[1]), dtype=np.float32)

    for i, t in enumerate(lob_times):
        valid = news_times <= t
        if not np.any(valid):
            # no news before this time: use zero vector or first available
            out[i] = news_sentiment_vectors[0] if len(news_sentiment_vectors) else 0
            continue
        # index of latest news time <= t
        valid_times = np.where(valid, news_times, -np.inf)
        idx = np.argmax(valid_times)
        out[i] = news_sentiment_vectors[idx]

    return out


def main():
    parser = argparse.ArgumentParser(description="Bridge: FinBERT hidden state -> LOBFrame input")
    parser.add_argument("--csv", type=str, required=True, help="CSV with text column (e.g. headline)")
    parser.add_argument("--text_column", type=str, default="headline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--pool", choices=("cls", "mean"), default="cls")
    parser.add_argument("--out", type=str, default=None, help="Output .npz (vectors) or .csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df, vectors = run_on_csv(
        args.csv,
        text_column=args.text_column,
        model_name=args.model,
        max_length=args.maxlen,
        pool=args.pool,
        output_path=args.out or "sentiment_vectors.npz",
        batch_size=args.batch_size,
    )
    print(f"Computed {len(vectors)} sentiment vectors, shape {vectors.shape}")


if __name__ == "__main__":
    main()
