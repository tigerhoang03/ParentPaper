"""
LOB dataset + sentiment from Part 1 CSV.

Use when you have:
  - An LOB dataset that returns (lob_tensor, label).
  - Part 1 CSV (main.py or datasplit output) with sent_score and optionally
    sent_probs_neg, sent_probs_neu, sent_probs_pos.

Sentiment is built from CSV columns: (N, 1) from sent_score or (N, 3) from probs.
LOB and CSV rows are aligned by index (same order).
"""

from __future__ import annotations

from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset


def sentiment_columns_from_csv(
    df: pd.DataFrame,
    use_probs: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Build sentiment tensor from Part 1 CSV columns.

    Args:
        df: DataFrame with at least sent_score; optionally sent_probs_neg, sent_probs_neu, sent_probs_pos.
        use_probs: If True and prob columns exist, return (N, 3) and sentiment_dim=3.

    Returns:
        sentiment: (N, 1) or (N, 3) float32 tensor
        sentiment_dim: 1 or 3
    """
    if "sent_score" not in df.columns:
        raise ValueError(f"DataFrame must have 'sent_score'. Columns: {list(df.columns)}")
    if use_probs:
        prob_cols = ["sent_probs_neg", "sent_probs_neu", "sent_probs_pos"]
        if all(c in df.columns for c in prob_cols):
            arr = df[prob_cols].astype("float32").values
            return torch.tensor(arr, dtype=torch.float32), 3
    arr = df["sent_score"].astype("float32").values.reshape(-1, 1)
    return torch.tensor(arr, dtype=torch.float32), 1


class LOBWithSentimentDataset(Dataset):
    """
    Wraps an LOB dataset and sentiment from Part 1 DataFrame/CSV.
    __getitem__ returns (lob_tensor, sentiment_tensor, label).
    Sentiment is (1,) or (3,) per sample from sent_score or sent_probs.
    """

    def __init__(
        self,
        lob_dataset: Dataset,
        sentiment_df: pd.DataFrame | None = None,
        sentiment_csv_path: str | None = None,
        sentiment_columns: List[str] | None = None,
        use_probs: bool = False,
    ):
        """
        Args:
            lob_dataset: Dataset that returns (lob, label) in __getitem__.
            sentiment_df: Part 1 DataFrame (must have sent_score; optionally sent_probs_*).
            sentiment_csv_path: Path to Part 1 CSV; used if sentiment_df is None.
            sentiment_columns: Optional explicit list, e.g. ["sent_score"] or
                ["sent_probs_neg","sent_probs_neu","sent_probs_pos"]. If None, inferred from use_probs.
            use_probs: If True, use 3-d probs when available.
        """
        self.lob_dataset = lob_dataset
        if sentiment_df is not None:
            df = sentiment_df
        elif sentiment_csv_path is not None:
            df = pd.read_csv(sentiment_csv_path)
        else:
            raise ValueError("Provide either sentiment_df or sentiment_csv_path")

        if sentiment_columns is not None:
            if "sent_score" in sentiment_columns and len(sentiment_columns) == 1:
                self.sentiment = torch.tensor(df["sent_score"].astype("float32").values.reshape(-1, 1), dtype=torch.float32)
            else:
                self.sentiment = torch.tensor(df[sentiment_columns].astype("float32").values, dtype=torch.float32)
        else:
            self.sentiment, _ = sentiment_columns_from_csv(df, use_probs=use_probs)

        n_lob = len(self.lob_dataset)
        n_sent = self.sentiment.shape[0]
        if n_lob != n_sent:
            raise ValueError(
                f"LOB dataset length ({n_lob}) must match sentiment length ({n_sent}). "
                "Align by index (same row order)."
            )

    def __len__(self) -> int:
        return len(self.lob_dataset)

    def __getitem__(self, idx: int):
        lob, label = self.lob_dataset[idx]
        sent = self.sentiment[idx]
        return lob, sent, label


def collate_lob_sentiment(batch):
    """DataLoader collate_fn when batch is (lob, sentiment, label)."""
    lobs = torch.stack([b[0] for b in batch])
    sents = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return (lobs, sents), labels
