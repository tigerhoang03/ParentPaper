"""
Example wrapper: LOBFrame CustomDataset + precomputed sentiment vectors.

Use this when you have:
  - LOBFrame's CustomDataset (or a .pt saved dataset) that returns (lob_tensor, label).
  - A .npz file from bridge_sentiment_to_lob.py with key "sentiment_vectors", shape (N, dim).
    (Sentiment can be produced from ParentPaper-style CSVs without modifying ParentPaper.)

We assume the same ordering and length: sample index i in the LOB dataset corresponds to
row i in the sentiment file (e.g. you built both from the same time-ordered data).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LOBWithSentimentDataset(Dataset):
    """
    Wraps an LOB dataset and a (N, sentiment_dim) tensor so that __getitem__ returns
    (lob_tensor, sentiment_tensor, label).
    """

    def __init__(
        self,
        lob_dataset: Dataset,
        sentiment_vectors: torch.Tensor | None = None,
        sentiment_npz_path: str | None = None,
    ):
        """
        Args:
            lob_dataset: Any Dataset that returns (lob, label) in __getitem__.
            sentiment_vectors: (N, sentiment_dim) tensor; optional if sentiment_npz_path given.
            sentiment_npz_path: Path to .npz with key "sentiment_vectors". Loaded if sentiment_vectors is None.
        """
        self.lob_dataset = lob_dataset
        if sentiment_vectors is not None:
            self.sentiment = sentiment_vectors
        elif sentiment_npz_path is not None:
            import numpy as np
            data = np.load(sentiment_npz_path)
            self.sentiment = torch.tensor(data["sentiment_vectors"], dtype=torch.float32)
        else:
            raise ValueError("Provide either sentiment_vectors or sentiment_npz_path")

        n_lob = len(self.lob_dataset)
        n_sent = self.sentiment.shape[0]
        if n_lob != n_sent:
            raise ValueError(
                f"LOB dataset length ({n_lob}) must match sentiment vectors length ({n_sent}). "
                "Align by index or use align_sentiment_to_lob_indices() first."
            )

    def __len__(self) -> int:
        return len(self.lob_dataset)

    def __getitem__(self, idx: int):
        lob, label = self.lob_dataset[idx]
        sent = self.sentiment[idx]
        return lob, sent, label


def collate_lob_sentiment(batch):
    """Use as DataLoader collate_fn when batch is (lob, sentiment, label)."""
    lobs = torch.stack([b[0] for b in batch])
    sents = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return (lobs, sents), labels


# Usage with Lightning:
# In training_step: (inputs, targets) = batch
#   if isinstance(inputs, tuple):
#       lob, sentiment = inputs
#       logits = self.model(lob, sentiment=sentiment)
#   else:
#       logits = self.model(inputs)
