"""
Single script: Engine 1 (FinBERT) -> Engine 2 (DeepLOB) together.

Builds on ParentPaper (Andrew's repo): we use his load_finbert(), demo_news(), and
load_news_csv() when the fusion folder is inside ParentPaper. We only add
hidden-state extraction (768-d vectors) and Engine 2; we do not rewrite his code.

Usage:
  # Demo: uses ParentPaper's demo_news() if available, else built-in headlines
  python run_both_engines.py --demo

  # With your own news CSV (uses ParentPaper's load_news_csv if available)
  python run_both_engines.py --news_csv news.csv --text_column headline

  # With news CSV + precomputed sentiment .npz (skip Engine 1)
  python run_both_engines.py --sentiment_npz sentiment_vectors.npz

  # With LOBFrame saved dataset .pt (real LOB) + news CSV
  python run_both_engines.py --news_csv news.csv --lob_dataset path/to/training_dataset.pt

Requires: torch, transformers. When run inside ParentPaper repo, reuses his main.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Engine 1: reuse ParentPaper's FinBERT loading; we only add hidden-state extraction
from parent_paper_engine1 import (
    SENTIMENT_DIM,
    load_finbert_and_tokenizer,
    text_to_sentiment_vectors,
    load_news_from_parent_paper,
)

FINBERT_MODEL = "ProsusAI/finbert"


def engine1_text_to_vectors(
    texts: list[str],
    device: str | None = None,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Engine 1: raw text -> sentiment vectors (batch, 768). Uses ParentPaper's
    load_finbert when available; otherwise standalone. Output feeds into Engine 2.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, device = load_finbert_and_tokenizer(FINBERT_MODEL, device)
    return text_to_sentiment_vectors(texts, tokenizer, model, device, max_length=max_length)


# -----------------------------------------------------------------------------
# Engine 2: DeepLOB that accepts LOB + sentiment (from Engine 1) as input
# -----------------------------------------------------------------------------

class DeepLOBSentimentModule(nn.Module):
    """
    Engine 2: LOB + optional sentiment. Same architecture as LOBFrame's
    DeepLOBSentiment but plain nn.Module (no Lightning) so this file has no pl dependency.
    """

    def __init__(self, lighten: bool = True, sentiment_dim: int = 768):
        super().__init__()
        self.sentiment_dim = sentiment_dim
        self._lob_feature_dim = 192

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        k = 5 if lighten else 10
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, k)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(
            input_size=self._lob_feature_dim + self.sentiment_dim,
            hidden_size=64, num_layers=1, batch_first=True,
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, lob: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        """LOB: (batch, 1, T, F). Sentiment: (batch, 768). Returns logits (batch, 3)."""
        x = self.conv1(lob)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), self._lob_feature_dim)
        sent = sentiment.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat((x, sent), dim=2)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


# -----------------------------------------------------------------------------
# Pipeline: run both engines together
# -----------------------------------------------------------------------------

def run_demo():
    """Demo: Engine 1 on headlines (from ParentPaper's demo_news() when available), Engine 2 with synthetic LOB."""
    print("=" * 60)
    print("DEMO: Engine 1 (FinBERT) -> Engine 2 (DeepLOB) in one script")
    print("=" * 60)

    # Use ParentPaper's demo_news() when fusion is inside ParentPaper; else built-in headlines
    df = load_news_from_parent_paper(demo=True)
    if df is not None and "headline" in df.columns:
        headlines = df["headline"].astype(str).tolist()
        print("[Using ParentPaper demo_news() for headlines]")
    else:
        headlines = [
            "Apple stock soars after strong iPhone demand",
            "Fed signals rate cut amid weak jobs data",
            "Oil prices drop on rising supply concerns",
            "Tech earnings beat expectations, shares rally",
        ]
    batch_size = len(headlines)

    print("\n[Engine 1] Running FinBERT on", batch_size, "headlines...")
    sentiment_vectors = engine1_text_to_vectors(headlines)
    print("  -> Sentiment output shape:", sentiment_vectors.shape, "(this is input to Engine 2)")

    # Synthetic LOB (Engine 2 input): (batch, 1, window_size, n_features)
    # lighten=True expects 20 features -> conv output width 1; lighten=False expects 40.
    window_size = 100
    lighten = True
    n_features = 20 if lighten else 40
    lob_batch = torch.randn(batch_size, 1, window_size, n_features).float() * 0.01

    print("\n[Engine 2] Running DeepLOB with LOB + sentiment from Engine 1...")
    model = DeepLOBSentimentModule(lighten=lighten, sentiment_dim=SENTIMENT_DIM)
    model.eval()
    with torch.no_grad():
        logits = model(lob_batch, sentiment_vectors)
    print("  -> LOB input shape:", lob_batch.shape)
    print("  -> Sentiment input shape (from Engine 1):", sentiment_vectors.shape)
    print("  -> Logits output shape:", logits.shape, "(batch, 3 classes: down / flat / up)")

    print("\n[Both engines together] One forward pass: text -> sentiment -> LOB+sentiment -> prediction.")
    print("Done. Both systems work together in this single file.")
    return logits


def run_with_news_csv(news_csv: str, text_column: str = "headline", max_samples: int | None = None):
    """Run Engine 1 on CSV (uses ParentPaper's load_news_csv when available), then Engine 2 with synthetic LOB."""
    import pandas as pd
    try:
        df = load_news_from_parent_paper(csv_path=news_csv)
    except Exception:
        df = None
    if df is None:
        df = pd.read_csv(news_csv)
    if text_column not in df.columns:
        raise ValueError(f"CSV must have column '{text_column}'")
    texts = df[text_column].astype(str).tolist()
    if max_samples:
        texts = texts[:max_samples]
    print("[Engine 1] Extracting sentiment from", len(texts), "rows in", news_csv)
    sentiment_vectors = engine1_text_to_vectors(texts)
    batch_size = sentiment_vectors.size(0)
    lob_batch = torch.randn(batch_size, 1, 100, 40).float() * 0.01
    print("[Engine 2] Forward pass with LOB + sentiment from Engine 1")
    model = DeepLOBSentimentModule(lighten=True, sentiment_dim=SENTIMENT_DIM)
    model.eval()
    with torch.no_grad():
        logits = model(lob_batch, sentiment_vectors)
    print("  -> Logits shape:", logits.shape)
    return logits, sentiment_vectors


def run_with_sentiment_npz(sentiment_npz: str, batch_size: int = 8):
    """Load precomputed sentiment from .npz (Engine 1 output), run Engine 2 with synthetic LOB."""
    import numpy as np
    data = np.load(sentiment_npz)
    vecs = data["sentiment_vectors"]
    sentiment_vectors = torch.tensor(vecs[:batch_size], dtype=torch.float32)
    lob_batch = torch.randn(batch_size, 1, 100, 40).float() * 0.01
    model = DeepLOBSentimentModule(lighten=True, sentiment_dim=sentiment_vectors.size(1))
    model.eval()
    with torch.no_grad():
        logits = model(lob_batch, sentiment_vectors)
    print("[Engine 2] Used precomputed sentiment from", sentiment_npz, "-> logits shape", logits.shape)
    return logits


def run_with_lob_dataset(lob_dataset_path: str, sentiment_vectors: torch.Tensor):
    """Run Engine 2 with real LOB from LOBFrame saved dataset + sentiment from Engine 1."""
    dataset = torch.load(lob_dataset_path)
    n = min(len(dataset), sentiment_vectors.size(0))
    model = DeepLOBSentimentModule(lighten=True, sentiment_dim=sentiment_vectors.size(1))
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(n):
            lob, label = dataset[i]
            lob = lob.unsqueeze(0)
            sent = sentiment_vectors[i : i + 1]
            logits = model(lob, sent)
            logits_list.append(logits)
    logits = torch.cat(logits_list, dim=0)
    print("[Engine 2] Ran on", n, "samples from LOB dataset + Engine 1 sentiment -> logits", logits.shape)
    return logits


def main():
    p = argparse.ArgumentParser(description="Run Engine 1 (FinBERT) and Engine 2 (DeepLOB) together")
    p.add_argument("--demo", action="store_true", help="Demo with synthetic data (no files)")
    p.add_argument("--news_csv", type=str, help="CSV with text column; Engine 1 runs on it")
    p.add_argument("--text_column", type=str, default="headline")
    p.add_argument("--sentiment_npz", type=str, help="Precomputed sentiment .npz (skip Engine 1)")
    p.add_argument("--lob_dataset", type=str, help="Path to LOBFrame .pt dataset (real LOB)")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    if args.demo:
        run_demo()
        return

    if args.sentiment_npz:
        run_with_sentiment_npz(args.sentiment_npz)
        return

    if args.news_csv:
        logits, sentiment_vectors = run_with_news_csv(
            args.news_csv, text_column=args.text_column, max_samples=args.max_samples
        )
        if args.lob_dataset:
            run_with_lob_dataset(args.lob_dataset, sentiment_vectors)
        return

    # No args: run demo
    print("No --demo, --news_csv, or --sentiment_npz given. Running demo.")
    run_demo()


if __name__ == "__main__":
    main()
