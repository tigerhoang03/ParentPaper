"""
Part 2: LOB + sentiment (supply/demand). Reads Part 1 output directly.

Uses main.py / datasplit.py CSV output (sent_score, optionally sent_probs_*).
No FinBERT or 768-d vectors in this script.

Usage:
  # Demo: synthetic LOB + synthetic sentiment from small built-in CSV
  python run_part2.py --demo

  # Part 1 CSV (e.g. data_splits/train.csv or results.csv)
  python run_part2.py --csv data_splits/train.csv

  # Use 3-d probs if CSV has sent_probs_neg, sent_probs_neu, sent_probs_pos
  python run_part2.py --csv results.csv --use_probs

  # With real LOB dataset (.pt)
  python run_part2.py --csv data_splits/train.csv --lob_dataset path/to/dataset.pt

Run from repo root: python fusion/run_part2.py ...
Or from fusion/: python run_part2.py ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Build sentiment tensor from Part 1 CSV
# -----------------------------------------------------------------------------

def load_sentiment_from_csv(
    csv_path: str,
    use_probs: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Load Part 1 CSV; require sent_score. Optionally use sent_probs_neg/neu/pos for 3-d.

    Returns:
        sentiment: (N, 1) or (N, 3) float32 tensor
        sentiment_dim: 1 or 3
    """
    df = pd.read_csv(csv_path)
    if "sent_score" not in df.columns:
        raise ValueError(f"CSV must have column 'sent_score'. Columns: {list(df.columns)}")

    if use_probs:
        prob_cols = ["sent_probs_neg", "sent_probs_neu", "sent_probs_pos"]
        if all(c in df.columns for c in prob_cols):
            arr = df[prob_cols].astype("float32").values
            return torch.tensor(arr, dtype=torch.float32), 3
    # Default: 1-d from sent_score
    arr = df["sent_score"].astype("float32").values.reshape(-1, 1)
    return torch.tensor(arr, dtype=torch.float32), 1


# -----------------------------------------------------------------------------
# DeepLOB (plain nn.Module) — LOB + sentiment_dim (1 or 3)
# -----------------------------------------------------------------------------

class DeepLOBSentimentModule(nn.Module):
    """
    LOB + sentiment. Same architecture as LOBFrame DeepLOBSentiment; plain nn.Module.
    sentiment_dim can be 1 (sent_score) or 3 (probs) when using Part 1 CSV.
    """

    def __init__(self, lighten: bool = True, sentiment_dim: int = 1):
        super().__init__()
        self.sentiment_dim = sentiment_dim
        self._lob_feature_dim = 192
        k = 5 if lighten else 10

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
        """LOB: (batch, 1, T, F). Sentiment: (batch, sentiment_dim). Returns logits (batch, 3)."""
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
# Pipeline
# -----------------------------------------------------------------------------

# lighten=True requires 20 features so conv output width is 1 (192-d LOB feature vector)
WINDOW_SIZE = 100
LIGHTEN = True
N_FEATURES = 20 if LIGHTEN else 40


def run_demo():
    """Demo with tiny built-in CSV and synthetic LOB."""
    print("=" * 60)
    print("Part 2 demo: Part 1 CSV -> sentiment tensor -> DeepLOB (synthetic LOB)")
    print("=" * 60)

    # Minimal Part 1-style CSV (sent_score only)
    demo_df = pd.DataFrame({
        "date": ["2025-01-06", "2025-01-07", "2025-01-08"],
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "headline": ["Apple stock soars", "Apple faces downgrade", "Regulatory approval boosts Apple"],
        "sent_score": [-0.53, 0.01, -0.90],
    })
    n = len(demo_df)
    sentiment = torch.tensor(demo_df["sent_score"].astype("float32").values.reshape(-1, 1), dtype=torch.float32)
    sentiment_dim = 1

    lob_batch = torch.randn(n, 1, WINDOW_SIZE, N_FEATURES).float() * 0.01

    model = DeepLOBSentimentModule(lighten=LIGHTEN, sentiment_dim=sentiment_dim)
    model.eval()
    with torch.no_grad():
        logits = model(lob_batch, sentiment)

    print(f"  Rows: {n}, sentiment shape: {sentiment.shape}, LOB shape: {lob_batch.shape}")
    print(f"  Logits shape: {logits.shape} (batch, 3 classes: down / flat / up)")
    print("Done.")
    return logits


def run_with_csv(
    csv_path: str,
    use_probs: bool = False,
    lob_dataset_path: str | None = None,
    max_samples: int | None = None,
):
    """Load Part 1 CSV, build sentiment, run DeepLOB (synthetic or real LOB)."""
    print("=" * 60)
    print("Part 2: Part 1 CSV -> DeepLOB")
    print("=" * 60)

    sentiment, sentiment_dim = load_sentiment_from_csv(csv_path, use_probs=use_probs)
    if max_samples is not None:
        sentiment = sentiment[:max_samples]
    n = sentiment.size(0)
    print(f"  Loaded sentiment from {csv_path}: shape {sentiment.shape} (sentiment_dim={sentiment_dim})")

    if lob_dataset_path is not None:
        dataset = torch.load(lob_dataset_path)
        n_lob = min(n, len(dataset))
        model = DeepLOBSentimentModule(lighten=LIGHTEN, sentiment_dim=sentiment_dim)
        model.eval()
        logits_list = []
        with torch.no_grad():
            for i in range(n_lob):
                lob, label = dataset[i]
                lob = lob.unsqueeze(0)
                sent = sentiment[i : i + 1]
                logits = model(lob, sent)
                logits_list.append(logits)
        logits = torch.cat(logits_list, dim=0)
        print(f"  Ran on {n_lob} samples from LOB dataset -> logits {logits.shape}")
    else:
        lob_batch = torch.randn(n, 1, WINDOW_SIZE, N_FEATURES).float() * 0.01
        model = DeepLOBSentimentModule(lighten=LIGHTEN, sentiment_dim=sentiment_dim)
        model.eval()
        with torch.no_grad():
            logits = model(lob_batch, sentiment)
        print(f"  Synthetic LOB shape: {lob_batch.shape} -> logits {logits.shape}")

    return logits


def main():
    p = argparse.ArgumentParser(description="Part 2: LOB + sentiment from Part 1 CSV")
    p.add_argument("--demo", action="store_true", help="Demo with built-in small CSV and synthetic LOB")
    p.add_argument("--csv", type=str, help="Part 1 CSV (e.g. data_splits/train.csv or results.csv)")
    p.add_argument("--use_probs", action="store_true", help="Use sent_probs_neg/neu/pos (3-d) if present")
    p.add_argument("--lob_dataset", type=str, default=None, help="Path to .pt LOB dataset (optional)")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    if args.demo:
        run_demo()
        return

    if args.csv:
        run_with_csv(
            args.csv,
            use_probs=args.use_probs,
            lob_dataset_path=args.lob_dataset,
            max_samples=args.max_samples,
        )
        return

    print("No --demo or --csv given. Running demo.")
    run_demo()


if __name__ == "__main__":
    main()
