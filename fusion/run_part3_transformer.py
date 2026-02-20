from __future__ import annotations

import argparse
import pandas as pd
import torch

from fusion.run_part2 import load_sentiment_from_csv, WINDOW_SIZE, N_FEATURES, LIGHTEN
from fusion.models.transformer_lob import DeepLOBTransformerSentiment


def run_demo():
    print("=" * 60)
    print("Part 3 Demo: CNN + Transformer (synthetic LOB)")
    print("=" * 60)

    demo_df = pd.DataFrame({
        "date": ["2025-01-06", "2025-01-07", "2025-01-08"],
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "headline": ["Apple stock soars", "Apple faces downgrade", "Regulatory approval boosts Apple"],
        "sent_score": [-0.53, 0.01, -0.90],
    })

    n = len(demo_df)
    sentiment = torch.tensor(
        demo_df["sent_score"].astype("float32").values.reshape(-1, 1),
        dtype=torch.float32
    )
    lob_batch = torch.randn(n, 1, WINDOW_SIZE, N_FEATURES).float() * 0.01

    model = DeepLOBTransformerSentiment(lighten=LIGHTEN, sentiment_dim=1)
    model.eval()
    with torch.no_grad():
        logits = model(lob_batch, sentiment)

    preds = torch.argmax(logits, dim=1)
    print(f"Rows={n}")
    print(f"LOB shape={tuple(lob_batch.shape)} | sentiment shape={tuple(sentiment.shape)}")
    print(f"logits shape={tuple(logits.shape)} (B,3 classes)")
    print("Pred classes (0=down,1=flat,2=up):", preds.tolist())
    print("Done.")


def run_with_csv(csv_path: str, use_probs: bool, max_samples: int | None):
    print("=" * 60)
    print("Part 3: CNN + Transformer using Part 1 CSV sentiment")
    print("=" * 60)

    sentiment, sentiment_dim = load_sentiment_from_csv(csv_path, use_probs=use_probs)
    if max_samples is not None:
        sentiment = sentiment[:max_samples]

    n = sentiment.size(0)
    lob_batch = torch.randn(n, 1, WINDOW_SIZE, N_FEATURES).float() * 0.01

    model = DeepLOBTransformerSentiment(lighten=LIGHTEN, sentiment_dim=sentiment_dim)
    model.eval()
    with torch.no_grad():
        logits = model(lob_batch, sentiment)

    preds = torch.argmax(logits, dim=1)
    print(f"CSV={csv_path} | n={n} | sentiment_dim={sentiment_dim}")
    print(f"LOB shape={tuple(lob_batch.shape)} | logits shape={tuple(logits.shape)}")
    print("First 10 preds:", preds[:10].tolist())
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--use_probs", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    if args.demo:
        run_demo()
    elif args.csv:
        run_with_csv(args.csv, use_probs=args.use_probs, max_samples=args.max_samples)
    else:
        print("Use --demo OR --csv <path>")