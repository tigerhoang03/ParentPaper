"""
FinBERT sentiment -> next-day return regression + SHAP text explanation (basic baseline)

USAGE:

1) Demo (w/ built-in example headlines):
   python finbert_sentiment_return_shap.py --demo

2) With your own CSV:
   # CSV must have columns: date,ticker,headline
   # date format: YYYY-MM-DD
   python finbert_sentiment_return_shap.py --csv news.csv

Optional:
   --model ProsusAI/finbert
   --maxlen 128
   --shap_text "some text to explain"
   --no_shap   (skip SHAP)
"""

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# finance
import yfinance as yf

# NLP
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# explainability
import shap


# -----------------------------
# FinBERT utilities
# -----------------------------

def load_finbert(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def finbert_sentiment(
    text: str,
    tokenizer,
    model,
    max_length: int = 128,
) -> Tuple[str, float, np.ndarray]:
    """
    Returns:
      label: "negative" | "neutral" | "positive"
      score: positive_prob - negative_prob  (simple polarity)
      probs: [neg, neu, pos]
    """
    # ProsusAI/finbert standard label order: [negative, neutral, positive]
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    pred_idx = int(np.argmax(probs))
    pred_label = label_map[pred_idx]
    score = float(probs[2] - probs[0])

    return pred_label, score, probs


def predict_proba_texts(
    texts: List[str],
    tokenizer,
    model,
    max_length: int = 128,
) -> np.ndarray:
    """
    Batch predict class probabilities for a list of texts.
    Output shape: (n, 3) for [neg, neu, pos]
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# -----------------------------
# Market data utilities
# -----------------------------

def get_next_day_return(ticker: str, date_str: str) -> float:
    """
    Next trading day close-to-close return for given ticker and date.
    Returns np.nan if not available.
    """
    try:
        d = pd.to_datetime(date_str).date()
    except Exception:
        return np.nan

    start = pd.to_datetime(d) - pd.Timedelta(days=2)
    end = pd.to_datetime(d) + pd.Timedelta(days=7)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return np.nan

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # --- Get a 1D close series robustly (works with MultiIndex columns) ---
    if isinstance(df.columns, pd.MultiIndex):
        # try common patterns: ("Close", <ticker>) or ("Close", "")
        close_series = None
        for col in df.columns:
            if col[0] == "Close":
                close_series = df[col]
                break
        if close_series is None:
            return np.nan
    else:
        if "Close" not in df.columns:
            return np.nan
        close_series = df["Close"]

    # Ensure close_series is 1D numeric series
    close_series = pd.to_numeric(close_series.squeeze(), errors="coerce")

    # Find row position for the given date
    matches = np.where(df["Date"].to_numpy() == d)[0]
    if matches.size == 0:
        return np.nan

    pos = int(matches[0])
    if pos + 1 >= len(df):
        return np.nan

    close_today = float(close_series.iloc[pos])
    close_next = float(close_series.iloc[pos + 1])

    if not np.isfinite(close_today) or not np.isfinite(close_next) or close_today == 0:
        return np.nan

    return (close_next / close_today) - 1.0


# -----------------------------
# Data loading
# -----------------------------

def demo_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-06", "2025-01-07", "2025-01-08"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "headline": [
                "Apple stock soars after strong iPhone demand",
                "Apple faces downgrade amid weak China sales",
                "Regulatory approval boosts investor confidence in Apple",
            ],
        }
    )


def load_news_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["date", "ticker", "headline"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df = df[cols].copy()
    df["date"] = df["date"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["headline"] = df["headline"].astype(str)
    return df


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(args):
    print(f"Loading model: {args.model}")
    tokenizer, model = load_finbert(args.model)

    if args.demo:
        news = demo_news()
        print("Loaded demo dataset.")
    else:
        news = load_news_csv(args.csv)
        print(f"Loaded {len(news)} rows from {args.csv}")

    # FinBERT sentiment scoring
    labels = []
    scores = []
    probs_list = []

    print("Scoring sentiment with FinBERT...")
    for text in news["headline"].tolist():
        lab, sc, pr = finbert_sentiment(text, tokenizer, model, max_length=args.maxlen)
        labels.append(lab)
        scores.append(sc)
        probs_list.append(pr)

    news["sent_label"] = labels
    news["sent_score"] = scores
    news["sent_probs_neg"] = [float(p[0]) for p in probs_list]
    news["sent_probs_neu"] = [float(p[1]) for p in probs_list]
    news["sent_probs_pos"] = [float(p[2]) for p in probs_list]

    # Next-day return
    print("Fetching next-day returns (yfinance)...")
    news["next_day_return"] = news.apply(
        lambda r: get_next_day_return(r["ticker"], r["date"]), axis=1
    )

    # Save results
    if args.out:
        news.to_csv(args.out, index=False)
        print(f"Saved enriched dataset to: {args.out}")

    # Regression: sent_score -> next_day_return
    df = news.dropna(subset=["sent_score", "next_day_return"]).copy()
    if len(df) < 2:
        print("\nNot enough rows with returns to fit regression.")
        print("Try providing more dates/tickers or ensure market dates align.")
    else:
        X = df[["sent_score"]].values
        y = df["next_day_return"].values

        reg = LinearRegression()
        reg.fit(X, y)
        pred = reg.predict(X)

        r2 = r2_score(y, pred)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))

        print("\n=== Regression: next_day_return ~ sent_score ===")
        print(f"n = {len(df)}")
        print(f"coef = {reg.coef_[0]:.6f}")
        print(f"intercept = {reg.intercept_:.6f}")
        print(f"R2 = {r2:.4f}")
        print(f"RMSE = {rmse:.6f}")

    # SHAP explanation for a single text
    if not args.no_shap:
        shap_text = args.shap_text
        if shap_text is None:
            shap_text = news.iloc[int(np.argmax(np.abs(news["sent_score"].values)))]["headline"]

        shap_text = str(shap_text)

        print("\nRunning SHAP text explanation for:")
        print(f"  {shap_text}")
        print("(If you're running in a terminal, the SHAP text plot may not render; fallback will print tokens.)")

        def f(texts):
            texts = [str(t) for t in texts]
            return predict_proba_texts(texts, tokenizer, model, max_length=args.maxlen)

        explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))
        shap_values = explainer([shap_text])

        try:
            shap.plots.text(shap_values[0])
        except Exception:
            tokens = shap_values.data[0]
            vals = shap_values.values[0, :, 2]  # positive class contributions
            top = np.argsort(np.abs(vals))[::-1][:10]
            print("\nTop token contributions (positive class):")
            for i in top:
                print(f"  {tokens[i]!r}: {vals[i]:+.4f}")

    print("\nDone.")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Run with built-in sample headlines.")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV with date,ticker,headline.")
    p.add_argument("--out", type=str, default="results.csv", help="Output CSV path.")
    p.add_argument("--model", type=str, default="ProsusAI/finbert", help="HF model name.")
    p.add_argument("--maxlen", type=int, default=128, help="Tokenizer max length.")
    p.add_argument("--no_shap", action="store_true", help="Skip SHAP step.")
    p.add_argument("--shap_text", type=str, default=None, help="Custom text to explain with SHAP.")
    args = p.parse_args()

    if not args.demo and not args.csv:
        p.error("Choose one: --demo OR --csv news.csv")

    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print("\nERROR:", str(e))
        sys.exit(1)
