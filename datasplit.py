"""
Build Train/Test/Validation splits for a public financial-news dataset,
restricted to S&P 500 tickers.

Dataset:
  - Hugging Face: ashraq/financial-news
    Columns (as of recent versions): headline, url, publisher, date, stock

Pipeline:
  1) Load dataset from HF (optionally subsample for speed)
  2) Fetch current S&P 500 ticker list (Wikipedia, with DataHub fallback)
  3) Normalize + filter dataset to S&P 500 tickers only
  4) Compute FinBERT sentiment score (polarity = P(pos) - P(neg))
  5) Compute next-day return using yfinance (download ONCE per ticker, not per row)
  6) Split into train/test/val (70/20/10 default)
  7) Save CSVs: train.csv, test.csv, val.csv

Rule reminder:
  - Do NOT open/use val.csv until you are satisfied with train/test.

USAGE (recommended):
  # Fast sanity check (no returns/sentiment):
  python datasplit_sp500.py --limit_rows 20000 --skip_sentiment --skip_returns

  # Then add sentiment:
  python datasplit_sp500.py --limit_rows 20000 --skip_returns

  # Then full pipeline (will be slower):
  python datasplit_sp500.py --limit_rows 20000

OUTPUT:
  out_dir/train.csv, out_dir/test.csv, out_dir/val.csv
"""

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import re
import argparse
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

from datasets import load_dataset
import yfinance as yf

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split


# -----------------------------
# Ticker normalization + S&P 500 list
# -----------------------------

def normalize_ticker(t: str) -> str:
    """
    Normalize dataset tickers to Yahoo-like format.
    Examples:
      "$AAPL" -> "AAPL"
      "BRK.B" -> "BRK-B"
      "BF.B"  -> "BF-B"
    """
    t = str(t).strip().upper()
    t = re.sub(r"^\$", "", t)   # remove leading $
    t = t.replace(".", "-")     # dot share classes -> hyphen (Yahoo)
    t = t.replace("/", "-")
    return t


def fetch_sp500_tickers() -> set:
    """
    Fetch current S&P 500 tickers from Wikipedia, with DataHub fallback.
    Requires internet at runtime.
    """
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    datahub_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

    # 1) Try Wikipedia table read
    try:
        tables = pd.read_html(wiki_url)
        # The first table is typically the constituents table with "Symbol"
        # We'll search for a table containing "Symbol".
        sp_table = None
        for t in tables:
            if "Symbol" in t.columns:
                sp_table = t
                break
        if sp_table is None:
            raise ValueError("No table with 'Symbol' column found on Wikipedia.")
        syms = sp_table["Symbol"].astype(str).tolist()
        syms = {normalize_ticker(s) for s in syms}
        # common Yahoo adjustments
        # (Wikipedia already uses BRK.B and BF.B; normalize handles to BRK-B / BF-B)
        syms.discard("")
        return syms
    except Exception as e1:
        print(f"[WARN] Wikipedia S&P 500 fetch failed: {e1}")

    # 2) Fallback: DataHub CSV
    try:
        df = pd.read_csv(datahub_url)
        if "Symbol" not in df.columns:
            raise ValueError(f"DataHub CSV missing 'Symbol'. Columns: {list(df.columns)}")
        syms = {normalize_ticker(s) for s in df["Symbol"].astype(str).tolist()}
        syms.discard("")
        return syms
    except Exception as e2:
        raise RuntimeError(
            f"Failed to fetch S&P 500 tickers from both sources.\n"
            f"Wikipedia error: {e1}\n"
            f"DataHub error: {e2}"
        )


# -----------------------------
# FinBERT
# -----------------------------

def load_finbert(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def finbert_polarity_scores_batch(
    texts: list,
    tokenizer,
    model,
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Returns polarity scores for each text: P(pos) - P(neg).
    FinBERT label order (ProsusAI/finbert): [neg, neu, pos]
    """
    scores = np.empty(len(texts), dtype=np.float32)
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, 3)
        scores[i:i + len(batch)] = (probs[:, 2] - probs[:, 0]).astype(np.float32)

    return scores


# -----------------------------
# Returns via yfinance (download ONCE per ticker)
# -----------------------------

def _extract_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    # Handles possible MultiIndex columns from yfinance
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # look for ("Close", <ticker>) shape
        for col in df.columns:
            if col[0] == "Close":
                return pd.to_numeric(df[col].squeeze(), errors="coerce")
        return None
    if "Close" not in df.columns:
        return None
    return pd.to_numeric(df["Close"].squeeze(), errors="coerce")


def build_next_day_return_map_for_ticker(
    ticker: str,
    dates_needed: set,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Dict[str, float]:
    """
    Download a ticker once, then compute next-trading-day close-to-close return
    for specific calendar dates in dates_needed.

    Returns dict: date_str -> return (float)
    """
    out = {}
    df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)

    if df is None or df.empty:
        return out

    df = df.reset_index()
    # yfinance uses "Date" when reset_index() is called
    if "Date" not in df.columns:
        return out

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    close_series = _extract_close_series(df)
    if close_series is None or close_series.empty:
        return out

    # Map date -> positional index in the downloaded trading-day series
    date_arr = df["Date"].to_numpy()
    pos_by_date = {d: i for i, d in enumerate(date_arr)}

    for d_str in dates_needed:
        try:
            d = pd.to_datetime(d_str).date()
        except Exception:
            continue
        if d not in pos_by_date:
            continue
        pos = pos_by_date[d]
        if pos + 1 >= len(df):
            continue

        c0 = close_series.iloc[pos]
        c1 = close_series.iloc[pos + 1]
        if not np.isfinite(c0) or not np.isfinite(c1) or c0 == 0:
            continue
        out[d_str] = float((c1 / c0) - 1.0)

    return out


def compute_next_day_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute next-day returns for each row using ONE download per ticker.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Determine global date window needed
    min_d = pd.to_datetime(df["date"]).min()
    max_d = pd.to_datetime(df["date"]).max()
    start_dt = min_d - pd.Timedelta(days=5)
    end_dt = max_d + pd.Timedelta(days=10)

    # Group requested dates per ticker
    dates_by_ticker = (
        df.groupby("ticker")["date"]
          .apply(lambda s: set(s.astype(str).tolist()))
          .to_dict()
    )

    ret_values = np.full(len(df), np.nan, dtype=np.float64)

    # Build return maps per ticker, then fill
    for tkr, dates_needed in dates_by_ticker.items():
        ret_map = build_next_day_return_map_for_ticker(
            tkr, dates_needed=dates_needed, start_dt=start_dt, end_dt=end_dt
        )
        if not ret_map:
            continue

        idxs = df.index[df["ticker"] == tkr].to_numpy()
        for i in idxs:
            d = str(df.at[i, "date"])
            if d in ret_map:
                ret_values[i] = ret_map[d]

    return pd.Series(ret_values, index=df.index, name="next_day_return")


# -----------------------------
# Dataset loading + filtering to S&P 500
# -----------------------------

def load_public_news(limit_rows: int, seed: int) -> pd.DataFrame:
    """
    Loads ashraq/financial-news from Hugging Face.
    Dataset columns (current): headline, url, publisher, date, stock
    Standardize to: date, ticker, headline
    """
    ds = load_dataset("ashraq/financial-news", split="train")

    # Subsample BEFORE converting to pandas (critical for speed/memory)
    if limit_rows and limit_rows > 0 and len(ds) > limit_rows:
        ds = ds.shuffle(seed=seed).select(range(limit_rows))

    df = ds.to_pandas()

    # validate columns
    needed = {"headline", "stock", "date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}. Columns: {list(df.columns)}")

    df = df[["date", "stock", "headline"]].rename(columns={"stock": "ticker"}).copy()

    # clean
    df["headline"] = df["headline"].astype(str).str.strip()
    df["ticker"] = df["ticker"].apply(normalize_ticker)

    # normalize date to YYYY-MM-DD strings
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    df = df.dropna(subset=["date", "ticker", "headline"])
    df = df[df["headline"].str.len() > 0]
    df = df[df["ticker"].str.len() > 0]

    return df.reset_index(drop=True)


# -----------------------------
# Split logic (70/20/10 default)
# -----------------------------

def split_train_test_val(
    df: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if not np.isclose(train_ratio + test_ratio + val_ratio, 1.0):
        raise ValueError("train_ratio + test_ratio + val_ratio must sum to 1.0")

    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_ratio), random_state=seed, shuffle=True
    )

    val_share_in_temp = val_ratio / (test_ratio + val_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_share_in_temp), random_state=seed, shuffle=True
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data_splits", help="Where to write train/test/val CSVs.")
    p.add_argument("--limit_rows", type=int, default=20000, help="Subsample rows for speed (0 = no limit).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--val_ratio", type=float, default=0.10)

    p.add_argument("--model", type=str, default="ProsusAI/finbert")
    p.add_argument("--maxlen", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)

    # optional: skip expensive parts while debugging
    p.add_argument("--skip_returns", action="store_true", help="Skip yfinance next-day return computation.")
    p.add_argument("--skip_sentiment", action="store_true", help="Skip FinBERT sentiment computation.")

    args = p.parse_args()

    print("Fetching S&P 500 tickers...")
    sp500 = fetch_sp500_tickers()
    print(f"S&P 500 tickers fetched: {len(sp500)}")

    print("Loading public dataset from Hugging Face: ashraq/financial-news")
    df = load_public_news(limit_rows=args.limit_rows, seed=args.seed)
    print(f"Loaded rows: {len(df)} (before S&P 500 filter)")

    # Filter to S&P 500
    df = df[df["ticker"].isin(sp500)].reset_index(drop=True)
    print(f"Rows after S&P 500 filter: {len(df)} (unique tickers: {df['ticker'].nunique()})")

    if len(df) == 0:
        raise RuntimeError("After filtering to S&P 500 tickers, no rows remain. Try a larger --limit_rows.")

    # Sentiment
    if not args.skip_sentiment:
        print(f"Loading FinBERT model: {args.model}")
        tokenizer, model = load_finbert(args.model)
        print("Computing FinBERT polarity scores...")
        df["sent_score"] = finbert_polarity_scores_batch(
            df["headline"].tolist(),
            tokenizer=tokenizer,
            model=model,
            max_length=args.maxlen,
            batch_size=args.batch_size,
        )
    else:
        df["sent_score"] = np.nan

    # Returns (one download per ticker)
    if not args.skip_returns:
        print("Computing next-day returns (yfinance) using one download per ticker...")
        df["next_day_return"] = compute_next_day_returns(df)
        before = len(df)
        df = df.dropna(subset=["next_day_return"]).reset_index(drop=True)
        print(f"Dropped rows missing next_day_return: {before - len(df)} -> remaining {len(df)}")
    else:
        df["next_day_return"] = np.nan

    if len(df) < 50:
        print("[WARN] Very small dataset after filtering/return computation. Consider increasing --limit_rows.")

    # Split
    train_df, test_df, val_df = split_train_test_val(
        df,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    val_path = os.path.join(args.out_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("\nWrote:")
    print(f"  {train_path}  (train)")
    print(f"  {test_path}   (test)")
    print(f"  {val_path}    (validation / unseen)")

    print("\nRule:")
    print("  Do NOT open/use val.csv until you are happy with your training + test pipeline.")


if __name__ == "__main__":
    main()
