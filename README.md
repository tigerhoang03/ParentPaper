# ParentPaper

Implementation of a two-part financial pipeline: **Part 1** assigns sentiment scores to stocks from news headlines and links them to next-day returns; **Part 2** uses those scores together with limit-order-book (LOB) style inputs for supply/demand–style prediction (down / flat / up).

---

## Introduction

This project combines **news sentiment** (Part 1) with **order-book–inspired modeling** (Part 2). Part 1 uses FinBERT to score financial headlines and fits a simple regression of next-day stock returns on sentiment. Part 2 reads Part 1’s CSV output and feeds the sentiment (and optionally LOB data) into a DeepLOB-style model that predicts price direction. The design keeps Part 1 and Part 2 separate so you can run sentiment and return analysis on its own, then plug the same outputs into the LOB pipeline.

---

## Project Summary

| Part   | Purpose |
|--------|--------|
| **Part 1** | Score headlines with FinBERT, fetch next-day returns (yfinance), and fit **next_day_return ~ sent_score**. Optional SHAP explanations. Output: CSV with sentiment and returns. |
| **Part 2** | Load Part 1 CSV (sent_score or 3-d probs), optionally load LOB data, and run a **DeepLOB + sentiment** model for 3-class prediction (down / flat / up). |

Data flow: **News CSV or demo** → **Part 1 (main.py or datasplit.py)** → **CSV (sent_score, optional probs)** → **Part 2 (fusion/run_part2.py)** → **Logits / predictions**.

---

## What Each File Does and How They Work Together

### Root

| File | Role |
|------|------|
| **main.py** | Part 1 entry: loads FinBERT, scores headlines (sentiment label + score + probs), fetches next-day returns via yfinance, fits linear regression (next_day_return ~ sent_score), and optionally runs SHAP. Writes an enriched CSV (e.g. `results.csv`). Run with `--demo` or `--csv <path>`. |
| **datasplit.py** | Part 1 data pipeline: loads Hugging Face financial-news data, restricts to S&P 500 tickers, computes FinBERT polarity and next-day returns, then splits into train/val/test (default 70/20/10) and writes `data_splits/train.csv`, `test.csv`, `val.csv`. Those CSVs have `sent_score` (and optionally can be extended with probs) for use in Part 2. |
| **results.csv** | Example Part 1 output: one row per headline with date, ticker, headline, sent_label, sent_score, sent_probs_neg/neu/pos, next_day_return. |
| **train.csv**, **val.csv**, **test.csv** | Part 1 split outputs (at root or under `data_splits/`): same schema as above but from datasplit; used as input to Part 2. |

### fusion/ (Part 2)

| File | Role |
|------|------|
| **run_part2.py** | Part 2 entry: reads a Part 1 CSV (`sent_score` and optionally 3-d probs), builds a sentiment tensor (N×1 or N×3), and runs a plain PyTorch DeepLOB+sentiment model. Uses synthetic LOB by default (20 features, `lighten=True`); can take a real LOB `.pt` dataset. No FinBERT here—sentiment comes only from the CSV. |
| **dataset_with_sentiment.py** | Dataset helper: builds (LOB, sentiment, label) batches from an LOB dataset and a Part 1 DataFrame/CSV (sentiment from `sent_score` or sent_probs columns). Used when training Part 2 with a DataLoader. |
| **models/DeepLob/deeplob_sentiment.py** | PyTorch Lightning DeepLOB module that accepts optional sentiment (any dimension). Used if you train Part 2 with Lightning; `run_part2.py` uses its own plain `nn.Module` copy. |

### How they work together

1. **Part 1 only**  
   Run `main.py --demo` or `main.py --csv news.csv` to get sentiment and return regression; output goes to e.g. `results.csv`.  
   Or run `datasplit.py` to produce train/val/test CSVs under `data_splits/`.

2. **Part 1 → Part 2**  
   Use a Part 1 CSV as input to Part 2:  
   `python fusion/run_part2.py --csv results.csv` (1-d sent_score) or  
   `python fusion/run_part2.py --csv results.csv --use_probs` (3-d probs).  
   Part 2 loads that CSV, builds sentiment tensors, and runs DeepLOB with synthetic or real LOB.

3. **Training Part 2**  
   Use `dataset_with_sentiment.py` with a Part 1 CSV and your LOB dataset so each batch is (LOB, sentiment, label); train the Lightning model in `models/DeepLob/deeplob_sentiment.py` or the module in `run_part2.py` with your own training loop.

---

## Results (Example)

- **Part 1 demo** (3 AAPL headlines, `main.py --demo --no_shap`):  
  Regression **next_day_return ~ sent_score**: n=3, coefficient ≈ 0.028, R² ≈ 0.99, RMSE ≈ 0.001.  
  Enriched rows written to `results.csv`.

- **Part 2** (e.g. `run_part2.py --csv results.csv --use_probs`):  
  Loads 3 rows, sentiment shape (3, 3), synthetic LOB (3, 1, 100, 20), output logits (3, 3) for down/flat/up per sample.

*(These are illustrative runs on tiny data; performance on full datasets will depend on data quality and tuning.)*

---

## Libraries and Algorithms

### Libraries

| Library | Use |
|---------|-----|
| **torch** | DeepLOB model, tensors, training. |
| **transformers** | FinBERT tokenizer and model (ProsusAI/finbert). |
| **pandas** | CSV I/O, tables, alignment. |
| **numpy** | Arrays, metrics. |
| **scikit-learn** | Linear regression, train/test split, R², RMSE. |
| **yfinance** | Next-day stock returns. |
| **shap** | Token-level explanations for sentiment (main.py). |
| **datasets** (Hugging Face) | Loading ashraq/financial-news in datasplit.py. |
| **pytorch-lightning** | Optional: training DeepLOB+sentiment in `fusion/models/DeepLob/deeplob_sentiment.py`. |

### Algorithms

| Component | Algorithm / model |
|-----------|-------------------|
| **Sentiment** | **FinBERT** (ProsusAI/finbert): BERT for sequence classification; outputs negative/neutral/positive label and probabilities; we use sent_score = P(positive) − P(negative) and optionally the 3-d probability vector. |
| **Return prediction (Part 1)** | **Linear regression**: next_day_return ~ sent_score (single feature). |
| **Explainability** | **SHAP** (text explainer) on FinBERT for selected headlines. |
| **Part 2 model** | **DeepLOB + sentiment**: convolutional + inception-style blocks on LOB input, then LSTM with input = 192-d LOB features + sentiment_dim (1 or 3 from Part 1 CSV); output 3 classes (down / flat / up). |

---

## Setup and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Part 1: demo (sentiment + return regression, writes results.csv)
python main.py --demo --no_shap

# Part 1: with your CSV (columns: date, ticker, headline)
python main.py --csv news.csv --out results.csv

# Part 2: run on Part 1 output (sent_score only)
python fusion/run_part2.py --csv results.csv

# Part 2: run with 3-d probs if CSV has sent_probs_neg/neu/pos
python fusion/run_part2.py --csv results.csv --use_probs

# Part 2: demo (built-in small CSV + synthetic LOB)
python fusion/run_part2.py --demo
```

---

## Optional: Data splits (Part 1)

To build train/val/test from public financial news (S&P 500, FinBERT, next-day returns):

```bash
# Quick sanity check (no returns/sentiment)
python datasplit.py --limit_rows 20000 --skip_sentiment --skip_returns

# With sentiment, no returns
python datasplit.py --limit_rows 20000 --skip_returns

# Full pipeline (slower)
python datasplit.py --limit_rows 20000
```

Outputs go to `data_splits/train.csv`, `test.csv`, `val.csv`. Use those paths as `--csv` in Part 2 when aligning LOB data by index.
