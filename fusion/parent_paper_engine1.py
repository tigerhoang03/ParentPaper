"""
Engine 1 built on top of ParentPaper (Andrew's repo). We reuse his code; we only add
hidden-state extraction so the output can feed into Engine 2 (DeepLOB).

When the fusion folder lives inside ParentPaper (or main is on PYTHONPATH), we use:
  - load_finbert(), load_news_csv(), demo_news() from ParentPaper's main.py
  - ParentPaper's model (AutoModelForSequenceClassification); we use model.bert to get
    last_hidden_state for 768-d vectors (ParentPaper itself only uses logits/probs).

When ParentPaper is not available, we fall back to standalone loading so the pipeline
still runs (e.g. in CI or without cloning ParentPaper).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# ParentPaper uses ProsusAI/finbert; classification model has .bert (BertModel)
SENTIMENT_DIM = 768
DEFAULT_MODEL_NAME = "ProsusAI/finbert"


def _ensure_parent_paper_on_path():
    """If this package lives in ParentPaper/fusion/, add ParentPaper root to path."""
    this_dir = Path(__file__).resolve().parent
    parent_root = this_dir.parent
    main_py = parent_root / "main.py"
    if main_py.exists() and str(parent_root) not in sys.path:
        sys.path.insert(0, str(parent_root))


def _load_finbert_from_parent_paper(model_name: str = DEFAULT_MODEL_NAME):
    """Use ParentPaper's load_finbert (no code duplication). Returns (tokenizer, model) or None."""
    _ensure_parent_paper_on_path()
    try:
        from main import load_finbert
        tokenizer, model = load_finbert(model_name)
        return tokenizer, model
    except ImportError:
        return None


def load_finbert_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
    """
    Load FinBERT the way ParentPaper does (reuse his code). If ParentPaper is not
    available, load tokenizer and model ourselves so the rest of the pipeline still works.

    Returns:
        tokenizer, model, device
        model is either ParentPaper's classifier (use .bert for hidden state) or our AutoModel.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pair = _load_finbert_from_parent_paper(model_name)
    if pair is not None:
        tokenizer, model = pair
        model.to(device)
        model.eval()
        return tokenizer, model, device

    # Fallback when ParentPaper is not on path (no rewrite of his logic; we load for standalone use)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def text_to_sentiment_vectors(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 128,
    use_cls_token: bool = True,
) -> torch.Tensor:
    """
    Get 768-d sentiment vectors from FinBERT. Builds on ParentPaper's model:
    we use the same tokenizer and model, but take last_hidden_state from the
    base BERT (model.bert) instead of the classification logits.

    Args:
        texts: list of strings (e.g. headlines)
        tokenizer, model, device: from load_finbert_and_tokenizer (ParentPaper's or fallback)
        max_length: same as ParentPaper's --maxlen
        use_cls_token: if True, use [CLS] token as vector; else mean pool

    Returns:
        (N, 768) tensor, CPU.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # ParentPaper's model is BertForSequenceClassification; .bert is the base encoder
    if hasattr(model, "bert"):
        with torch.no_grad():
            out = model.bert(**enc)
            hidden = out.last_hidden_state
    else:
        # standalone AutoModel path
        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state

    if use_cls_token:
        vectors = hidden[:, 0, :].cpu()
    else:
        mask = enc["attention_mask"].unsqueeze(-1).float()
        lengths = (mask.sum(dim=1).clamp(min=1e-9))
        vectors = (hidden * mask).sum(dim=1).div(lengths).cpu()

    return vectors


def load_news_from_parent_paper(csv_path: str | None = None, demo: bool = False):
    """
    Reuse ParentPaper's data loading when available: load_news_csv(path) or demo_news().
    Returns DataFrame with at least 'headline' (or None if ParentPaper not available).
    """
    _ensure_parent_paper_on_path()
    try:
        from main import load_news_csv, demo_news
        if demo:
            return demo_news()
        if csv_path:
            return load_news_csv(csv_path)
    except ImportError:
        pass
    return None
