from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shap
from transformers import pipeline

from .. import config, data

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


@dataclass(slots=True)
class ShapConfig:
    checkpoint_path: Path
    num_samples: int = 32
    data_limit: int | None = None
    sample_seed: int = 42
    test_ratio: float = 0.2
    algorithm: str = "partition"


def load_test_texts(cfg: ShapConfig) -> list[str]:
    """Fetch a subset of the RAID test split for explanation."""
    config.ensure_directories()
    dataset = data.load_raid(
        limit=cfg.data_limit,
        sample_seed=cfg.sample_seed,
        test_ratio=cfg.test_ratio,
    )
    texts = dataset["test"][config.TEXT_FIELD][: cfg.num_samples]
    return list(texts)


def build_text_classifier(checkpoint_path: Path):
    """Create a Hugging Face pipeline for the fine-tuned checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        model_path = (Path.cwd() / checkpoint_path).resolve()
    else:
        model_path = checkpoint_path.resolve()
    return pipeline(
        "text-classification",
        model=model_path.as_posix(),
        tokenizer=model_path.as_posix(),
        truncation=True,
        padding=True,
        top_k=None,
    )


def compute_shap_values(
    clf,
    texts: list[str],
    algorithm: str = "partition",
) -> shap._explanation.Explanation:
    """Compute SHAP values for a set of texts using the provided pipeline."""
    masker = shap.maskers.Text(clf.tokenizer)
    explainer = shap.Explainer(clf, masker=masker, algorithm=algorithm)
    return explainer(texts)


def aggregate_token_importance(shap_values, top_k: int = 20) -> pd.DataFrame:
    """Aggregate absolute SHAP scores per token across all samples."""
    if shap_values.values.ndim < 3:
        raise ValueError("Expected SHAP values with shape (n_samples, n_classes, seq_len).")

    scores = shap_values.values[:, 0, :]
    tokens = shap_values.data[:, 0, :]
    records: dict[str, dict[str, float | int]] = {}

    for sample_scores, sample_tokens in zip(scores, tokens, strict=False):
        for token, score in zip(sample_tokens, sample_scores, strict=False):
            token = token.strip()
            if not token or token in _SPECIAL_TOKENS:
                continue
            entry = records.setdefault(
                token,
                {"token": token, "count": 0, "mean_score": 0.0, "mean_abs_score": 0.0},
            )
            entry["count"] += 1
            entry["mean_score"] += float(score)
            entry["mean_abs_score"] += float(abs(score))

    if not records:
        return pd.DataFrame(columns=["token", "count", "mean_score", "mean_abs_score"])

    df = pd.DataFrame(records.values())
    df["mean_score"] = df["mean_score"] / df["count"]
    df["mean_abs_score"] = df["mean_abs_score"] / df["count"]
    df = df.sort_values("mean_abs_score", ascending=False).head(top_k).reset_index(drop=True)
    return df


def summarize_examples(
    clf,
    shap_values,
    texts: List[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Create a per-example summary with predictions and top contributing tokens."""
    scores = shap_values.values[:, 0, :]
    tokens = shap_values.data[:, 0, :]
    raw_preds = clf(texts, top_k=1)

    rows = []
    for idx, (text, pred, token_scores, token_strings) in enumerate(
        zip(texts, raw_preds, scores, tokens, strict=False), start=1
    ):
        pred_entry = pred[0] if isinstance(pred, list) else pred
        label = pred_entry["label"]
        confidence = float(pred_entry["score"])

        order = np.argsort(-np.abs(token_scores))
        highlights = []
        for position in order:
            token = token_strings[position].strip()
            if not token or token in _SPECIAL_TOKENS:
                continue
            highlights.append(f"{token} ({token_scores[position]:+.3f})")
            if len(highlights) >= top_k:
                break

        snippet = text[:160] + ("..." if len(text) > 160 else "")
        rows.append(
            {
                "example": idx,
                "predicted_label": label,
                "confidence": confidence,
                "top_tokens": ", ".join(highlights),
                "text_snippet": snippet,
            }
        )

    return pd.DataFrame(rows)
