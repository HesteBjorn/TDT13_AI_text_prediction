from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shap
from datasets import Dataset
from transformers import pipeline

from .. import config, data

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def _label_str_is_ai(label: str) -> bool:
    """Best-effort detection of whether a pipeline label corresponds to the AI class."""
    label_lower = label.lower()
    if label_lower.isdigit():
        return label_lower == "1"
    if label_lower.startswith("label_"):
        return label_lower.endswith("1")
    if "ai" in label_lower or "machine" in label_lower:
        return True
    if "human" in label_lower or "reference" in label_lower:
        return False
    # Fall back to checking trailing digit if present.
    digits = "".join(char for char in label_lower if char.isdigit())
    if digits:
        return digits[-1] == "1"
    return False


def _select_indices_with_ratio(labels: list[int], total: int, ratio: float, seed: int) -> list[int]:
    """Pick indices targeting a desired positive-class ratio."""
    if total <= 0:
        return []
    if not 0 <= ratio <= 1:
        raise ValueError("class_ratio must be between 0 and 1.")

    pos_idx = [i for i, label in enumerate(labels) if int(label) == 1]
    neg_idx = [i for i, label in enumerate(labels) if int(label) == 0]

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    target_pos = min(len(pos_idx), int(round(total * ratio)))
    target_neg = min(len(neg_idx), total - target_pos)

    selected = pos_idx[:target_pos] + neg_idx[:target_neg]

    if len(selected) < total:
        remaining = pos_idx[target_pos:] + neg_idx[target_neg:]
        if remaining:
            remaining = np.array(remaining)
            rng.shuffle(remaining)
            selected.extend(remaining[: total - len(selected)])

    selected = selected[:total]
    selected.sort()
    return selected


def _prepare_shap_arrays(shap_values):
    """
    SHAP sometimes returns ragged object arrays (shape like (n_samples, None, 2)).
    This helper normalizes them into iterables of per-sample (scores, tokens).
    """
    values = np.asarray(shap_values.values, dtype=object)
    tokens = np.asarray(shap_values.data, dtype=object)

    if values.dtype != object and values.ndim >= 2:
        if values.ndim == 3:
            return values[:, 0, :], tokens[:, 0, :]
        return values, tokens

    score_list: list[np.ndarray] = []
    token_list: list[np.ndarray] = []

    for sample_scores, sample_tokens in zip(values, tokens, strict=False):
        score_arr = np.asarray(sample_scores)
        token_arr = np.asarray(sample_tokens)

        if score_arr.ndim == 2:
            score_arr = score_arr[:, 0]
        score_arr = score_arr.reshape(-1)

        if token_arr.ndim == 2:
            token_arr = token_arr[:, 0]
        token_arr = token_arr.reshape(-1)

        limit = min(score_arr.shape[0], token_arr.shape[0])
        score_list.append(score_arr[:limit])
        token_list.append(token_arr[:limit])

    return score_list, token_list


@dataclass(slots=True)
class ShapConfig:
    checkpoint_path: Path
    num_samples: int = 32
    data_limit: int | None = None
    sample_seed: int = 42
    test_ratio: float = 0.2
    algorithm: str = "partition"
    class_ratio: float = 0.5  # Desired fraction of AI-labelled samples in SHAP slice


def load_test_texts(cfg: ShapConfig, return_labels: bool = False) -> list[str] | tuple[list[str], list[int]]:
    """Fetch a subset of the RAID test split for explanation.

    When `return_labels` is True, the ground-truth labels are returned alongside the texts.
    """
    config.ensure_directories()
    dataset = data.load_raid(
        limit=cfg.data_limit,
        sample_seed=cfg.sample_seed,
        test_ratio=cfg.test_ratio,
    )
    test_split = dataset["test"]
    total_available = len(test_split)
    sample_count = min(cfg.num_samples, total_available)
    if sample_count <= 0:
        return ([], []) if return_labels else []

    all_texts = test_split[config.TEXT_FIELD]
    all_labels = [int(label) for label in test_split[config.LABEL_FIELD]]

    if cfg.class_ratio is None:
        indices = list(range(sample_count))
    else:
        indices = _select_indices_with_ratio(all_labels, sample_count, cfg.class_ratio, cfg.sample_seed)

    texts = [all_texts[i] for i in indices]
    labels = [all_labels[i] for i in indices]

    if return_labels:
        return texts, labels
    return texts


def build_text_classifier(checkpoint_path: Path):
    """Create a Hugging Face pipeline for the fine-tuned checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_absolute():
        candidate_paths = [checkpoint_path.resolve()]
    else:
        candidate_paths = [
            (config.PROJECT_ROOT / checkpoint_path).resolve(),
            (Path.cwd() / checkpoint_path).resolve(),
        ]

    for path in candidate_paths:
        if path.exists():
            model_path = path
            break
    else:
        raise FileNotFoundError(
            f"Could not locate checkpoint at {checkpoint_path!s}. "
            f"Tried: {', '.join(str(p) for p in candidate_paths)}"
        )

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
    batch_size: int = 8,
) -> shap._explanation.Explanation:
    """Compute SHAP values for a set of texts using the provided pipeline."""
    masker = shap.maskers.Text(clf.tokenizer)
    explainer = shap.Explainer(clf, masker=masker, algorithm=algorithm)
    if not texts:
        raise ValueError("No texts provided for SHAP computation.")

    if batch_size is None or batch_size <= 0:
        batch_size = len(texts)

    dataset = Dataset.from_dict({config.TEXT_FIELD: texts})
    return explainer(dataset[config.TEXT_FIELD], batch_size=batch_size)


def aggregate_token_importance(shap_values, top_k: int = 20) -> pd.DataFrame:
    """Aggregate absolute SHAP scores per token across all samples."""
    scores, tokens = _prepare_shap_arrays(shap_values)
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
    true_labels: List[int] | None = None,
) -> pd.DataFrame:
    """Create a per-example summary with predictions and top contributing tokens."""
    if true_labels is not None and len(true_labels) != len(texts):
        raise ValueError("Number of true labels must match number of texts.")

    scores, tokens = _prepare_shap_arrays(shap_values)
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
        row = {
            "example": idx,
            "predicted_label": label,
            "confidence": confidence,
            "top_tokens": ", ".join(highlights),
            "text_snippet": snippet,
        }

        if true_labels is not None:
            true_label_value = true_labels[idx - 1]
            true_label_is_ai = int(true_label_value) == 1
            pred_is_ai = _label_str_is_ai(label)
            row["true_label"] = "AI" if true_label_is_ai else "Human"
            row["true_label_is_ai"] = true_label_is_ai
            row["prediction_is_ai"] = pred_is_ai
            row["prediction_correct"] = pred_is_ai == true_label_is_ai
            row["truth_annotation"] = f"Truth: {row['true_label']}"

        rows.append(row)

    return pd.DataFrame(rows)
