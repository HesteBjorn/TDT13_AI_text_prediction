from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple

import evaluate
import pandas as pd
from datasets import Dataset, DatasetDict
from raid.utils import load_data
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import config

_LABEL_NORMALIZATION = {
    "human": 0,
    "human-written": 0,
    "human_written": 0,
    "human_reference": 0,
    "human-reference": 0,
    "ai": 1,
    "ai_written": 1,
    "ai-written": 1,
    "machine": 1,
    "machine-generated": 1,
    "machine_generated": 1,
}

TEXT_SOURCE_COLUMN = "generation"
LABEL_SOURCE_COLUMN = "label"
MODEL_SOURCE_COLUMN = "model"

_HUMAN_MODEL_TOKENS = {"human", "human-reference", "human_reference", "reference"}


def load_raid(
    *,
    include_adversarial: bool = config.DEFAULT_INCLUDE_ADVERSARIAL,
    test_ratio: float = 0.2,
    seed: int = 42,
    raw: bool = False,
    limit: int | None = None,
    sample_seed: int = 42,
) -> DatasetDict | Dict[str, pd.DataFrame]:
    """
    Load the RAID train split via the official raid-bench helper and derive
    train/test subsets locally.
    When `limit` is set, the base train split is randomly subsampled (after download) to that many rows.
    """
    base_df = _prepare_dataframe(load_data(split="train", include_adversarial=include_adversarial))
    base_df = _limit_dataframe(base_df, limit, sample_seed)
    train_df, test_df = _train_test_split(
        base_df,
        test_ratio=test_ratio,
        seed=seed,
    )

    if raw:
        return {"train": train_df, "test": test_df}

    return DatasetDict(
        {
            "train": _to_dataset(train_df),
            "test": _to_dataset(test_df),
        }
    )


def _limit_dataframe(
    df: pd.DataFrame,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df.reset_index(drop=True)
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if limit < 2:
        raise ValueError("limit must be at least 2 to create train/test splits.")
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def _train_test_split(
    df: pd.DataFrame,
    *,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    stratify = df[config.LABEL_FIELD]
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if TEXT_SOURCE_COLUMN not in df.columns:
        raise KeyError(f"Expected '{TEXT_SOURCE_COLUMN}' column in RAID dataframe.")
    texts = df[TEXT_SOURCE_COLUMN]
    labels = _extract_labels(df)
    valid_mask = texts.notna() & labels.notna()
    if not valid_mask.any():
        raise ValueError("No valid rows remained after filtering empty text/label entries.")

    prepared = df.loc[valid_mask, [TEXT_SOURCE_COLUMN]].rename(columns={TEXT_SOURCE_COLUMN: config.TEXT_FIELD}).copy()
    prepared[config.LABEL_FIELD] = labels.loc[valid_mask].to_numpy(dtype="int64")
    return prepared.reset_index(drop=True)


def _extract_labels(df: pd.DataFrame) -> pd.Series:
    if LABEL_SOURCE_COLUMN in df.columns:
        series = df[LABEL_SOURCE_COLUMN]
        return _normalize_label_series(series)

    if MODEL_SOURCE_COLUMN not in df.columns:
        raise KeyError("RAID dataframe did not contain a label column nor a 'model' column to derive labels from.")
    series = (~df[MODEL_SOURCE_COLUMN].astype(str).str.lower().isin(_HUMAN_MODEL_TOKENS)).astype(int)
    return series


def _normalize_label_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("int64")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64")
    normalized = series.astype(str).str.lower().map(_LABEL_NORMALIZATION)
    if normalized.isnull().any():
        unknown = series[normalized.isnull()].unique()
        raise ValueError(f"Encountered unknown label values while normalizing RAID labels: {unknown[:5]!r}")
    return normalized.astype("int64")


def _to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str | None = None) -> PreTrainedTokenizerBase:
    """Lazily creates a tokenizer for the desired backbone."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name or config.DEFAULT_TRAINING_CONFIG.model_name,
        cache_dir=str(config.DATA_DIR / "hf-cache"),
    )
    # DistilBERT lacks a pad token by default; align to CLS token if needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    return tokenizer


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        tokens = tokenizer(
            batch[config.TEXT_FIELD],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = batch[config.LABEL_FIELD]
        return tokens

    tokenized = dataset.map(tokenize, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized = tokenized.remove_columns(
        [col for col in tokenized["train"].column_names if col not in keep_cols]
    )
    tokenized.set_format("torch")
    return tokenized


def build_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels)["f1"],
        }

    return compute_metrics
