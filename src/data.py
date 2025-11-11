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
    "ai": 1,
    "machine": 1,
    "machine-generated": 1,
    "machine_generated": 1,
}


def load_raid(
    *,
    include_adversarial: bool = True,
    validation_ratio: float = 0.1,
    seed: int = 42,
    raw: bool = False,
) -> DatasetDict | Dict[str, pd.DataFrame]:
    """
    Load RAID splits using the official raid-bench helper.
    Returns pandas DataFrames when `raw=True`, otherwise wraps them
    into a Hugging Face DatasetDict with a validation fold.
    """
    train_df = load_data(split="train", include_adversarial=include_adversarial)
    test_df = load_data(split="test", include_adversarial=include_adversarial)
    extra_df = load_data(split="extra", include_adversarial=include_adversarial)

    if raw:
        return {"train": train_df, "test": test_df, "extra": extra_df}

    train_df, val_df = _split_train_validation(train_df, validation_ratio, seed)
    test_df = _prepare_dataframe(test_df).reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": _to_dataset(train_df),
            "validation": _to_dataset(val_df),
            "test": _to_dataset(test_df),
        }
    )
    return dataset


def _split_train_validation(df: pd.DataFrame, validation_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    df = _prepare_dataframe(df)
    stratify = df[config.LABEL_FIELD]
    train_df, val_df = train_test_split(
        df,
        test_size=validation_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[config.TEXT_FIELD, config.LABEL_FIELD]].dropna(subset=[config.TEXT_FIELD, config.LABEL_FIELD]).copy()
    label_series = df[config.LABEL_FIELD]
    if label_series.dtype == object:
        normalized = label_series.astype(str).str.lower().map(_LABEL_NORMALIZATION)
        if normalized.isnull().any():
            raise ValueError("Encountered unknown label values while normalizing RAID labels.")
        df[config.LABEL_FIELD] = normalized.astype("int64")
    elif label_series.dtype == bool:
        df[config.LABEL_FIELD] = label_series.astype("int64")
    return df


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
