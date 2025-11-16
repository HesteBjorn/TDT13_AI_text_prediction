from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .. import config
from .perplexity_detector import PerplexityDetector, PerplexityStatistics


def _build_feature_matrix(stats: PerplexityStatistics) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack the statistics into a [num_examples, 3] matrix and return a finite mask.

    Returns:
        features: column stack of (nll, mean_logp, std_logp)
        mask: boolean array indicating which rows are fully finite
    """
    features = np.column_stack([stats.nll, stats.mean_logp, stats.std_logp]).astype(np.float32)
    mask = np.all(np.isfinite(features), axis=1)
    return features, mask


@dataclass(slots=True)
class PerplexityLogisticClassifier:
    """
    Wraps a PerplexityDetector with a LogisticRegression head that learns a data-driven boundary.
    """

    detector_config: dict
    logistic_model: LogisticRegression = field(repr=False)
    detector: PerplexityDetector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.detector = PerplexityDetector(**self.detector_config)

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "perplexity_classifier.pkl"
        with path.open("wb") as f:
            pickle.dump(
                {
                    "detector_config": self.detector_config,
                    "logistic_model": self.logistic_model,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> "PerplexityLogisticClassifier":
        path = Path(path)
        if path.is_dir():
            path = path / "perplexity_classifier.pkl"
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(detector_config=payload["detector_config"], logistic_model=payload["logistic_model"])

    def predict(self, texts: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        stats = self.detector.score(texts)
        features, mask = _build_feature_matrix(stats)
        probs = np.zeros(len(features), dtype=np.float32)
        preds = np.zeros(len(features), dtype=int)
        if mask.any():
            proba_valid = self.logistic_model.predict_proba(features[mask])[:, 1]
            probs[mask] = proba_valid
            preds[mask] = (proba_valid >= 0.5).astype(int)
        preds[~mask] = 1  # default to AI when we cannot score
        probs[~mask] = 0.5
        return preds, probs

    def evaluate(self, texts: Iterable[str], labels: Iterable[int]) -> dict:
        labels_arr = np.asarray(list(labels), dtype=int)
        preds, probs = self.predict(texts)
        metrics = {
            "accuracy": float(accuracy_score(labels_arr, preds)),
            "precision": float(precision_score(labels_arr, preds)),
            "recall": float(recall_score(labels_arr, preds)),
            "f1": float(f1_score(labels_arr, preds)),
            "roc_auc": float(roc_auc_score(labels_arr, probs)),
        }
        return metrics


def train_perplexity_classifier(
    detector_config: dict,
    train_texts: Iterable[str],
    train_labels: Iterable[int],
) -> LogisticRegression:
    detector = PerplexityDetector(**detector_config)
    stats = detector.score(train_texts)
    features, mask = _build_feature_matrix(stats)
    labels_arr = np.asarray(list(train_labels), dtype=int)
    if not mask.any():
        raise ValueError("No valid perplexity scores available to train the classifier.")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(features[mask], labels_arr[mask])
    return clf
