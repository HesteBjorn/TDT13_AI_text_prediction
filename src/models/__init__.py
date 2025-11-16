"""
Model wrappers for the AI text detection playground.
"""

from .distilbert_classifier import DistilBERTClassifier
from .llm_prompt import HeuristicDetector, PromptDetector
from .perplexity_classifier import PerplexityLogisticClassifier, train_perplexity_classifier
from .perplexity_detector import PerplexityDetector, PerplexityStatistics

__all__ = [
    "DistilBERTClassifier",
    "HeuristicDetector",
    "PerplexityDetector",
    "PerplexityStatistics",
    "PromptDetector",
    "PerplexityLogisticClassifier",
    "train_perplexity_classifier",
]
