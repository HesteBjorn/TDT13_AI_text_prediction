"""
Model wrappers for the AI text detection playground.
"""

from .distilbert_classifier import DistilBERTClassifier
from .llm_prompt import HeuristicDetector, PromptDetector

__all__ = ["DistilBERTClassifier", "HeuristicDetector", "PromptDetector"]
