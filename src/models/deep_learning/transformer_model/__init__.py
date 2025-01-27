"""
Transformer Model Module for Sentiment Analysis
This module provides a PyTorch implementation of a transformer-based model for sentiment analysis.
"""

from .config import TransformerConfig
from .model import TransformerSentimentClassifier
from .dataset import ReviewDataset, DataProcessor
from .trainer import CustomTrainer
from .utils import setup_logging, compute_metrics

__version__ = "1.0.0"
__author__ = "IronHack NLP Project" 