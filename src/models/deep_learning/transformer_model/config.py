"""
Configuration module for the Transformer model.

This module defines parameters and settings for training and inference of the Transformer model.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Project paths
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

class TransformerConfig:
    """Configuration class for the Transformer model"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        # Model paths
        self.MODEL_PATH = MODELS_DIR / "bert_sentiment"
        self.TOKENIZER_PATH = MODELS_DIR / "bert_tokenizer"
        self.T5_MODEL_PATH = MODELS_DIR / "t5_summarizer"
        self.LOG_DIR = LOGS_DIR / "transformer_model"
        
        # Model configuration
        self.model_name = "bert-base-multilingual-cased"
        self.max_length = 512
        self.batch_size = 16
        self.num_labels = 2
        
        # Training parameters
        self.learning_rate = 2e-5
        self.num_train_epochs = 3
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        
        # Logging and saving
        self.logging_dir = LOGS_DIR / "transformer_logs"
        self.logging_steps = 100
        self.save_steps = 1000
        self.eval_steps = 500
        self.log_level = logging.INFO
        
        # Create directories if they don't exist
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify model paths exist
        if not (self.MODEL_PATH / "model.safetensors").exists():
            raise FileNotFoundError(f"Model not found at {self.MODEL_PATH}")
        if not (self.TOKENIZER_PATH / "tokenizer_config.json").exists():
            raise FileNotFoundError(f"Tokenizer not found at {self.TOKENIZER_PATH}")
    
    def save(self, path: Path) -> None:
        """
        Save configuration to JSON file
        
        Args:
            path: Path to save the configuration
        """
        config_dict = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'num_epochs': self.num_train_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'logging_steps': self.logging_steps,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'max_grad_norm': self.max_grad_norm,
            'log_level': logging.getLevelName(self.log_level)
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: Path) -> 'TransformerConfig':
        """
        Load configuration from JSON file
        
        Args:
            path: Path to the configuration file
            
        Returns:
            TransformerConfig: Loaded configuration
        """
        config = cls()
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        for key, value in config_dict.items():
            setattr(config, key, value)
            
        return config 