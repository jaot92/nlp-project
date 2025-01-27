"""
Custom trainer class for the Transformer model.
Extends the Hugging Face Trainer with custom functionality.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy
)
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from pathlib import Path
import logging
from tqdm.auto import tqdm
import dataclasses
from .utils import compute_metrics, plot_confusion_matrix, plot_training_history, save_predictions

class CustomTrainer(Trainer):
    """Custom trainer class with additional functionality"""
    
    def __init__(self,
                 config: 'TransformerConfig',
                 class_weights: Optional[torch.Tensor] = None,
                 early_stopping_patience: int = 3,
                 early_stopping_threshold: float = 0.01,
                 **kwargs):
        """
        Initialize CustomTrainer
        
        Args:
            config: Configuration object
            class_weights: Optional tensor of class weights for imbalanced datasets
            early_stopping_patience: Number of epochs to wait before early stopping
            early_stopping_threshold: Minimum change in the monitored quantity to qualify as an improvement
            **kwargs: Additional arguments passed to Trainer
        """
        # Configurar early stopping
        callbacks = kwargs.get('callbacks', [])
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        )
        kwargs['callbacks'] = callbacks
        
        self.config = config
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(kwargs['model'].device)
        
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation with class weights
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs: If True, returns the model outputs along with the loss
            
        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The loss or the loss and model outputs
        """
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def train(self,
              resume_from_checkpoint: Optional[Union[str, bool]] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              **kwargs) -> torch.nn.Module:
        """
        Custom training with additional logging and visualization
        
        Args:
            resume_from_checkpoint: Path to a checkpoint to resume from
            trial: Trial object for hyperparameter optimization
            **kwargs: Additional arguments
            
        Returns:
            torch.nn.Module: The trained model
        """
        # Entrenar el modelo
        result = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            **kwargs
        )
        
        # Guardar historial de entrenamiento
        if hasattr(self, 'config') and hasattr(self.config, 'LOG_DIR'):
            plot_training_history(
                self.state.log_history,
                save_path=self.config.LOG_DIR / "training_history.png"
            )
        
        return result
    
    def evaluate(self, 
                eval_dataset: Optional[torch.utils.data.Dataset] = None,
                ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "eval") -> Dict[str, float]:
        """
        Custom evaluation with additional metrics and logging
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model outputs
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Dict[str, float]: The evaluation metrics
        """
        # Evaluación estándar
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Obtener predicciones detalladas
        predictions = self.predict(eval_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Calcular métricas adicionales
        detailed_metrics = compute_metrics(pred_labels, true_labels)
        metrics.update({
            f"{metric_key_prefix}_{k}": v 
            for k, v in detailed_metrics.items()
        })
        
        # Guardar matriz de confusión
        if hasattr(self, 'config') and hasattr(self.config, 'LOG_DIR'):
            plot_confusion_matrix(
                true_labels=true_labels,
                pred_labels=pred_labels,
                save_path=self.config.LOG_DIR / f"{metric_key_prefix}_confusion_matrix.png"
            )
        
        return metrics
    
    def predict(self,
                test_dataset: torch.utils.data.Dataset,
                ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "test") -> 'PredictionOutput':
        """
        Run predictions and save results
        
        Args:
            test_dataset: Dataset to predict on
            ignore_keys: Keys to ignore in the model outputs
            metric_key_prefix: Prefix for metric names
            
        Returns:
            PredictionOutput: Object containing predictions
        """
        predictions = super().predict(
            test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Guardar predicciones si hay un directorio de logs configurado
        if hasattr(self, 'config') and hasattr(self.config, 'LOG_DIR'):
            # Obtener textos originales si están disponibles
            if hasattr(test_dataset, 'texts'):
                texts = test_dataset.texts
            else:
                texts = [f"Text_{i}" for i in range(len(predictions.predictions))]
            
            save_predictions(
                texts=texts,
                true_labels=predictions.label_ids,
                pred_labels=np.argmax(predictions.predictions, axis=1),
                probabilities=predictions.predictions,
                output_file=self.config.LOG_DIR / f"{metric_key_prefix}_predictions.csv"
            )
        
        return predictions
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Custom logging
        
        Args:
            logs: Dictionary of logs to record
        """
        # Logging estándar
        super().log(logs) 