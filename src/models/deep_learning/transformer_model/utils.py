"""
Utility functions for the Transformer model.
Includes logging setup, metrics computation, and visualization tools.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Any
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def setup_logging(config: 'TransformerConfig') -> logging.Logger:
    """
    Configure logging for the model
    
    Args:
        config: TransformerConfig instance
        
    Returns:
        logging.Logger: Configured logger
    """
    # Crear el directorio de logs si no existe
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar el nombre del archivo de log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"transformer_model_{timestamp}.log"
    
    # Configurar el logger
    logger = logging.getLogger("transformer_model")
    logger.setLevel(config.log_level)
    
    # Configurar el formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def compute_metrics(pred_labels: Union[np.ndarray, List[int]], 
                   true_labels: Union[np.ndarray, List[int]]) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        pred_labels: Predicted labels
        true_labels: True labels
        
    Returns:
        Dict with metrics (accuracy, precision, recall, f1)
    """
    # Convertir a numpy arrays si son listas
    if isinstance(pred_labels, list):
        pred_labels = np.array(pred_labels)
    if isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    
    # Calcular métricas básicas
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        pred_labels, 
        average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(true_labels: np.ndarray, 
                         pred_labels: np.ndarray,
                         save_path: Path = None) -> None:
    """
    Plot and optionally save confusion matrix
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Path = None) -> None:
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training')
    if 'eval_loss' in history:
        plt.plot(history['eval_loss'], label='Validation')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    for metric in history.keys():
        if metric.startswith('eval_') and metric != 'eval_loss':
            plt.plot(history[metric], label=metric.replace('eval_', ''))
    plt.title('Metrics vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_predictions(texts: List[str],
                    true_labels: List[int],
                    pred_labels: List[int],
                    probabilities: np.ndarray,
                    output_file: Path) -> None:
    """
    Save prediction results to file
    
    Args:
        texts: Input texts
        true_labels: True labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        output_file: Path to save results
    """
    import pandas as pd
    
    results = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': pred_labels,
        'confidence': np.max(probabilities, axis=1)
    })
    
    # Añadir probabilidades por clase
    for i in range(probabilities.shape[1]):
        results[f'prob_class_{i}'] = probabilities[:, i]
    
    results.to_csv(output_file, index=False)

def get_prediction_errors(texts: List[str],
                         true_labels: List[int],
                         pred_labels: List[int],
                         probabilities: np.ndarray,
                         n_examples: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get examples of prediction errors for analysis
    
    Args:
        texts: Input texts
        true_labels: True labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        n_examples: Number of examples to return
        
    Returns:
        Dictionary with error examples
    """
    errors = []
    for i, (text, true, pred, probs) in enumerate(zip(texts, true_labels, pred_labels, probabilities)):
        if true != pred:
            errors.append({
                'text': text,
                'true_label': true,
                'predicted_label': pred,
                'confidence': float(probs[pred]),
                'all_probabilities': probs.tolist()
            })
    
    # Ordenar por confianza de predicción
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'high_confidence_errors': errors[:n_examples],
        'low_confidence_errors': errors[-n_examples:]
    } 