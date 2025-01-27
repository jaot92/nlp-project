"""
Main Transformer model class for sentiment analysis.
This module provides the main interface for training and using the model.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
import logging
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import TransformerConfig
from .dataset import DataProcessor, ReviewDataset
from .trainer import CustomTrainer
from .utils import setup_logging, compute_metrics

logger = logging.getLogger(__name__)

class TransformerSentimentClassifier:
    """Main class for the Transformer-based sentiment classifier"""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize the model
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar tokenizer y modelo
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.to(self.device)
        
        self.logger = setup_logging(config)
        self.logger.info(f"Modelo inicializado en dispositivo: {self.device}")
        
    def train(self,
              train_dataset: ReviewDataset,
              val_dataset: Optional[ReviewDataset] = None,
              class_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            class_weights: Optional tensor of class weights for imbalanced datasets
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Iniciando entrenamiento del modelo...")
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=self.config.MODEL_PATH,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.LOG_DIR,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=True,
            **self.config.training_args
        )
        
        # Inicializar trainer
        trainer = CustomTrainer(
            config=self.config,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            class_weights=class_weights,
            compute_metrics=compute_metrics
        )
        
        # Entrenar modelo
        train_result = trainer.train()
        metrics = train_result.metrics
        
        # Evaluar en conjunto de validación
        if val_dataset:
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            
        # Guardar modelo
        trainer.save_model(self.config.MODEL_PATH)
        self.logger.info(f"Modelo guardado en {self.config.MODEL_PATH}")
        
        return metrics
    
    def predict(self,
                texts: Union[str, List[str]],
                batch_size: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Predict sentiment for given texts.
        
        Args:
            texts: Single text or list of texts to analyze.
            batch_size: Batch size for processing. If None, uses config batch_size.
            
        Returns:
            List of dictionaries containing prediction probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if batch_size is None:
            batch_size = self.config.batch_size
            
        self.model.eval()
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and move to device
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            # Convert to list of dictionaries
            batch_predictions = [
                {
                    "negative": float(p[0]),
                    "positive": float(p[1])
                }
                for p in probs.cpu().numpy()
            ]
            predictions.extend(batch_predictions)
            
        return predictions
    
    def evaluate(self,
                test_dataset: ReviewDataset,
                output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
            output_dir: Optional directory to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Evaluando modelo en conjunto de prueba...")
        
        # Configurar argumentos de evaluación
        eval_args = TrainingArguments(
            output_dir=self.config.MODEL_PATH,
            per_device_eval_batch_size=self.config.batch_size,
            **self.config.eval_args
        )
        
        # Inicializar evaluador
        evaluator = CustomTrainer(
            config=self.config,
            model=self.model,
            args=eval_args,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Evaluar y obtener métricas
        metrics = evaluator.evaluate()
        
        # Obtener predicciones detalladas
        predictions = evaluator.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        # Realizar análisis de errores si se especifica directorio
        if output_dir:
            error_analysis = self.analyze_errors(
                texts=test_dataset.texts,
                true_labels=predictions.label_ids,
                pred_labels=pred_labels,
                probabilities=predictions.predictions,
                output_dir=output_dir
            )
            metrics.update(error_analysis)
        
        return metrics

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer from local files.
        
        Returns:
            The loaded tokenizer.
        """
        logger.info(f"Loading tokenizer from {self.config.TOKENIZER_PATH}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.TOKENIZER_PATH,
                local_files_only=True
            )
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
            
    def _load_model(self) -> PreTrainedModel:
        """
        Load the model from local files.
        
        Returns:
            The loaded model.
        """
        logger.info(f"Loading model from {self.config.MODEL_PATH}")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.MODEL_PATH,
                num_labels=self.config.num_labels,
                local_files_only=True,
                device_map="auto"
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def analyze_errors(self,
                      texts: List[str],
                      true_labels: List[int],
                      pred_labels: List[int],
                      probabilities: np.ndarray,
                      output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Perform detailed error analysis
        
        Args:
            texts: List of input texts
            true_labels: True labels
            pred_labels: Predicted labels
            probabilities: Prediction probabilities
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing error analysis results
        """
        # Crear DataFrame con predicciones
        results_df = pd.DataFrame({
            'text': texts,
            'true_label': true_labels,
            'pred_label': pred_labels,
            'confidence': np.max(probabilities, axis=1)
        })
        
        # Identificar errores
        results_df['is_error'] = results_df['true_label'] != results_df['pred_label']
        
        # Análisis de errores por clase
        error_analysis = {
            'total_errors': results_df['is_error'].sum(),
            'error_rate': results_df['is_error'].mean(),
            'errors_by_class': results_df[results_df['is_error']].groupby('true_label').size().to_dict(),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels)
        }
        
        # Análisis de confianza
        error_analysis.update({
            'mean_confidence_correct': results_df[~results_df['is_error']]['confidence'].mean(),
            'mean_confidence_errors': results_df[results_df['is_error']]['confidence'].mean(),
            'high_confidence_errors': results_df[
                (results_df['is_error']) & (results_df['confidence'] > 0.9)
            ].shape[0]
        })
        
        # Guardar resultados si se especifica un directorio
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar errores más significativos
            significant_errors = results_df[
                (results_df['is_error']) & (results_df['confidence'] > 0.8)
            ].sort_values('confidence', ascending=False)
            
            significant_errors.to_csv(
                output_dir / 'significant_errors.csv',
                index=False
            )
            
            # Visualizar distribución de confianza
            self._plot_confidence_distribution(
                results_df,
                output_dir / 'confidence_distribution.png'
            )
            
        return error_analysis
    
    def test_robustness(self,
                       texts: List[str],
                       labels: List[int],
                       perturbation_types: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Test model robustness against various perturbations
        
        Args:
            texts: List of input texts
            labels: True labels
            perturbation_types: Types of perturbations to test
            
        Returns:
            Dictionary with robustness metrics
        """
        if perturbation_types is None:
            perturbation_types = ['typos', 'word_swap', 'punctuation']
            
        robustness_results = {}
        
        for pert_type in perturbation_types:
            # Aplicar perturbación
            perturbed_texts = self._apply_perturbation(texts, pert_type)
            
            # Obtener predicciones
            predictions = self.predict(perturbed_texts)
            pred_labels = np.argmax(predictions, axis=1)
            
            # Calcular métricas
            metrics = compute_metrics(pred_labels, labels)
            robustness_results[pert_type] = metrics
            
        return robustness_results
    
    def _apply_perturbation(self, texts: List[str], perturbation_type: str) -> List[str]:
        """
        Apply specified perturbation to texts
        
        Args:
            texts: List of input texts
            perturbation_type: Type of perturbation to apply
            
        Returns:
            List of perturbed texts
        """
        perturbed_texts = []
        
        for text in texts:
            if perturbation_type == 'typos':
                perturbed = self._introduce_typos(text)
            elif perturbation_type == 'word_swap':
                perturbed = self._swap_words(text)
            elif perturbation_type == 'punctuation':
                perturbed = self._modify_punctuation(text)
            else:
                perturbed = text
                
            perturbed_texts.append(perturbed)
            
        return perturbed_texts
    
    def _introduce_typos(self, text: str) -> str:
        """Introduce random typos in text"""
        # Implementación básica: cambiar algunas letras
        chars = list(text)
        n_typos = max(1, len(text) // 50)  # 1 typo por cada 50 caracteres
        
        for _ in range(n_typos):
            idx = np.random.randint(0, len(chars))
            if chars[idx].isalpha():
                # Cambiar por una letra cercana en el teclado
                keyboard_neighbors = {
                    'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf',
                    'e': 'wr', 'f': 'dg', 'g': 'fh', 'h': 'gj',
                    'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
                    'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o',
                    'q': 'a', 'r': 'et', 's': 'ad', 't': 'ry',
                    'u': 'yi', 'v': 'cb', 'w': 'e', 'x': 'c',
                    'y': 'tu', 'z': 'x'
                }
                char = chars[idx].lower()
                if char in keyboard_neighbors:
                    replacement = np.random.choice(list(keyboard_neighbors[char]))
                    chars[idx] = replacement
                    
        return ''.join(chars)
    
    def _swap_words(self, text: str) -> str:
        """Swap some words in the text"""
        words = text.split()
        if len(words) < 2:
            return text
            
        n_swaps = max(1, len(words) // 20)  # 1 swap por cada 20 palabras
        
        for _ in range(n_swaps):
            idx1 = np.random.randint(0, len(words))
            idx2 = np.random.randint(0, len(words))
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def _modify_punctuation(self, text: str) -> str:
        """Modify punctuation in text"""
        # Eliminar o agregar signos de puntuación aleatorios
        punctuation = ',.!?'
        chars = list(text)
        
        n_modifications = max(1, len(text) // 50)
        
        for _ in range(n_modifications):
            if np.random.random() < 0.5 and len(chars) > 0:
                # Eliminar puntuación
                idx = np.random.randint(0, len(chars))
                if chars[idx] in punctuation:
                    chars.pop(idx)
            else:
                # Agregar puntuación
                idx = np.random.randint(0, len(chars))
                chars.insert(idx, np.random.choice(list(punctuation)))
                
        return ''.join(chars)
    
    def _plot_confidence_distribution(self,
                                   results_df: pd.DataFrame,
                                   output_path: Path) -> None:
        """
        Plot distribution of prediction confidence
        
        Args:
            results_df: DataFrame with prediction results
            output_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Distribución para predicciones correctas e incorrectas
        sns.histplot(
            data=results_df,
            x='confidence',
            hue='is_error',
            multiple="layer",
            bins=30
        )
        
        plt.title('Distribución de Confianza en Predicciones')
        plt.xlabel('Confianza')
        plt.ylabel('Frecuencia')
        
        plt.savefig(output_path)
        plt.close()

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save the model and tokenizer.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def to(self, device: Union[str, torch.device]) -> None:
        """Move the model to the specified device.
        
        Args:
            device: Device to move the model to.
        """
        self.device = torch.device(device)
        self.model.to(self.device) 