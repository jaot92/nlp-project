"""
Dataset and data processing classes for the Transformer model.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging
from transformers import PreTrainedTokenizer
from .utils import setup_logging
import matplotlib.pyplot as plt
import seaborn as sns

class ReviewDataset(Dataset):
    """Dataset class for reviews"""
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 128):
        """
        Initialize ReviewDataset
        
        Args:
            texts: List of text reviews
            labels: List of labels
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenizar todos los textos de una vez
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        self.labels_tensor = torch.tensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        item = {
            key: val[idx] for key, val in self.encodings.items()
        }
        item['labels'] = self.labels_tensor[idx]
        return item

class DataProcessor:
    """Class for handling all data processing tasks"""
    
    def __init__(self, config: 'TransformerConfig'):
        """
        Initialize DataProcessor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logging(config)
        self.label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def load_and_preprocess_data(self, 
                                data_path: Union[str, Path],
                                text_column: str = 'reviews.text_processed',
                                label_column: str = 'sentiment',
                                test_size: float = 0.2,
                                val_size: float = 0.2,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the data
        
        Args:
            data_path: Path to the data file
            text_column: Name of the text column
            label_column: Name of the label column
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Iniciando carga y preprocesamiento de datos...")
        
        # Cargar datos
        df = pd.read_csv(data_path)
        self.logger.info(f"Dataset cargado con {len(df)} registros")
        
        # Validar columnas
        required_columns = [text_column, label_column]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Columnas faltantes en el dataset: {missing}")
        
        # Limpieza básica
        df = df.dropna(subset=[text_column, label_column])
        
        # Convertir etiquetas
        df['label'] = df[label_column].map(self.label_mapping)
        
        # División de datos
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_column]
        )
        
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_df[label_column]
        )
        
        # Calcular y guardar pesos de clase
        self.class_weights = self._compute_class_weights(train_df['label'])
        
        # Analizar y visualizar distribución de datos
        self._analyze_and_visualize_data(train_df, val_df, test_df, text_column, label_column)
        
        return train_df, val_df, test_df
    
    def create_datasets(self,
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       tokenizer: PreTrainedTokenizer,
                       text_column: str = 'reviews.text_processed') -> Tuple[ReviewDataset, ReviewDataset, ReviewDataset]:
        """
        Create PyTorch datasets
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            tokenizer: Tokenizer to use
            text_column: Name of text column
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        self.logger.info("Creando datasets de PyTorch...")
        
        train_dataset = ReviewDataset(
            texts=train_df[text_column].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        val_dataset = ReviewDataset(
            texts=val_df[text_column].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        test_dataset = ReviewDataset(
            texts=test_df[text_column].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        self.logger.info(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
        self.logger.info(f"Tamaño del conjunto de validación: {len(val_dataset)}")
        self.logger.info(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _compute_class_weights(self, labels: pd.Series) -> torch.Tensor:
        """Compute class weights for imbalanced dataset"""
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(weights)
    
    def _analyze_and_visualize_data(self,
                                  train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  text_column: str,
                                  label_column: str) -> None:
        """
        Analyze and visualize dataset characteristics
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            text_column: Name of text column
            label_column: Name of label column
        """
        # Log class distribution
        self._log_class_distribution(train_df, val_df, test_df, label_column)
        
        # Analyze text lengths
        text_stats = self.analyze_text_lengths(train_df, text_column)
        
        # Visualize class distribution
        self._plot_class_distribution(train_df, val_df, test_df, label_column)
        
        # Visualize text length distribution
        self._plot_text_length_distribution(train_df, text_column)
    
    def _log_class_distribution(self,
                              train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              label_column: str) -> None:
        """Log class distribution for all splits"""
        self.logger.info("Distribución de clases en cada conjunto:")
        
        self.logger.info("\nEntrenamiento:")
        self.logger.info(train_df[label_column].value_counts(normalize=True))
        
        self.logger.info("\nValidación:")
        self.logger.info(val_df[label_column].value_counts(normalize=True))
        
        self.logger.info("\nPrueba:")
        self.logger.info(test_df[label_column].value_counts(normalize=True))
    
    def analyze_text_lengths(self,
                           df: pd.DataFrame,
                           text_column: str = 'reviews.text_processed') -> Dict[str, float]:
        """
        Analyze text lengths in the dataset
        
        Args:
            df: DataFrame with texts
            text_column: Name of text column
            
        Returns:
            Dictionary with text length statistics
        """
        lengths = df[text_column].str.split().str.len()
        
        stats = {
            'mean': lengths.mean(),
            'median': lengths.median(),
            'p95': lengths.quantile(0.95),
            'max': lengths.max(),
            'min': lengths.min(),
            'std': lengths.std()
        }
        
        self.logger.info("Estadísticas de longitud de texto:")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value:.2f}")
        
        return stats
    
    def _plot_class_distribution(self,
                               train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               label_column: str) -> None:
        """
        Plot class distribution for all splits
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            label_column: Name of label column
        """
        plt.figure(figsize=(15, 5))
        
        # Training set
        plt.subplot(131)
        train_dist = train_df[label_column].value_counts(normalize=True)
        sns.barplot(x=train_dist.index, y=train_dist.values)
        plt.title('Training Set Distribution')
        plt.ylabel('Proportion')
        
        # Validation set
        plt.subplot(132)
        val_dist = val_df[label_column].value_counts(normalize=True)
        sns.barplot(x=val_dist.index, y=val_dist.values)
        plt.title('Validation Set Distribution')
        
        # Test set
        plt.subplot(133)
        test_dist = test_df[label_column].value_counts(normalize=True)
        sns.barplot(x=test_dist.index, y=test_dist.values)
        plt.title('Test Set Distribution')
        
        plt.tight_layout()
        plt.savefig(self.config.LOG_DIR / 'class_distribution.png')
        plt.close()
    
    def _plot_text_length_distribution(self,
                                     df: pd.DataFrame,
                                     text_column: str) -> None:
        """
        Plot text length distribution
        
        Args:
            df: DataFrame with texts
            text_column: Name of text column
        """
        lengths = df[text_column].str.split().str.len()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=lengths, bins=50)
        plt.title('Text Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        
        # Add vertical lines for statistics
        plt.axvline(lengths.mean(), color='r', linestyle='--', label=f'Mean: {lengths.mean():.1f}')
        plt.axvline(lengths.median(), color='g', linestyle='--', label=f'Median: {lengths.median():.1f}')
        plt.axvline(lengths.quantile(0.95), color='b', linestyle='--', label=f'95th Percentile: {lengths.quantile(0.95):.1f}')
        
        plt.legend()
        plt.savefig(self.config.LOG_DIR / 'text_length_distribution.png')
        plt.close() 