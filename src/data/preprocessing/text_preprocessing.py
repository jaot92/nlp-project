"""
Módulo para preprocesamiento de texto de las reseñas de Amazon.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Union
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Clase para el preprocesamiento de texto de las reseñas de Amazon.
    
    Attributes:
        stop_words (set): Conjunto de stop words en inglés
        lemmatizer (WordNetLemmatizer): Instancia del lematizador
        contractions (dict): Diccionario de contracciones en inglés
    """
    
    def __init__(self):
        """Inicializa el preprocesador de texto."""
        logger.info("Inicializando TextPreprocessor...")
        
        # Descargar recursos necesarios de NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.error(f"Error descargando recursos NLTK: {str(e)}")
            raise
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Diccionario común de contracciones en inglés
        self.contractions = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            # ... más contracciones
        }
        
        logger.info("TextPreprocessor inicializado correctamente")
    
    def expand_contractions(self, text: str) -> str:
        """
        Expande las contracciones en el texto.
        
        Args:
            text (str): Texto a procesar
            
        Returns:
            str: Texto con contracciones expandidas
        """
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto removiendo caracteres especiales y normalizando.
        
        Args:
            text (str): Texto a limpiar
            
        Returns:
            str: Texto limpio
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Expandir contracciones
        text = self.expand_contractions(text)
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remover emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remover números y caracteres especiales
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remover espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza el texto en palabras.
        
        Args:
            text (str): Texto a tokenizar
            
        Returns:
            List[str]: Lista de tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remueve las stop words de la lista de tokens.
        
        Args:
            tokens (List[str]): Lista de tokens
            
        Returns:
            List[str]: Lista de tokens sin stop words
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lematiza la lista de tokens.
        
        Args:
            tokens (List[str]): Lista de tokens
            
        Returns:
            List[str]: Lista de tokens lematizados
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Aplica todo el pipeline de preprocesamiento al texto.
        
        Args:
            text (str): Texto a procesar
            
        Returns:
            str: Texto procesado
        """
        try:
            # Limpiar texto
            clean = self.clean_text(text)
            
            # Tokenizar
            tokens = self.tokenize(clean)
            
            # Remover stopwords
            tokens_no_stop = self.remove_stopwords(tokens)
            
            # Lematizar
            tokens_lemma = self.lemmatize(tokens_no_stop)
            
            # Unir tokens
            processed_text = ' '.join(tokens_lemma)
            
            return processed_text
        
        except Exception as e:
            logger.error(f"Error procesando texto: {str(e)}")
            return ""

def process_dataset(input_file: str, output_file: str, text_columns: List[str]):
    """
    Procesa el dataset aplicando el preprocesamiento de texto.
    
    Args:
        input_file (str): Ruta al archivo de entrada
        output_file (str): Ruta al archivo de salida
        text_columns (List[str]): Lista de columnas de texto a procesar
    """
    logger.info(f"Iniciando procesamiento del dataset: {input_file}")
    
    try:
        # Crear directorio de salida si no existe
        output_path = Path(output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar preprocesador
        preprocessor = TextPreprocessor()
        
        # Leer dataset en chunks para manejar archivos grandes
        chunk_size = 10000
        first_chunk = True
        
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            # Procesar cada columna de texto
            for col in text_columns:
                if col in chunk.columns:
                    chunk[f"{col}_processed"] = chunk[col].apply(preprocessor.preprocess)
            
            # Guardar chunk procesado
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(output_file, mode=mode, header=header, index=False)
            first_chunk = False
            
            logger.info(f"Procesado chunk de {len(chunk)} registros")
        
        logger.info(f"Procesamiento completado. Archivo guardado en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error procesando dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuración
    input_file = "data/processed/consolidated_reviews.csv"
    output_file = "data/processed/reviews_preprocessed.csv"
    text_columns = ["reviews.text", "reviews.title"]
    
    # Procesar dataset
    process_dataset(input_file, output_file, text_columns) 