import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMSentimentClassifier:
    """
    Clase para entrenar y evaluar un modelo LSTM bidireccional para clasificación de sentimiento.
    """
    
    def __init__(self, max_words=10000, max_len=200, embedding_dim=100):
        """
        Inicializa el clasificador LSTM.
        
        Args:
            max_words (int): Número máximo de palabras en el vocabulario
            max_len (int): Longitud máxima de las secuencias
            embedding_dim (int): Dimensión del embedding
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        
    def prepare_data(self, data_file: str):
        """
        Prepara los datos para el entrenamiento.
        
        Args:
            data_file (str): Ruta al archivo de datos preprocesados
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, class_weights
        """
        logger.info("Cargando y preparando datos...")
        
        # Cargar datos
        df = pd.read_csv(data_file)
        
        # Verificar y reportar valores nulos
        null_counts = df[['reviews.text_processed', 'sentiment']].isnull().sum()
        logger.info(f"Valores nulos antes de limpieza:\n{null_counts}")
        
        # Limpiar valores nulos
        df['reviews.text_processed'] = df['reviews.text_processed'].fillna('')
        df = df.dropna(subset=['sentiment'])
        
        # Verificar datos después de limpieza
        logger.info(f"Registros después de limpieza: {len(df)}")
        
        # Tokenizar textos
        self.tokenizer.fit_on_texts(df['reviews.text_processed'])
        sequences = self.tokenizer.texts_to_sequences(df['reviews.text_processed'])
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Preparar etiquetas
        y = pd.get_dummies(df['sentiment']).values
        
        # Calcular class weights para balancear el entrenamiento
        class_counts = df['sentiment'].value_counts()
        total = len(df)
        base_weights = {
            i: total / (len(class_counts) * count)
            for i, count in enumerate(class_counts.sort_index())
        }
        
        # Ajustar pesos para dar más énfasis a clases minoritarias
        max_weight = max(base_weights.values())
        class_weights = {
            k: min(v * 2.0, max_weight * 5.0)  # Más énfasis en clases minoritarias
            for k, v in base_weights.items()
        }
        logger.info(f"\nPesos por clase:\n{class_weights}")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df['sentiment']
        )
        
        logger.info(f"Datos divididos - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test, class_weights
    
    def build_model(self):
        """
        Construye el modelo LSTM bidireccional con arquitectura mejorada.
        """
        # Regularizadores
        lstm_regularizer = l1_l2(l1=1e-5, l2=1e-4)
        dense_regularizer = l1_l2(l1=1e-6, l2=1e-5)
        
        self.model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(256, return_sequences=True, 
                             kernel_regularizer=lstm_regularizer)),
            Dropout(0.2),
            Bidirectional(LSTM(128, return_sequences=True,
                             kernel_regularizer=lstm_regularizer)),
            Dropout(0.2),
            Bidirectional(LSTM(64, kernel_regularizer=lstm_regularizer)),
            Dropout(0.2),
            Dense(256, activation='relu', kernel_regularizer=dense_regularizer),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu', kernel_regularizer=dense_regularizer),
            BatchNormalization(),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        # Construir el modelo con shape de entrada conocido
        self.model.build((None, self.max_len))
        
        # Usar Adam con learning rate más alto y clipnorm
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Resumen del modelo:")
        logger.info(self.model.summary())
    
    def train(self, X_train, y_train, X_test, y_test, class_weights, epochs=20, batch_size=32):
        """
        Entrena el modelo LSTM con class weights ajustados.
        """
        logger.info("Iniciando entrenamiento del modelo LSTM...")
        
        # Ajustar class weights para ser menos agresivos
        adjusted_weights = {
            k: min(v, 3.0) for k, v in class_weights.items()
        }
        logger.info(f"Class weights ajustados: {adjusted_weights}")
        
        # Callbacks mejorados
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Cambiado a accuracy
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Reducción más suave
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            class_weight=adjusted_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test, output_dir: str):
        """
        Evalúa el modelo y guarda los resultados.
        
        Args:
            X_test, y_test: Datos de prueba
            output_dir (str): Directorio para guardar resultados
        """
        # Crear directorio si no existe
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Reporte de clasificación
        report = classification_report(y_test_classes, y_pred_classes, zero_division=1)
        logger.info("\nClassification Report:")
        logger.info(f"\n{report}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Visualizar y guardar matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - LSTM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_path / "lstm_confusion_matrix.png")
        plt.close()
        
        # Guardar modelo y tokenizer
        self.model.save(output_path / "lstm_model.keras")
        joblib.dump(self.tokenizer, output_path / "lstm_tokenizer.joblib")
        
        return report, cm

def main():
    """Función principal para ejecutar el entrenamiento del modelo LSTM."""
    # Configuración
    data_file = "data/processed/reviews_preprocessed.csv"
    output_dir = "models/deep_learning"
    
    logger.info("Iniciando entrenamiento del modelo LSTM...")
    
    try:
        # Inicializar clasificador
        classifier = LSTMSentimentClassifier()
        
        # Preparar datos
        X_train, X_test, y_train, y_test, class_weights = classifier.prepare_data(data_file)
        
        # Construir modelo
        classifier.build_model()
        
        # Entrenar modelo
        history = classifier.train(X_train, y_train, X_test, y_test, class_weights)
        
        # Evaluar y guardar resultados
        report, cm = classifier.evaluate(X_test, y_test, output_dir)
        
        logger.info("\nEntrenamiento del modelo LSTM completado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 