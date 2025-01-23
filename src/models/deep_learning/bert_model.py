import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTSentimentClassifier:
    def __init__(self, max_length=128, learning_rate=2e-5, warmup_steps=0, dropout_rate=0.2):
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.dropout_rate = dropout_rate
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        
        # Configuración de dispositivo
        self.strategy = self._setup_strategy()
        
    def _setup_strategy(self):
        """
        Configura la estrategia de entrenamiento basada en el hardware disponible.
        """
        try:
            # Intentar usar TPU si está disponible
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            logger.info("Entrenando en TPU")
        except:
            # Si no hay TPU, intentar usar GPU
            if len(tf.config.list_physical_devices('GPU')) > 0:
                strategy = tf.distribute.MirroredStrategy()
                logger.info(f"Entrenando en {strategy.num_replicas_in_sync} GPU(s)")
            else:
                strategy = tf.distribute.get_strategy()
                logger.info("Entrenando en CPU")
        
        return strategy
        
    def _validate_input_data(self, df):
        """
        Valida los datos de entrada.
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío")
            
        required_columns = ['reviews.text_processed', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes: {missing_columns}")
            
        # Verificar valores nulos
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Valores nulos encontrados:\n{null_counts}")
            
        # Verificar longitud de textos
        text_lengths = df['reviews.text_processed'].str.len()
        logger.info(f"Estadísticas de longitud de texto:\n{text_lengths.describe()}")
        
        return df
        
    def prepare_data(self, data_file: str):
        """
        Prepara los datos para el entrenamiento con BERT.
        """
        logger.info("Cargando y preparando datos...")
        
        # Cargar y validar datos
        df = pd.read_csv(data_file)
        df = self._validate_input_data(df)
        
        # Limpiar valores nulos
        df['reviews.text_processed'] = df['reviews.text_processed'].fillna('')
        df = df.dropna(subset=['sentiment'])
        
        # Tokenizar textos con manejo de errores
        try:
            encodings = self.tokenizer(
                df['reviews.text_processed'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='tf'
            )
        except Exception as e:
            logger.error(f"Error en la tokenización: {str(e)}")
            raise
        
        # Preparar etiquetas
        y = pd.get_dummies(df['sentiment']).values
        
        # Calcular class weights
        class_counts = df['sentiment'].value_counts()
        total = len(df)
        class_weights = {
            i: total / (len(class_counts) * count)
            for i, count in enumerate(class_counts.sort_index())
        }
        
        logger.info(f"Pesos por clase:\n{class_weights}")
        
        # División train/test
        train_idx, test_idx = train_test_split(
            range(len(df)), test_size=0.2, random_state=42,
            stratify=df['sentiment']
        )
        
        # Preparar datos
        X_train = {
            'input_ids': encodings['input_ids'][train_idx],
            'attention_mask': encodings['attention_mask'][train_idx]
        }
        X_test = {
            'input_ids': encodings['input_ids'][test_idx],
            'attention_mask': encodings['attention_mask'][test_idx]
        }
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        logger.info(f"Datos divididos - Train: {len(train_idx)}, Test: {len(test_idx)}")
        return X_train, X_test, y_train, y_test, class_weights
    
    def build_model(self):
        """
        Construye el modelo BERT con capas adicionales para clasificación.
        """
        with self.strategy.scope():
            # Cargar modelo base BERT
            bert = TFBertModel.from_pretrained('bert-base-uncased')
            
            # Inputs
            input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
            attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
            
            # BERT layer
            bert_outputs = bert(input_ids, attention_mask=attention_mask)[0]
            
            # Usar el token [CLS] para clasificación
            cls_output = bert_outputs[:, 0, :]
            
            # Capas adicionales con dropout configurable
            x = tf.keras.layers.Dense(256, activation='relu')(cls_output)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            
            # Construir modelo
            self.model = tf.keras.Model(
                inputs=[input_ids, attention_mask],
                outputs=outputs
            )
            
            # Learning rate schedule con warmup
            total_steps = 1000  # Ajustar según el tamaño del dataset
            warmup_steps = self.warmup_steps
            
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=total_steps - warmup_steps,
                end_learning_rate=self.learning_rate * 0.1
            )
            
            if warmup_steps:
                lr_schedule = WarmUp(
                    initial_learning_rate=self.learning_rate,
                    decay_schedule_fn=lr_schedule,
                    warmup_steps=warmup_steps
                )
            
            # Compilar modelo con gradient clipping
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=1.0
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            logger.info("Resumen del modelo:")
            logger.info(self.model.summary())
    
    def train(self, X_train, y_train, X_test, y_test, class_weights, epochs=5, batch_size=32):
        """
        Entrena el modelo BERT con validación cruzada.
        """
        logger.info("Iniciando entrenamiento del modelo BERT...")
        
        # Configurar callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=1,
                min_lr=1e-7
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Entrenamiento principal
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Validación cruzada
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train['input_ids'])):
            logger.info(f"Entrenando fold {fold + 1}/5")
            
            # Preparar datos para este fold
            X_train_fold = {
                'input_ids': X_train['input_ids'][train_idx],
                'attention_mask': X_train['attention_mask'][train_idx]
            }
            X_val_fold = {
                'input_ids': X_train['input_ids'][val_idx],
                'attention_mask': X_train['attention_mask'][val_idx]
            }
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            # Entrenar en este fold
            self.build_model()  # Reiniciar modelo para cada fold
            history_fold = self.model.fit(
                X_train_fold,
                y_train_fold,
                epochs=2,  # Menos épocas para CV
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                class_weight=class_weights,
                verbose=0
            )
            
            # Guardar score
            cv_scores.append(history_fold.history['val_accuracy'][-1])
        
        logger.info(f"Scores de validación cruzada: {cv_scores}")
        logger.info(f"Media CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return history
    
    def evaluate(self, X_test, y_test, output_dir: str):
        """
        Evalúa el modelo y guarda los resultados.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Reporte de clasificación
        report = classification_report(y_test_classes, y_pred_classes)
        logger.info("\nClassification Report:")
        logger.info(f"\n{report}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Visualizaciones
        self._plot_confusion_matrix(cm, output_path)
        self._plot_roc_curves(y_test, y_pred, output_path)
        self._plot_precision_recall_curves(y_test, y_pred, output_path)
        
        # Guardar modelo y tokenizer
        self.model.save_pretrained(output_path / "bert_model")
        self.tokenizer.save_pretrained(output_path / "bert_tokenizer")
        
        # Guardar métricas en formato JSON
        metrics = {
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return report, cm
    
    def _plot_confusion_matrix(self, cm, output_path):
        """
        Visualiza y guarda la matriz de confusión.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - BERT')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_path / "bert_confusion_matrix.png")
        plt.close()
    
    def _plot_roc_curves(self, y_test, y_pred, output_path):
        """
        Genera y guarda las curvas ROC.
        """
        plt.figure(figsize=(10, 8))
        
        for i in range(y_test.shape[1]):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Clase {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curvas ROC por Clase')
        plt.legend()
        plt.savefig(output_path / "bert_roc_curves.png")
        plt.close()
    
    def _plot_precision_recall_curves(self, y_test, y_pred, output_path):
        """
        Genera y guarda las curvas de Precision-Recall.
        """
        plt.figure(figsize=(10, 8))
        
        for i in range(y_test.shape[1]):
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
            plt.plot(recall, precision, label=f'Clase {i}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall por Clase')
        plt.legend()
        plt.savefig(output_path / "bert_precision_recall_curves.png")
        plt.close()

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implementación de warmup para learning rate.
    """
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        with tf.name_scope("learning_rate"):
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done
            
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps)
            )
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
        }

def main():
    """Función principal para ejecutar el entrenamiento del modelo BERT."""
    data_file = "data/processed/reviews_preprocessed.csv"
    output_dir = "models/deep_learning"
    
    logger.info("Iniciando entrenamiento del modelo BERT...")
    
    try:
        # Inicializar clasificador
        classifier = BERTSentimentClassifier()
        
        # Preparar datos
        X_train, X_test, y_train, y_test, class_weights = classifier.prepare_data(data_file)
        
        # Construir modelo
        classifier.build_model()
        
        # Entrenar modelo
        history = classifier.train(X_train, y_train, X_test, y_test, class_weights)
        
        # Evaluar y guardar resultados
        report, cm = classifier.evaluate(X_test, y_test, output_dir)
        
        logger.info("Entrenamiento del modelo BERT completado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 