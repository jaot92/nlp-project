import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.model_evaluation import plot_confusion_matrix, plot_class_distribution, plot_model_comparison

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    """
    Clase para entrenar y evaluar modelos de clasificación de sentimiento.
    """
    
    def __init__(self):
        """Inicializa el clasificador con diferentes modelos."""
        self.models = {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', MultinomialNB())
            ]),
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', LogisticRegression(max_iter=1000))
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', LinearSVC(max_iter=1000))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', RandomForestClassifier(n_estimators=100))
            ])
        }
        
        self.trained_models = {}
        self.results = {}
    
    def prepare_data(self, data_file: str):
        """
        Prepara los datos para el entrenamiento.
        
        Args:
            data_file (str): Ruta al archivo de datos preprocesados
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
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
        
        # Usar el texto preprocesado
        X = df['reviews.text_processed']
        y = df['sentiment']
        
        # Verificar balance de clases
        class_distribution = y.value_counts()
        logger.info(f"\nDistribución de clases:\n{class_distribution}")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Datos divididos - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, output_dir: str):
        """
        Entrena y evalúa todos los modelos.
        
        Args:
            X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
            output_dir (str): Directorio para guardar resultados
        """
        results = {}
        
        # Crear directorio si no existe
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            logger.info(f"\nEntrenando modelo: {name}")
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            
            # Convertir a numpy arrays si son pandas Series
            y_true_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
            y_pred_np = np.array(y_pred)
            
            # Verificar que los arrays tienen la forma correcta
            logger.info(f"Forma de y_true: {y_true_np.shape}")
            logger.info(f"Forma de y_pred: {y_pred_np.shape}")

            # Guardar resultados
            results[name] = {
                'y_true': y_true_np,  # Añadir y_true
                'y_pred': y_pred_np,  # Añadir y_pred
                'classification_report': classification_report(y_true_np, y_pred_np),
                'confusion_matrix': confusion_matrix(y_true_np, y_pred_np)
            }
            
            # Verificar que los resultados se guardaron correctamente
            logger.info(f"Claves en results[{name}]: {list(results[name].keys())}")

            # Guardar modelo
            model_file = output_path / f"{name}_model.joblib"
            joblib.dump(model, model_file)
            
            # Imprimir resultados
            logger.info(f"\nResultados para {name}:")
            logger.info("\nClassification Report:")
            logger.info(f"\n{results[name]['classification_report']}")
            
            # Visualizar matriz de confusión
            plot_confusion_matrix(
                results[name]['confusion_matrix'],
                labels=sorted(set(y_test)),
                title=f'Matriz de Confusión - {name}'
            )
            
            # Realizar validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            logger.info(f"\nValidación cruzada (5-fold):")
            logger.info(f"Media: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.results = results
        return results
    
    def _plot_confusion_matrix(self, cm, model_name: str, output_file: str):
        """
        Visualiza y guarda la matriz de confusión.
        
        Args:
            cm: Matriz de confusión
            model_name (str): Nombre del modelo
            output_file (str): Ruta para guardar la visualización
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_file)
        plt.close()

def main():
    """Función principal para ejecutar el entrenamiento de modelos."""
    # Configuración
    data_file = "data/processed/reviews_preprocessed.csv"
    output_dir = "models/traditional"
    
    logger.info("Iniciando entrenamiento de modelos tradicionales...")
    
    try:
        # Inicializar clasificador
        classifier = SentimentClassifier()
        
        # Preparar datos
        X_train, X_test, y_train, y_test = classifier.prepare_data(data_file)
        
        # Entrenar y evaluar modelos
        results = classifier.train_and_evaluate(
            X_train, X_test, y_train, y_test, output_dir
        )
        
        logger.info("\nEntrenamiento completado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 