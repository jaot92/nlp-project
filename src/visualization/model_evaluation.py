import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

def plot_confusion_matrix(cm, labels, title="Matriz de Confusión"):
    """Visualiza la matriz de confusión."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, title="Distribución de Clases"):
    """Visualiza la distribución de clases."""
    plt.figure(figsize=(10, 6))
    sns.countplot(y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results):
    """Visualiza la comparación de modelos."""
    if not results:
        raise ValueError("El diccionario de resultados está vacío")
    
    metrics = []
    for name, result in results.items():
        try:
            # Extraer métricas del classification_report (que es un string)
            report_lines = result['classification_report'].split('\n')
            
            # Buscar la línea de accuracy
            for line in report_lines:
                if 'accuracy' in line:
                    accuracy = float(line.strip().split()[-1])
                    break
            
            # Buscar la línea de weighted avg
            for line in report_lines:
                if 'weighted avg' in line:
                    weighted_f1 = float(line.strip().split()[-2])
                    break
            
            metrics.append({
                'model': name,
                'accuracy': accuracy,
                'weighted_f1': weighted_f1
            })
        except Exception as e:
            print(f"Error procesando resultados para {name}: {str(e)}")
            continue
    
    if not metrics:
        raise ValueError("No se pudieron procesar métricas para ningún modelo")
    
    df_metrics = pd.DataFrame(metrics)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_metrics, x='model', y='accuracy')
    plt.title('Comparación de Accuracy entre Modelos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()