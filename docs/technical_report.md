# Análisis de Sentimientos en Reseñas de Amazon: Un Enfoque de Aprendizaje Profundo
## Reporte Técnico

## Tabla de Contenidos
1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Análisis Exploratorio de Datos](#2-análisis-exploratorio-de-datos)
3. [Modelos Tradicionales](#3-modelos-tradicionales)
4. [Modelo LSTM](#4-modelo-lstm)
5. [Modelo Transformer](#5-modelo-transformer)
6. [Conclusiones y Recomendaciones](#6-conclusiones-y-recomendaciones)

## 1. Resumen Ejecutivo

Este proyecto aborda el desafío de clasificar automáticamente las reseñas de productos de Amazon en tres categorías de sentimiento: positivo, neutral y negativo. Se implementaron y evaluaron tres enfoques diferentes:
- Modelos tradicionales de aprendizaje automático
- Redes neuronales recurrentes (LSTM)
- Modelos basados en transformers (BERT)

### Dataset
- **Tamaño total**: 67,992 reseñas
- **Distribución**: 91.99% positivas, 4.27% neutrales, 3.74% negativas
- **Características**: 14 columnas incluyendo texto, títulos, calificaciones y metadatos

[ESPACIO PARA GRÁFICO DE DISTRIBUCIÓN DE CLASES]

## 2. Análisis Exploratorio de Datos

### 2.1 Características del Dataset

#### Estadísticas Textuales
- **Longitud de Reseñas por Sentimiento**:
  - Positivas: Media de 145.04 palabras
  - Neutrales: Media de 185.56 palabras
  - Negativas: Media de 23.50 palabras

[ESPACIO PARA GRÁFICO DE DISTRIBUCIÓN DE LONGITUD DE RESEÑAS]

#### Complejidad Textual
- Las reseñas negativas tienden a ser más largas y usar vocabulario más complejo
- Mayor número de oraciones en reseñas negativas
- Palabras por oración más consistente en reseñas positivas

[ESPACIO PARA GRÁFICO DE COMPLEJIDAD TEXTUAL]

### 2.2 Análisis de Utilidad
- Reseñas negativas: Media de 2.15 votos útiles
- Reseñas neutrales: Media de 0.69 votos útiles
- Reseñas positivas: Media de 0.39 votos útiles

[ESPACIO PARA GRÁFICO DE VOTOS ÚTILES POR SENTIMIENTO]

## 3. Modelos Tradicionales

### 3.1 Metodología
- Preprocesamiento:
  - Limpieza y normalización de texto
  - Vectorización TF-IDF
  - Balanceo de clases (SMOTE + RandomUnderSampler)

### 3.2 Resultados Comparativos

| Modelo | Accuracy | Macro F1-score | Cross-validation |
|--------|----------|----------------|------------------|
| Naive Bayes | 0.79 | 0.52 | 0.797 (±0.010) |
| Regresión Logística | 0.84 | 0.57 | 0.837 (±0.008) |
| SVM | 0.85 | 0.58 | 0.851 (±0.012) |
| Random Forest | 0.95 | 0.75 | 0.947 (±0.001) |

[ESPACIO PARA MATRIZ DE CONFUSIÓN DEL MEJOR MODELO]

## 4. Modelo LSTM

### 4.1 Arquitectura
- Bidirectional LSTM con 3 capas (256, 128, 64 unidades)
- Dropout (0.2) para regularización
- Batch Normalization
- Dense layers con regularización L1/L2
- Total parámetros: 2,619,715

### 4.2 Resultados
- Accuracy de entrenamiento: 0.8923
- Accuracy de validación: 0.8745
- Loss de entrenamiento: 0.2834
- Loss de validación: 0.3156

#### Métricas por Clase
| Clase | Precision | Recall | F1-score |
|-------|-----------|---------|-----------|
| Negativo | 0.87 | 0.85 | 0.86 |
| Neutral | 0.79 | 0.76 | 0.77 |
| Positivo | 0.91 | 0.93 | 0.92 |

[ESPACIO PARA GRÁFICO DE CURVAS DE ENTRENAMIENTO]

## 5. Modelo Transformer

### 5.1 Configuración
- Modelo base: BERT (bert-base-uncased)
- Tokenización máxima: 128 tokens
- Batch Size: 32
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Épocas: 4

### 5.2 Resultados
- Accuracy: 0.960290
- F1-Score: 0.959570
- Precision: 0.958969
- Recall: 0.960290

#### Progresión del Entrenamiento
| Época | F1-Score |
|-------|-----------|
| 1 | 0.904339 |
| 2 | 0.918984 |
| 3 | 0.944916 |
| 4 | 0.952680 |

[ESPACIO PARA GRÁFICO DE PROGRESIÓN DEL ENTRENAMIENTO]

## 6. Conclusiones y Recomendaciones

### 6.1 Comparación de Modelos
1. **Transformer (BERT)**
   - Mejor rendimiento general (96% accuracy)
   - Mayor capacidad de generalización
   - Requiere más recursos computacionales

2. **LSTM**
   - Buen balance rendimiento/recursos (87% accuracy)
   - Efectivo en capturar dependencias temporales
   - Menor tiempo de entrenamiento que BERT

3. **Random Forest**
   - Mejor modelo tradicional (95% accuracy)
   - Fácil de implementar y mantener
   - Menor costo computacional

### 6.2 Recomendaciones
1. Implementar el modelo BERT en producción para casos que requieran máxima precisión
2. Utilizar LSTM como alternativa cuando los recursos computacionales sean limitados
3. Mantener Random Forest como baseline y para casos de baja latencia

### 6.3 Trabajo Futuro
1. Explorar técnicas de data augmentation para clases minoritarias
2. Implementar sistema de umbral de confianza para casos dudosos
3. Desarrollar pipeline de actualización continua del modelo 