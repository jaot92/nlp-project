# Guía de Desarrollo

## Estructura del Proyecto
```
project/
├── data/
│   ├── raw/          # Datos originales sin procesar
│   └── processed/    # Datos procesados para entrenamiento
├── notebooks/
│   ├── 1_eda.ipynb                # Análisis exploratorio de datos
│   ├── 2_traditional_models.ipynb  # Modelos tradicionales de ML
│   ├── 3_lstm_model.ipynb         # Modelo LSTM
│   └── 4_transformer_model.ipynb   # Modelo Transformer
├── src/
│   ├── data/
│   │   └── preprocessing/     # Scripts de preprocesamiento
│   ├── models/
│   │   ├── traditional/      # Implementación de modelos tradicionales
│   │   └── deep_learning/    # Implementación de modelos deep learning
│   ├── visualization/
│   │   └── dashboard/        # Código del dashboard
│   └── web/                  # Aplicación web
├── tests/                    # Tests unitarios
└── docs/                     # Documentación
```

## Configuración del Entorno

### Usando Conda (Recomendado)
```bash
# Crear el entorno
conda env create -f environment.yml

# Activar el entorno
conda activate nlp-reviews
```

### Dataset
El proyecto utiliza el dataset de Amazon Customer Reviews de Kaggle. Para obtener los datos:

1. Descargar de: [Amazon Customer Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data)
2. Colocar los archivos CSV en la carpeta `data/raw/`

## Organización del Trabajo

### Equipos y Responsabilidades

#### Equipo 1: EDA y Preprocesamiento
- Análisis exploratorio completo
- Limpieza y preparación de datos
- Generación de features
- Archivos principales:
  - `notebooks/1_eda.ipynb`
  - `src/data/preprocessing/`

#### Equipo 2: Modelos Tradicionales
- Implementación de modelos base
- Optimización y evaluación
- Archivos principales:
  - `notebooks/2_traditional_models.ipynb`
  - `src/models/traditional/`

#### Equipo 3: Deep Learning
- Implementación de LSTM y Transformers
- Fine-tuning y evaluación
- Archivos principales:
  - `notebooks/3_lstm_model.ipynb`
  - `notebooks/4_transformer_model.ipynb`
  - `src/models/deep_learning/`

#### Equipo 4: Visualización y API
- Dashboard interactivo
- API REST
- Documentación
- Archivos principales:
  - `src/visualization/`
  - `src/web/`

## Flujo de Trabajo Git

### Ramas Principales
- `main`: Código estable y producción
- `develop`: Desarrollo activo e integración

### Ramas de Características
- `feature/eda`: Análisis exploratorio
- `feature/preprocessing`: Preprocesamiento
- `feature/traditional-models`: Modelos tradicionales
- `feature/deep-learning`: Modelos deep learning
- `feature/dashboard`: Visualización
- `feature/api`: API REST

### Convenciones de Commits
- feat: Nueva característica
- fix: Corrección de bug
- docs: Documentación
- style: Formato
- refactor: Refactorización
- test: Tests
- chore: Mantenimiento

Ejemplo: `feat: Implementar limpieza de texto`

## Estándares de Código
- Usar Python 3.9+
- Seguir PEP 8
- Documentar funciones y clases
- Incluir docstrings
- Nombres descriptivos en inglés
- Tests unitarios para funciones críticas

## Entregables por Fase

### Fase 1: EDA y Preprocesamiento
- [ ] Análisis exploratorio completo
- [ ] Pipeline de preprocesamiento
- [ ] Documentación de hallazgos

### Fase 2: Modelos Base
- [ ] Implementación de modelos tradicionales
- [ ] Evaluación y métricas
- [ ] Selección de mejor modelo

### Fase 3: Deep Learning
- [ ] Implementación de LSTM
- [ ] Implementación de Transformers
- [ ] Comparación de resultados

### Fase 4: Visualización y API
- [ ] Dashboard interactivo
- [ ] API REST funcional
- [ ] Documentación técnica 