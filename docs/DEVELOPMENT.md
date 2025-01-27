# Guía de Desarrollo

Esta guía proporciona información detallada sobre cómo configurar el entorno de desarrollo y contribuir al proyecto.

## Configuración de Entorno

### Requisitos Previos

- Python 3.8+
- CUDA compatible GPU (recomendado)
- Git
- Visual Studio Code (recomendado)

### Configuración del Entorno Virtual

1. Crear entorno virtual:
```bash
python -m venv venv
```

2. Activar entorno:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Configuración de Git

1. Configurar usuario:
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

2. Configurar pre-commit hooks:
```bash
pre-commit install
```

## Estructura del Código

### Organización de Directorios

```
src/
├── data/                  # Procesamiento de datos
│   ├── preprocessing/     # Limpieza y transformación
│   └── loading/          # Carga de datos
├── models/               # Implementación de modelos
│   ├── traditional/      # Modelos clásicos
│   └── deep_learning/    # Modelos de deep learning
└── visualization/        # Dashboard y visualizaciones
    └── dashboard/        # Aplicación Streamlit
```

### Convenciones de Código

- Seguir PEP 8
- Docstrings en formato Google
- Type hints para funciones principales
- Tests unitarios para funciones críticas

## Flujo de Desarrollo

### 1. Preparación de Datos

```python
# src/data/preprocessing/text_preprocessing.py
class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza texto.
        
        Args:
            text: Texto a limpiar.
            
        Returns:
            Texto limpio y normalizado.
        """
        pass
```

### 2. Entrenamiento de Modelos

```python
# src/models/deep_learning/trainer.py
class ModelTrainer:
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Entrena el modelo.
        
        Args:
            model: Modelo a entrenar.
            train_loader: Datos de entrenamiento.
            val_loader: Datos de validación.
            
        Returns:
            Métricas de entrenamiento.
        """
        pass
```

### 3. Evaluación

```python
# src/models/evaluation.py
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader
) -> Dict[str, float]:
    """Evalúa el modelo en conjunto de prueba.
    
    Args:
        model: Modelo a evaluar.
        test_loader: Datos de prueba.
        
    Returns:
        Métricas de evaluación.
    """
    pass
```

## Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/models/test_lstm.py
```

### Escribir Tests

```python
# tests/models/test_lstm.py
def test_lstm_forward():
    model = LSTMModel()
    x = torch.randn(32, 100)
    output = model(x)
    assert output.shape == (32, 3)
```

## Logging y Monitoreo

### MLflow

```python
with mlflow.start_run():
    mlflow.log_param("learning_rate", lr)
    mlflow.log_metric("accuracy", acc)
    mlflow.pytorch.log_model(model, "model")
```

### Weights & Biases

```python
wandb.init(project="amazon-reviews")
wandb.watch(model)
wandb.log({"loss": loss, "accuracy": acc})
```

## Dashboard

### Desarrollo Local

1. Iniciar servidor:
```bash
streamlit run src/visualization/dashboard/app.py
```

2. Hot reload activado por defecto

### Deployment

1. Preparar requirements:
```bash
pip freeze > requirements.txt
```

2. Configurar Streamlit:
```toml
# .streamlit/config.toml
[server]
port = 8501
```

## CI/CD

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest
```

## Mejores Prácticas

1. **Código**
   - Usar type hints
   - Documentar funciones principales
   - Seguir principios SOLID

2. **Git**
   - Commits atómicos
   - Mensajes descriptivos
   - Pull requests pequeños

3. **Testing**
   - Tests unitarios
   - Tests de integración
   - Coverage > 80%

4. **Documentación**
   - Mantener README actualizado
   - Documentar cambios importantes
   - Incluir ejemplos de uso

## Troubleshooting

### Problemas Comunes

1. **CUDA Out of Memory**
   - Reducir batch size
   - Usar gradient accumulation
   - Implementar mixed precision

2. **Dependencias**
   - Usar virtual env
   - Especificar versiones exactas
   - Documentar conflictos conocidos

3. **Performance**
   - Profiling con cProfile
   - Optimizar data loading
   - Usar caching cuando sea posible

## Recursos

- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Hugging Face](https://huggingface.co/docs)
- [Streamlit](https://docs.streamlit.io)
- [MLflow](https://www.mlflow.org/docs/latest/index.html)