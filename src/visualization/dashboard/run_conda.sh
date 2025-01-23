#!/bin/bash

# Activar el ambiente Conda
eval "$(conda shell.bash hook)"
conda activate nlp-reviews

# Ejecutar Streamlit
streamlit run app.py --server.port 8501