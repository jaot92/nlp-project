import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_chunk(chunk):
    """
    Procesa un chunk de datos seleccionando y limpiando las columnas relevantes.
    """
    # Columnas base que deberían estar en todos los archivos
    base_columns = [
        'id',                  # ID del producto
        'name',                # Nombre del producto
        'brand',               # Marca
        'categories',          # Categorías
        'reviews.id',          # ID de la reseña
        'reviews.date',        # Fecha de la reseña
        'reviews.doRecommend', # Recomendación
        'reviews.numHelpful',  # Votos útiles
        'reviews.rating',      # Calificación
        'reviews.text',        # Texto de la reseña
        'reviews.title',       # Título de la reseña
        'reviews.username'     # Usuario
    ]
    
    # Verificar qué columnas están disponibles
    available_columns = [col for col in base_columns if col in chunk.columns]
    print(f"Columnas disponibles: {available_columns}")
    
    # Seleccionar columnas disponibles
    df = chunk[available_columns].copy()
    
    # Limpieza básica
    if 'reviews.text' in df.columns:
        df['reviews.text'] = df['reviews.text'].fillna('')
    if 'reviews.title' in df.columns:
        df['reviews.title'] = df['reviews.title'].fillna('')
    if 'reviews.rating' in df.columns:
        df['reviews.rating'] = df['reviews.rating'].fillna(0)
    if 'reviews.numHelpful' in df.columns:
        df['reviews.numHelpful'] = df['reviews.numHelpful'].fillna(0)
    
    # Convertir fechas si está disponible
    if 'reviews.date' in df.columns:
        df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
    
    # Agregar columnas derivadas
    if 'reviews.text' in df.columns:
        df['text_length'] = df['reviews.text'].str.len()
    if 'reviews.rating' in df.columns:
        df['sentiment'] = df['reviews.rating'].apply(
            lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral'
        )
    
    return df

def consolidate_datasets(input_files, output_file, chunksize=10000):
    """
    Consolida múltiples archivos CSV en uno solo, procesando por chunks.
    
    Args:
        input_files (list): Lista de rutas a los archivos CSV de entrada
        output_file (str): Ruta donde se guardará el archivo consolidado
        chunksize (int): Tamaño del chunk para procesamiento
    """
    # Crear directorio si no existe
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada archivo
    first_chunk = True
    for file in input_files:
        print(f"\nProcesando archivo: {file}")
        
        try:
            # Leer y procesar en chunks
            for i, chunk in enumerate(pd.read_csv(file, chunksize=chunksize)):
                processed_chunk = load_and_process_chunk(chunk)
                
                # Guardar
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                processed_chunk.to_csv(output_file, mode=mode, header=header, index=False)
                first_chunk = False
                
                # Mostrar progreso
                if (i + 1) % 10 == 0:
                    print(f"Procesados {(i + 1) * chunksize} registros...")
            
            print(f"Archivo {file} procesado exitosamente")
            
        except Exception as e:
            print(f"Error procesando archivo {file}: {str(e)}")
            continue

if __name__ == "__main__":
    # Definir archivos de entrada y salida
    input_files = [
        'data/raw/1429_1.csv',
        'data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv',
        'data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
    ]
    
    output_file = 'data/processed/consolidated_reviews.csv'
    
    print("Iniciando proceso de consolidación...")
    print(f"Archivos a procesar: {len(input_files)}")
    
    # Ejecutar consolidación
    consolidate_datasets(input_files, output_file)
    print(f"\nProceso completado. Datos consolidados guardados en: {output_file}") 