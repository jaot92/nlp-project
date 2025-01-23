import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import textwrap

@st.cache_resource
def load_summarization_model():
    """Cargar el modelo T5 para generación de resúmenes"""
    try:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        return tokenizer, model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None

def generate_summary_for_reviews(reviews, tokenizer, model, max_length=150):
    """Genera un resumen de las reviews usando T5"""
    try:
        # Concatenar reviews con un límite de tokens
        combined_reviews = " ".join(reviews.tolist())
        combined_reviews = textwrap.shorten(combined_reviews, width=1000)
        
        # Preparar input
        input_text = "summarize: " + combined_reviews
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generar resumen
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    except Exception as e:
        st.error(f"Error en la generación del resumen: {str(e)}")
        return "Error al generar el resumen."

def process_categories(df):
    # Consolidar categorías
    df['categories'] = df['categories'].fillna('')
    df['categories_list'] = df['categories'].str.split(',')
    
    # Obtener categorías principales
    main_categories = []
    for cats in df['categories_list']:
        if isinstance(cats, list) and len(cats) > 0:
            main_categories.append(cats[0].strip())
        else:
            main_categories.append('Others')
    
    df['main_category'] = main_categories
    return df

def process_ratings(df):
    # Convertir ratings a grupos
    df['rating_group'] = df['reviews.rating'].apply(
        lambda x: 'negative' if x <= 2 else 
                 'neutral' if x == 3 else 
                 'positive'
    )
    return df

def create_visualizations(df):
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de ratings
        fig_ratings = px.histogram(
            df, 
            x='reviews.rating',
            title='Distribución de Ratings',
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig_ratings)
        
        # Distribución por categoría
        category_counts = df['main_category'].value_counts()
        fig_categories = px.bar(
            category_counts,
            title='Reviews por Categoría',
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig_categories)
    
    with col2:
        # Sentimiento por categoría
        sentiment_by_category = pd.crosstab(
            df['main_category'], 
            df['rating_group']
        )
        fig_sentiment = px.bar(
            sentiment_by_category,
            title='Sentimiento por Categoría',
            barmode='group'
        )
        st.plotly_chart(fig_sentiment)
        
        # Evolución temporal
        try:
            # Convertir fechas usando formato ISO8601
            df['reviews.date'] = pd.to_datetime(df['reviews.date'], format='ISO8601')
            
            # Agrupar por mes
            monthly_ratings = df.groupby(
                df['reviews.date'].dt.to_period('M')
            )['reviews.rating'].mean()
            
            # Convertir el índice a datetime para plotly
            monthly_ratings.index = monthly_ratings.index.astype(str)
            
            fig_temporal = px.line(
                x=monthly_ratings.index,
                y=monthly_ratings.values,
                title='Evolución de Ratings',
                labels={'x': 'Fecha', 'y': 'Rating Promedio'}
            )
            st.plotly_chart(fig_temporal)
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico temporal: {str(e)}")

def generate_summaries_section(df):
    st.header("Generador de Resúmenes")
    
    # Cargar modelo de resumen
    tokenizer, model = load_summarization_model()
    
    if tokenizer is None or model is None:
        st.error("No se pudo cargar el modelo de resúmenes.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox(
            "Seleccionar Categoría",
            options=sorted(df['main_category'].unique())
        )
    
    with col2:
        selected_rating = st.selectbox(
            "Seleccionar Rating",
            options=['positive', 'neutral', 'negative']
        )
    
    if st.button("Generar Resumen"):
        # Filtrar reviews
        filtered_reviews = df[
            (df['main_category'] == selected_category) &
            (df['rating_group'] == selected_rating)
        ]['reviews.text'].head(50)
        
        if len(filtered_reviews) > 0:
            with st.spinner("Generando resumen..."):
                # Mostrar cantidad de reviews analizadas
                st.info(f"Analizando {len(filtered_reviews)} reviews...")
                
                # Generar y mostrar resumen
                summary = generate_summary_for_reviews(filtered_reviews, tokenizer, model)
                
                # Mostrar estadísticas
                st.success("¡Resumen generado!")
                st.write("### Resumen de las Reviews:")
                st.write(summary)
                
                # Mostrar estadísticas adicionales
                st.write("### Estadísticas:")
                avg_rating = df[
                    (df['main_category'] == selected_category) &
                    (df['rating_group'] == selected_rating)
                ]['reviews.rating'].mean()
                
                st.metric(
                    "Rating Promedio",
                    f"{avg_rating:.2f}/5.0"
                )
        else:
            st.warning("No hay suficientes reviews para esta combinación de categoría y rating.")


def main():
    st.set_page_config(
        page_title="Amazon Reviews Analysis",
        layout="wide"
    )
    
    st.title("Amazon Reviews Analysis Dashboard")
    
    # Suprimir warnings específicos
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    uploaded_file = st.sidebar.file_uploader(
        "Cargar archivo CSV de reviews",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Cargar datos con manejo de tipos mixtos
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        # Procesar datos
        df = process_categories(df)
        df = process_ratings(df)
        
        # Mostrar visualizaciones
        create_visualizations(df)
        
        # Sección de resúmenes
        generate_summaries_section(df)
    else:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis.")

if __name__ == "__main__":
    main()