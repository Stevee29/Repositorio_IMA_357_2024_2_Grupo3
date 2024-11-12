import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Función para cargar el archivo CSV desde GitHub
@st.cache
def load_data():
    url = "https://github.com/Stevee29/Repositorio_IMA_357_2024_2_Grupo3/blob/main/cuerpo_documentos_p2_gr_3.csv"
    response = requests.get(url)
    data = StringIO(response.text)
    try:
        df = pd.read_csv(data)
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return None
    except Exception as e:
        st.error(f"Un error inesperado ocurrió al leer el archivo CSV: {e}")
        return None

# Función para encontrar el documento con mayor frecuencia de una palabra
def find_max_frequency_document(df, word):
    df['word_count'] = df['Cuerpo'].apply(lambda x: str(x).lower().split().count(word.lower()))
    max_freq_doc = df.loc[df['word_count'].idxmax()]
    return max_freq_doc['Titular'], max_freq_doc['word_count']

# Función para encontrar el documento más similar usando similitud coseno
def find_most_similar_document(df, sentence):
    vectorizer = TfidfVectorizer().fit_transform(df['Cuerpo'].values)
    vectors = vectorizer.transform([sentence])
    cosine_similarities = cosine_similarity(vectors, vectorizer).flatten()
    most_similar_doc_idx = cosine_similarities.argmax()
    return df.loc[most_similar_doc_idx]['Titular'], cosine_similarities[most_similar_doc_idx]

# Función para encontrar el documento con mayor suma de frecuencias de tokens de la oración
def find_max_sum_frequency_document(df, sentence):
    tokens = sentence.lower().split()
    df['sum_token_count'] = df['Cuerpo'].apply(lambda x: sum([str(x).lower().split().count(token) for token in tokens]))
    max_sum_freq_doc = df.loc[df['sum_token_count'].idxmax()]
    return max_sum_freq_doc['Titular'], max_sum_freq_doc['sum_token_count']

# Cargar los datos
df = load_data()

if df is not None:
    # Verificar los datos cargados
    st.dataframe(df)
    st.write(df.columns)  # Para verificar los nombres de las columnas

    # Convertir la columna 'Cuerpo' a str y llenar valores nulos
    df['Cuerpo'] = df['Cuerpo'].astype(str)
    df['Cuerpo'].fillna('', inplace=True)

    # Inputs de texto
    word_input = st.text_input('Input de palabra')
    sentence_input = st.text_area('Input de oración')

    # Procesamiento y visualización
    if word_input:
        title, word_count = find_max_frequency_document(df, word_input)
        st.subheader('Documento con mayor frecuencia de la palabra ingresada:')
        st.write(f"Título: {title}")
        st.write(f"Frecuencia de la palabra '{word_input}': {word_count}")

    if sentence_input:
        most_similar_doc_title, cosine_sim = find_most_similar_document(df, sentence_input)
        max_sum_freq_doc_title, sum_freq_count = find_max_sum_frequency_document(df, sentence_input)
        st.subheader('Documento más similar a la oración ingresada (similitud coseno):')
        st.write(f"Título: {most_similar_doc_title}")
        st.write(f"Similitud coseno: {cosine_sim:.4f}")
        st.subheader('Documento con mayor suma de frecuencias de tokens de la oración ingresada:')
        st.write(f"Título: {max_sum_freq_doc_title}")
        st.write(f"Suma de frecuencias de tokens: {sum_freq_count}")
else:
    st.error("No se pudo cargar el archivo CSV.")
