import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# Configuración de la página
st.set_page_config(page_title="Simulador de precios Airbnb Santiago", layout="wide")

# Función ultra-robusta para encontrar archivos buscando alternativas de carpetas y mayúsculas
def buscar_archivo_insensible(directorio_base, partes_ruta):
    ruta_actual = directorio_base
    for parte in partes_ruta:
        if not os.path.exists(ruta_actual):
            return None
        # Listar contenido para buscar coincidencia insensible a mayúsculas/minúsculas
        coincidencias = [item for item in os.listdir(ruta_actual) if item.lower() == parte.lower()]
        if coincidencias:
            ruta_actual = os.path.join(ruta_actual, coincidencias[0])
        else:
            return None
    return ruta_actual

# Cargar el modelo y los datos de forma robusta adaptada a la nube y local
@st.cache_resource
def cargar_componentes():
    dir_actual = os.path.dirname(os.path.abspath(__file__))

    # Si app.py se ejecuta desde 'notebooks', subimos un nivel para ir a la raíz
    if os.path.basename(dir_actual) == 'notebooks':
        dir_raiz = os.path.dirname(dir_actual)
    else:
        dir_raiz = dir_actual

    # 1. Buscar dinámicamente el archivo de columnas_entrenamiento.pkl
    ruta_columnas = buscar_archivo_insensible(dir_raiz, ['models', 'columnas_entrenamiento.pkl'])
    if not ruta_columnas:
        # Intento alternativo en raíz si no está en models/
        ruta_columnas = buscar_archivo_insensible(dir_raiz, ['columnas_entrenamiento.pkl'])
    if not ruta_columnas:
        # Ruta por defecto por si todo falla
        ruta_columnas = os.path.join(dir_raiz, 'models', 'columnas_entrenamiento.pkl')

    # 2. Definir ruta para el archivo modelo_airbnb.pkl (insensible a mayúsculas para la carpeta models)
    dir_models_real = buscar_archivo_insensible(dir_raiz, ['models'])
    if not dir_models_real:
        dir_models_real = os.path.join(dir_raiz, 'models')
        os.makedirs(dir_models_real, exist_ok=True)

    archivo_modelo = os.path.join(dir_models_real, 'modelo_airbnb.pkl')

    # 3. Buscar el CSV usando la función de rastreo dinámico
    possible_dirs = [
        ['data', 'processed', 'airbnb_santiago_clean.csv'],
        ['data', 'airbnb_santiago_clean.csv'],
        ['airbnb_santiago_clean.csv']
    ]

    ruta_csv = None
    for p_dir in possible_dirs:
        tmp = buscar_archivo_insensible(dir_raiz, p_dir)
        if tmp and os.path.exists(tmp):
            ruta_csv = tmp
            break

    if not ruta_csv:
        ruta_csv = os.path.join(dir_raiz, 'data', 'processed', 'airbnb_santiago_clean.csv')

    # 4. Descargar modelo de Drive si no existe localmente
    if not os.path.exists(archivo_modelo):
        with st.spinner('Descargando modelo predictivo de IA... (Esto solo toma unos segundos la primera vez)'):
            id_modelo_drive = "1yCPNrclsoaT_1SjjjmnyLHduouK4Ehps" 
            url_modelo = f"https://drive.google.com/uc?id={id_modelo_drive}"
            gdown.download(url_modelo, archivo_modelo, quiet=True)

    # Cargar los componentes
    df = pd.read_csv(ruta_csv, sep=';')
    columnas_x = joblib.load(ruta_columnas)
    modelo = joblib.load(archivo_modelo)

    return modelo, columnas_x, df

modelo, columnas_x, df = cargar_componentes()

# --- INTERFAZ DE USUARIO ---
st.title("Simulador de Precios Airbnb Santiago (Machine Learning)")
st.caption("Esta aplicación predice en tiempo real el precio óptimo usando un modelo RandomForest.")

st.sidebar.header("Filtros de la Propiedad")

comunas_disponibles = sorted(df['neighbourhood_cleansed'].unique())
comuna_sel = st.sidebar.selectbox("Selecciona la Comuna", comunas_disponibles)

room_types = sorted(df['room_type'].unique())
room_sel = st.sidebar.selectbox("Tipo de Habitación", room_types)

minutos_metro_sel = st.sidebar.slider("Minutos Caminando al Metro", 0, 30, 5)
accommodates_sel = st.sidebar.slider("Capacidad de Huéspedes", int(df['accommodates'].min()), int(df['accommodates'].max()), 2)
bedrooms_sel = st.sidebar.slider("Dormitorios", int(df['bedrooms'].min()), int(df['bedrooms'].max()), 1)

# Slider de baños configurado de 0.5 en 0.5
bathrooms_sel = st.sidebar.slider(
    "Baños", 
    float(df['bathrooms_num'].min()), 
    float(df['bathrooms_num'].max()), 
    1.0,
    step=0.5
)

min_nights_sel = st.sidebar.slider("Noches Mínimas", int(df['minimum_nights'].min()), 30, 1)

# --- FILTRADO DINÁMICO ---
df_filtrado = df[
    (df['neighbourhood_cleansed'] == comuna_sel) &
    (df['room_type'] == room_sel) &
    (df['accommodates'] >= accommodates_sel) &
    (df['bedrooms'] == bedrooms_sel) &
    (df['bathrooms_num'] == bathrooms_sel) &
    (df['minimum_nights'] <= min_nights_sel) &
    (df['minutos_al_metro'].between(minutos_metro_sel - 3, minutos_metro_sel + 3))
].copy().reset_index(drop=True)

# --- MACHINE LEARNING ---
input_data = pd.DataFrame(0, index=[0], columns=columnas_x)
input_data['minutos_al_metro'] = minutos_metro_sel
input_data['accommodates'] = accommodates_sel
input_data['bedrooms'] = bedrooms_sel
input_data['bathrooms_num'] = bathrooms_sel
input_data['minimum_nights'] = min_nights_sel

col_room = f"room_type_{room_sel}"
col_neigh = f"neighbourhood_cleansed_{comuna_sel}"

if col_room in input_data.columns:
    input_data[col_room] = 1.0
if col_neigh in input_data.columns:
    input_data[col_neigh] = 1.0

precio_predicho = modelo.predict(input_data)[0]
mae = 21815

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Precio Mínimo Sugerido", value=f"${int(precio_predicho - mae):,}".replace(",", "."))
with col2:
    st.metric(label="PRECIO SUGERIDO", value=f"${int(precio_predicho):,}".replace(",", "."), delta="Recomendado")
with col3:
    st.metric(label="Precio Máximo Sugerido", value=f"${int(precio_predicho + mae):,}".replace(",", "."))

st.markdown("---")

st.subheader(f"Propiedades encontradas con tus características ({len(df_filtrado)} disponibles)")

# MAPEADO DE COORDENADAS CON SOPORTE COMPLETO A 'price'
if not df_filtrado.empty:
    if 'latitude' in df_filtrado.columns and 'longitude' in df_filtrado.columns:
        columnas_mapa = ['latitude', 'longitude']
        if 'price' in df_filtrado.columns:
            columnas_mapa.append('price')

        df_mapa = df_filtrado[columnas_mapa].rename(columns={'latitude': 'lat', 'longitude': 'lon'}).dropna()
        df_mapa['lat'] = df_mapa['lat'].astype(float)
        df_mapa['lon'] = df_mapa['lon'].astype(float)

        if 'price' in df_mapa.columns:
            df_mapa['price'] = df_mapa['price'].astype(float)
            st.map(df_mapa, size='price')
        else:
            st.map(df_mapa)
    else:
        st.warning("El dataset no cuenta con las columnas 'latitude' o 'longitude' para poder renderizar el mapa.")
else:
    st.warning("No se encontraron propiedades exactas con esta combinación de filtros. Intenta flexibilizar los criterios.")
