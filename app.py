import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# Configuración de la página
st.set_page_config(page_title="Simulador de precios Airbnb Santiago", layout="wide")

# Cargar el modelo y los datos desde la nube de forma asíncrona
@st.cache_resource
def cargar_componentes():
    archivo_modelo = 'modelo_airbnb.pkl'
    archivo_csv = 'airbnb_santiago_clean.csv'
    columnas_x = 'columnas_entrenamiento.pkl' # Este queda local desde GitHub

    # 1. El CSV y las columnas se leen directo de la carpeta (ya que están en GitHub)
    df = pd.read_csv('airbnb_santiago_clean.csv', sep=';')
    columnas_x = joblib.load('columnas_entrenamiento.pkl')

    # 2. Descarga del Modelo usando el endpoint de la API para evadir el bloqueo de +100MB
    if not os.path.exists(archivo_modelo):
        with st.spinner('Descargando modelo predictivo de IA... (Esto solo toma unos segundos la primera vez)'):
            id_modelo_drive = "1yCPNrclsoaT_1SjjjmnyLHduouK4Ehps" 
            url_modelo = f"https://www.googleapis.com/drive/v3/files/{id_modelo_drive}?alt=media"

            # Configuramos un User-Agent para que Google Drive acepte la petición de Streamlit
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)

            urllib.request.urlretrieve(url_modelo, archivo_modelo)

    modelo = joblib.load(archivo_modelo)
    columnas_x_data = joblib.load(columnas_x)
    df = pd.read_csv(archivo_csv, sep=';')
    return modelo, columnas_x_data, df

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
bathrooms_sel = st.sidebar.slider("Baños", float(df['bathrooms_num'].min()), float(df['bathrooms_num'].max()), 1.0)
min_nights_sel = st.sidebar.slider("Noches Mínimas", int(df['minimum_nights'].min()), 30, 1)

# --- FILTRADO DINÁMICO CON COPIA LIMPIA ---
df_filtrado = df[
    (df['neighbourhood_cleansed'] == comuna_sel) &
    (df['room_type'] == room_sel) &
    (df['accommodates'] >= accommodates_sel) &
    (df['bedrooms'] == bedrooms_sel) &
    (df['bathrooms_num'] == bathrooms_sel) &
    (df['minimum_nights'] <= min_nights_sel) &
    (df['minutos_al_metro'].between(minutos_metro_sel - 3, minutos_metro_sel + 3))
].copy().reset_index(drop=True)

# --- PROCESAMIENTO DE MACHINE LEARNING EN VIVO ---
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

# --- DISEÑO DEL DASHBOARD (MÉTRICAS) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Precio Mínimo Sugerido", value=f"${int(precio_predicho - mae):,}".replace(",", "."))
with col2:
    st.metric(label="PRECIO SUGERIDO", value=f"${int(precio_predicho):,}".replace(",", "."), delta="Recomendado")
with col3:
    st.metric(label="Precio Máximo Sugerido", value=f"${int(precio_predicho + mae):,}".replace(",", "."))

st.markdown("---")

# --- MAPA URBANO REACTIVO ---
st.subheader(f"Propiedades encontradas con tus características ({len(df_filtrado)} disponibles)")

if not df_filtrado.empty:
    df_mapa = df_filtrado[['latitude', 'longitude', 'price']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}).dropna()
    map_key = f"mapa_{comuna_sel}_{room_sel}_{minutos_metro_sel}_{accommodates_sel}_{bedrooms_sel}"
    st.map(df_mapa, size='price', key=map_key)
else:
    st.warning("No se encontraron propiedades exactas con esta combinación de filtros en la base de datos. Intenta flexibilizar los criterios.")
