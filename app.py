import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# Configuración de la página
st.set_page_config(page_title="Simulador de precios Airbnb Santiago", layout="wide")

# Cargar el modelo y los datos de forma robusta adaptada a la nube y local
@st.cache_resource
def cargar_componentes():
    # Obtener la carpeta donde está guardado este archivo app.py
    dir_actual = os.path.dirname(os.path.abspath(__file__))

    # Si app.py se está ejecutando desde 'notebooks', subimos un nivel para ir a la raíz
    if os.path.basename(dir_actual) == 'notebooks':
        dir_raiz = os.path.dirname(dir_actual)
    else:
        dir_raiz = dir_actual

    # Definir rutas absolutas apuntando exactamente a la estructura de tu proyecto
    archivo_modelo = os.path.join(dir_raiz, 'models', 'modelo_airbnb.pkl')
    ruta_columnas = os.path.join(dir_raiz, 'models', 'columnas_entrenamiento.pkl')
    ruta_csv = os.path.join(dir_raiz, 'data', 'processed', 'airbnb_santiago_clean.csv')

    # 1. Asegurar la existencia de la carpeta 'models' por si acaso
    os.makedirs(os.path.join(dir_raiz, 'models'), exist_ok=True)

    # 2. DESCARGAR PRIMERO: Si el modelo no existe (como en la nube de Streamlit), se descarga del Drive
    if not os.path.exists(archivo_modelo):
        with st.spinner('Descargando modelo predictivo de IA... (Esto solo toma unos segundos la primera vez)'):
            id_modelo_drive = "1yCPNrclsoaT_1SjjjmnyLHduouK4Ehps" 
            url_modelo = f"https://drive.google.com/uc?id={id_modelo_drive}"
            gdown.download(url_modelo, archivo_modelo, quiet=True)

    # 3. CARGAR DESPUÉS: Una vez descargado y asegurado en la carpeta 'models', se carga
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
    df_mapa = df_filtrado[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}).dropna()
    df_mapa['lat'] = df_mapa['lat'].astype(float)
    df_mapa['lon'] = df_mapa['lon'].astype(float)
    st.map(df_mapa)
else:
    st.warning("No se encontraron propiedades exactas con esta combinación de filtros en la base de datos. Intenta flexibilizar los criterios.")
