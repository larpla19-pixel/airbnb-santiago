import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# Configuración de la página
st.set_page_config(page_title="Simulador de precios Airbnb Santiago", layout="wide")

@st.cache_resource
def cargar_componentes():
    archivo_modelo = 'modelo_airbnb.pkl'
    df = pd.read_csv('airbnb_santiago_clean.csv', sep=';')
    columnas_x = joblib.load('columnas_entrenamiento.pkl')

    if not os.path.exists(archivo_modelo) or os.path.getsize(archivo_modelo) < 1_000_000:
        with st.spinner('Descargando modelo predictivo de IA...'):
            id_modelo_drive = "1O_z9kjnG7UG0LFivAeaeBrdVQjHtExyF"
            url_modelo = f"https://drive.google.com/uc?id={id_modelo_drive}"
            try:
                gdown.download(url_modelo, archivo_modelo, quiet=False)
            except Exception as e:
                st.error(f"No se pudo descargar el modelo: {e}")
                st.stop()

    if not os.path.exists(archivo_modelo) or os.path.getsize(archivo_modelo) < 1_000_000:
        st.error("El archivo descargado no es válido. Verifica el ID y los permisos en Drive.")
        st.stop()

    modelo = joblib.load(archivo_modelo)
    return modelo, columnas_x, df

def predecir_con_incertidumbre(modelo, input_data):
    preds_arboles = np.array([arbol.predict(input_data) for arbol in modelo.estimators_])
    pred_media = preds_arboles.mean(axis=0)[0]
    pred_std = preds_arboles.std(axis=0)[0]
    return pred_media, pred_std

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

df_comuna = df[df['neighbourhood_cleansed'] == comuna_sel]

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

precio_predicho, incertidumbre = predecir_con_incertidumbre(modelo, input_data)

# --- DISEÑO DEL DASHBOARD (MÉTRICAS) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Precio Mínimo Sugerido", value=f"${int(precio_predicho - incertidumbre):,}".replace(",", "."))
with col2:
    st.metric(label="PRECIO SUGERIDO IA", value=f"${int(precio_predicho):,}".replace(",", "."), delta="Recomendado")
with col3:
    st.metric(label="Precio Máximo Sugerido", value=f"${int(precio_predicho + incertidumbre):,}".replace(",", "."))

st.markdown("---")

st.subheader(f"📍 Distribución de propiedades en {comuna_sel}")

df_filtrado = df[
    (df['neighbourhood_cleansed'] == comuna_sel) &
    (df['room_type'] == room_sel) &
    (df['accommodates'] >= accommodates_sel) &
    (df['bedrooms'] == bedrooms_sel) &
    (df['bathrooms_num'] == bathrooms_sel) &
    (df['minimum_nights'] <= min_nights_sel)
].copy()

df_mapa = df_filtrado[['latitude', 'longitude', 'price']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}).dropna()

if not df_mapa.empty:
    df_mapa['size_normalizado'] = (df_mapa['price'] / df_mapa['price'].max()) * 50 + 10
    st.map(df_mapa, size='size_normalizado')
else:
    st.warning("No hay propiedades para mostrar en esta comuna.")
