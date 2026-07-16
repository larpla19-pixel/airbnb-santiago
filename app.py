import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Simulador de precios Airbnb Santiago", layout="wide")

# Cargar el modelo y los datos usando el CSV con delimitador de punto y coma (;)
@st.cache_resource
def cargar_componentes():
    modelo = joblib.load('modelo_airbnb.pkl')
    columnas_x = joblib.load('columnas_entrenamiento.pkl')
    df = pd.read_csv('airbnb_santiago_clean.csv', sep=';')
    return modelo, columnas_x, df

modelo, columnas_x, df = cargar_componentes()

# --- INTERFAZ DE USUARIO ---
st.title("📊 Simulador de Precios Airbnb Santiago (Machine Learning)")
st.caption("Esta aplicación predice en tiempo real el precio óptimo usando un modelo RandomForest.")

st.sidebar.header("⚙️ Filtros de la Propiedad")

comunas_disponibles = sorted(df['neighbourhood_cleansed'].unique())
comuna_sel = st.sidebar.selectbox("Selecciona la Comuna", comunas_disponibles)

room_types = sorted(df['room_type'].unique())
room_sel = st.sidebar.selectbox("Tipo de Habitación", room_types)

# 🚇 NUEVO SLIDER: Variable de alta importancia predictiva
minutos_metro_sel = st.sidebar.slider("Minutos Caminando al Metro", 0, 30, 5)

accommodates_sel = st.sidebar.slider("Capacidad de Huéspedes", int(df['accommodates'].min()), int(df['accommodates'].max()), 2)
bedrooms_sel = st.sidebar.slider("Dormitorios", int(df['bedrooms'].min()), int(df['bedrooms'].max()), 1)
bathrooms_sel = st.sidebar.slider("Baños", float(df['bathrooms_num'].min()), float(df['bathrooms_num'].max()), 1.0)
min_nights_sel = st.sidebar.slider("Noches Mínimas", int(df['minimum_nights'].min()), 30, 1)

df_comuna = df[df['neighbourhood_cleansed'] == comuna_sel]

# --- PROCESAMIENTO DE MACHINE LEARNING EN VIVO ---
input_data = pd.DataFrame(0, index=[0], columns=columnas_x)

# Llenamos las variables numéricas incluyendo los minutos al metro
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
    st.metric(label="🎯 PRECIO SUGERIDO IA", value=f"${int(precio_predicho):,}".replace(",", "."), delta="Recomendado")
with col3:
    st.metric(label="Precio Máximo Sugerido", value=f"${int(precio_predicho + mae):,}".replace(",", "."))

st.markdown("---")

st.subheader(f"📍 Distribución de propiedades en {comuna_sel}")

df_mapa = df_comuna[['latitude', 'longitude', 'price']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}).dropna()
st.map(df_mapa, size='price')
