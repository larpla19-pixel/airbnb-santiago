import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
import pydeck as pdk

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Simulador de precios Airbnb Santiago", layout="wide")

# ==============================================================================
# CARGA DE MODELO Y DATOS (con descarga blindada desde Google Drive)
# ==============================================================================
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

# ==============================================================================
# DATOS FIJOS: ESTACIONES DE METRO DE SANTIAGO
# ==============================================================================
datos_metro = {
    'estacion': [
        'Los Dominicos', 'Escuela Militar', 'Tobalaba', 'Baquedano', 'La Moneda', 'Estación Central', 'Ecuador', 'San Pablo',
        'Vespucio Norte', 'Cal y Canto', 'Los Héroes', 'Franklin', 'La Cisterna', 'Hospital El Pino',
        'Plaza Quilicura', 'Universidad de Chile', 'Irarrázaval', 'Plaza Egaña', 'Fernando Castillo Velasco',
        'Francisco Bilbao', 'Macul', 'Plaza de Puente Alto',
        'Plaza de Maipú', 'Laguna Sur', 'Quinta Normal', 'Santa Ana', 'Vicente Valdés',
        'Cerrillos', 'Lo Valledor', 'Ñuble', 'Inés de Suárez', 'Los Leones'
    ],
    'linea': [
        'L1','L1','L1','L1','L1','L1','L1','L1',
        'L2','L2','L2','L2','L2','L2',
        'L3','L3','L3','L3','L3',
        'L4','L4','L4',
        'L5','L5','L5','L5','L5',
        'L6','L6','L6','L6','L6'
    ],
    'lat_metro': [
        -33.40800,-33.41429,-33.41724,-33.43695,-33.44485,-33.45087,-33.45626,-33.45147,
        -33.36906,-33.43265,-33.44521,-33.47598,-33.53724,-33.57861,
        -33.36214,-33.44292,-33.45422,-33.45263,-33.45305,
        -33.43232,-33.50428,-33.61205,
        -33.51001,-33.46788,-33.44215,-33.44026,-33.52445,
        -33.48316,-33.47587,-33.46797,-33.43981,-33.42234
    ],
    'lon_metro': [
        -70.55560,-70.58463,-70.60105,-70.62886,-70.65480,-70.67915,-70.70109,-70.73038,
        -70.66270,-70.65191,-70.66014,-70.64812,-70.66367,-70.63842,
        -70.73428,-70.65039,-70.62772,-70.57018,-70.55014,
        -70.58554,-70.60831,-70.57564,
        -70.75731,-70.74902,-70.68112,-70.66046,-70.59775,
        -70.70311,-70.68266,-70.62534,-70.61271,-70.60813
    ]
}
df_metro_mapa = pd.DataFrame(datos_metro)

color_lineas = {
    'L1': [227, 6, 19],
    'L2': [255, 209, 0],
    'L3': [141, 76, 33],
    'L4': [0, 51, 141],
    'L5': [0, 166, 80],
    'L6': [116, 44, 140],
}
df_metro_mapa['color'] = df_metro_mapa['linea'].map(color_lineas)

# ==============================================================================
# INTERFAZ DE USUARIO — SIDEBAR DE FILTROS
# ==============================================================================
st.title("📊 Simulador de Precios Airbnb Santiago (Machine Learning)")
st.caption("Esta aplicación predice en tiempo real el precio óptimo usando un modelo RandomForest.")

st.sidebar.header("⚙️ Filtros de la Propiedad")

comunas_disponibles = sorted(df['neighbourhood_cleansed'].unique())
comuna_sel = st.sidebar.selectbox("Selecciona la Comuna", comunas_disponibles)

room_types = sorted(df['room_type'].unique())
room_sel = st.sidebar.selectbox("Tipo de Habitación", room_types)

minutos_metro_sel = st.sidebar.slider("Minutos Caminando al Metro", 0, 30, 5)
accommodates_sel = st.sidebar.slider("Capacidad de Huéspedes", int(df['accommodates'].min()), int(df['accommodates'].max()), 2)
bedrooms_sel = st.sidebar.slider("Dormitorios", int(df['bedrooms'].min()), int(df['bedrooms'].max()), 1)
bathrooms_sel = st.sidebar.slider("Baños", float(df['bathrooms_num'].min()), float(df['bathrooms_num'].max()), 1.0)
min_nights_sel = st.sidebar.slider("Noches Mínimas", int(df['minimum_nights'].min()), 30, 1)

# ==============================================================================
# FILTRADO DINÁMICO (todos los filtros del sidebar aplicados juntos)
# ==============================================================================
df_filtrado = df[
    (df['neighbourhood_cleansed'] == comuna_sel) &
    (df['room_type'] == room_sel) &
    (df['accommodates'] >= accommodates_sel) &
    (df['bedrooms'] == bedrooms_sel) &
    (df['bathrooms_num'] == bathrooms_sel) &
    (df['minimum_nights'] <= min_nights_sel)
].copy()

# ==============================================================================
# PREDICCIÓN DE PRECIO CON INCERTIDUMBRE
# ==============================================================================
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

# ==============================================================================
# DASHBOARD DE MÉTRICAS
# ==============================================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Precio Mínimo Sugerido", value=f"${int(precio_predicho - incertidumbre):,}".replace(",", "."))
with col2:
    st.metric(label="🎯 PRECIO SUGERIDO IA", value=f"${int(precio_predicho):,}".replace(",", "."), delta="Recomendado")
with col3:
    st.metric(label="Precio Máximo Sugerido", value=f"${int(precio_predicho + incertidumbre):,}".replace(",", "."))

st.markdown("---")

# ==============================================================================
# MAPA: PROPIEDADES FILTRADAS + ESTACIONES DE METRO POR LÍNEA
# ==============================================================================
st.subheader(f"📍 Distribución de propiedades en {comuna_sel} ({len(df_filtrado)} encontradas)")

df_mapa = df_filtrado[['latitude', 'longitude', 'price']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}).dropna()

if not df_mapa.empty:
    df_mapa['size_normalizado'] = (df_mapa['price'] / df_mapa['price'].max()) * 40 + 30

    capa_propiedades = pdk.Layer(
        "ScatterplotLayer",
        data=df_mapa,
        get_position='[lon, lat]',
        get_fill_color='[30, 30, 30, 160]',
        get_radius='size_normalizado',
        pickable=True,
    )

    capa_metro = pdk.Layer(
        "ScatterplotLayer",
        data=df_metro_mapa,
        get_position='[lon_metro, lat_metro]',
        get_fill_color='color',
        get_radius=80,
        pickable=True,
    )

    vista = pdk.ViewState(
        latitude=df_mapa['lat'].mean(),
        longitude=df_mapa['lon'].mean(),
        zoom=12,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[capa_metro, capa_propiedades],
        initial_view_state=vista,
        tooltip={"text": "{estacion}
{linea}"}
    ))

    st.caption("🔴 L1  🟡 L2  🟤 L3  🔵 L4  🟢 L5  🟣 L6  ⚫ Propiedades")
else:
    st.warning("No se encontraron propiedades con esta combinación de filtros. Intenta flexibilizar los criterios.")
