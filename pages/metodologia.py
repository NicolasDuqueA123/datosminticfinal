import pandas as pd
import utilidades as util
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns

#Página de presentación o índex
st.set_page_config(
    page_title="Metodología",
    initial_sidebar_state='expanded',
    layout="wide",
    page_icon="🌐"
)

#Llamamos la función para las columnas

util.generarMenu()

#Estructura de presentación
left_col, center_col, right_col = st.columns([0.5,8,0.5],vertical_alignment="center")
# #dito la columna central
with center_col:
    st.title("Metodología proyecto analisis sobre datos")
    st.write("""
            Quien hace puede equivocarse, quien no hace ya está equivocado.
            "DANIEL KON"
    """)
    st.write("""
Metodología Procesamiento de Datos:

    """)
    st.write("""
Dados los objetivos del estudio, donde se requirió una búsqueda de bases de datos de fuentes secundarias y descubrir relaciones entre las variables objeto de estudio, se usó el modelo KDD (Knowledge Discovery in Database).

    """)
    imagen2 = Image.open("media/kdd.jpg")
    st.image(imagen2, use_container_width=True,width=350,
            caption="Figura 1:      Pasos a desarrollar en el procesamiento de datos modelo KDD")

    st.write("""
    Fase 1: Recolección de Datos.""")

    st.write("""
    Fase 2: Limpieza de los datos. Usando Pandas, Numpy y el software JASP.""")

    st.write("""
    Fase 3: Transformación de datos. Usando la prueba de Shapiro Willks en JASP, encontrando normalidad y justificando el motivo de las variables con poca normalidad.""")

    st.write("""
    Fase 4: Técnicas estadísticas.""")

    st.write("""
    Correlaciones de Pearson.""")

    st.write("""
    Gráfico de pie para evolución de generación eléctrica total.""")

    st.write("""
    Gráfico de lineas para evlución generación fuentes de energía.""")

    st.write("""
    Barras apiladas porcentajes de Generación en TWH en periodo estudiado.""")

    st.write("""
    Fase 5: Análisis e interpretación de resultados. Mostrado en informe.""")

