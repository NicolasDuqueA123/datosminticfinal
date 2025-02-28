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
    imagen2 = Image.open("Media/kdd.jpg")
    st.image(imagen2, use_container_width=True,width=500,
            caption="Figura 1:      Pasos a desarrollar en el procesamiento de datos modelo KDD")

    st.write("""
    Es importante mencionar que cada estudio debe adaptarse a sus necesidades, en este caso, se realizaron las siguientes acciones:""")


    st.write("""
    Fase 1: Recolección de Datos.""")




    st.write("""
    Fase 2: Limpieza de los datos.""")

    st.write("""
    Utilizando la librería Pandas, se eliminaron las variables que contenían valores perdidos, de igual manera, considerando los objetivos del estudio, se eliminaron las variables que no son útiles para el estudio y las que estaban duplicadas por su forma de cálculo original, en la sección *********, se presentan los scripts correspondientes a la limpieza de datos.
    Se realizó un análisis exploratorio de datos, con el fin de detectar errores de digitación, valores atípicos y/o extremos según la regla de Yates que se plasma en el gráfico de bigotes, además, de analizar las estadísticas descriptivas de resumen, contando el coeficiente de variación para mejor entendimiento de la variabilidad relativa de cada una de las variables continuas. Para las variables categóricas (país, …) se encontraron las frecuencias simples, sólo para identificar posibles comparaciones por país.
    """)

    st.write("""
    Fase 3: Transformación de datos.""")

    st.write("""
    Es fundamental en cualquier técnica estadística validar los supuestos que se deben cumplir para satisfacer los objetivos de la investigación, cuando se trata de variables cuantitativas es útil conocer la forma de la distribución de ellas, donde se prueba la normalidad (datos acampanados) es imperativa, para ello, se utilizó la prueba de Shapiro Willks y …, 
    Para validar la normalidad de los datos en Python, se pueden utilizar varias librerías y métodos estadísticos. Aquí están algunas de las más comunes:
    """)

    st.write("""
    Librerías y Métodos para Validar Normalidad.""")


    st.write("""
    1.	SciPy
    La librería SciPy es muy utilizada para realizar pruebas estadísticas, incluyendo la prueba de Shapiro-Wilk y la prueba de Kolmogorov-Smirnov, que son dos de las más comunes para verificar la normalidad.
    •	Prueba de Shapiro-Wilk: Ideal para muestras pequeñas y también aplicable a muestras más grandes. Se puede implementar de la siguiente manera:
            """)


    st.write("""
    Fase 4: Minería de datos y/o técnicas estadísticas.""")

    st.write("""
    Considerando que los objetivos son claros y se tienen técnicas estadísticas definidas para ello, no fue necesario utilizar alguna técnica de minería de datos, se utilizaron las siguientes técnicas estadísticas supervisadas:""")


    st.write("""
    Fase 4: Minería de datos y/o técnicas estadísticas.""")

    st.write("""
    Gráfico de pastel:""")

    st.write("""
    Series de tiempo (gráfico de líneas):""")

    st.write("""
    Correlaciones (gráficos de dispersión);""")


    st.write("""
    Regresión lineal múltiple:""")


    st.write("""
    Fase 5: Creación de conocimiento (Análisis e interpretación de resultados):""")


    st.write("""
    Series de tiempo (gráfico de líneas):.""")
