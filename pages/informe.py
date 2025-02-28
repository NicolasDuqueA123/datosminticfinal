import pandas as pd
import streamlit as st
import utilidades as util
from matplotlib import pyplot as plt
import seaborn as sns

#configuramos encabezados de la pagina
st.set_page_config(
    page_title='Informe',
    page_icon='📊',
    initial_sidebar_state='expanded',
    layout='centered'
)

util.generarMenu()

#Visualización 
st.markdown("<h2 style='text-align: center;'>Gráficas Informe transición sobre datos</h2>", unsafe_allow_html=True)
ruta = 'data/data_energy_3.3.1.csv'
df = pd.read_csv(ruta)

tex = 'Goles marcados liga Betplay 2024-2'

util.visualizardata(df,tex)
