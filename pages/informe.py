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
st.title('Datos Transición Energética')
ruta = 'data/data_energy_3.3.1.csv'
df = pd.read_csv(ruta)

tex = 'Goles marcados liga Betplay 2024-2'

util.visualizardata(df,tex)

#Matriz de correlaciones
corr_matrix = df
corr_matrix = corr_matrix.drop(columns=['country', 'iso_code', 'nuclear_electricity', 'nuclear_share_elec', 'year',
                                        'electricity_demand', 'electricity_generation', 'net_elec_imports', 'greenhouse_gas_emissions',
                                        'population', 'gdp'])
corr_matrix = corr_matrix.loc[:, ~corr_matrix.columns.str.contains('share_elec')]
corr_matrix = corr_matrix.corr()

#Gráfico de la matriz de correlaciones
plt.figure(figsize=(7,5))
sns.heatmap(corr_matrix, annot = True, vmin = -1, vmax = 1, cmap = "YlGnBu").set_title('Correlation Matrix')
st.pyplot(plt)