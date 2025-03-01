# # Análisis del dataframe

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import streamlit as st
import utilidades as util
from PIL import Image


st.set_page_config(
    page_title='Modelo Predictivo',
    page_icon='🌲',
    initial_sidebar_state='expanded',
    layout='centered'
)

util.generarMenu()

st.title("🌍🔢 Modelo Predictivo - Random Forest Regresor")

########## Creamos un DataFrame con los valores ENSO para Colombia ##########


years = np.arange(1983, 2025)  # Años desde 1983 hasta 2024

# Asignación de valores ENSO basada en eventos históricos
enso_values = {
    (1982, 1983): -1, (1986, 1987): -0.6, (1991, 1992): -0.7, (1997, 1998): -1,
    (2002, 2003): -0.5, (2009, 2010): -0.8, (2014, 2016): -0.9, (2018, 2019): -0.4,
    (2023, 2024): -1, (1988, 1989): 0.7, (1995, 1996): 0.6, (1998, 2001): 1,
    (2007, 2008): 0.7, (2010, 2011): 1, (2016, 2017): 0.5, (2020, 2023): 0.8}

enso_series = []      # Generar la lista de valores ENSO
for year in years:
    enso_value = 0  # Neutro por defecto
    for (start, end), value in enso_values.items():
        if start <= year <= end:
            enso_value = value
            break
    enso_series.append(enso_value)

enso_df = pd.DataFrame({'year': years, 'ENSO': enso_series})  # Crear DataFrame

# Verificar que las columnas tengan el mismo número de datos
assert len(enso_df['year']) == len(enso_df['ENSO']), "Error: Desajuste en la longitud de las columnas."



#####
################# Relacion entre el fenomeno ENSO y la prodducción de energia por hidroelectricas en colombia ##################
######
#

# Cargamos el dataset Modificado
data_energy = pd.read_csv("data/data_energy_2.csv", index_col=None)



# %%
############
#################### Modelo predictivo #######################
############

data_col = data_energy[data_energy['country'] == 'Colombia']
data_hidroCol = data_col[['year', 'population', 'hydro_electricity']]

# %%
df_predict = pd.merge(data_hidroCol, enso_df, on='year')

def mover_columna(df, columna, nueva_pos):
    col = df.pop(columna)  # Extraer la columna
    df.insert(nueva_pos, columna, col)  # Insertar en la nueva posición
    return df

df_predict = mover_columna(df_predict, 'hydro_electricity', 0)  # Mueve 'B' a la primera posición
print(df_predict)



# %%
############# Modelo predictivo random forest regresor
#####
#

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as asc
from sklearn.ensemble import RandomForestRegressor as rfr


def modelo_rf_regresor(df_p):
    st.subheader("📊 Datos de Producción de Energía Hidroeléctrica en TWh")
    st.dataframe(df_p)  # Mostrar los primeros datos

    Y = df_p.iloc[:, 0]  # Variable a predecir (numérica continua)
    X = df_p.iloc[:, 1:]  # Variables predictoras

    # División de los datos en entrenamiento y prueba
    X_entrenar, X_prueba, Y_entrenar, Y_prueba = tts(X, Y, train_size=0.8, random_state=42)

    st.write(f"📌 Datos de entrenamiento: **{X_entrenar.shape[0]} muestras**")
    st.write(f"📌 Datos de prueba: **{X_prueba.shape[0]} muestras**")

    # Crear y entrenar el modelo de regresión
    forest = rfr()
    forest.fit(X_entrenar, Y_entrenar)

    # Hacer la predicción
    Y_prediccion = forest.predict(X_prueba)

    # Evaluación con R² en lugar de accuracy
    from sklearn.metrics import r2_score
    score = r2_score(Y_prueba, Y_prediccion)

    st.subheader("📈 Desempeño del Modelo")
    st.write(f"🔹 **R² Score obtenido:** `{score:.4f}`")  # Formato con 4 decimales

    return score, forest




# %%
# Entrenar el modelo
score, forest = modelo_rf_regresor(df_predict)

# Entrada de nuevos datos para predicción
st.subheader("🔮 Hacer una Predicción")
with st.form("prediction_form"):
    year = st.number_input("Año", min_value=2020, max_value=2100, value=2025)
    population = st.number_input("Población", min_value=1000000.0, value=31003052.0, step=100000.0)
    enso = st.slider("ENSO", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    
    submit_button = st.form_submit_button("Predecir")

# Procesar la predicción cuando el usuario presiona el botón
if submit_button:
    nuevos_datos = pd.DataFrame([[year, population, enso]], columns=['year', 'population', 'ENSO'])
    Y_prediccion = forest.predict(nuevos_datos)

    st.success(f"🔮 **Predicción de Producción de Energía:** `{Y_prediccion[0]:.2f} TWh`")