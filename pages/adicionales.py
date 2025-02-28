
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
    page_title='Others',
    page_icon='✈️',
    initial_sidebar_state='expanded',
    layout='centered'
)

util.generarMenu()

st.markdown("<h2 style='text-align: center;'>Gráficas adicionales</h2>", unsafe_allow_html=True)

# Cargamos el dataset Modificado
data_energy = pd.read_csv("data/data_energy_2.csv", index_col=None)


# %%
#Generamos un datafreme sin la demanda energetica
data_tip = data_energy.drop(columns=['electricity_demand', 'electricity_generation',
                                            'net_elec_imports', 'greenhouse_gas_emissions'])

# Solo Colombia
data_col = data_energy[data_energy['country'] == 'Colombia']

# Solo Latino America
data_lat = data_energy[data_energy['iso_code'].isin(["COL", "BRA", "CHL", "MEX",
                                                            "ARG", "PER", "ECU"])]

# Paises Desarrollados
data_des = data_energy[data_energy['iso_code'].isin(["NOR", "DEU", "ESP", "NLD", "FRA", "USA", "CHN",
                                                            "IND", "CAN", "JPN", "AUS"])]


# %%
# selecciono solo la columna de year y las de share_elec
data_col1 = data_col.filter(regex="year|share", axis=1)
data_col1 = data_col1[data_col1["year"] >= 2000]


# %% [markdown]
# Relacion entre el fenomeno ENSO y la prodducción de energia por hidroelectricas en colombia

# %%
# Años desde 1983 hasta 2024
years = np.arange(1983, 2025)

# Asignación de valores ENSO basada en eventos históricos
enso_values = {
    (1982, 1983): -1, (1986, 1987): -0.6, (1991, 1992): -0.7, (1997, 1998): -1,
    (2002, 2003): -0.5, (2009, 2010): -0.8, (2014, 2016): -0.9, (2018, 2019): -0.4,
    (2023, 2024): -1, (1988, 1989): 0.7, (1995, 1996): 0.6, (1998, 2001): 1,
    (2007, 2008): 0.7, (2010, 2011): 1, (2016, 2017): 0.5, (2020, 2023): 0.8
}

# Generar la lista de valores ENSO
enso_series = []
for year in years:
    enso_value = 0  # Neutro por defecto
    for (start, end), value in enso_values.items():
        if start <= year <= end:
            enso_value = value
            break
    enso_series.append(enso_value)

# Crear DataFrame
enso_df = pd.DataFrame({'year': years, 'ENSO': enso_series})

# Verificar que las columnas tengan el mismo número de datos
assert len(enso_df['year']) == len(enso_df['ENSO']), "Error: Desajuste en la longitud de las columnas."

# Mostrar el DataFrame
print(enso_df.head())


# %%
data_hidrCol = data_col.filter(regex="year|hydro_share_elec", axis=1)

# %%
# Configuración de estilo oscuro
plt.style.use('dark_background')
sns.set_palette("bright")  # Colores vibrantes

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(12, 6))

# Dibujar línea
ax.plot(data_hidrCol['year'], data_hidrCol["hydro_share_elec"], label= "Hidroelectrica", linewidth=2)

# Personalización del gráfico
ax.set_facecolor("#030764")  # Fondo de la grafica
ax.grid(color='gray', linestyle='dashdot', linewidth=0.4)  # Rejilla sutil
ax.set_title("Relacion entre el ENSO y la energia hidroelectrica", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Año", fontsize=12, color='white')
ax.set_ylabel("Energía en porcentaje", fontsize=12, color='white')
ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.9), fontsize=12, frameon=False)
ax.tick_params(axis='both', colors='white')  # Color de los números en ejes
ax.set_xticks(np.arange(data_hidrCol['year'].min(), data_hidrCol['year'].max() + 1, 1))  # Configurar la grilla para que aparezca cada año
ax.set_xticklabels([str(year) if year % 5 == 0 else "" for year in data_hidrCol['year']], rotation=0)  # Configurar etiquetas

# Agregar un segundo eje para ENSO
ax2 = ax.twinx()
ax2.plot(enso_df['year'], enso_df['ENSO'], label='ENSO Index', color='magenta', linestyle='dashed', linewidth=2)
ax2.set_ylabel(" Niño   <----------------- ENSO ----------------->   Niña", fontsize=12, color='magenta')
ax2.tick_params(axis='y', colors='magenta')
ax2.grid(color='black', linestyle='dotted', linewidth=0.2)  # Puedes aumentar el valor de linewidth

# Agregar leyenda para ENSO
ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.95), fontsize=12, frameon=False)



# Mostrar gráfico
st.markdown("<h5 style='text-align: center;'>1. Gráfica ENSO hidroeléctrica</h5>", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'>Es visible como en periodos donde hay menor cantidad de lluvias la generación hidroeléctrica disminuye y viceversa.</h7>", unsafe_allow_html=True)
# Gráfico ENSO
st.pyplot(plt)

# Gráfica de barras apiladas latam
st.markdown("<h5 style='text-align: center;'>2. Gráfica de barras apiladas LATAM</h5>", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'>Se visualiza la distribución estádistica de paises latam y su comparación con Colombia.</h7>", unsafe_allow_html=True)
genlatam = Image.open("media/genlatam.jpeg")
st.image(genlatam, use_container_width=False, width=650, caption=" ")

# Mapamundi
st.markdown("<h5 style='text-align: center;'>2. Mapamundi de generación renovable</h5>", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'>Se visualiza cuales son los paises que generan más energia de fuentes renovables a medida que el tono del color se oscurece</h7>", unsafe_allow_html=True)
mundo = Image.open("media/mapamundi.jpeg")
st.image(genlatam, use_container_width=False, width=650, caption=" ")


# COMIENZA MODELO PREDICTIVO
data_hidroCol = data_col[['year', 'population', 'hydro_electricity']]

# %%
df_predict = pd.merge(data_hidroCol, enso_df, on='year')

# %%
def mover_columna(df, columna, nueva_pos):
    col = df.pop(columna)  # Extraer la columna
    df.insert(nueva_pos, columna, col)  # Insertar en la nueva posición
    return df

df_predict = mover_columna(df_predict, 'hydro_electricity', 0)  # Mueve 'B' a la primera posición
print(df_predict)

# %%
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as asc
from sklearn.ensemble import RandomForestRegressor as rfr

# %%
# Modelo predictivo random forest regresor

def modelo_rf_regresor(df_p):
    print("## Datos produccion de energia hidroelectrica en Twh")
    print(df_p.head())
    print("\n--- Resultado del modelo Random Forest ---\n")

    Y = df_p.iloc[:, 0]  # Variable a predecir (numérica continua)
    X = df_p.iloc[:, 1:]  # Variables predictoras

    # División de los datos en entrenamiento y prueba
    X_entrenar, X_prueba, Y_entrenar, Y_prueba = tts(X, Y, train_size=0.8, random_state=42)

    print("### Separamos los datos")
    print(f"Datos de entrenamiento: {X_entrenar.shape[0]} muestras")
    print(f"Datos de prueba: {X_prueba.shape[0]} muestras\n")

    # Crear y entrenar el modelo de regresión
    forest = rfr()
    forest.fit(X_entrenar, Y_entrenar)

    # Hacer la predicción
    Y_prediccion = forest.predict(X_prueba)

    # Evaluación con R² en lugar de accuracy
    from sklearn.metrics import r2_score
    score = r2_score(Y_prueba, Y_prediccion)

    print("\nMétrica de desempeño del modelo:")
    print(f"R² Score obtenido: {score:.4f}")  # Formato con 4 decimales

    return score, forest




# %%
modelo_rf_regresor(df_predict)


# %%
nuevos_datos = pd.DataFrame([[2025, 31003052.0, 1], [2025, 31003052.0, -1]], columns=['year', 'population', 'ENSO'])

# Hacer la predicción

forest = modelo_rf_regresor(df_predict)[1]
Y_prediccion = forest.predict(nuevos_datos)

print(Y_prediccion)

# %%
from sklearn.ensemble import RandomForestClassifier as rfc
#Función del modelo predictivo

def modelo_rf(df_p):
    print("## Datos enfermedades de pacientes")
    print(df_p.head())  # Muestra las primeras filas del DataFrame
    print("\n--- Resultado del modelo Random Forest ---\n")

    # Variable a predecir
    Y = df_p.iloc[:, 0]
    # Variables predictoras
    X = df_p.iloc[:, 1:]

    # División de los datos en entrenamiento y prueba
    X_entrenar, X_prueba, Y_entrenar, Y_prueba = tts(X, Y, train_size=0.8, random_state=42)

    print("### Separamos los datos")
    print("Datos de entrenamiento:")
    print(f"Muestra de las variables de entrenamiento: {X_entrenar.shape[0]} datos")
    print(f"Muestra de las variables de prueba: {X_prueba.shape[0]} datos\n")

    # Crear el modelo Random Forest
    forest = rfc()

    # Entrenar el modelo
    forest.fit(X_entrenar, Y_entrenar)

    # Hacer la predicción
    Y_prediccion = forest.predict(X_prueba)
    accuracy = asc(Y_prueba, Y_prediccion)

    print("\nMétrica de precisión del modelo:")
    print(f"Precisión obtenida: {accuracy:.4f}")  # Formato de 4 decimales

    return accuracy



