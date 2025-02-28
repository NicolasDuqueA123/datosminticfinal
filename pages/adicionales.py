
# # Análisis del dataframe

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import streamlit as st
import utilidades as util
from PIL import Image
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

#Generamos un datafreme sin la demanda energetica
data_tip = data_energy.drop(columns=['electricity_demand', 'electricity_generation',
                                            'net_elec_imports', 'greenhouse_gas_emissions'])

# Solo Colombia
data_col = data_energy[data_energy['country'] == 'Colombia']

# Solo Latino America
data_lat = data_energy[data_energy['iso_code'].isin(["COL", "BRA", "CHL", "MEX",
                                                            "ARG", "PER", "ECU"])]

# selecciono solo la columna de year y las de share_elec
data_col1 = data_col.filter(regex="year|share", axis=1)
data_col1 = data_col1[data_col1["year"] >= 2000]


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

data_hidrCol = data_col.filter(regex="year|hydro_share_elec", axis=1)

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

#INICIA GRÁFICA
# Gráfica de barras apiladas latam
st.markdown("<h5 style='text-align: center;'>2. Gráfica de barras apiladas LATAM</h5>", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'>Se visualiza la distribución estádistica de paises latam y su comparación con Colombia.</h7>", unsafe_allow_html=True)
#genlatam = Image.open("media/genlatam.jpeg")
#st.image(genlatam, use_container_width=False, width=700, caption=" ")
Latam_2023 = data_lat[data_lat['year'] == 2023]
# Filtrar las columnas relevantes
Latam_2023_share = Latam_2023.filter(regex="country|_share_elec", axis=1)

# Modificar el valor de biofuel en Chile
Latam_2023_share.loc[Latam_2023_share['country'] == 'Chile', 'biofuel_share_elec'] = 0.0

# Definir colores personalizados para cada fuente de energía
colores = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]


# Crear la figura
fig, ax4 = plt.subplots(figsize=(10, 6))

# Inicializar la acumulación de valores en 0    
apil = 0

# Crear las barras apiladas con colores personalizados
for i, col in enumerate(Latam_2023_share.columns[1:]):  # Excluir el país
    bars = ax4.bar(Latam_2023_share['country'], Latam_2023_share[col],
                   bottom=apil, label=col.replace("_share_elec", "").capitalize(),
                   linewidth=2, color=colores[i % len(colores)])  # Asignar color

    # Agregar etiquetas con valores
    for bar, value in zip(bars, Latam_2023_share[col]):
        if value > 5:
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}%",
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Acumular valores para la siguiente iteración
    apil += Latam_2023_share[col]

# Etiquetas y formato
ax4.set_ylabel("Porcentaje de Generación Eléctrica")
ax4.set_title("Generación de Energía por Fuente en Países de LATAM (2023)")
plt.xticks(rotation=45)

# Mover la leyenda fuera de la gráfica
ax4.legend(title="Fuente de Energía", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar diseño para evitar recortes
plt.tight_layout()

# Mostrar la gráfica
st.pyplot(plt)


# INICIA GRÁFICA
# Mapamundi
st.markdown("<h5 style='text-align: center;'>2. Mapamundi de generación renovable</h5>", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'>Se visualiza cuales son los paises que generan más energia de fuentes renovables a medida que el tono del color se oscurece</h7>", unsafe_allow_html=True)
#mundo = Image.open("media/mapamundi.jpeg")
#st.image(mundo, use_container_width=False, width=700, caption=" ")

data_energy = pd.read_csv("data/owid_energy_data.csv", index_col=None)
# Cargar el Shapefile con GeoPandas
world = gpd.read_file("data/ne_50m_admin_0_countries.shp")
# Estandarizar los nombres de los países en el Shapefile (mayúsculas y sin espacios
world["ADMIN"] = world["ADMIN"].str.upper().str.strip()
# Generar Mapa base del mundo
world.plot(figsize=(15, 13), edgecolor='black', facecolor='none')
plt.title('Mapa base del mundo', fontsize=14)
plt.axis('off')
data_energy_2023 = data_energy[data_energy['year'] == 2023]
data_energy_2023 = data_energy_2023[['iso_code', 'renewables_share_elec']]
#.filter(regex="iso_code|renewables_share_elec", axis=1)

#convierto la la columna world['ISO_A3'] en una lista
iso_codes = world['ISO_A3'].tolist()

# Seleccionamos aquellos paises que estan en la lista creada
data_energy_2023 = data_energy_2023[data_energy_2023['iso_code'].isin(iso_codes)]
data_energy_2023.reset_index(drop=True, inplace=True)
total           = data_energy_2023.isnull().sum().sort_values(ascending=False)
percent         = (data_energy_2023.isnull().sum()/data_energy_2023.isnull().count()*100).sort_values(ascending=False)
missing_train1  = pd.concat([total,percent],axis=1,keys=["Total","Percent"])
world_filt = world[['ADMIN', 'ISO_A3', 'geometry']]

# Seleccionamos aquellos paises que estan en la lista creada
world_filt.reset_index(drop=True, inplace=True)
world_filt = world_filt.rename(columns={'ISO_A3': 'iso_code'})
world_energy = world_filt.merge(data_energy_2023, on='iso_code', how='left')
# Cambiar el color de fondo del área de dibujo
ax.set_facecolor("lightgray")  # Fondo del mapa
plt.gca().set_facecolor("lightgray")  # Asegurar que se aplique en toda la zona de gráficos
fig.patch.set_facecolor("whitesmoke")  # Fondo exterior de la figura

# Graficar el DataFrame GeoPandas
world_energy.plot(column='renewables_share_elec',
                  cmap='Blues',
                  linewidth=0.8,
                  ax=ax,
                  edgecolor='black',  # Se mantiene un solo edgecolor
                  legend=True,
                  legend_kwds={'label': "proporción de energía renovable",
                               'orientation': "vertical",
                               'shrink': 0.4,   # Ajusta el tamaño de la barra de colores (1 = tamaño normal)
                               'aspect': 20},
                  missing_kwds={"color": "purple", "edgecolor": "black"}  # Para áreas sin datos
                  )

# Agregar leyenda para zonas sin datos
legend_elements = [Patch(facecolor='purple', edgecolor='black', label='Sin datos')]
ax.legend(handles=legend_elements, loc='lower left')

# Título y eliminación de ejes
ax.set_title('Porcentaje de energía renovable - 2023', fontsize=16)
ax.axis('off')

# Mostrar el gráfico
st.pyplot(plt)



# LINKS A COLAB
st.markdown("<h7 style='text-align: center;'>El desarrollo de estás gráficas es visible en los siguientes links:</h7>", unsafe_allow_html=True)
st.markdown('<a href="https://colab.research.google.com/drive/1uF-1jIGjC7Gi5VjOUI-s21ihNOPaZJlt#scrollTo=PczYi5GfJtvj" target="_blank">Ir a gráficas LATAM</a>', unsafe_allow_html=True)
st.markdown('<a href="https://colab.research.google.com/drive/1Lhy4pbWL6LRIM3w3qr04e_czjpdH0quf" target="_blank">Ir a gráfica mundial</a>', unsafe_allow_html=True)


# COMIENZA MODELO PREDICTIVO
data_hidroCol = data_col[['year', 'population', 'hydro_electricity']]

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



