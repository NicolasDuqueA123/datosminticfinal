
# # Análisis del dataframe


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

# Cargamos el dataset Modificado
data_energy = pd.read_csv("D:/0_Edwin_Duque/AnalisisDatosIntegrador/Proyect_Energy/Data/data_energy_2.csv", index_col=None)

# %% [markdown]
# Generamos varios dataset para segmentar su posterior análisis

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

# %% [markdown]
# ## Colombia
# 

# %% [markdown]
# ¿De donde proviene la energia de colombia?

# %%
# mostramos las columnas que tiene el dataframe:
data_col.columns

# %% [markdown]
# Fuentes de energia en porcentaje por año

# %%
# selecciono solo la columna de year y las de share_elec
data_col1 = data_col.filter(regex="year|share", axis=1)
data_col1 = data_col1[data_col1["year"] >= 2000]
data_col1.columns

# %%
# Configuración de estilo oscuro
plt.style.use('dark_background')
sns.set_palette("bright")  # Colores vibrantes

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(12, 6))

# Dibujar cada línea
for col in data_col1.columns[1:]:  # Excluir el año
    ax.plot(data_col1['year'], data_col1[col], label=col.replace("_share_elec", "").capitalize(), linewidth=2)

# Personalización del gráfico
ax.set_facecolor("#030764")  # Fondo de la grafica
ax.grid(color='gray', linestyle='dashdot', linewidth=0.4)  # Rejilla sutil
ax.set_title("Evolución del Mix Energético en Colombia", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Año", fontsize=12, color='white')
ax.set_ylabel("Porcentaje de Energía", fontsize=12, color='white')
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12, frameon=True)
ax.tick_params(axis='both', colors='white')  # Color de los números en ejes
ax.set_xticks(np.arange(data_col1['year'].min(), data_col1['year'].max() + 1, 1))  # Configurar la grilla para que aparezca cada año
ax.set_xticklabels([str(year) if year % 5 == 0 else "" for year in data_col1['year']], rotation=0)  # Configurar etiquetas
ax.set_yticks(np.arange(0, 100, 10))  # Configurar la grilla para que aparezca
ax.set_yticklabels([str(y) if y % 20 == 0 else "" for y in (np.arange(0, 100, 10))], rotation=0)  # Configurar etiquetas


# Mostrar gráfico
plt.show()

# %%
# Configuración del estilo de Seaborn
sns.set_theme(style="whitegrid")

# Crear el gráfico de área apilada
plt.figure(figsize=(12, 6))
plt.stackplot(data_col1['year'],
              data_col1['biofuel_share_elec'], data_col1['coal_share_elec'], data_col1['gas_share_elec'], data_col1['nuclear_share_elec'],
              data_col1['oil_share_elec'], data_col1['solar_share_elec'], data_col1['wind_share_elec'], data_col1['hydro_share_elec'],
              labels=['Biocombustible', 'Carbon', 'Gas', 'Nuclear', 'Petroleo', 'Solar', 'Eólica', 'Hidráulica' ],
              colors=['#ff7f0e', '#808080', '#2ca02c', '#9467bd', '#B22222', '#FFD700', '#ADD8E6', '#1f77b4'],
              alpha=0.8)

# Personalización del gráfico
plt.title("Evolución de la Matriz Energética en Colombia (%)", fontsize=14, fontweight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Porcentaje de Energía", fontsize=12)
plt.legend(loc="upper left", title="Tipo de Energía")
plt.xticks(data_col1['year'], rotation=90)
plt.yticks(range(0, 101, 10))
plt.ylim(0, 100)

# Mostrar el gráfico
plt.show()

# %%
# Definir colores personalizados para cada fuente de energía
colores = ["#003f5c", "#808080", "#665191", "#a05195", "#d45087", "#f95d6a", "#FFD700", "#ffa600"]
#colores=['#ff7f0e', '#808080', '#2ca02c', '#9467bd', '#B22222', '#FFD700', '#ADD8E6', '#1f77b4']

# Crear la figura
fig, ax4 = plt.subplots(figsize=(10, 6))

# Inicializar la acumulación de valores en 0
apil = 0

# Crear las barras apiladas con colores personalizados
for i, col in enumerate(data_col1.columns[1:]):  # Excluir el año
    bars = ax4.bar(data_col1['year'], data_col1[col],
                   bottom=apil, label=col.replace("_share_elec", "").capitalize(),
                   linewidth=2, color=colores[i % len(colores)])  # Asignar color

    # Acumular valores para la siguiente iteración
    apil += data_col1[col]

# Etiquetas y formato
ax4.set_ylabel("Porcentaje de Generación Eléctrica")
ax4.set_title("Generación de Energía por Fuente en Colombia")
plt.xticks(rotation=45)

# Mover la leyenda fuera de la gráfica
ax4.legend(title="Fuente de Energía", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar diseño para evitar recortes
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# %% [markdown]
# Fuentes de energia en TWh

# %%
# selecciono solo la columna de year y las de electricity
data_col2 = data_col.filter(regex="year|electricity", axis=1)
data_col2 = data_col2[data_col2["year"] >= 2000]

# Elimino algunas columnas que se me saltaron al filtro
data_col2 = data_col2.drop(columns=['electricity_demand', 'electricity_generation'])
data_col2.columns

# %%
# Configuración de estilo oscuro
plt.style.use('dark_background')
sns.set_palette("bright")  # Colores vibrantes

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(12, 6))

# Dibujar cada línea
for col in data_col2.columns[1:]:  # Excluir el año
    ax.plot(data_col2['year'], data_col2[col], label=col.replace("_electricity", "").capitalize(), linewidth=2)

# Personalización del gráfico
ax.set_facecolor("#030764")  # Fondo de la grafica
ax.grid(color='gray', linestyle='dashdot', linewidth=0.4)  # Rejilla sutil
ax.set_title("Evolución del Mix Energético en Colombia", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Año", fontsize=12, color='white')
ax.set_ylabel("Energía en TWh", fontsize=12, color='white')
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12, frameon=True)
ax.tick_params(axis='both', colors='white')  # Color de los números en ejes
ax.set_xticks(np.arange(data_col1['year'].min(), data_col1['year'].max() + 1, 1))  # Configurar la grilla para que aparezca cada año
ax.set_xticklabels([str(year) if year % 5 == 0 else "" for year in data_col1['year']], rotation=0)  # Configurar etiquetas
#ax.set_yticks(np.arange(0, 100, 10))  # Configurar la grilla para que aparezca
#ax.set_yticklabels([str(y) if y % 20 == 0 else "" for y in (np.arange(0, 100, 10))], rotation=0)  # Configurar etiquetas


# Mostrar gráfico
plt.show()

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
plt.show()

# %%
#Matriz de correlaciones
corr_matrix = data_col
corr_matrix = corr_matrix.drop(columns=['country', 'iso_code', 'nuclear_electricity', 'nuclear_share_elec', 'year',
                                        'electricity_demand', 'electricity_generation', 'net_elec_imports', 'greenhouse_gas_emissions',
                                        'population', 'gdp'])
corr_matrix = corr_matrix.loc[:, ~corr_matrix.columns.str.contains('share_elec')]
corr_matrix = corr_matrix.corr()

#Gráfico de la matriz de correlaciones
plt.figure(figsize=(7,5))
sns.heatmap(corr_matrix, annot = True, vmin = -1, vmax = 1, cmap = "YlGnBu").set_title('Correlation Matrix')
plt.show()





# %% [markdown]
# ## Latinoamerica
# ¿Cuales son el tipo de energia que utilizan otros paises representativos de latinoamerica?

# %%
Latam_2023 = data_lat[data_lat['year'] == 2023]
Latam_2023.columns

# %%
# Primero visualisaremos el panorama que tienen los paises de latam, visualizando su poblacion, demanda de energia etc para el año 2023
# Crear las barras
fig, ax3 = plt.subplots(figsize=(10, 5))  # Ajusta el tamaño de la figura
bar1 = ax3.bar(Latam_2023['country'], Latam_2023['population']/1_000_000, label='Población', color='green')

# Añadir etiquetas y título
ax3.set_ylabel('Población (millones)')
ax3.set_title('Población de paises representativos de Latinoamerica al año 2023')

# Mostrar gráfico
plt.tight_layout()
plt.show()


# %%
# Demanda de electricidad
# Crear las barras
fig, ax4 = plt.subplots(figsize=(10, 5))  # Ajusta el tamaño de la figura
bar1 = ax4.bar(Latam_2023['country'], Latam_2023['electricity_demand'], label='Población', color='green')
bar2 = ax4.bar(Latam_2023['country'], Latam_2023['electricity_generation'], 0.55, label='Población', color='blue')

# Añadir etiquetas y título
ax4.set_ylabel('Electricidad (Twh)')
ax4.set_title('Demanda y generación de energia al año 2023')

# Mostrar gráfico
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

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
plt.show()


# %% [markdown]
# ## Paises Desarrollados

# %%
des_2023 = data_des[data_des['year'] == 2023]

# %%
# Demanda de electricidad
# Crear las barras
fig, ax5 = plt.subplots(figsize=(10, 5))  # Ajusta el tamaño de la figura
bar1 = ax5.bar(des_2023['country'], des_2023['electricity_demand'], label='Población', color='green')
bar2 = ax5.bar(des_2023['country'], des_2023['electricity_generation'], 0.55, label='Población', color='blue')

# Añadir etiquetas y título
ax5.set_ylabel('Electricidad (Twh)')
ax5.set_title('Demanda y generación de energia al año 2023')

# Mostrar gráfico
plt.tight_layout()
plt.show()

# %%
# Filtrar las columnas relevantes
des_2023_share = des_2023.filter(regex="country|_share_elec", axis=1)

# Definir colores personalizados para cada fuente de energía
colores = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]


# Crear la figura
fig, ax4 = plt.subplots(figsize=(10, 6))

# Inicializar la acumulación de valores en 0
apil = 0

# Crear las barras apiladas con colores personalizados
for i, col in enumerate(des_2023_share.columns[1:]):  # Excluir el país
    bars = ax4.bar(des_2023_share['country'], des_2023_share[col],
                   bottom=apil, label=col.replace("_share_elec", "").capitalize(),
                   linewidth=2, color=colores[i % len(colores)])  # Asignar color

    # Agregar etiquetas con valores
    for bar, value in zip(bars, des_2023_share[col]):
        if value > 5:
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}%",
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Acumular valores para la siguiente iteración
    apil += des_2023_share[col]

# Etiquetas y formato
ax4.set_ylabel("Porcentaje de Generación Eléctrica")
ax4.set_title("Generación de Energía por Fuente en Países Desarrollados (2023)")
plt.xticks(rotation=45)

# Mover la leyenda fuera de la gráfica
ax4.legend(title="Fuente de Energía", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar diseño para evitar recortes
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# %%
# Filtrar las columnas relevantes
des_2023_share = des_2023.filter(regex="country|_share_elec", axis=1)

# Definir colores personalizados para cada fuente de energía
colores = ["#003f5c", "#000000", "#665191", "#a05195", "#A52A2A", "#f95d6a", "#ffa600", "#00FFFF"]


# Crear la figura
fig, ax4 = plt.subplots(figsize=(10, 6))

# Inicializar la acumulación de valores en 0
apil = 0

# Crear las barras apiladas con colores personalizados
for i, col in enumerate(des_2023_share.columns[1:]):  # Excluir el país
    bars = ax4.bar(des_2023_share['country'], des_2023_share[col],
                   bottom=apil, label=col.replace("_share_elec", "").capitalize(),
                   linewidth=2, color=colores[i % len(colores)])  # Asignar color

    # Agregar etiquetas con valores
    for bar, value in zip(bars, des_2023_share[col]):
        if value > 5:
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}%",
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Acumular valores para la siguiente iteración
    apil += des_2023_share[col]

# Etiquetas y formato
ax4.set_ylabel("Porcentaje de Generación Eléctrica")
ax4.set_title("Generación de Energía por Fuente en Países Desarrollados (2023)")
plt.xticks(rotation=45)

# Mover la leyenda fuera de la gráfica
ax4.legend(title="Fuente de Energía", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar diseño para evitar recortes
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# %%
des_2023_share.columns

# %%
des_2023_share["Energia_renovable"] = des_2023_share[['biofuel_share_elec', 'hydro_share_elec', 'solar_share_elec', 'wind_share_elec']].sum(axis=1)
des_2023_share["Energia_no_renovable"] = des_2023_share[['coal_share_elec', 'gas_share_elec', 'oil_share_elec']].sum(axis=1)

# %%
des_2023_share = des_2023_share.drop(columns=['biofuel_share_elec', 'hydro_share_elec', 'solar_share_elec', 'wind_share_elec', 'coal_share_elec', 'gas_share_elec', 'oil_share_elec'])

# %%
colores = ["#A52A2A", "#228B22", "#003f5c"]

# Crear la figura
fig, ax4 = plt.subplots(figsize=(10, 6))

# Inicializar la acumulación de valores en 0
apil = 0

# Crear las barras apiladas con colores personalizados
for i, col in enumerate(des_2023_share.columns[1:]):  # Excluir el país
    bars = ax4.bar(des_2023_share['country'], des_2023_share[col],
                   bottom=apil, label=col.replace("_share_elec", "").capitalize(),
                   linewidth=2, color=colores[i % len(colores)])  # Asignar color

    # Agregar etiquetas con valores
    for bar, value in zip(bars, des_2023_share[col]):
        if value > 5:
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}%",
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Acumular valores para la siguiente iteración
    apil += des_2023_share[col]

# Etiquetas y formato
ax4.set_ylabel("Porcentaje de Generación Eléctrica")
ax4.set_title("Generación de Energía por Fuente en Países Desarrollados (2023)")
plt.xticks(rotation=45)

# Mover la leyenda fuera de la gráfica
ax4.legend(title="Fuente de Energía", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar diseño para evitar recortes
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# %% [markdown]
# Lineas de tendencia

# %%

# Pivotar el DataFrame para tener países como columnas
df_pivot = data_tip.pivot_table(values='biofuel_share_elec', index='year', columns='country')

# Crear gráfico de líneas
plt.figure(figsize=(10, 6))
df_pivot.plot(kind='line', marker='o', ax=plt.gca())
plt.xlabel('Año')
plt.ylabel('Electricidad generada por bioenergía (%)')
plt.title('Tendencia de Electricidad de Biocombustibles por País')
plt.legend(title='País')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Modelo predictivo
# 

# %%
# mostramos las columnas que tiene el dataframe:
data_col.columns

# %%
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



