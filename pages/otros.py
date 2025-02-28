# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

# Cargamos el dataset original
data_energy = pd.read_csv("D:/0_Edwin_Duque/AnalisisDatosIntegrador/Proyect_Energy/Data/owid_energy_data.csv", index_col=None)

# %%
# Ruta al archivo .shapefile descargado de Natural Earth
#shapefile_path_world = r"C:\Users\edwin\Downloads\ne_50m_admin_0_countries.shp"

# Cargar el Shapefile con GeoPandas
world = gpd.read_file("D:/0_Edwin_Duque/AnalisisDatosIntegrador/Proyect_Energy/Data/ne_50m_admin_0_countries.shp")

# %%
# Estandarizar los nombres de los países en el Shapefile (mayúsculas y sin espacios
world["ADMIN"] = world["ADMIN"].str.upper().str.strip()
# Generar Mapa base del mundo
world.plot(figsize=(15, 13), edgecolor='black', facecolor='none')
plt.title('Mapa base del mundo', fontsize=14)
plt.axis('off')
plt.show()

# %%
world

# %%
data_energy_2023 = data_energy[data_energy['year'] == 2023]
data_energy_2023 = data_energy_2023[['iso_code', 'renewables_share_elec']]
#.filter(regex="iso_code|renewables_share_elec", axis=1)

#convierto la la columna world['ISO_A3'] en una lista
iso_codes = world['ISO_A3'].tolist()

# Seleccionamos aquellos paises que estan en la lista creada
data_energy_2023 = data_energy_2023[data_energy_2023['iso_code'].isin(iso_codes)]
data_energy_2023.reset_index(drop=True, inplace=True)

data_energy_2023


# %%
#Porcentaje de datos faltantes para cada variable del dataset
total           = data_energy_2023.isnull().sum().sort_values(ascending=False)
percent         = (data_energy_2023.isnull().sum()/data_energy_2023.isnull().count()*100).sort_values(ascending=False)
missing_train1  = pd.concat([total,percent],axis=1,keys=["Total","Percent"])
missing_train1

# %%
world_filt = world[['ADMIN', 'ISO_A3', 'geometry']]

# Seleccionamos aquellos paises que estan en la lista creada
world_filt.reset_index(drop=True, inplace=True)
world_filt = world_filt.rename(columns={'ISO_A3': 'iso_code'})

world_filt

# %%
world_energy = world_filt.merge(data_energy_2023, on='iso_code', how='left')

# %%
world_energy

# %%

# Crear la figura y los ejes
fig, ax = plt.subplots(1, 1, figsize=(18, 16))

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
plt.show()



