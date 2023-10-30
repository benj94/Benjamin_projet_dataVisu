
# In[0]:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import geopandas as gpd 
import matplotlib as mlt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import datetime as dt
from matplotlib.pyplot import pie, axis, show
import streamlit as st
sns.set()
# In[1]:

data1 = pd.read_csv("carcteristiques-2022.csv", delimiter=";")
data2 = pd.read_csv("lieux-2022.csv", delimiter=";" )
data3 = pd.read_csv("usagers-2022.csv", delimiter=";" )
data4 = pd.read_csv("vehicules-2022.csv", delimiter=";" )

data1.info()
# In[19]:
data1 = data1.drop(['adr'], axis=1)

data1['long'] = data1['long'].str.replace(',', '.', regex=True).astype(float)
data1['lat'] = data1['lat'].str.replace(',', '.', regex=True).astype(float)

data1.info()
# In[19]:
data2 = data2.drop(['voie','v1','v2','pr','pr1','lartpc'], axis=1)
data2.info()
data2 = data2.rename(columns={'Num_Acc': 'Accident_Id'})
# In[19]:
data3 = data3.rename(columns={'Num_Acc': 'Accident_Id'})
print(data3)
# In[19]:
merged_data_1 = pd.merge(data1, data2, on='Accident_Id', how='inner')
print(merged_data_1)
# In[19]:
print(data3.duplicated())
print(data3[['Accident_Id','grav']])

merged_data = pd.merge(merged_data_1, data3[['Accident_Id','grav']], on='Accident_Id', how='inner')
print(merged_data)

data3

merged_data.isnull().sum()


nouveaux_noms = {
    'hrmn': 'heure min',
    'int': 'intersection',
    'atm': 'meteo',
    'col': 'type colision',
    'catr': 'type de route',
    'nbv': 'nombre de voie',
    'vosp': 'voie reserve',
    'prof': 'profil de la route',
    'larrout': 'largeur chausse',
    'situ': 'situation accident',
    'lum': 'luminosite',
}


merged_data = merged_data.rename(columns=nouveaux_noms)
merged_data.info()


# Création d'un dictionnaire pour associer les codes aux descriptions de la colonne luminosite pour plus de clarté
mappings = {1: 'Plein jour',
            2: 'Crépuscule ou aube',
            3: 'Nuit sans éclairage public',
            4: 'Nuit avec éclairage public non allumé',
            5: 'Nuit avec éclairage public allumé'}

# Remplacement des codes par les descriptions
merged_data['luminosite'] = merged_data['luminosite'].replace(mappings)
print(merged_data)


# Filtrer les données pour conserver uniquement les valeurs de VMA supérieures ou égales à 130
filtered_data = merged_data[merged_data['vma'] <= 130]
# Créez un histogramme de la vitesse maximale autorisée (VMA) à partir des données filtrées
fig = px.histogram(filtered_data, x='vma', title="Histogramme du nombre d'accidents en fonction de la vitesse limite (VMA >= 130)")
# Personnalisez l'histogramme (si nécessaire)
fig.update_xaxes(title='VMA')
fig.update_yaxes(title='accidents')
fig.update_traces(marker_color='blue') 
# Affichez l'histogramme
fig.show()


# Création d'un dictionnaire pour associer les codes aux descriptions de la colonne grav pour plus de clarté
mappings = {1:'Indemne',
2:'Tué',
3:'Blessé hospitalisé',
4: 'Blessé léger',
-1:'NC'}
# Remplacement des codes par les descriptions
merged_data['grav'] = merged_data['grav'].replace(mappings)
print(merged_data)

dataMetropole = merged_data[merged_data['dep'].str.len() <= 2]

from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, factor_mark
from bokeh.io import output_notebook

output_notebook()

GRAV = dataMetropole["grav"].unique()
MARKERS = ['hex', 'triangle']


p = figure(background_fill_color="#fafafa")
p.xaxis.axis_label = 'long'
p.yaxis.axis_label = 'lat'

couleurs = ['orange','blue','green','red']

p.scatter("long", "lat", source=dataMetropole,

          legend_group="grav", fill_alpha=0.4, size=2,
          marker=factor_mark('grav', MARKERS, GRAV),
          color=factor_cmap('grav', couleurs, GRAV))

 

p.legend.location = "top_left"
p.legend.title = "grav"

 

show(p)



dataTue = dataMetropole[dataMetropole['grav'] == 'Tué']


from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, factor_mark
from bokeh.io import output_notebook

output_notebook()

GRAV = dataTue["grav"].unique()
MARKERS = ['hex', 'triangle']


p = figure(background_fill_color="#fafafa")
p.xaxis.axis_label = 'long'
p.yaxis.axis_label = 'lat'

couleurs = ['red']

p.scatter("long", "lat", source=dataTue,

          legend_group="grav", fill_alpha=0.4, size=5,
          marker=factor_mark('grav', MARKERS, GRAV),
          color=factor_cmap('grav', couleurs, GRAV))

 

p.legend.location = "top_left"
p.legend.title = "grav"

 

show(p)


import pandas as pd

# Fonction pour extraire l'heure à partir de l'heure:minutes
def extract_hour(time):
    hour, _ = time.split(':')
    return int(hour)

# Appliquez la fonction à la colonne 'Heures'
merged_data['heure min'] = merged_data['heure min'].apply(extract_hour)

print(merged_data)



import pandas as pd
import matplotlib.pyplot as plt

# Filtrer les données pour les accidents en agglomération
accidents_agglomeration = merged_data[merged_data['agg'] == 2]

# Filtrer les données pour les accidents hors agglomération
accidents_hors_agglomeration = merged_data[merged_data['agg'] == 1]

# Créer un histogramme pour les accidents en agglomération
plt.hist(accidents_agglomeration['heure min'], bins=24, alpha=0.5, label='En Agglo', color='blue')

# Créer un histogramme pour les accidents hors agglomération
plt.hist(accidents_hors_agglomeration['heure min'], bins=24, alpha=0.5, label='Hors Agglo', color='red')
plt.xlabel('Heures de la journée')
plt.ylabel('Nombre d\'accidents')
plt.title('Accidents en Agglomération vs. Hors Agglomération par heure')
plt.legend()
plt.show()

merged_data.info()
merged_data.axes


pd.set_option('display.max_rows', 150) 
gdf = gpd.read_file("contour-des-departements.geojson")
print(gdf)

gdf.to_csv('gdf1.csv', index=False)
gdf= gdf.rename(columns={'code': 'dep'})
map_data = pd.merge(merged_data, gdf, on='dep', how='inner')
map_data2 = merged_data.groupby('dep')['Accident_Id'].count()
map_data2

map_data = pd.merge(map_data2, gdf, on='dep', how='inner')
map_data


mappings = {
    1 : 'Autoroute',
    2 : 'Route nationale',
    3 : 'Route Départementale',
    4 : 'Voie Communales',
    5 : 'Hors réseau public',
    6 : 'Parc de stationnement ouvert à la circulation publique',
    7 : 'Routes de métropole urbaine',
    9 : 'autre'
}

# Remplacez les codes par leurs descriptions
merged_data['type de route'] = merged_data['type de route'].replace(mappings)





# Sélectionnez les colonnes 'type de route' et 'surf'
subset_data = merged_data[['type de route', 'grav']]

# Effectuez un groupement pour compter le nombre d'occurrences de chaque combinaison de valeurs
grouped_data = subset_data.groupby(['type de route', 'grav']).size().unstack().fillna(0)

# Utilisez Plotly Express pour créer une heatmap
fig = px.imshow(grouped_data, x=grouped_data.columns, y=grouped_data.index,
                color_continuous_scale=[(0, 'white'), (0.5, 'green'), (1, 'blue')],
                 title='Heatmap de la Relation entre le Type de Route et la Surface')

# Personnalisez le graphique (facultatif)
fig.update_xaxes(side='top')  # Pour afficher l'axe des surfaces en haut

# Affichez la heatmap
fig.show()


# Filtrer le DataFrame pour ne conserver que les catégories 'Tué' et 'Blessé hospitalisé'
filtered_data = merged_data[merged_data['grav'].isin(['Tué', 'Blessé hospitalisé'])]

# Sélectionnez les colonnes 'type de route' et 'grav'
subset_data = filtered_data[['type de route', 'grav']]

# Effectuez un groupement pour compter le nombre d'occurrences de chaque combinaison de valeurs
grouped_data = subset_data.groupby(['type de route', 'grav']).size().unstack().fillna(0)

# Utilisez Plotly Express pour créer une heatmap avec une palette de couleurs personnalisée
fig = px.imshow(grouped_data, x=grouped_data.columns, y=grouped_data.index,
                color_continuous_scale=[(0, 'white'), (0.5, 'green'), (1, 'blue')],
                title='Heatmap de la Relation entre le Type de Route et la Gravité des Accidents (Tué et Blessé hospitalisé)')

# Personnalisez le graphique (facultatif)
fig.update_xaxes(side='top')  # Pour afficher l'axe des surfaces en haut

# Affichez la heatmap
fig.show()


import geopandas as gpd
import pandas as pd
import plotly.express as px

# Charger les données géospatiales des départements
url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
departements = gpd.read_file(url)


# Créer une carte choroplèthe avec Plotly Express
fig = px.choropleth(map_data,
    geojson=departements,  
    locations='dep',  
    featureidkey="properties.code",  
    color='Accident_Id',
    color_continuous_scale=[(0, 'white'), (0.5, 'green'), (1, 'blue')],
    title="Moyenne d'Accident par département en France métropolitaine")

fig.update_geos(projection_type="mercator")
fig.update_geos(
    projection_scale=25, 
    center={"lon": 2.0, "lat": 47.0}  # Centrer sur la France
)

# Afficher la carte
fig.show()
# %%
