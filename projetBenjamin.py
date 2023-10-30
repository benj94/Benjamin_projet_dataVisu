
# In[0]:
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os
import geopandas as gpd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.pyplot import pie, axis, show
import streamlit as st
sns.set()
# In[1]:
data1 = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7", delimiter=";")
data2 = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/a6ef711a-1f03-44cb-921a-0ce8ec975995", delimiter=";")
data3 = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8", delimiter=";")
data4 = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/c9742921-4427-41e5-81bc-f13af8bc31a0", delimiter=";")

st.sidebar.markdown("[GitHub](https://github.com/benj94/benben)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/feed/)")

data1.info()
# In[2]:
data1 = data1.drop(['adr'], axis=1)

data1['long'] = data1['long'].str.replace(',', '.', regex=True).astype(float)
data1['lat'] = data1['lat'].str.replace(',', '.', regex=True).astype(float)

data1.info()
# In[3]:
data2 = data2.drop(['voie','v1','v2','pr','pr1','lartpc'], axis=1)
data2.info()
data2 = data2.rename(columns={'Num_Acc': 'Accident_Id'})
# In[4]:
data3 = data3.rename(columns={'Num_Acc': 'Accident_Id'})
print(data3)
# In[5]:
merged_data_1 = pd.merge(data1, data2, on='Accident_Id', how='inner')
print(merged_data_1)
# In[6]:
print(data3.duplicated())
print(data3[['Accident_Id','grav']])

merged_data = pd.merge(merged_data_1, data3[['Accident_Id','grav']], on='Accident_Id', how='inner')
merged_data.isnull().sum()
#print(merged_data)
# In[7]

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

st.markdown('<a name="top-section"></a>',  unsafe_allow_html=True)

st.markdown("<h1> Les accidents de la route: Un fléau annuel </h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center'>
        <h2>Analyse des accidents de la route pour déterminer où concentrer nos efforts afin de les réduire</h2>
    </div>
""", unsafe_allow_html=True)

print("\n")

mappings = {-1: 'NC',
            1: 'Homme',
            2: 'Femme'}
# Remplacez les codes par leurs descriptions
data3['sexe'] = data3['sexe'].replace(mappings)

from bokeh.plotting import figure
st.header("Quel genre est le plus impliqué dans les accidents ?")
st.markdown('<a name="first-section"></a>',  unsafe_allow_html=True)

def accidentgender(data3):
    # Ici on retire les lignes avec la valeur "NC" dans la colonne "sexe".
    df = data3[data3['sexe'] != 'NC']
    # Opération de group by par sexe et comptez les occurrences d'Accident_Id.
    grouped_data = df.groupby('sexe')['Accident_Id'].count().reset_index()
    # Création du graphe à barres empilées.
    p = figure(x_range=grouped_data['sexe'], 
            toolbar_location=None, tools="")
    # Création des barres empilées pour la variable "Accident_Id".
    p.vbar(x='sexe', top='Accident_Id', source=grouped_data, width=0.5, color="blue")
    p.xaxis.axis_label = "Sexe"
    p.yaxis.axis_label = "Nombre d'accidents"
    st.bokeh_chart(p)
accidentgender(data3)


st.markdown('<a name="second-section"></a>',  unsafe_allow_html=True)

def infrastructure(data2):

    import pandas as pd
    import plotly.express as px


    st.write("Légende:")
    st.write("- 1 : Souterrain tunnel")
    st.write("- 2 : Pont autopont")
    st.write("- 3 : bretelle d'échargeur ")
    st.write("- 4 : Voie ferrée")
    st.write("- 5 : Carrefour aménagé")
    st.write("- 6 : Zone piétonne")
    st.write("- 7 : Zone de péage ")
    st.write("- 8 : Chantier")
    st.write("- 9 : Autres")
    data2 = data2[data2['infra'] != 0]
    data2 = data2[data2['infra'] != -1]
    counts = data2['infra'].value_counts()

    # Création d'un pie plot
    fig = px.pie(names=counts.index, values=counts, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Pour séparer les secteurs légèrement

    # Titre du graphique.
    fig.update_layout(title_text='Répartition des accidents par infrastructure')

    # Affichez le graphique
    st.plotly_chart(fig)
infrastructure(data2)

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

st.markdown('<a name="third-section"></a>',  unsafe_allow_html=True)
st.header("Localisation des accidents en France métropolitaine")
def locaAccident(dataMetropole):

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

    st.bokeh_chart(p)

locaAccident(dataMetropole)

dataTue = dataMetropole[dataMetropole['grav'] == 'Tué']



# Fonction pour extraire l'heure à partir de l'heure:minutes
def extract_hour(time):
    hour, _ = time.split(':')
    return int(hour)

# On applique la fonction à la colonne 'Heures'
merged_data['heure min'] = merged_data['heure min'].apply(extract_hour)
#print(merged_data)

st.header("Accidents en Agglomération vs. Hors Agglomération par heure")
st.markdown('<a name="fourth-section"></a>',  unsafe_allow_html=True)

#---------------------------------- histograme qui compare les accident hors et dans l'agglomération en fonction de l'heure----------------------

def hist(merged_data):
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
    plt.legend()
    st.pyplot(plt)
hist(merged_data)


st.header("Quels sont les départements où il y a le plus d'accident")
st.markdown('<a name="eleventh"></a>',  unsafe_allow_html=True)

# Charger les données géospatiales des départements
url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
departements = gpd.read_file(url)



url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
departements = gpd.read_file(url)
 
def map(map_data):
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
    st.plotly_chart(fig)
#--------------------------------------------------------------- Map permettant de voir le npmbre d'accident par département dans la métropole--------------------------

st.subheader("Moyenne d'Accident par département en France métropolitaine")
st.markdown('<a name="fith-section"></a>',  unsafe_allow_html=True)

def map(map_data):
    # Créer une carte choroplèthe avec Plotly Express
    fig = px.choropleth(map_data,
        geojson=departements,  
        locations='dep',  
        featureidkey="properties.code",  
        color='Accident_Id',
        color_continuous_scale=[(0, 'white'), (0.5, 'green'), (1, 'blue')]
        )

    fig.update_geos(projection_type="mercator")
    fig.update_geos(
        projection_scale=25,  
        # Centrer sur la France
        center={"lon": 2.0, "lat": 47.0} 
    )

    # Afficher la carte
    st.plotly_chart(fig)


map(map_data)

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

# Remplacement des codes par leurs descriptions
merged_data['type de route'] = merged_data['type de route'].replace(mappings)
# Sélection des colonnes 'type de route' et 'surf'
subset_data = merged_data[['type de route', 'grav']]
# On fait un groupement pour compter le nombre d'occurrences de chaque combinaison de valeurs
grouped_data = subset_data.groupby(['type de route', 'grav']).size().unstack().fillna(0)


#---------------------------------------heatmap de la gravité de l'accident en fonction du type de route utilisé-------------------

st.subheader("Relation entre le Type de Route et la gravité de l'accident")
st.markdown('<a name="sixth-section"></a>',  unsafe_allow_html=True)

def heatmap (grouped_data):
    # création d'une heatmap
    fig = px.imshow(grouped_data, x=grouped_data.columns, y=grouped_data.index,
                    color_continuous_scale=[(0, 'white'), (0.5, 'green'), (1, 'blue')])
    fig.update_xaxes(side='top')  # Pour afficher l'axe des surfaces en haut
    st.plotly_chart(fig)

heatmap(grouped_data)




# Filtrer le DataFrame pour ne conserver que les catégories 'Tué' et 'Blessé hospitalisé'
filtered_data = merged_data[merged_data['grav'].isin(['Tué', 'Blessé hospitalisé'])]
# Sélection des colonnes 'type de route' et 'grav'
subset_data = filtered_data[['type de route', 'grav']]
# Groupement pour compter le nombre d'occurrences de chaque combinaison de valeurs
grouped_data = subset_data.groupby(['type de route', 'grav']).size().unstack().fillna(0)



#-------------------------------------------- Heatmap de la Relation entre le Type de Route et la Gravité des Accidents (Tué et Blessé hospitalisé)--------------------------------------------

st.subheader("Relation entre le Type de Route et la Gravité des Accidents (Tué et Blessé hospitalisé)")
st.markdown('<a name="seventh-section"></a>',  unsafe_allow_html=True)

def heatmap2(grouped_data): 
    fig = px.imshow(grouped_data, x=grouped_data.columns, y=grouped_data.index,
                    color_continuous_scale=[(0, 'white'), (0.5, 'green'), (1, 'blue')])
    fig.update_xaxes(side='top')  # Pour afficher l'axe des surfaces en haut

    # Affichez la heatmap
    st.plotly_chart(fig)

heatmap2(grouped_data)


#----------------------------------- lineplot de la gravité des accidents en fonction de l'age de la victime ---------------------------------

st.subheader("L'âge joue t-elle sur la conduite ?")
st.markdown('<a name="eighth-section"></a>',  unsafe_allow_html=True)

def gravAge(merged_data):

    merged_data = pd.merge(data1, data2, on='Accident_Id', how='inner')
    data_usa = pd.merge(merged_data, data3, on='Accident_Id', how='inner')
    annee_actuelle = pd.Timestamp.now().year
    data_usa['age'] = annee_actuelle - data_usa['an_nais']
    st.subheader("Etat des personnes en fonction de leur âge ")
    st.write("Légende:")
    st.write("- 1 : indemne")
    st.write("- 2 : tué")
    st.write("- 3 : blessé hospitalisé")
    st.write("- 4 : blessé léger")
    st.write("Nombre de cas qui associe grave et l'age ")
    st.write("Age")
    st.line_chart(data_usa.groupby(['age', 'grav']).size().unstack(), use_container_width=True)
gravAge(merged_data)

#-------------------------------------- pieplot pourcentage d'accident par infrastructure et aménagement ---------------------------------------

st.subheader("Doit-t-on revoir certains aménagements et infrastructure ?")
st.markdown('<a name="nineth-section"></a>',  unsafe_allow_html=True)

def infrastructure(data2):


    st.write("Légende:")
    st.write("- 1 : Souterrain tunnel")
    st.write("- 2 : Pont autopont")
    st.write("- 3 : bretelle d'échargeur ")
    st.write("- 4 : Voie ferrée")
    st.write("- 5 : Carrefour aménagé")
    st.write("- 6 : Zone piétonne")
    st.write("- 7 : Zone de péage ")
    st.write("- 8 : Chantier")
    st.write("- 9 : Autres")
    data2 = data2[data2['infra'] != 0]
    data2 = data2[data2['infra'] != -1]
    counts = data2['infra'].value_counts()

    # Création d'un pie plot
    fig = px.pie(names=counts.index, values=counts, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Pour séparer les secteurs légèrement

    # Titre du graphique.
    fig.update_layout(title_text='Répartition des accidents par infrastructure/aménagement')
    st.plotly_chart(fig)



#----------------------------- skroll bar pour séléctionner un scatter plot en fonction du type de la gravité de l'accident-------------------

databless = dataMetropole[dataMetropole['grav'] == 'Blessé hospitalisé']
dataindem = dataMetropole[dataMetropole['grav'] == 'Indemne']
datablessleg = dataMetropole[dataMetropole['grav'] == 'Blessé léger']
categories = ['Blessé hospitalisé']

st.subheader("Localisation des accidents par type de gravité")
st.markdown('<a name="tenth-section"></a>',  unsafe_allow_html=True)

def garphe(df):  
    GRAV = df["grav"].unique()
    MARKERS = [ 'triangle']
    
    
    p = figure(background_fill_color="#fafafa")
    p.xaxis.axis_label = 'long'
    p.yaxis.axis_label = 'lat'
    
    couleurs = ['orange']
    
    p.scatter("long", "lat", source=df,
    
            legend_group="grav", fill_alpha=0.4, size=5,
            marker=factor_mark('grav', MARKERS, GRAV),
            color=factor_cmap('grav', couleurs, GRAV))
    
    
    
    p.legend.location = "top_left"
    p.legend.title = "grav"
    
    
    
    st.bokeh_chart(p)


categories = ['Blessé hospitalisé', "Tué", "Blessé léger", "Indemne"] 
st.header('Localisation des accidens par gravité')
selected_category = st.selectbox("Sélectionnez un type d'essence", categories)

if selected_category == 'Blessé hospitalisé':
    garphe(databless)
elif selected_category == 'Tué':
    garphe(dataTue)
elif selected_category == 'Blessé léger':
    garphe(datablessleg)
elif selected_category == 'Indemne':
    garphe(dataindem)


with st.sidebar:
    st.header('#datavz2023efrei')
    st.sidebar.text("Benjamin")
    st.sidebar.text("Rousseau")
    st.sidebar.text("Promo 2025")
    st.sidebar.text("Classe : BIA2")

st.sidebar.markdown("[Les accidents de la routes](#top-section)")
st.sidebar.markdown("[Quel genre est le plus impliqué dans les accidents ?](#first-section)")
st.sidebar.markdown("[Répartition des accidents par infrastructure](#second-section)")
st.sidebar.markdown("[Localisation des accidents en France métropolitaine](#third-section)")
st.sidebar.markdown("[Nombre d'accident par département](#eleventh-section)")
st.sidebar.markdown("[Moyenne d'Accident par département en France métropolitaine](#fith-section)")
st.sidebar.markdown("[Relation entre le Type de Route et la gravité de l'accident](#sixth-section)")
st.sidebar.markdown("[Tué et Blessé hospitalisé](#seventh-section)")
st.sidebar.markdown("[L'âge joue t-elle sur la conduite ?](#eighth-section)")
st.sidebar.markdown("[Doit-t-on revoir certains aménagements et infrastructure ?](#nineth-section)")
st.sidebar.markdown("[Doit-t-on revoir certains aménagements et infrastructure ?](#tenth-section)")
