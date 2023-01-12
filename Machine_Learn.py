### Import Libraries 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
#import plotly.express as px

final_geo = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/datathon_realestate/main/final_geo.csv')
df_2022_ml = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/datathon_realestate/main/df_2022_ml_st.csv')

# Dashboard


image_map_area = Image.open('Images/area_mean_map.png')
image_map_area = image_map_area.resize((400, 400))

image_map_value = Image.open('Images/value_mean_map.png')
image_map_value = image_map_value.resize((400, 400))

image_evolution_price = Image.open('Images/evolution_price.png')
image_evolution_price = image_evolution_price.resize((400, 400))

image_trans_depart = Image.open('Images/trans_depart.png')
image_trans_depart = image_trans_depart.resize((400, 400))


# Machine Learning


image_house = Image.open('house_apt_picto.png')
image_house = image_house.resize((90, 80))

image_geo = Image.open('geo.png')
image_geo = image_geo.resize((90, 80))

image_sur = Image.open('surface.png')
image_sur = image_sur.resize((90, 80))

image_rooms = Image.open('rooms.png')
image_rooms = image_rooms.resize((90, 80))

##ML alg

X = df_2022_ml[['Actual_built_surface','Nb_of_main_rooms','Apartment','House','75 - Paris','77 - Seine-et-Marne','78 - Yvelines', "91 - l'Essonne", '92 - Hauts-de-Seine','93 - Seine-Saint-Denis', '94 - Val-de-Marne', "95 - Val-d'Oise"]]
y = df_2022_ml['Property_value']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=29, train_size = 0.75)

# Create and fit a scaler model
scaler = StandardScaler().fit(X)
# Your scaler model can now transform your data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


dtr = DecisionTreeRegressor(random_state= 39, min_samples_split = 7, min_samples_leaf= 46, max_depth = 12).fit(X_train_scaled, y_train)

y_pred = dtr.predict(X_test_scaled)

## end alg


tab1, tab2 = st.tabs(["Dashboard", "Machine Learning"])

with tab1:
    #st.title("")
    col1,col2 = st.columns(2)
    with col1:
        #st.title("")
        st.image(image_map_area)
    with col2:
        #st.title("")
        st.image(image_evolution_price)
    col1,col2 = st.columns(2)
    with col1:
        #st.title("")
        st.image(image_map_value)
    with col2:
        #st.title("")
        st.image(image_trans_depart)
    
    
with tab2:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Find the price of your dream place")
    col1,col2 = st.columns(2)
    with col1:
        ### Insert Filter 
        st.image(image_geo)
        ChooseDep1 = st.radio("Choose the department you want:",('75 - Paris','77 - Seine-et-Marne','78 - Yvelines',"91 - l'Essonne",'92 - Hauts-de-Seine','93 - Seine-Saint-Denis',
                                                                    '94 - Val-de-Marne', "95 - Val-d'Oise"))
    with col2:
        st.image(image_house)
        ChooseType1 = st.radio("Choose type of place you want:",('Appartment','House'))
        
    col1,col2 = st.columns(2)
    with col1:
        st.image(image_sur)
        Actual_built_surface = st.number_input('Choose a surface area:')
    with col2:
        st.image(image_rooms)
        Nb_of_main_rooms = st.number_input('Choose the number of main rooms:')
        
        
        Paris_75 = int(ChooseDep1=='75 - Paris')
        Seine_et_Marne_77 = int(ChooseDep1=='77 - Seine-et-Marne')
        Yvelines_78 = int(ChooseDep1=='78 - Yvelines')	
        Essonne_91= int(ChooseDep1=="91 - l'Essonne")
        Hauts_de_Seine_92 = int(ChooseDep1=='92 - Hauts-de-Seine')
        Seine_Saint_Denis_93 = int(ChooseDep1=='93 - Seine-Saint-Denis')
        Val_de_Marne_94 =int(ChooseDep1=='94 - Val-de-Marne')
        Val_dOise_95 =int(ChooseDep1=="95 - Val-d'Oises")
        Apartment = int(ChooseDep1=='Appartment')
        House = int(ChooseDep1=='House')
        

        X_list = list([Paris_75,Seine_et_Marne_77,Yvelines_78,Essonne_91,Hauts_de_Seine_92,Seine_Saint_Denis_93,Val_de_Marne_94,Val_dOise_95, Apartment,
                       House,Actual_built_surface, Nb_of_main_rooms])

        newhouse_prediction = dtr.predict(np.array(X_list).reshape(1,-1)).astype(int)
        
    if st.button('Click to see the price'):
        st.subheader('The price is:')
        st.subheader(["{:,}".format(x) for x in newhouse_prediction])
        #st.write(f"This is the predicted price {newhouse_prediction}") 