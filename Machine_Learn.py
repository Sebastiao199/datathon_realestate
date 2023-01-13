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

st.set_page_config(layout="wide")

# Dashboard

image_map_area = Image.open('Images/area_mean_map.png')
image_map_area = image_map_area.resize((400, 400))

image_map_value2014 = Image.open('Images/value_2014_map.png')
image_map_value2014 = image_map_value2014.resize((400, 400))

image_map_value2022 = Image.open('Images/value_2022_map.png')
image_map_value2022 = image_map_value2022.resize((400, 400))

image_evolution_price = Image.open('Images/evolution_price.png')
image_evolution_price = image_evolution_price.resize((500, 400))

image_trans_depart = Image.open('Images/trans_depart_1.png')
image_trans_depart = image_trans_depart.resize((500, 400))

# Machine Learning

image_building = Image.open('Images/building.png')
image_building = image_building.resize((90, 80))

image_geo = Image.open('Images/map.png')
image_geo = image_geo.resize((90, 80))

image_sur = Image.open('Images/surface.png')
image_sur = image_sur.resize((90, 80))

image_rooms = Image.open('Images/nb_rooms.png')
image_rooms = image_rooms.resize((90, 80))

##ML alg
df_2022_ml[['Actual_built_surface','Nb_of_main_rooms']]=df_2022_ml[['Actual_built_surface','Nb_of_main_rooms']].astype(int)
X = df_2022_ml[['Actual_built_surface','Nb_of_main_rooms','Apartment','House','75 - Paris','77 - Seine-et-Marne','78 - Yvelines', "91 - l'Essonne", '92 - Hauts-de-Seine','93 - Seine-Saint-Denis', '94 - Val-de-Marne', "95 - Val-d'Oise"]]
y = df_2022_ml['Property_value']


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=29, train_size = 0.75)


scaler = StandardScaler()
# Your scaler model can now transform your data
X_train[['Actual_built_surface_sca', 'Nb_of_main_rooms_sca']] = scaler.fit_transform(X_train[['Actual_built_surface', 'Nb_of_main_rooms']])
X_test[['Actual_built_surface_sca', 'Nb_of_main_rooms_sca']] = scaler.fit_transform(X_test[['Actual_built_surface', 'Nb_of_main_rooms']])

X_train_sca = X_train[['Actual_built_surface_sca', 'Nb_of_main_rooms_sca', 'Apartment','House','75 - Paris','77 - Seine-et-Marne','78 - Yvelines', "91 - l'Essonne", '92 - Hauts-de-Seine','93 - Seine-Saint-Denis', '94 - Val-de-Marne', "95 - Val-d'Oise" ]]
X_test_sca = X_test[['Actual_built_surface_sca', 'Nb_of_main_rooms_sca', 'Apartment','House','75 - Paris','77 - Seine-et-Marne','78 - Yvelines', "91 - l'Essonne", '92 - Hauts-de-Seine','93 - Seine-Saint-Denis', '94 - Val-de-Marne', "95 - Val-d'Oise" ]]


dtr = DecisionTreeRegressor(random_state= 39, min_samples_split = 7, min_samples_leaf= 46, max_depth = 12)
dtr.fit(X_train_sca, y_train)

y_pred_train = dtr.predict(X_train_sca)
y_pred_test = dtr.predict(X_test_sca)

## end alg


tab1, tab2 = st.tabs(["Dashboard", "Machine Learning"])

with tab1:
    st.title("Insights on the real estate market in Ile-de-France")
    st.subheader("*Difference between property values between 2014 and 2022*")    
    
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Map of the property value in 2014**")
        st.image(image_map_value2014)
    with col2:
        st.markdown("**Map of the property value in 2022**")
        st.image(image_map_value2022)        
    

    
    st.subheader("*The impact of covid-19 on the real estate market in Ile-de-France*")    
        
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Evolution of the price per square meter between 2014 and 2022**")
        st.image(image_evolution_price)
    with col2:
        st.markdown("**Evolution of the transactions per year by departments between 2014 and 2022**")
        st.image(image_trans_depart)
 
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Which departement has the biggest/smallest Surface Area?**")
        st.image(image_map_area)
    with col2:
        st.empty()
    
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
        st.image(image_building)
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
        

        #X_list = list([Paris_75,Seine_et_Marne_77,Yvelines_78,Essonne_91,Hauts_de_Seine_92,Seine_Saint_Denis_93,Val_de_Marne_94,Val_dOise_95, Apartment,
         #              House,Actual_built_surface, Nb_of_main_rooms])
        
        #X_scaled_list = scaler.transform(np.array(X_list).reshape(1,-1))
        #newhouse_prediction = dtr.predict(X_scaled_list).astype(int)
        
        
        X_array = np.array([Actual_built_surface, Nb_of_main_rooms, Apartment, House, Paris_75, Seine_et_Marne_77, Yvelines_78, Essonne_91, Hauts_de_Seine_92, Seine_Saint_Denis_93, Val_de_Marne_94, Val_dOise_95]).reshape(1,12)
                       
        X_df = pd.DataFrame(X_array, columns=['Actual_built_surface', 'Nb_of_main_rooms', 'Apartment', 'House','Paris_75','Seine_et_Marne_77','Yvelines_78','Essonne_91','Hauts_de_Seine_92', 'Seine_Saint_Denis_93','Val_de_Marne_94','Val_dOise_95'])
        
        X_df[['Actual_built_surface','Nb_of_main_rooms']] = scaler.transform(X_df[['Actual_built_surface','Nb_of_main_rooms']])
        X_df[['Paris_75','Seine_et_Marne_77','Yvelines_78','Essonne_91','Hauts_de_Seine_92','Seine_Saint_Denis_93','Val_de_Marne_94','Val_dOise_95', 'Apartment', 'House']] = X_df[['Paris_75','Seine_et_Marne_77','Yvelines_78','Essonne_91','Hauts_de_Seine_92','Seine_Saint_Denis_93','Val_de_Marne_94','Val_dOise_95', 'Apartment', 'House']].astype(int)


        newhouse_prediction = dtr.predict(X_df).astype(int)
        newhouse_prediction = np.array([newhouse_prediction])[0]
        newhouse_prediction = newhouse_prediction[0]
        
    if st.button('Click to see the price'):
        st.subheader('The price is:')
        #newhouse_prediction1
        st.subheader("{:,} Euros".format(newhouse_prediction))
        #st.write(f"This is the predicted price {newhouse_prediction}") 