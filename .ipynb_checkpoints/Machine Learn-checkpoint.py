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

df_2022_ml = pd.read_csv('https://raw.githubusercontent.com/Sebastiao199/datathon_realestate/main/df_2022_ml_st.csv')

##ML alg

X = df_2022_ml[['Actual_built_surface','Nb_of_main_rooms','Apartment','House','75 - Paris','77 - Seine-et-Marne','78 - Yvelines', "91 - l'Essonne", '92 - Hauts-de-Seine','93 - Seine-Saint-Denis', '94 - Val-de-Marne', "95 - Val-d'Oise"]]
y = df_2022_ml['Property_value']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=29, train_size = 0.75)

# Create and fit a scaler model
scaler = StandardScaler().fit(X)
# Your scaler model can now transform your data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


dtr = DecisionTreeRegressor(random_state= 39, min_samples_split = 7, min_samples_leaf= 46, max_depth = 12)
dtr.fit(X_train_scaled, y_train)

y_pred = dtr.predict(X_test_scaled)

## end alg


tab1, tab2 = st.tabs(["Dashboard", "Machine Learning"])

with tab1:
    st.title('The unicorns of the world')

    
with tab2:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Where is the best neighbourhood to invest?')
    col1,col2 = st.columns(2)
    with col1:
        ### Insert Filter 
        #ChooseDep1 = st.selectbox('Choose your department:', df_2022_ml['Country'].unique(), 1)
        ChooseDep1 = st.radio("Choose your department:",('75 - Paris','77 - Seine-et-Marne','78 - Yvelines',"91 - l'Essonne",'92 - Hauts-de-Seine','93 - Seine-Saint-Denis',
                                                                    '94 - Val-de-Marne', "95 - Val-d'Oise"))
        
        ChooseType1 = st.radio("Choose your department:",('Appartment','House'))
        
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
        Actual_built_surface = st.number_input('Choose a surface area')
        Nb_of_main_rooms = st.number_input('Choose the number of main rooms')


        X_list = list([Paris_75,Seine_et_Marne_77,Yvelines_78,Essonne_91,Hauts_de_Seine_92,Seine_Saint_Denis_93,Val_de_Marne_94,Val_dOise_95, Apartment,
                       House,Actual_built_surface, Nb_of_main_rooms])

        newhouse_prediction = dtr.predict(np.array(X_list).reshape(1,-1))
        newhouse_prediction = newhouse_prediction.style.format({'newhouse_prediction': ':.0f'})
    with col2:
        st.write(f"This is the predicted price {newhouse_prediction}") 