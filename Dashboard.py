### Import Libraries 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np


df_2022_ml = pd.read_csv('Documents/GitHub/datathon_realestate/df_2022_ml_st.csv')

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
        ChooseDep1 = st.radio("Choose your department:",df_2022_ml[['75 - Paris','77 - Seine-et-Marne',	'78 - Yvelines',"91 - l'Essonne",	'92 - Hauts-de-Seine',	'93 - Seine-Saint-Denis',
                                                                    '94 - Val-de-Marne',	"95 - Val-d'Oise"]])

    
