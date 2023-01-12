### Import Libraries 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np


df_2022_ml = pd.read_csv()

tab1, tab2 = st.tabs(["Dashboard", "Machine Learning"])

with tab1:
    st.title('The unicorns of the world')

    
with tab2:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Where is the best neighbourhood to invest?')
    col1,col2 = st.columns(2)
    with col1:
        ### Insert Filter 
        ChooseCountry1 = st.selectbox('Choose your country:', unicorns['Country'].unique(), 1)
        
    
