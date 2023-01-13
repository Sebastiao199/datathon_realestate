# Datathon - Global analysis of real estate in Ile-de-France 
Team: Sebastião Oliveira, Juliette Hanau, Ana Quintino, Francisco Câmara

# Context 
We merged several datasets from the French Public Finance Department open data (governmental institution) to get the data on the French real estate from 2014 to 2022.
Link to the data: https://www.data.gouv.fr/fr/datasets/5c4ae55a634f4117716d5656/#description
We decided to focus on the houses and apartments of the 8 departments in Ile-de-France ((Paris (75), Hauts-de-Seine (92), Seine-Saint-Denis (93), Val-de-Marne (94), Seine-et-Marne (77), Yvelines (78), Essonne (91), Val-d'Oise (95))

# Deliverables 
1) Insights with plots and maps 

2) A price prediction system using Maching learning:
More information below

# Pre-processing & Cleaning explanation
Not a lot of cleaning to do, but there were some duplicates in the datasets and missing values for 2022. 
 
# Machine Learning explanation 
We used the Decision Tree, a Supervised machine learning algorithm that can be used for Regression.
Our goal was to target the price of the property, based on:
- if it's an appartement / house
- the surface
- the departement of Ile-de-France 
- the number of the main rooms
