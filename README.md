# Datathon - Global analysis of real estate in Ile-de-France 
Team: Sebastião Oliveira, Juliette Hanau, Ana Quintino, Francisco Câmara

# Context 
We merged several datasets from the French Public Finance Department open data (governmental institution) to get the data on the French real estate from 2014 to 2022.
Link to the data: https://www.data.gouv.fr/fr/datasets/5c4ae55a634f4117716d5656/#description
We decided to focus on the houses and apartments of the 8 departments in Ile-de-France ((Paris (75), Hauts-de-Seine (92), Seine-Saint-Denis (93), Val-de-Marne (94), Seine-et-Marne (77), Yvelines (78), Essonne (91), Val-d'Oise (95))

# Deliverables 
1) Insights with plots and maps: 
> Difference between property values between 2014 and 2022
 - Map of the property value in 2014
 - Map of the property value in 2022

> The impact of covid-19 on the real estate market in Ile-de-France
 - Evolution of the price per square meter between 2014 and 2022
 - Evolution of the transactions per year by departments between 2014 and 2022

> Which departement has the biggest/smallest Surface area?

2) A price prediction system using Maching learning:
More information below

# Pre-processing & Cleaning explanation
Not a lot of cleaning to do, but there were some duplicates in the datasets and missing values for 2022. 
In the beggining we visualized some observations and insights on the dataframes with the library pandas-profiling, more specifically ProfileReport.

# Maps Creation
We used the GeoPandas library to make the maps, since we had a .geojson file with the France departments.
Then we filtered to have only the 8 Ile-de-France departments.
After that we merged this GeoDataset with the Dataset that has all the property transactions between 2014 and 2022.
For the 2014 and 2022 maps we filtered the dataset by those years and created one for each year.

 
# Machine Learning explanation 
We used the Decision Tree, a Supervised machine learning algorithm that can be used for Regression.
Target: Predict house prices
Variables: Department, Type of house, Area and Number of main rooms 
Optimization: RandomSearch

