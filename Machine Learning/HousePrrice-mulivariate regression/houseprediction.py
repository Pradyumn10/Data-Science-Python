#%%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# %%
#reading csv file
df = pd.read_csv('homeprices.csv')
# %%
df
# %%
#Handling null values
#using median
medium_bedrooms = df.bedrooms.median()
medium_bedrooms
# %%
df.bedrooms = df.bedrooms.fillna(medium_bedrooms)
# %%
df
# %%
#Multivariate Equation : price=m1*area+m2*bedrooms+m3*age
reg = LinearRegression()
# %%
reg.fit(df[['area','bedrooms','age']], df['price'])
# %%
#values of m
reg.coef_
# %%
#values of intecept
reg.intercept_
#Multivariate Equation : price = 112.0624*area+23388.880077*bedrooms-3231.7179*price
# %%
reg.predict([[3000,3,40]])
# %%
