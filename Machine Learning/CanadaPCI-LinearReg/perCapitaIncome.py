#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# %%
#reading the csv file
df = pd.read_csv('canada_per_capita_income.csv')
# %%
df
#%%
#renaming the column
df=df.rename(columns={'per capita income (US$)':'pci'})
# %%
#Visualization of points
plt.xlabel("year")
plt.ylabel("CANADA-Per Capita Income")
plt.scatter(df['year'],df['pci'],color='blue',marker='*')
# %%
#Creating model
model = LinearRegression()
# %%
model.fit(df[['year']],df['pci'])
# %%
#m
model.coef_
# %%
#x
model.intercept_
# Linear Equation
# pci=m*year+c
# pci=828.465*year-1632210.7578554575
# %%
model.score(df[['year']],df['pci'])
# %%
#Plotting best fit line of model
plt.xlabel("year")
plt.ylabel("CANADA-Per Capita Income")
plt.scatter(df['year'],df['pci'],color='blue',marker='*')
plt.plot(df['year'],model.predict(df[['year']]),color='yellow')
# %%
model.predict([[2016]])
# %%
