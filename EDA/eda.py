'''
* @FileName : eda.py
* @Author : Pradyumn Joshi
* @Brief : Doing exploratory data analysis
* @Date : 24 Jan 2021
*
* Copyright (C) 2021
'''
#%%
#Doing exploratory data analysis on data
#importing modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

# %%
#reading csv
df = pd.read_csv("./data.csv")
# %%
df.head()
# %%
#checking data types
df.dtypes
# %%
#dropping irrelevant column
df = df.drop(['Engine Fuel Type', 'Market Category' , 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'], axis=1)
# %%
df
# %%
#Renaming columns
df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price" })
# %%
df
# %%
#dropping duplicate values
df.shape
# %%
duplicate_row_df = df[df.duplicated()]
print("Number of duplicate: ", duplicate_row_df.shape)
# %%
df.count()
# %%
df = df.drop_duplicates()
# %%
df
# %%
df.count()
# %%
#checking null values
print(df.isnull().sum())
# %%
df =df.dropna()
# %%
df.count()
# %%
#detecting outliers
#lower extreme, lower quartile(25%), median, upper quartile(75%), upper extreme, outliers
sns.boxplot(x=df['Price'])
# %%
sns.boxplot(x=df['HP'])
# %%
sns.boxplot(x=df['Cylinders'])
# %%
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
print(IQR)
# %%
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
# %%
#Visualization
#Histogram = used for frequency plotting
df.Make.value_counts().nlargest(40).plot(kind='bar',figsize=(10,5))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make')
# %%
#heart map = finding dependent variables
#correlation values lies in +1 to -1 ; highly correalted to negatively correlated
plt.figure(figsize=(10,5))
c=df.corr() #correlation
sns.heatmap(c,cmap="BrBG", annot=True)
c
# %%
#scatter plot
#finding correlation between two columns
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['Price'])
ax.set_xlabel('HP')
ax.set_ylabel('Price')
plt.show()
# %%
