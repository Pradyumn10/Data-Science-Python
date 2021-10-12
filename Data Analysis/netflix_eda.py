# -*- coding: utf-8 -*-
"""Netflix EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G9ONl41H1DG2CxO9S6m-Rq6B_PZQcCXx
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat
import plotly.express as px #used for graphs

import warnings
sns.set(style='darkgrid', font_scale=1.2, color_codes=True)
warnings.filterwarnings('ignore')

raw_data = pd.read_csv('https://raw.githubusercontent.com/Pradyumn10/Data-Science-Python/master/Data%20Analysis/netflix_titles.csv')

raw_data.head()

#checking data types of the column
raw_data.dtypes

#some data is missing (director, cast, country etc)
raw_data.describe()

#data preprocssing
raw_data.info()

#converting date to formatted date
raw_data["date_added"] = pd.to_datetime(raw_data['date_added'])
raw_data['year_added'] = raw_data['date_added'].dt.year

raw_data

raw_data.shape

raw_data.info()

raw_data.columns

raw_data.isnull().sum()

#Data Cleaning and Filling
#Filling the null values by dropping some and filling some

#plotting missing value graph
missing = raw_data.isnull().sum().to_frame()

missing

plt.figure(figsize = (10, 7))
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : '22'}
plt.rc('font', **font)
missing.plot( title="Missing Values", figsize=(12,7), kind="bar")

(raw_data.country).mode()

raw_data.year_added.mode()

#Filling the data with mode for country and year_added
#Filling the columns for director and rating with DNK and TV_MA 
#dropping date added, description and cast
raw_data['rating'] = raw_data['rating'].fillna('TV-MA')
raw_data['country'] = raw_data['country'].fillna('United States')
raw_data['year_added'] = raw_data['year_added'].fillna(2019)
raw_data['director'] = raw_data['director'].fillna('DNK')

raw_data.drop(['date_added'],axis=1,inplace=True)
raw_data.drop([ 'description'], axis=1, inplace=True)
raw_data.dropna(subset =['cast'],inplace=True)

raw_data.isnull().sum()

raw_data.info()

df=raw_data.reset_index()

#Exploratory Data Analysis
print(df['release_year'].min(), df['release_year'].max())

df_tv_show = df[df['type'] == 'TV Show']
df_movies = df[df['type'] == 'Movie']

df

font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : '22'}
plt.rc( 'font',**font)

#checking number of movies and TV shows
vs = df.type.value_counts()
vs.plot.bar(color  ="blue")
plt.xticks(rotation = 0)
plt.title("Movies VS Tv shows")
plt.show()

vs.plot.pie( autopct='%1.1f%%', shadow=True, startangle=90 , explode = (0.07,0.14) )
plt.show()

#contry analysis..comparing which country produces more content
df['country'] = df['country'].apply(lambda x: x.split(",")[0])

cont = df.country.value_counts().head(10)
plt.figure(figsize = (17, 10) )
cont.plot.bar(color = 'burlywood' )
plt.xticks(rotation = 0)
plt.xlabel("COUNTRIES")
plt.ylabel("NUMBER OF MOVIES")
plt.title("TOP 10 CONTENT PRODUCING COUNTRIES")
plt.show()

sns.displot(df,x='release_year',hue='type', height=6 ,  aspect=2)
plt.title('Released year vs Type of content')
plt.show()

#checking release years
fig = px.strip(df, x='release_year', y="type", orientation="h", color="type")
fig.show()

#content and rating
sns.displot(df,x='rating', hue ="type",height=6 ,  aspect=2  )
plt.title('RATINGS Vs COUNT')
plt.xlabel("RATINGS")
plt.ylabel("Count")

#Replacing rating index for better understanding
df['rating'] = df['rating'].replace({'TV-PG': 'Kids','TV-MA': 'Adults','TV-Y7-FV':'Kids','TV-Y7': 'Kids','TV-14': 'Teens','R': 'Adults',
                                         'TV-Y': 'Kids','NR': 'Adults','PG-13': 'Teens', 'TV-G': 'Kids', 
                                         'PG': 'Kids', 'G': 'Kids','UR': 'Adults',
                                         'NC-17': 'Adults'})

df.rating.value_counts()

sns.displot(df,x='rating', hue ="type",height=5,  aspect=2 )
plt.title('RATINGS Vs COUNT')
plt.xlabel("RATINGS")
plt.ylabel("COUNT")

#There is wrong data entered in rating i.e duration data is entered in rating which are 74min, 84min and 66min
#Popular Actors
plt.figure(figsize = (11, 8))
df_movies.cast.value_counts().head(4).plot.bar(color = 'grey')
plt.xlabel("ACTORS")
plt.xticks(rotation =0)
plt.yticks([int(n) for n in range(0,12,2)])
plt.ylabel("Count ")
plt.title("Popular Actors in Movies " )
plt.show()

#splitting genre because there are more genre seperated by ','
df['genre'] = df['listed_in'].apply(lambda x: x.split(",")[0] )

plt.figure(figsize= (10 ,8))
genre = df.genre.value_counts() 
genre

explode = [0.19, 0.11, 0.11, 0.11, 0.11]
colors = ['red' , 'blue' , 'lime','orange' ,'skyblue']

df.genre.value_counts()[:5].plot(kind='pie',figsize=(6,6),title = "TOP 5 GENRES"  , autopct='%.0f%%', shadow = True,startangle = 120,colors = colors, explode = explode  )

dir = df.director.value_counts()[1:11]
dir.plot(kind='bar',figsize=(12,7), title= "TOP 10 Directors based on number of Contents" , color = 'yellow')
plt.ylabel("Count ")
plt.xlabel("DIRECTORS")
plt.yticks([int(n) for n in range(0,22,2)] )
plt.show()
dir

#genres and ratings
fig = px.density_heatmap(df, x="genre", y="rating", marginal_x="rug", marginal_y="histogram" )
plt.figure(figsize= (16,10))
fig.show()

fig = px.parallel_categories(df,dimensions=['type', 'rating'])
fig.show()

