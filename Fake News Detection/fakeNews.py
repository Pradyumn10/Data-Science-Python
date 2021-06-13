'''
* @FileName : fakeNews.py
* @Author : Pradyumn Joshi
* @Brief : Doing exploratory data analysis
* @Date : 22 March 2021
*
* Copyright (C) 2021
'''
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
#reading csv files
df = pd.read_csv("./news.csv")
# %%
df
# %%
df.shape
# %% 
df.dtypes
# %%
labels = df.label
# %%
labels
# %%
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
# %%
# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
# %%
#initialize a passive aggresive classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
# %%
#taking predictions 
y_pred = pac.predict(tfidf_test)
# %%
#accuracy
score = accuracy_score(y_test, y_pred)
# %%
print(score)
# %%
#creating confusion matrix
conf = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
# %%
sns.heatmap(conf,annot=True, fmt="d")
# %%
