#%%
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from word2number import w2n
# %%
df = pd.read_csv('hiring.csv')
# %%
df
# %%
#Handling null values
df=df.rename(columns={'test_score(out of 10)' : 'score1' , 'interview_score(out of 10)':'score2'})
# %%
df
#%%
df['experience']= df.fillna('zero')
#%%
df.experience = df.experience.apply(w2n.word_to_num)
# %%
df
# %%
