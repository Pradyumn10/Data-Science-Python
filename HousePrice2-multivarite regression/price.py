#%%
import pandas as pd 
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('./homeprices.csv')
# %%
df
# %%
#dummy variable & one hot encoding
dummies = pd.get_dummies(df.town)
# %%
dummies
# %%
merged = pd.concat([df,dummies],axis='columns')
# %%
final = merged.drop(['town','west windsor'],axis=1)
# %%
final
# %%
from sklearn.linear_model import LinearRegression
# %%
model = LinearRegression()
# %%
X=final.drop('price',axis=1)
# %%
X
# %%
y=final.price
# %%
y
# %%
model.fit(X,y)
# %%
#m
model.coef_
# %%
#intercept
model.intercept_
# %%
model.predict([[2800,0,1]])
# %%
model.predict([[3400,0,0]])
# %%
model.score(X,y)*100
#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#%%
dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle
#%%
X = dfle[['town','area']].values
#%%
y = dfle.price.values
#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')
# %%
X = ct.fit_transform(X)
X
# %%
X = X[:,1:]

# %%
model.fit(X,y)
# %%
model.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor
# %%
model.predict([[1,0,2800]]) # 2800 sqr ft home in robbinsville
# %%
