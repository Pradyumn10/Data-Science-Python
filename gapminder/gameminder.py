'''
* @FileName : heartHealthPrediction.py
* @Author : Pradyumn Joshi
* @Brief : Given conditions of health predict whether a female can fertile or not
* @Date : 24 Jan 2021
*
* Copyright (C) 2021
'''
#%%
import pandas as pd
# %%
df = pd.read_csv("gapminder.csv")
# %%
df
# %%
y=df["fertility"]
# %%
X=df.drop(['fertility'],axis=1)
# %%
X.isna().sum()
# %%
y.isna().sum()
# %%
X.dtypes
# %%
#Converting the object data type into numeric
#Checking unique values
X["Region"].unique()
#%%
#Doing one hot encoding; it will create 6new columns for 6 values
X=pd.get_dummies(X,columns=["Region"], drop_first=True)
# %%
X
# %%
type(X)
# %%
X.shape
# %%
from sklearn.model_selection import train_test_split as tts 
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3, random_state=42)
# %%
X.head
# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# %%
model.fit(X_train,y_train)
# %%
model.score(X_test, y_test)
# %%
data = pd.DataFrame({"First":[100],"Second":[0.7],"Third":[55],"Fourth":[30],"Fifth":[12000],"Sixth":[200],"Seventh":[70],"Eighth":[130],"Ningth":[0],"Tenth":[0],"Eleventh":[0],"Twelve":[1],"Thirteen":[0]})
# %%
data = scaler.transform(data)
# %%
model.predict(data)
# %%
