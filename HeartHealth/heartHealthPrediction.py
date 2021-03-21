#%%
'''
* @FileName : heartHealthPrediction.py
* @Author : Pradyumn Joshi
* @Brief : Given conditions of health predict whether a person has some heart disease using Decision Tree
* @Date : 24 Jan 2021
*
* Copyright (C) 2021
'''
#%%
'''
1. Extract Features
    a. Features and target should not have null value
    b. Features should be numeric in nature
    c. Features should be of the type array/dataframe
    d. Features should have some rows and columns  
2. Split the dataset into training and testing datasets.
    e. Features should be on same scale
3. Train the model on training dataset
4. Test the model on testing dataset
'''
#%%
import pandas as pd
import time
# %%
s_time = time.time()
mainFile = pd.read_csv("heart.csv")
# %%
mainFile
# %%
y = mainFile["target"]
# %%
X = mainFile.drop(['target'],axis=1)
# %%
X
#%%
X=X.drop(["sex"],axis=1)
# %%
#Checking for null values
X.isna().sum()
# %%
y.isna().sum()
# %%
X.dtypes
# %%
type(X)
# %%
X.shape

# %%
#Splitting the data into train and test
from sklearn.model_selection import train_test_split as tts 
X_train,X_test, y_train, y_test = tts(X,y, test_size=0.25, random_state=15, stratify=y) 
# %%
from sklearn.preprocessing import MinMaxScaler as mms
scaler = mms()
# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=76)
# %%
model.fit(X_train,y_train)
# %%
score = model.score(X_test,y_test)*100
#%%
print("Model Accuracy Score : ", score)
#%%
data = pd.DataFrame({0:[59], 1:[1], 2:[2], 3:[135], 4:[244], 5:[0], 6:[1], 7:[179], 8:[0], 9:[1.0],10:[1],11:[1],12:[2]})
# %%
data
# %%
data = data.drop(1, axis=1)
# %%
data
# %%
model.predict(data)
# %%
print("Time Consumed : ",(time.time()-s_time))
# %%
