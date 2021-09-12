'''
* @FileName : irisSpeciePrediction.py
* @Author : Pradyumn Joshi
* @Brief : Model on Iris dataset predicting its species. There are 3 types of species in iris flower
* @Date : 23 Jan 2020
*
* @copyright (c) 2020

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
#%%
df = pd.read_csv("Iris.csv")
# %%
df
# %%
#Prediciting this column ;so we drop for training our model
y=df["Species"]
# %%
X = df.drop(['Species'],axis=1)

# %%
X.head()
# %%
#Dropping unwanted column
X=X.drop("Id",axis=1)

# %%
X
# %%
#Checking for null values
X.isna().sum()
# %%
y.isna().sum()
# %%
#Checking if the model is having all numeric values
X.dtypes
# %%
type(X)
#%%
X.shape
# %%
#Splitting the data
from sklearn.model_selection import train_test_split as tts
X_train,X_test, y_train, y_test = tts(X,y,test_size=0.3, random_state=42,stratify=y)

# %%
X.describe()
# %%
from sklearn.preprocessing import MinMaxScaler as mms 
scaler = mms()
# %%
X_train = scaler.fit_transform(X_train)
# %%
X_test = scaler.transform(X_test)
# %%
#Importing the model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
# %%
model.fit(X_train,y_train)
# %%
model.score(X_test, y_test)*100
# %%
#Predicting
data = pd.DataFrame({0:[4.5,5.7], 1:[9.4,4],2:[1.9,2],3:[0.6,0.9]})
# %%
data
# %%
data = scaler.transform(data)
# %%
model.predict(data)
# %%
