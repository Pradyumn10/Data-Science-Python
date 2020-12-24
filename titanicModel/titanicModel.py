#%%
'''
* @Filename : titanicModel.py
* @author : Pradyumn Joshi
* @breif : A machine learning model on titanic dataset predicting the number of people survived using random forest
* @version : 0.1.0
*
* @copyright (c) 2020
'''
#Importing Modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SC
from sklearn.metrics import accuracy_score
print("Modules imported")
#%%
train_file = pd.read_csv('train.csv')
#%%
train_file
#%%
train_file.describe
#%%
#Plot for survived
sns.countplot(train_file['Survived'],color='#36b8c7')
# %%
#Plot for passenger class
sns.countplot(train_file['Pclass'])
# %%
#Checking null data
train_file.isna().sum()
# %%
for val in train_file:
    print(train_file[val].value_counts())
    print()

#%%
#Filling the data using backfil
train_file = train_file.fillna(method='bfill')
# %%
#Creating encoder instance and converting the values
l_encoder=LabelEncoder()
l_encoder.fit(train_file['Sex'])
train_file['Sex'] = l_encoder.transform(train_file['Sex'])
#%%
#Converting value of embarked
def value_emb(frame):
    frame["Embarked"]=frame["Embarked"].map({"S":0,"C":1,"Q":2})
    return frame
#%%
train_file = value_emb(train_file)
#%%
train_file
#%%
train_file = train_file.drop(['Name','Ticket','Cabin'],axis=1)
# %%
train_file = train_file.drop(['Fare'],axis=1)
#%%
train_file
# %%
#Splitting the data in train and validation set
train, valid = tts(train_file, train_size = 0.70, random_state=1)
# %%
def xAndy(f_frame):
    indent = f_frame.drop(["Survived"], axis = 1)
    target = f_frame["Survived"]
    return indent, target
# %%
x_train, y_train = xAndy(train)
x_valid, y_valid = xAndy(valid)
# %%
sc=SC()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_valid = sc.transform(x_valid)
# %%
#Using random forest
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
forest.fit(x_train,y_train)
# %%
#Printing training accuracy
print("Random Forest Training Accuracy : ",forest.score(x_train,y_train)*100)
# %%
accuracy=accuracy_score(y_valid,forest.predict(x_valid))
accuracy = accuracy*100
print('Model Validation Accuracy="{}"'.format(accuracy))
# %%
#Reading the test file
test_file = pd.read_csv("test.csv")
# %%
test_file
#%%
#Checking null data
test_file.isna().sum()
#%%
#Filling the data using backfill
test_file = test_file.fillna(method='bfill')
#%%
#Checking again if some null values is left or not
test_file.isna().sum()
#%%
#Filling null vales using forward fill
test_file = test_file.fillna(method='ffill')
#%%
#Checking again
test_file.isna().sum()
# %%
test_file = test_file.drop(['Name','Ticket','Fare','Cabin'],axis=1)
# %%
test_file
#%%
test_file = value_emb(test_file)
#%%
test_file
#%%
l_encoder.fit(test_file['Sex'])
test_file['Sex'] = l_encoder.transform(test_file['Sex'])
#%%
test_file
# %%
#scalling the data
sc.fit(test_file)
test_file = sc.transform(test_file)
# %%
prediction = forest.predict(test_file)
# %%
print(prediction)
# %%
#Creating submission file
submission = pd.read_csv('test.csv')
#%%
submission['predictions'] = prediction
# %%
submission
# %%
submission = submission.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
# %%
submission
# %%
submission.to_csv('predictions.csv',index=False)
# %%
