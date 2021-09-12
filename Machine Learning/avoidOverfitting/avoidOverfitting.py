'''
* @Filename : avoidOverfitting.py
* @author : Pradyumn Joshi
* @breif : A logistic regression  model on MNIST fashion dataset avoiding overfitting in the model
* @version : 0.1.0
*
* @copyright (c) 2020
'''
#%%
#importing modules
import pandas as pd
import numpy as np
import seaborn as sns
import time,sys,os,warnings
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score as f1s
from sklearn import metrics
print("Modules Printed")
# %%
#Reading CSV
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
train_data=pd.read_csv("./train.csv")
test_data=pd.read_csv("./test.csv")
print('csv imported')
#%%
#Viewing the data
train_data.head()
# %%
test_data.head()
# %%
train_data.describe()
# %%
test_data.describe()
# %%
#Checking for null data
train_data.isnull().sum()
train_data.isnull().sum().sum()
test_data.isnull().sum()
test_data.isnull().sum().sum()
#Therefore there is no null data

# %%
#Extracting dependent and independent variable
train_data=train_data.drop(["id"],axis=1)
id=test_data["id"]
test_data=test_data.drop(["id"],axis=1)

# %%
#Plotting graph
sns.countplot(train_data["target"])
# %%
import scipy.stats as stats
#  depen = Categorical, inden = Continous ==> pointbiserial test
features = []
alpha = 0.07
for i in train_data.columns:
    target,pvalue = stats.pointbiserialr(train_data["target"],train_data[i])
    
    if pvalue>alpha:
        pass
    else:
        features.append(i)
print(features)
#%%
train_data = train_data[features]
features.pop(0)
test_data = test_data[features]
#%%
#Splitting the data
train, test = tts(train_data, test_size = 0.2, random_state = 1)
def x_and_y(frame):
    inden = frame.drop(["target"], axis = 1)
    target = frame["target"]
    return inden, target

x_train, y_train = x_and_y(train)
x_test , y_test = x_and_y(test)
#%%
#Training
model=LR()
model.fit(x_train,y_train)

train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

train_f1 = f1s(train_pred, y_train)
test_f1 = f1s(test_pred, y_test)

print("\nThe F1 Score for the Training Set is: {:.2f}%\n".format(train_f1*100))
print("\nThe F1 Score for the Testing Set is: {:.2f}%\n".format(test_f1*100))

score_matrix =confusion_matrix(y_test,test_pred)
print("Confusion Matrix for the tested data is :\n",score_matrix,"\n")

#%%
#Predicting For Test Data set
test_pred = model.predict(test_data)
class my_dictionary(dict): 

    # init function 
    def init(self): 
        self = dict() 

    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

result=my_dictionary()

for i in range(len(id)):
    result.add(id[i],test_pred[i])
    
# %%
print(result)
# %%
