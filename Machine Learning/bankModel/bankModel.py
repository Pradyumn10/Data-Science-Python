'''
* @Filename : bankModel.py
* @author : Pradyumn Joshi
* @breif : A Logistic Regression model on bank dataset
* @version : 0.1.0
*
* @copyright (c) 2020
'''
#%%
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#%%
#Reading CSV file
bank = pd.read_csv('./bank.csv',sep=';')
bank.head()

#%%
bank.shape
bank.describe()

#%%
bank['y'].value_counts()
sns.countplot(bank['y'])

#%%
#Grouping By balance and yes/no
bank["y"] =bank["y"].map({"yes":1, "no":0})
bank.groupby('balance')[['y']].mean()
# %%
#Sorting through balance (0-4000,4000-15000)
balance=pd.cut(bank['balance'],[0,4000,15000])
bank.pivot_table('y',['loan',balance],'age')

#%%
#Checking the total numbers of 
bank.isna().sum()
# %%
#To see the count of each values involved in the column fields in the dataset
for val in bank:
    print(bank[val].value_counts())
    print()

#%%
#Removing unnecessary data
bank=bank.drop(['job','marital','education','contact','campaign'],axis=1)

bank.shape
bank.dtypes

#%%
#Creating Encoder instance and converting their values
l_encoder = LabelEncoder()
#converting default columns values
l_encoder.fit(bank['default'])
bank['default'] = l_encoder.transform(bank['default'])

#converting housing values
l_encoder.fit(bank['housing'])
bank['housing'] = l_encoder.transform(bank['housing'])

#converting loan values
l_encoder.fit(bank['loan'])
bank['loan'] = l_encoder.transform(bank['loan'])

#converting month values
l_encoder.fit(bank['month'])
bank['month'] = l_encoder.transform(bank['month'])

#converting poutcome values
l_encoder.fit(bank['poutcome'])
bank['poutcome'] = l_encoder.transform(bank['poutcome'])

#%%
bank.head()

#%%
bank.dtypes

#%%
#Splitting data into dependent 'x' and 'y' variables
x=bank.iloc[:,1:8].values
y=bank.iloc[:,0].values

train, test = tts(bank,test_size=0.2, random_state=1)

#%%
#%%
cols=['age','default','housing','loan','previous','poutcome']
n_rows=3
n_cols=2
fig,axs=plt.subplots(n_rows,n_cols,figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):
        i=r*n_cols+c
        ax=axs[r][c]
        sns.countplot(train[cols[i]],hue=train['y'],ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="y",loc="upper right")
plt.tight_layout()

#%%
def xAndy(f_data):
    inden = f_data.drop(['y'],axis = 1)
    target = f_data['y']
    return inden, target

x_train, y_train = xAndy(train)
x_test, y_test = xAndy(test)

#%%
#scale the data
l_sc = StandardScaler()
l_sc.fit(x_train)
x_train = l_sc.transform(x_train)
x_test = l_sc.transform(x_test)

#%%
#creating instance of logistic regression and training it with the data
l_regression = LogisticRegression(random_state=0)
l_regression.fit(x_train, y_train)

#print('Logistic Regression Training Accuracy: {}% '.format(l_regression.score(x_train,y_train)*100))

#%%
cm=confusion_matrix(y_test,l_regression.predict(x_test))

#Extract TrueNegative,TruePsitive,FalseNegative,FalsePositive
TN,FP,FN,TP=confusion_matrix(y_test,l_regression.predict(x_test)).ravel()
test_score=((TP+TN)/(TP+TN+FP+FN))*100

print(cm)
#print('Logistic Regeression Testing Accuracy="{}%"'.format(test_score))
print()
    
#%%
#Compairing the values with the actual values
l_prediction = l_regression.predict(x_test)
print(l_prediction)
print()

print(y_test)

#%%
#Checking the model
l_modelCheck = [[45,0,844,0,0,5,6,1018,-1,0,3]]
l_survival = l_sc.transform(l_modelCheck)

#%%
#using Logistic Regeression to predict
print('Logistic Regression Training Accuracy: {}% '.format(l_regression.score(x_train,y_train)*100))
print('Logistic Regeression Testing Accuracy="{}%"'.format(test_score))
pred = l_regression.predict(l_survival)
#print(pred)

#%%
if pred==0:
    print("Sorry! The answer is NO for data {}".format(l_modelCheck))
else:
    print("The answer is yes!! \n Enjoy for data {}".format(l_modelCheck))
# %%