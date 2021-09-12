#%%
'''
* @Filename : churnModel.py
* @author : Pradyumn Joshi
* @breif : A machine learning model on churn dataset predicting if customer is going to stop using the product or not
* @version : 0.1.0
*
* @copyright (c) 2020
'''
import pandas as pd
import numpy as np
import sys, time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
print("Modules Imported")

#%%
#Reading the file
file_train = pd.read_csv('train.csv')
#%%
file_train
#%%
file_train.describe()

# %%
#Creating Encoder instance and converting their values
l_encoder = LabelEncoder()
#converting target column values
l_encoder.fit(file_train['churn'])
file_train['churn'] = l_encoder.transform(file_train['churn'])

#%%
file_train
#%%
#Plotting the graph for target value
file_train['churn'].value_counts()
sns.countplot(file_train['churn'])
#%%
#Checking for null data
file_train.isnull().sum()
file_train.isnull().sum().sum()
#Therefore there is no null data
# %%
l_encoder.fit(file_train['international_plan'])
file_train['international_plan'] = l_encoder.transform(file_train['international_plan'])
l_encoder.fit(file_train['voice_mail_plan'])
file_train['voice_mail_plan'] = l_encoder.transform(file_train['voice_mail_plan'])

l_encoder.fit(file_train['state'])
file_train['state'] = l_encoder.transform(file_train['state'])
l_encoder.fit(file_train['area_code'])
file_train['area_code'] = l_encoder.transform(file_train['area_code'])

# %%
file_train
# %%
#to remove some data
file_train = file_train.drop(['account_length','number_customer_service_calls'],axis=1)
#%%
file_train
# %%
#Splitting into train data and validation data
train, valid = tts(file_train,train_size = 0.75, random_state = 1)
# %%
def xAndy(f_frame):
    inden = f_frame.drop(["churn"], axis = 1)
    target = f_frame["churn"]
    return inden, target

# %%
x_train, y_train = xAndy(train)
x_valid, y_valid = xAndy(valid)

#%%
#Scale the data
sc=StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_valid=sc.transform(x_valid)
# %%
def models(x_train,y_train):

    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(x_train,y_train)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    knn.fit(x_train,y_train)

    #SVC (linear kernel)
    from sklearn.svm import SVC
    svc_lin=SVC(kernel='linear',random_state=0)
    svc_lin.fit(x_train,y_train)

    #SVC(RBF kernel)
    from sklearn.svm import SVC
    svc_rbf=SVC(kernel='rbf',random_state=0)
    svc_rbf.fit(x_train,y_train)

    #GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gauss=GaussianNB()
    gauss.fit(x_train,y_train)

    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(x_train,y_train)

    #RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(x_train,y_train)
    
    #Print Training Accuracy
    print('[0]Logistic Regression Training Accuracy: ',(log.score(x_train,y_train)*100))
    print('[1]K Neighbours Training Accuracy: ',(knn.score(x_train,y_train)*100))
    print('[2]SVC Linear Training Accuracy: ',(svc_lin.score(x_train,y_train)*100))
    print('[3]SVC RBF Training Accuracy: ',(svc_rbf.score(x_train,y_train)*100))
    print('[4]Gaussian Training Accuracy: ',(gauss.score(x_train,y_train)*100))
    print('[5]Decission Tree Training Accuracy: ',(tree.score(x_train,y_train)*100))
    print('[6]Random Forest Training Accuracy: ',(forest.score(x_train,y_train)*100))


    return log,knn,svc_lin,svc_rbf,gauss,tree,forest
# %%
best_model = models(x_train,y_train)
# %%
#The best training accuracy was of Random Forest with 98%
for i in range(len(best_model)):
     accuracy=accuracy_score(y_valid,best_model[i].predict(x_valid))
     accuracy = accuracy*100
     print('Model[{}] Validation Accuracy="{}"'.format(i,accuracy))
     print()

# %%
#The best accuracy was given by Random Forest

#Reading test file
file_test = pd.read_csv('test.csv')
file_test
# %%
#Encoding the data
l_encoder.fit(file_test['international_plan'])
file_test['international_plan'] = l_encoder.transform(file_test['international_plan'])
l_encoder.fit(file_test['voice_mail_plan'])
file_test['voice_mail_plan'] = l_encoder.transform(file_test['voice_mail_plan'])
l_encoder.fit(file_test['state'])
file_test['state'] = l_encoder.transform(file_test['state'])
l_encoder.fit(file_test['area_code'])
file_test['area_code'] = l_encoder.transform(file_test['area_code'])
# %%
#Dropping data
file_test = file_test.drop(['account_length','number_customer_service_calls'],axis=1)

# %%
file_test
#%%
def x_and_y(frame):
    inden = frame.drop(["id"], axis = 1)
    target = frame["id"]
    return inden, target
file_test, id = x_and_y(file_test)
'''
file_test = file_test.drop(['id'],axis=0)
id = file_test['id']
'''
#%%
#scalling the data
sc.fit(file_test)
file_test = sc.transform(file_test)

#%%
#Random Forest gave the best training and test accuracy
forest = best_model[6]
prediction = forest.predict(file_test)
#%%
print(prediction)
#%%
#Plotting the graph for prediction value
sns.countplot(prediction)
# %%
sub = pd.read_csv('sampleSubmission.csv')
#%%
sub['predictions'] = prediction
#%%
sub
#%%
sub.drop('churn',axis=1)
#%%
sub.to_csv('sample.csv',index=False)
# %%
