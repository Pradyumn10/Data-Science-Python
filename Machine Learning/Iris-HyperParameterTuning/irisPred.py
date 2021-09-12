#%%
from sklearn import svm, datasets
from matplotlib import pyplot as plt
import numpy as np
# %%
iris = datasets.load_iris()
# %%
iris
# %%
import pandas as pd 
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# %%
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x:iris.target_names[x])
# %%
df
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.3)
# %%
model = svm.SVC(kernel='rbf', C=30, gamma='auto')
model.fit(X_train, y_train)
# %%
model.score(X_test,y_test)
# %%
from sklearn.model_selection import cross_val_score
# %%
cross_val_score(svm.SVC(kernel='linear', C=10, gamma='auto'),iris.data,iris.target, cv=5)
# %%
cross_val_score(svm.SVC(kernel='rbf', C=10, gamma='auto'),iris.data,iris.target,cv=5)
# %%
cross_val_score(svm.SVC(kernel='rbf', C=20, gamma='auto'),iris.data,iris.target,cv=5)
# %%
#Doing the same cross val approch with loop
kernels = ['rbf','linear']
C=[1,10,20]
avg_scores = {}
for kval in kernels:
    for Cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval, C=Cval, gamma='auto'),iris.data,iris.target,cv=5)
        avg_scores[kval + ' _ ' +str(Cval)] = np.average(cv_scores)
avg_scores
# %%
#using grid search cv
#it will do the same thing like last cell but it will be efficient as it
#wont use loop
#you can use n-number of parameters and it will give you result
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel' : ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
clf.cv_results_
# %%
df = pd.DataFrame(clf.cv_results_)
# %%
df
# %%
#this will tell the best parameters which we can use
df[['param_C', 'param_kernel','mean_test_score']]
# %%
clf.best_score_
# %%
clf.best_params_
# %%
#Grid Search have one con that is it will be difficult when we will have
#many data. So to overcome this we have Randomize Search
# %%
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},
                        cv = 5,
                        return_train_score=False,
                        n_iter=2
                        )
rs.fit(iris.data,iris.target)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel', 'mean_test_score']]
# %%
#choosing best model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# %%
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
# %%
scores=[]
for model_name, mp in model_params.items():
     clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
     clf.fit(iris.data, iris.target)
     scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df
# %%
